# Import Python Standard Library dependencies
import datetime
import json
import math
import multiprocessing
import random
from functools import partial
from pathlib import Path

# Import matplotlib for creating plots
import matplotlib.pyplot as plt
# Import the pandas package
import pandas as pd
# Import utility functions
from cjm_pil_utils.core import resize_img, get_img_files, stack_imgs
from cjm_pytorch_utils.core import tensor_to_pil, get_torch_device, set_seed, move_data_to_device
from cjm_torchvision_tfms.core import ResizeMax, PadSquare, CustomRandomIoUCrop
# Import the distinctipy module
from distinctipy import distinctipy

# Import numpy

# Set options for Pandas DataFrame display
pd.set_option('max_colwidth', None)  # Do not truncate the contents of cells in the DataFrame
pd.set_option('display.max_rows', None)  # Display all rows in the DataFrame
pd.set_option('display.max_columns', None)  # Display all columns in the DataFrame

# Import PIL for image manipulation
from PIL import Image

# Import PyTorch dependencies
import torch
from torch.amp import autocast
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision

torchvision.disable_beta_transforms_warning()
from torchvision.tv_tensors import BoundingBoxes, Mask
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks
import torchvision.transforms.v2 as transforms

# Import Mask R-CNN
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

# Import tqdm for progress bar
from tqdm.auto import tqdm

from windows_utils import create_polygon_mask
from windows_utils import LabelMeDataset

def run_epoch(model, dataloader: DataLoader, optimizer, lr_scheduler, device, scaler, epoch_id, is_training):
    """
    Function to run a single training or evaluation epoch.

    Args:
        model: A PyTorch model to train or evaluate.
        dataloader: A PyTorch DataLoader providing the data.
        optimizer: The optimizer to use for training the model.
        loss_func: The loss function used for training.
        device: The device (CPU or GPU) to run the model on.
        scaler: Gradient scaler for mixed-precision training.
        is_training: Boolean flag indicating whether the model is in training or evaluation mode.

    Returns:
        The average loss for the epoch.
    """
    # Set the model to training mode
    model.train()

    epoch_loss = 0  # Initialize the total loss for this epoch
    progress_bar = tqdm(total=len(dataloader), desc="Train" if is_training else "Eval")  # Initialize a progress bar

    # Loop over the data
    for batch_id, (inputs, targets) in enumerate(dataloader):
        # Move inputs and targets to the specified device
        inputs = torch.stack(inputs).to(device)

        # Forward pass with Automatic Mixed Precision (AMP) context manager
        with autocast(torch.device(device).type):
            if is_training:
                losses = model(inputs.to(device), move_data_to_device(targets, device))
            else:
                with torch.no_grad():
                    losses = model(inputs.to(device), move_data_to_device(targets, device))

            # Compute the loss
            loss = sum([loss for loss in losses.values()])  # Sum up the losses

        # If in training mode, backpropagate the error and update the weights
        if is_training:
            if scaler:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                old_scaler = scaler.get_scale()
                scaler.update()
                new_scaler = scaler.get_scale()
                if new_scaler >= old_scaler:
                    lr_scheduler.step()
            else:
                loss.backward()
                optimizer.step()
                lr_scheduler.step()

            optimizer.zero_grad()

        # Update the total loss
        loss_item = loss.item()
        epoch_loss += loss_item

        # Update the progress bar
        progress_bar_dict = dict(loss=loss_item, avg_loss=epoch_loss / (batch_id + 1))
        if is_training:
            progress_bar_dict.update(lr=lr_scheduler.get_last_lr()[0])
        progress_bar.set_postfix(progress_bar_dict)
        progress_bar.update()

        # If loss is NaN or infinity, stop training
        if is_training:
            stop_training_message = f"Loss is NaN or infinite at epoch {epoch_id}, batch {batch_id}. Stopping training."
            assert not math.isnan(loss_item) and math.isfinite(loss_item), stop_training_message

    # Cleanup and close the progress bar
    progress_bar.close()

    # Return the average loss for this epoch
    return epoch_loss / (batch_id + 1)


def train_loop(model,
               train_dataloader,
               valid_dataloader,
               optimizer,
               lr_scheduler,
               device,
               epochs,
               checkpoint_path,
               use_scaler=False):
    """
    Main training loop.

    Args:
        model: A PyTorch model to train.
        train_dataloader: A PyTorch DataLoader providing the training data.
        valid_dataloader: A PyTorch DataLoader providing the validation data.
        optimizer: The optimizer to use for training the model.
        lr_scheduler: The learning rate scheduler.
        device: The device (CPU or GPU) to run the model on.
        epochs: The number of epochs to train for.
        checkpoint_path: The path where to save the best model checkpoint.
        use_scaler: Whether to scale graidents when using a CUDA device

    Returns:
        None
    """
    # Initialize a gradient scaler for mixed-precision training if the device is a CUDA GPU
    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' and use_scaler else None
    best_loss = float('inf')  # Initialize the best validation loss

    # Loop over the epochs
    for epoch in tqdm(range(epochs), desc="Epochs"):
        # Run a training epoch and get the training loss
        train_loss = run_epoch(model, train_dataloader, optimizer, lr_scheduler, device, scaler, epoch,
                               is_training=True)
        # Run an evaluation epoch and get the validation loss
        with torch.no_grad():
            valid_loss = run_epoch(model, valid_dataloader, None, None, device, scaler, epoch, is_training=False)

        # If the validation loss is lower than the best validation loss seen so far, save the model checkpoint
        if valid_loss < best_loss:
            best_loss = valid_loss
            torch.save(model.state_dict(), checkpoint_path)

            # Save metadata about the training process
            training_metadata = {
                'epoch': epoch,
                'train_loss': train_loss,
                'valid_loss': valid_loss,
                'learning_rate': lr_scheduler.get_last_lr()[0],
                'model_architecture': model.name
            }
            with open(Path(checkpoint_path.parent / 'training_metadata.json'), 'w') as f:
                json.dump(training_metadata, f)

    # If the device is a GPU, empty the cache
    if device.type != 'cpu':
        getattr(torch, device.type).empty_cache()


def load_data_and_visualize():
    # Set the seed for generating random numbers in PyTorch, NumPy, and Python's random module.
    seed = 1234
    set_seed(seed)



    project_name = "mask-rcnn-elba"
    dataset_name = 'elba/test_ann_labelme'

    dataset_dir = Path("datasets")
    project_dir = Path(f"out/{project_name}/")
    project_dir.mkdir(parents=True, exist_ok=True)
    dataset_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = Path(f'{dataset_dir}/{dataset_name}/')

    # Get a list of image files in the dataset
    img_file_paths = get_img_files(dataset_path)
    # Get a list of JSON files in the dataset
    annotation_file_paths = list(dataset_path.glob('*.json'))

    # Create a dictionary that maps file names to file paths
    img_dict = {file.stem: file for file in img_file_paths}

    # Print the number of image files
    print(f"Number of Images: {len(img_dict)}")

    # Create a generator that yields Pandas DataFrames containing the data from each JSON file
    cls_dataframes = (pd.read_json(f, orient='index').transpose() for f in tqdm(annotation_file_paths))

    # Concatenate the DataFrames into a single DataFrame
    annotation_df = pd.concat(cls_dataframes, ignore_index=False)

    # Assign the image file name as the index for each row
    annotation_df['index'] = annotation_df.apply(lambda row: row['imagePath'].split('.')[0], axis=1)
    annotation_df = annotation_df.set_index('index')

    # Keep only the rows that correspond to the filenames in the 'img_dict' dictionary
    annotation_df = annotation_df.loc[list(img_dict.keys())]

    # Explode the 'shapes' column in the annotation_df dataframe
    # Convert the resulting series to a dataframe and rename the 'shapes' column to 'shapes'
    # Apply the pandas Series function to the 'shapes' column of the dataframe
    shapes_df = annotation_df['shapes'].explode().to_frame().shapes.apply(pd.Series)

    # Get a list of unique labels in the 'annotation_df' DataFrame
    class_names = shapes_df['label'].unique().tolist()

    # class_counts = shapes_df['label'].value_counts()
    # # Plot the distribution
    # class_counts.plot(kind='bar')
    # plt.title('Class distribution')
    # plt.ylabel('Count')
    # plt.xlabel('Classes')
    # plt.xticks(range(len(class_counts.index)), class_names, rotation=75)  # Set the x-axis tick labels
    # plt.show()

    # Prepend a `background` class to the list of class names
    class_names = ['background'] + class_names

    # Generate a list of colors with a length equal to the number of labels
    colors = distinctipy.get_colors(len(class_names))

    # Make a copy of the color map in integer format
    int_colors = [tuple(int(c * 255) for c in color) for color in colors]

    # Set the name of the font file
    font_file = 'data/roboto.ttf'
    #
    # # Download the font file
    # download_file(f"https://fonts.gstatic.com/s/roboto/v30/{font_file}", "./")

    draw_bboxes = partial(draw_bounding_boxes, fill=False, width=2, font=font_file, font_size=25)

    # # Get the file ID of the first image file
    # file_id = list(img_dict.keys())[56]
    #
    # # Open the associated image file as a RGB image
    # sample_img = Image.open(img_dict[file_id]).convert('RGB')


    # Extract the labels for the sample
    # labels = [shape['label'] for shape in annotation_df.loc[file_id]['shapes']]
    # # Extract the polygon points for segmentation mask
    # shape_points = [shape['points'] for shape in annotation_df.loc[file_id]['shapes']]
    # # Format polygon points for PIL
    # xy_coords = [[tuple(p) for p in points] for points in shape_points]
    # # Generate mask images from polygons
    # mask_imgs = [create_polygon_mask(sample_img.size, xy) for xy in xy_coords]
    # Convert mask images to tensors
    # masks = torch.concat([Mask(transforms.PILToTensor()(mask_img), dtype=torch.bool) for mask_img in mask_imgs])
    # Generate bounding box annotations from segmentation masks
    # bboxes = torchvision.ops.masks_to_boxes(masks)

    # Annotate the sample image with segmentation masks
    # annotated_tensor = draw_segmentation_masks(
    #     image=transforms.PILToTensor()(sample_img),
    #     masks=masks,
    #     alpha=0.3,
    #     colors=[int_colors[i] for i in [class_names.index(label) for label in labels]]
    # )
    #
    # # Annotate the sample image with labels and bounding boxes
    # annotated_tensor = draw_bboxes(
    #     image=annotated_tensor,
    #     boxes=bboxes,
    #     labels=labels,
    #     colors=[int_colors[i] for i in [class_names.index(label) for label in labels]]
    # )
    #
    # plt.imshow(tensor_to_pil(annotated_tensor))
    # plt.show()
    # ------------------------------------------
    # Initialize a Mask R-CNN model with pretrained weights
    device = get_torch_device()
    dtype = torch.float32
    model = maskrcnn_resnet50_fpn_v2(weights='DEFAULT')

    # Get the number of input features for the classifier
    in_features_box = model.roi_heads.box_predictor.cls_score.in_features
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels

    # Get the numbner of output channels for the Mask Predictor
    dim_reduced = model.roi_heads.mask_predictor.conv5_mask.out_channels

    # Replace the box predictor
    model.roi_heads.box_predictor = FastRCNNPredictor(in_channels=in_features_box, num_classes=len(class_names))

    # Replace the mask predictor
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_channels=in_features_mask, dim_reduced=dim_reduced,
                                                       num_classes=len(class_names))

    # Set the model's device and data type
    model.to(device=device, dtype=dtype)

    # Add attributes to store the device and model name for later reference
    model.device = device
    model.name = 'maskrcnn_resnet50_fpn_v2'

    # test_inp = torch.randn(1, 3, 256, 256).to(device)
    #
    # summary_df = markdown_to_pandas(f"{get_module_summary(model.eval(), [test_inp])}")
    #
    # # # Filter the summary to only contain Conv2d layers and the model
    # summary_df = summary_df[summary_df.index == 0]
    #
    # summary_df.drop(['In size', 'Out size', 'Contains Uninitialized Parameters?'], axis=1)
    # print(summary_df)

    # Get the list of image IDs
    img_keys = list(img_dict.keys())

    # Shuffle the image IDs
    random.shuffle(img_keys)

    # Define the percentage of the images that should be used for training
    train_pct = 0.8
    # val_pct = 0.2

    # Calculate the index at which to split the subset of image paths into training and validation sets
    train_split = int(len(img_keys) * train_pct)
    # val_split = int(len(img_keys) * (train_pct + val_pct))

    # Split the subset of image paths into training and validation sets
    train_keys = img_keys[:train_split]
    val_keys = img_keys[train_split:]

    # Set training image size
    train_sz = 512

    # Create a RandomIoUCrop object
    iou_crop = CustomRandomIoUCrop(min_scale=0.3,
                                   max_scale=1.0,
                                   min_aspect_ratio=0.5,
                                   max_aspect_ratio=2.0,
                                   sampler_options=[0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
                                   trials=400,
                                   jitter_factor=0.25)

    # Create a `ResizeMax` object
    resize_max = ResizeMax(max_sz=train_sz)

    # Create a `PadSquare` object
    pad_square = PadSquare(shift=True, fill=0)

    # Extract the labels for the sample
    # labels = [shape['label'] for shape in annotation_df.loc[file_id]['shapes']]
    # # Extract the polygon points for segmentation mask
    # shape_points = [shape['points'] for shape in annotation_df.loc[file_id]['shapes']]
    # # Format polygon points for PIL
    # xy_coords = [[tuple(p) for p in points] for points in shape_points]
    # # Generate mask images from polygons
    # mask_imgs = [create_polygon_mask(sample_img.size, xy) for xy in xy_coords]
    # Convert mask images to tensors
    # masks = torch.concat([Mask(transforms.PILToTensor()(mask_img), dtype=torch.bool) for mask_img in mask_imgs])
    # # Generate bounding box annotations from segmentation masks
    # bboxes = BoundingBoxes(data=torchvision.ops.masks_to_boxes(masks), format='xyxy', canvas_size=sample_img.size[::-1])
    #
    # # Get colors for dataset sample
    # sample_colors = [int_colors[i] for i in [class_names.index(label) for label in labels]]

    # Prepare mask and bounding box targets
    # targets = {
    #     'masks': Mask(masks),
    #     'boxes': bboxes,
    #     'labels': torch.Tensor([class_names.index(label) for label in labels])
    # }
    #
    # # Crop the image
    # cropped_img, targets = iou_crop(sample_img, targets)

    # Resize the image
    # resized_img, targets = resize_max(cropped_img, targets)
    #
    # # Pad the image
    # padded_img, targets = pad_square(resized_img, targets)

    # Ensure the padded image is the target size
    # resize = transforms.Resize([train_sz] * 2, antialias=True)
    # resized_padded_img, targets = resize(padded_img, targets)
    # sanitized_img, targets = transforms.SanitizeBoundingBoxes()(resized_padded_img, targets)
    #
    # # Annotate the sample image with segmentation masks
    # annotated_tensor = draw_segmentation_masks(
    #     image=transforms.PILToTensor()(sanitized_img),
    #     masks=targets['masks'],
    #     alpha=0.3,
    #     colors=sample_colors
    # )

    # Annotate the sample image with labels and bounding boxes
    # annotated_tensor = draw_bboxes(
    #     image=annotated_tensor,
    #     boxes=targets['boxes'],
    #     labels=[class_names[int(label.item())] for label in targets['labels']],
    #     colors=sample_colors
    # )

    # # Display the annotated image
    # plt.imshow(tensor_to_pil(annotated_tensor))
    # plt.show()
    # print(pd.Series({
    #     "Source Image:": sample_img.size,
    #     "Cropped Image:": cropped_img.size,
    #     "Resized Image:": resized_img.size,
    #     "Padded Image:": padded_img.size,
    #     "Resized Padded Image:": resized_padded_img.size,
    # }).to_frame())



    # Compose transforms for data augmentation
    data_aug_tfms = transforms.Compose(
        transforms=[
            iou_crop,
            transforms.ColorJitter(
                brightness=(0.875, 1.125),
                contrast=(0.5, 1.5),
                saturation=(0.5, 1.5),
                hue=(-0.05, 0.05),
            ),
            transforms.RandomGrayscale(),
            transforms.RandomEqualize(),
            transforms.RandomPosterize(bits=3, p=0.5),
            transforms.RandomHorizontalFlip(p=0.5),
        ],
    )

    # Compose transforms to resize and pad input images
    resize_pad_tfm = transforms.Compose([
        resize_max,
        pad_square,
        transforms.Resize([train_sz] * 2, antialias=True)
    ])

    # Compose transforms to sanitize bounding boxes and normalize input data
    final_tfms = transforms.Compose([
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.SanitizeBoundingBoxes(),
    ])

    # Define the transformations for training and validation datasets
    train_tfms = transforms.Compose([
        data_aug_tfms,
        resize_pad_tfm,
        final_tfms
    ])
    valid_tfms = transforms.Compose([resize_pad_tfm, final_tfms])

    # Create a mapping from class names to class indices
    class_to_idx = {c: i for i, c in enumerate(class_names)}

    # Instantiate the datasets using the defined transformations
    train_dataset = LabelMeDataset(train_keys, annotation_df, img_dict, class_to_idx, train_tfms)
    valid_dataset = LabelMeDataset(val_keys, annotation_df, img_dict, class_to_idx, valid_tfms)

    dataset_sample = train_dataset[0]

    # Get colors for dataset sample
    sample_colors = [int_colors[int(i.item())] for i in dataset_sample[1]['labels']]

    # Annotate the sample image with segmentation masks
    annotated_tensor = draw_segmentation_masks(
        image=(dataset_sample[0] * 255).to(dtype=torch.uint8),
        masks=dataset_sample[1]['masks'],
        alpha=0.3,
        colors=sample_colors
    )

    # Annotate the sample image with bounding boxes
    annotated_tensor = draw_bboxes(
        image=annotated_tensor,
        boxes=dataset_sample[1]['boxes'],
        labels=[class_names[int(i.item())] for i in dataset_sample[1]['labels']],
        colors=sample_colors
    )

    plt.imshow(tensor_to_pil(annotated_tensor))
    plt.show()

    quit()
    dataset_sample = valid_dataset[0]

    # Get colors for dataset sample
    sample_colors = [int_colors[int(i.item())] for i in dataset_sample[1]['labels']]

    # Annotate the sample image with segmentation masks
    annotated_tensor = draw_segmentation_masks(
        image=(dataset_sample[0] * 255).to(dtype=torch.uint8),
        masks=dataset_sample[1]['masks'],
        alpha=0.3,
        colors=sample_colors
    )

    # Annotate the sample image with bounding boxes
    annotated_tensor = draw_bboxes(
        image=annotated_tensor,
        boxes=dataset_sample[1]['boxes'],
        labels=[class_names[int(i.item())] for i in dataset_sample[1]['labels']],
        colors=sample_colors
    )

    # plt.imshow(tensor_to_pil(annotated_tensor))
    # plt.show()

    from windows_utils import tuple_batch

    # Set the training batch size
    bs = 4

    # Set the number of worker processes for loading data.
    num_workers = multiprocessing.cpu_count() // 2

    # Define parameters for DataLoader
    data_loader_params = {
        'batch_size': bs,  # Batch size for data loading
        'num_workers': num_workers,  # Number of subprocesses to use for data loading
        'persistent_workers': True,
        # If True, the data loader will not shutdown the worker processes after a dataset has been consumed once. This allows to maintain the worker dataset instances alive.
        'pin_memory': 'cuda' in device,
        # If True, the data loader will copy Tensors into CUDA pinned memory before returning them. Useful when using GPU.
        'pin_memory_device': device if 'cuda' in device else '',
        # Specifies the device where the data should be loaded. Commonly set to use the GPU.
        'collate_fn': tuple_batch,
    }

    # Create DataLoader for training data. Data is shuffled for every epoch.
    train_dataloader = DataLoader(train_dataset, **data_loader_params, shuffle=True)

    # Create DataLoader for validation data. Shuffling is not necessary for validation data.
    valid_dataloader = DataLoader(valid_dataset, **data_loader_params)

    # Generate timestamp for the training session (Year-Month-Day_Hour_Minute_Second)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Create a directory to store the checkpoints if it does not already exist
    checkpoint_dir = Path(project_dir / f"{timestamp}")

    # Create the checkpoint directory if it does not already exist
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # The model checkpoint path
    checkpoint_path = checkpoint_dir / f"{model.name}.pth"

    print(checkpoint_path)

    # Create a color map and write it to a JSON file
    color_map = {'items': [{'label': label, 'color': color} for label, color in zip(class_names, colors)]}
    with open(f"{checkpoint_dir}/{dataset_path.name}-colormap.json", "w") as file:
        json.dump(color_map, file)

    # Print the name of the file that the color map was written to
    print(f"{checkpoint_dir}/{dataset_path.name}-colormap.json")

    # Learning rate for the model
    lr = 5e-4

    # Number of training epochs
    epochs = 40

    # AdamW optimizer; includes weight decay for regularization
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # Learning rate scheduler; adjusts the learning rate during training
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,
                                                       max_lr=lr,
                                                       total_steps=epochs * len(train_dataloader))

    train_loop(model=model,
               train_dataloader=train_dataloader,
               valid_dataloader=valid_dataloader,
               optimizer=optimizer,
               lr_scheduler=lr_scheduler,
               device=torch.device(device),
               epochs=epochs,
               checkpoint_path=checkpoint_path,
               use_scaler=True)

    # Choose a random item from the validation set
    file_id = random.choice(val_keys)

    # Retrieve the image file path associated with the file ID
    test_file = img_dict[file_id]

    # Open the test file
    test_img = Image.open(test_file).convert('RGB')

    # Resize the test image
    input_img = resize_img(test_img, target_sz=train_sz, divisor=1)

    # Calculate the scale between the source image and the resized image
    min_img_scale = min(test_img.size) / min(input_img.size)

    plt.imshow(test_img)
    plt.show()

    # Extract the polygon points for segmentation mask
    target_shape_points = [shape['points'] for shape in annotation_df.loc[file_id]['shapes']]
    # Format polygon points for PIL
    target_xy_coords = [[tuple(p) for p in points] for points in target_shape_points]
    # Generate mask images from polygons
    target_mask_imgs = [create_polygon_mask(test_img.size, xy) for xy in target_xy_coords]
    # Convert mask images to tensors
    target_masks = Mask(
        torch.concat([Mask(transforms.PILToTensor()(mask_img), dtype=torch.bool) for mask_img in target_mask_imgs]))

    # Get the target labels and bounding boxes
    target_labels = [shape['label'] for shape in annotation_df.loc[file_id]['shapes']]
    target_bboxes = BoundingBoxes(data=torchvision.ops.masks_to_boxes(target_masks), format='xyxy',
                                  canvas_size=test_img.size[::-1])

    # Set the model to evaluation mode
    model.eval()

    # Ensure the model and input data are on the same device
    model.to(device)
    input_tensor = transforms.Compose([transforms.ToImage(), transforms.ToDtype(torch.float32, scale=True)])(input_img)[
        None].to(device)

    # Make a prediction with the model
    with torch.no_grad():
        model_output = model(input_tensor)

    # Set the confidence threshold
    threshold = 0.5

    # Move model output to the CPU
    model_output = move_data_to_device(model_output, 'cpu')

    # Filter the output based on the confidence threshold
    scores_mask = model_output[0]['scores'] > threshold

    # Scale the predicted bounding boxes
    pred_bboxes = BoundingBoxes(model_output[0]['boxes'][scores_mask] * min_img_scale, format='xyxy',
                                canvas_size=input_img.size[::-1])

    # Get the class names for the predicted label indices
    pred_labels = [class_names[int(label)] for label in model_output[0]['labels'][scores_mask]]

    # Extract the confidence scores
    pred_scores = model_output[0]['scores']

    # Scale and stack the predicted segmentation masks
    pred_masks = F.interpolate(model_output[0]['masks'][scores_mask], size=test_img.size[::-1])
    pred_masks = torch.concat([Mask(torch.where(mask >= threshold, 1, 0), dtype=torch.bool) for mask in pred_masks])

    # Get the annotation colors for the targets and predictions
    target_colors = [int_colors[i] for i in [class_names.index(label) for label in target_labels]]
    pred_colors = [int_colors[i] for i in [class_names.index(label) for label in pred_labels]]

    # Convert the test images to a tensor
    img_tensor = transforms.PILToTensor()(test_img)

    # Annotate the test image with the target segmentation masks
    annotated_tensor = draw_segmentation_masks(image=img_tensor, masks=target_masks, alpha=0.3, colors=target_colors)
    # Annotate the test image with the target bounding boxes
    annotated_tensor = draw_bboxes(image=annotated_tensor, boxes=target_bboxes, labels=target_labels,
                                   colors=target_colors)
    # Display the annotated test image
    annotated_test_img = tensor_to_pil(annotated_tensor)

    # Annotate the test image with the predicted segmentation masks
    annotated_tensor = draw_segmentation_masks(image=img_tensor, masks=pred_masks, alpha=0.3, colors=pred_colors)
    # Annotate the test image with the predicted labels and bounding boxes
    annotated_tensor = draw_bboxes(
        image=annotated_tensor,
        boxes=pred_bboxes,
        labels=[f"{label}\n{prob * 100:.2f}%" for label, prob in zip(pred_labels, pred_scores)],
        colors=pred_colors
    )

    # Display the annotated test image with the predicted bounding boxes
    plt.imshow(stack_imgs([annotated_test_img, tensor_to_pil(annotated_tensor)]))
    plt.show()


if __name__ == '__main__':
    load_data_and_visualize()
