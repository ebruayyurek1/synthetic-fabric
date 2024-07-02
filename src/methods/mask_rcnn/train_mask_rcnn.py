# Import Python Standard Library dependencies
import datetime
import json
# import linecache
import multiprocessing
import os
import random
# import tracemalloc
from functools import partial
from pathlib import Path

# Import matplotlib for creating plots
import matplotlib.pyplot as plt
# Import the pandas package
import pandas as pd
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms.v2 as transforms
from PIL import Image
# Import utility functions
from cjm_pil_utils.core import resize_img, get_img_files, stack_imgs
from cjm_psl_utils.core import download_file
from cjm_pytorch_utils.core import tensor_to_pil, get_torch_device, set_seed, move_data_to_device
from cjm_torchvision_tfms.core import ResizeMax, PadSquare, CustomRandomIoUCrop
from distinctipy import distinctipy
from torch.utils.data import DataLoader
# Import Mask R-CNN
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
# torchvision.disable_beta_transforms_warning()
from torchvision.tv_tensors import BoundingBoxes, Mask
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks
from tqdm.auto import tqdm

from src.methods.mask_rcnn.train_utils import train_loop
from src.utils.io_utils import load_yaml
from windows_utils import LabelMeDataset, create_polygon_mask, tuple_batch


def main():
    # --- INIT ---
    # tracemalloc.start()
    config: dict = load_yaml("configs/train_params.yml")
    dataset_dir, project_dir = seed_and_prepare_folders(config)
    dataset_path = Path(f'{dataset_dir}/{config["train_ds"]}/')
    # Prepare bbox drawing function
    draw_bboxes = partial(draw_bounding_boxes, fill=False, width=2, font=config['font_file'], font_size=25)

    # --- LOAD DATA ---
    # Get filenams for imgs and annotations
    # FIXME: remove debug limitation
    img_file_paths = get_img_files(dataset_path)[:50]
    annotation_file_paths = list(dataset_path.glob('*.json'))[:50]
    img_dict = {file.stem: file for file in img_file_paths}  # dict file names -> file paths

    print(f"Number of Images: {len(img_dict)}")

    # -----------
    # # Read JSON files and store data in a list
    # data_list = list(read_json_files(annotation_file_paths))
    # # Create a dictionary to store the combined data
    # combined_data = {}
    # # Process each JSON file's data
    # for data in data_list:
    #     # for key, value in data.items():
    #     image_name = data['imagePath'].split('.')[0]
    #     if image_name in img_dict:
    #         combined_data[image_name] = data
    # # Explode 'shapes' column
    # shapes_list = []
    # for key, value in combined_data.items():
    #     for shape in value['shapes']:
    #         shape['imageName'] = key
    #         shapes_list.append(shape)
    #
    # # Convert shapes list to a structured array
    # dtype = [('label', 'U50'), ('points', 'O'), ('imageName', 'U50')]
    # shapes_array = np.array(shapes_list, dtype=dtype)
    #
    # # Get unique labels
    # class_names = np.unique(shapes_array['label']).tolist()
    #
    # # Prepend 'background' class
    # class_names.insert(0, 'background')
    #
    # # Generate colors
    # colors = distinctipy.get_colors(len(class_names))
    #
    # # Convert colors to integer format
    # int_colors = [tuple(int(c * 255) for c in color) for color in colors]
    # quit()
    # -----------
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

    # Prepend a `background` class to the list of class names
    class_names = ['background'] + class_names

    # Generate a list of colors with a length equal to the number of labels
    colors = distinctipy.get_colors(len(class_names))

    # Make a copy of the color map in integer format
    int_colors = [tuple(int(c * 255) for c in color) for color in colors]

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
    model.to(device=device, dtype=dtype)

    model.device = device
    model.name = config['model_name']
    img_keys = list(img_dict.keys())

    random.shuffle(img_keys)

    train_pct = 1 - config['val_split']
    train_split = int(len(img_keys) * train_pct)
    train_keys = img_keys[:train_split]
    val_keys = img_keys[train_split:]

    # Create a RandomIoUCrop object
    iou_crop = CustomRandomIoUCrop(min_scale=0.3,
                                   max_scale=1.0,
                                   min_aspect_ratio=0.5,
                                   max_aspect_ratio=2.0,
                                   sampler_options=[0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
                                   trials=400,
                                   jitter_factor=0.25)

    # Create a `ResizeMax` object
    resize_max = ResizeMax(max_sz=config['img_size'])

    # Create a `PadSquare` object
    pad_square = PadSquare(shift=True, fill=0)

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
        transforms.Resize([config['img_size']] * 2, antialias=True)
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

    # Set the number of worker processes for loading data.
    num_workers = multiprocessing.cpu_count() // 2

    # Define parameters for DataLoader
    data_loader_params = {
        'batch_size': config['batch_size'],  # Batch size for data loading
        'num_workers': num_workers,  # Number of subprocesses to use for data loading
        'persistent_workers': True,
        # If True, the data loader will not shutdown the worker processes after a dataset has been consumed once.
        # This allows to maintain the worker dataset instances alive.
        'pin_memory': 'cuda' in device,
        # If True, the data loader will copy Tensors into CUDA pinned memory before returning them.
        # Useful when using GPU.
        'pin_memory_device': device if 'cuda' in device else '',
        # Specifies the device where the data should be loaded. Commonly set to use the GPU.
        'collate_fn': tuple_batch,
    }

    train_dataloader = DataLoader(train_dataset, **data_loader_params, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, **data_loader_params)

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    checkpoint_dir = Path(project_dir / f"{timestamp}")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    # Create a color map and write it to a JSON file
    color_map = {'items': [{'label': label, 'color': color} for label, color in zip(class_names, colors)]}
    with open(f"{checkpoint_dir}/{dataset_path.name}-colormap.json", "w") as file:
        json.dump(color_map, file)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'])
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,
                                                       max_lr=config['lr'],
                                                       total_steps=config['epochs'] * len(train_dataloader))
    #
    # snapshot = tracemalloc.take_snapshot()
    # display_top(snapshot)
    # quit()
    train_loop(model=model,
               train_dataloader=train_dataloader,
               valid_dataloader=valid_dataloader,
               optimizer=optimizer,
               lr_scheduler=lr_scheduler,
               device=torch.device(device),
               epochs=config['epochs'],
               checkpoint_path=checkpoint_dir / f"{model.name}.pth",
               use_scaler=True)

    try_one_val_img(annotation_df, class_names, config, device, draw_bboxes, img_dict, int_colors, model, val_keys)


def try_one_val_img(annotation_df, class_names, config, device, draw_bboxes, img_dict, int_colors, model, val_keys):
    # Choose a random item from the validation set
    file_id = random.choice(val_keys)
    # Retrieve the image file path associated with the file ID
    test_file = img_dict[file_id]
    # Open the test file
    test_img = Image.open(test_file).convert('RGB')
    # Resize the test image
    input_img = resize_img(test_img, target_sz=config['img_size'], divisor=1)
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
    model_output = move_data_to_device(model_output, torch.device('cpu'))
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



# def display_top(snapshot, key_type='lineno', limit=3):
#     snapshot = snapshot.filter_traces((
#         tracemalloc.Filter(False, "<frozen importlib._bootstrap>"),
#         tracemalloc.Filter(False, "<unknown>"),
#     ))
#     top_stats = snapshot.statistics(key_type)
#
#     print("Top %s lines" % limit)
#     for index, stat in enumerate(top_stats[:limit], 1):
#         frame = stat.traceback[0]
#         # replace "/path/to/module/file.py" with "module/file.py"
#         filename = os.sep.join(frame.filename.split(os.sep)[-2:])
#         print("#%s: %s:%s: %.1f KiB"
#               % (index, filename, frame.lineno, stat.size / 1024))
#         line = linecache.getline(frame.filename, frame.lineno).strip()
#         if line:
#             print('    %s' % line)
#
#     other = top_stats[limit:]
#     if other:
#         size = sum(stat.size for stat in other)
#         print("%s other: %.1f KiB" % (len(other), size / 1024))
#     total = sum(stat.size for stat in top_stats)
#     print("Total allocated size: %.1f KiB" % (total / 1024))
def seed_and_prepare_folders(config):
    set_seed(config['seed'])
    if not Path(config['font_file']).exists():
        download_file(f"https://fonts.gstatic.com/s/roboto/v30/{config['font_file']}", "./")
    dataset_dir = Path("datasets")
    project_dir = Path(f"out/{config['project_name']}/")
    project_dir.mkdir(parents=True, exist_ok=True)
    dataset_dir.mkdir(parents=True, exist_ok=True)
    return dataset_dir, project_dir


if __name__ == '__main__':
    main()
