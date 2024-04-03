def calculate_aspect_ratio_fit(src_width, src_height, resize_side) -> [int, int]:
    """
    Calculate width and height of an image we want to resize based on a single side.
    The smallest side will be adapted to max_side.
    The aspect ratio will be preserved

    :param src_width: width of original image
    :param src_height: height of original image
    :param resize_side: side to which the smallest side is adapted keeping aspect ration
    :return: width, height of resized image
    """
    ratio: float = min(resize_side / src_width, resize_side / src_height)
    width: int = round(src_width * ratio)
    height: int = round(src_height * ratio)
    return width, height
