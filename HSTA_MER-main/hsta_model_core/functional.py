# functional.py

import torch
import numpy as np
from PIL import Image

def resize_clip(clip, size, interpolation='bilinear'):
    """
    Resize a list of (H x W x C) numpy.ndarray to the final size.
    Args:
        clip (list of np.ndarray or PIL.Image): List of images to be resized.
        size (tuple): Desired output size (width, height).
        interpolation (str): Interpolation method.
    Returns:
        list: Resized list of images.
    """
    if isinstance(clip[0], np.ndarray):
        return [np.array(Image.fromarray(img).resize(size, Image.BILINEAR if interpolation == 'bilinear' else Image.NEAREST)) for img in clip]
    elif isinstance(clip[0], Image.Image):
        return [img.resize(size, Image.BILINEAR if interpolation == 'bilinear' else Image.NEAREST) for img in clip]
    else:
        raise TypeError('Expected numpy.ndarray or PIL.Image')

def crop_clip(clip, top, left, height, width):
    """
    Crop a list of (H x W x C) numpy.ndarray or PIL.Image at the same location.
    Args:
        clip (list of np.ndarray or PIL.Image): List of images to be cropped.
        top (int): Vertical coordinate of the top left corner of the crop box.
        left (int): Horizontal coordinate of the top left corner of the crop box.
        height (int): Height of the crop box.
        width (int): Width of the crop box.
    Returns:
        list: Cropped list of images.
    """
    if isinstance(clip[0], np.ndarray):
        return [img[top:top+height, left:left+width] for img in clip]
    elif isinstance(clip[0], Image.Image):
        return [img.crop((left, top, left + width, top + height)) for img in clip]
    else:
        raise TypeError('Expected numpy.ndarray or PIL.Image')

def normalize(clip, mean, std):
    """
    Normalize a clip with mean and standard deviation.
    Args:
        clip (Tensor): Tensor clip of size (T, C, H, W) to be normalized.
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
    Returns:
        Tensor: Normalized Tensor clip.
    """
    mean = torch.tensor(mean).view(1, -1, 1, 1)
    std = torch.tensor(std).view(1, -1, 1, 1)
    return (clip - mean) / std