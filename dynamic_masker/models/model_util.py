import math
import copy
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

def label_smoothing(inputs, eps=0.1):
    """
    Label smoothing
    """
    return inputs * (1 - eps) + 0.5 * eps  # Smooth labels

def label_blurring(mask, kernel_size=5, sigma=1.0):
    kernel = get_gaussian_kernel(kernel_size, sigma).to(mask.device)
    padding = kernel_size // 2
    return F.conv2d(mask, kernel, padding=padding, groups=1)

def get_gaussian_kernel(kernel_size=5, sigma=1.0):
    """Returns a 2D Gaussian kernel as [1, 1, k, k] tensor."""
    x = torch.arange(kernel_size).float() - kernel_size // 2
    gauss = torch.exp(-0.5 * (x / sigma)**2)
    kernel_1d = gauss / gauss.sum()
    kernel_2d = kernel_1d[:, None] @ kernel_1d[None, :]
    return kernel_2d.view(1, 1, kernel_size, kernel_size)

def initialize_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        init.xavier_uniform_(m.weight)  # Xavier Uniform
        if m.bias is not None:
            init.zeros_(m.bias)  # Bias = 0

def recursive_clone(tensor):
    """
    Assumes tensor is a torch.tensor with 'clone()' method, possibly
    inside nested iterable.
    E.g., tensor = [(pytorch_tensor, pytorch_tensor), ...]
    """
    if hasattr(tensor, "clone"):
        return tensor.clone()
    try:
        return type(tensor)(recursive_clone(t) for t in tensor)
    except TypeError:
        print("{} is not iterable and has no clone() method.".format(tensor))


def copy_states(states):
    """
    Simple deepcopy if list of Nones, else clone.
    """
    if states[0] is None:
        return copy.deepcopy(states)
    return recursive_clone(states)


class ImagePadder(object):
    """
    From E-RAFT: https://github.com/uzh-rpg/E-RAFT
    """

    # =================================================================== #
    # In some networks, the image gets downsized. This is a problem, if   #
    # the to-be-downsized image has odd dimensions ([15x20]->[7.5x10]).   #
    # To prevent this, the input image of the network needs to be a       #
    # multiple of a minimum size (min_size)                               #
    # The ImagePadder makes sure, that the input image is of such a size, #
    # and if not, it pads the image accordingly.                          #
    # =================================================================== #

    def __init__(self, min_size=64):
        # --------------------------------------------------------------- #
        # The min_size additionally ensures, that the smallest image      #
        # does not get too small                                          #
        # --------------------------------------------------------------- #
        self.min_size = min_size
        self.pad_height = None
        self.pad_width = None

    def pad(self, image):
        # --------------------------------------------------------------- #
        # If necessary, this function pads the image on the left & top    #
        # --------------------------------------------------------------- #
        height, width = image.shape[-2:]
        if self.pad_width is None:
            self.pad_height = (self.min_size - height % self.min_size) % self.min_size
            self.pad_width = (self.min_size - width % self.min_size) % self.min_size
        else:
            pad_height = (self.min_size - height % self.min_size) % self.min_size
            pad_width = (self.min_size - width % self.min_size) % self.min_size
            if pad_height != self.pad_height or pad_width != self.pad_width:
                raise
        return torch.nn.ZeroPad2d((self.pad_width, 0, self.pad_height, 0))(image)

    def unpad(self, image):
        # --------------------------------------------------------------- #
        # Removes the padded rows & columns                               #
        # --------------------------------------------------------------- #
        return image[..., self.pad_height :, self.pad_width :]
