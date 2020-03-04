import numpy as np
from PIL import Image # for loading images
import matplotlib.pyplot as plt # for displaying images

# How much does each color contribute to the brightness of
# a pixel when converting to grayscale
grayscale_conversion_coeffs = np.array([0.299, 0.587, 0.114])

def imshow_grayscale(i):
    """Display a grayscale image data clamped between 0 and 1"""
    plt.imshow(i, cmap='gray', interpolation='none', vmin=0, vmax=1)
    plt.show()

def imload_grayscale(path):
    """Load an image and convert to normalized grayscale"""
    return np.sum(np.asarray(Image.open(path)) / 255. * grayscale_conversion_coeffs, axis=2)

def imsave_grayscale(i, path):
    """De-normalize a grayscale image, convert to RGB, and save it"""
    c = (i * 255).clip(0, 255).astype(np.uint8)
    Image.fromarray(np.stack([c] * 3, axis=2), mode='RGB').save(path)


def imshow(i):
    """Display a grayscale image data clamped between 0 and 1"""
    plt.imshow(i.clip(0, 1), interpolation='none')

def imload(path):
    """Load an image and convert to normalized grayscale"""
    return np.asarray(Image.open(path)) / 255.

def imsave(i, path):
    """De-normalize a grayscale image, convert to RGB, and save it"""
    c = (i * 255).clip(0, 255).astype(np.uint8)
    Image.fromarray(c, mode='RGB').save(path)
