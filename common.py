import os
import sys

import cv2
import skimage.io
import numpy as np
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), ".."))
from proposals.Proposal import Proposal


class ImageProposals:

    def __init__(self, imageName, proposals):

        self.image = imageName
        self.proposals = proposals


def resizeImage(image, width, height, maxSide):
    """
    Create a new image resized according to parameters
    :param image:
    :param width:
    :param height:
    :param maxSide:
    :return:
    """
    if width > height:
        # Landscape
        if width <= maxSide:
            return np.copy(image)
        newWidth = maxSide
        newHeight = int(height * float(newWidth) / float(width))
    else:
        # Portrait
        if height <= maxSide:
            return np.copy(image)
        newHeight = maxSide
        newWidth = int(width * float(newHeight) / float(height))
    return cv2.resize(image, (newWidth, newHeight))
