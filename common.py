import os
import sys

import cv2
import skimage.io
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


def prepareJSON(width, height, regions, maxRegions):

    proposals = []

    for region in tqdm(regions[:maxRegions], unit="Region"):
        relX = region['rect'][0] / float(width)
        relY = region['rect'][1] / float(height)
        relW = (region['rect'][2] + 1) / float(width)
        relH = (region['rect'][3] + 1) / float(height)

        assert relW <= 1.0 and relW > 0.0
        assert relH <= 1.0 and relH > 0.0

        proposal = Proposal(relX, relY, relW, relH)
        proposals.append(proposal)

    return proposals


def exportCrop(image, regions, outputDir, imageNameWithoutExtension, minSize, maxRegions):

    for index, region in tqdm(enumerate(regions[:maxRegions]), unit="Region"):

        try:

            x = region['rect'][0]
            y = region['rect'][1]
            w = region['rect'][2]
            h = region['rect'][3]

            assert w > minSize
            assert h > minSize

            crop = image[y:y+h, x:x+w]
            skimage.io.imsave(os.path.join(outputDir, imageNameWithoutExtension + "_" + str(index) + ".jpg"), crop)

        except Exception as e:
            print(e.message)