import skimage.data
import skimage.io
import selectivesearch.selectivesearch as ss
import argparse
import glob
import json
from tqdm import tqdm

import sys
import os

from common import ImageProposals, resizeImage

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), ".."))
from proposals.Proposal import Proposal


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


def do_parsing():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Script to export selective search proposals')
    parser.add_argument('--imagesDir', required=True, help='Folder containing input images')
    parser.add_argument('--maxSide', type=int, required=False, default=300, help='Max side of image preprocess resize')
    parser.add_argument('--outputDirCrop', type=str, required=False, help='Output dir where to put images crops')
    parser.add_argument('--outputDirJson', type=str, required=False, help='Output dir where to put JSON files')
    # Over 1000 regions the crops seem to be very similar
    parser.add_argument('--maxRegions', type=int, required=False, default=1000,
                        help='Maximum number of regions to export')
    parser.add_argument('--minSize', type=int, required=False, default=32,
                        help='Minimum image side dimension to save the file crop')
    return parser.parse_args()


def main():

    args = do_parsing()
    print(args)

    #Files to analyze, ordered by name
    imgFormats = ["jpg", "png", "jpeg"]
    for imgFormat in imgFormats:
        files = sorted(glob.glob(args.imagesDir + "/*." + imgFormat), key=lambda s: s.lower())
        for imageFile in tqdm(files, unit="Image"):

            imageNameWithoutExtension = imageFile[imageFile.rindex('/') + 1:imageFile.rindex('.')]

            # HWC image loading
            img = skimage.data.load(imageFile)
            width = img.shape[1]
            height = img.shape[0]

            # resize image, REALLY REALLY IMPORTANT FOR PERFORMANCES
            img_resized = resizeImage(img, width=width, height=height, maxSide=args.maxSide)

            # region x, y, w, h
            img_lbl, regions = ss.selective_search(img_resized, scale=500, sigma=0.9, min_size=10)

            # Proposals export in JSON for our object detection training pipeline
            if args.outputDirJson is not None:

                if os.path.exists(args.outputDirJson) is False:
                    os.makedirs(args.outputDirJson)

                proposals = prepareJSON(width, height, regions, maxRegions=args.maxRegions)
                # {
                # "image": "000001",
                # "proposals": [
                #    {
                #        "height": 1.0,
                #        "width": 0.725212454795837,
                #        "x": 0.0906515568494797,
                #        "y": 0.0
                #    },
                imageProposal = ImageProposals(imageName=imageNameWithoutExtension, proposals=proposals)
                with open(os.path.join(args.outputDirJson, imageNameWithoutExtension + ".json"), 'w') as outfile:
                    json.dump(imageProposal, outfile, default=lambda o: o.__dict__, indent=4, sort_keys=True)

            # Proposals crops export as images
            if args.outputDirCrop is not None:

                if os.path.exists(args.outputDirCrop) is False:
                    os.makedirs(args.outputDirCrop)

                exportCrop(img, regions, args.outputDirCrop, imageNameWithoutExtension,
                           minSize=args.minSize, maxRegions=args.maxRegions)


if __name__ == "__main__":
    main()

