import skimage.data
import skimage.io
import selectivesearch.selectivesearch as ss
import argparse
import glob
import json
from tqdm import tqdm

import sys
import os

from common import ImageProposals, prepareJSON, exportCrop

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), ".."))
from proposals.Proposal import Proposal


def do_parsing():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Script to export selective search proposals')
    parser.add_argument('--imagesDir', required=True, help='Folder containing input images')
    parser.add_argument('--outputDirCrop', type=str, required=False, help='Output dir where to put images crops')
    parser.add_argument('--outputDirJSON', type=str, required=False, help='Output dir where to put JSON files')
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
    imgFormats = ["jpg", "png"]
    for imgFormat in imgFormats:
        files = sorted(glob.glob(args.imagesDir + "/*." + imgFormat), key=lambda s: s.lower())
        for imageFile in tqdm(files, unit="Image"):

            imageNameWithoutExtension = imageFile[imageFile.rindex('/') + 1:imageFile.rindex('.')]

            # HWC image loading
            img = skimage.data.load(imageFile)
            width = img.shape[1]
            height = img.shape[0]

            # region x, y, w, h
            img_lbl, regions = ss.selective_search(img, scale=500, sigma=0.9, min_size=10)

            # Proposals export in JSON for our object detection training pipeline
            if args.outputDirJSON is not None:

                if os.path.exists(args.outputDirJSON) is False:
                    os.makedirs(args.outputDirJSON)

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
                with open(os.path.join(args.outputDirJSON, imageNameWithoutExtension + ".json"), 'w') as outfile:
                    json.dump(imageProposal, outfile, default=lambda o: o.__dict__, indent=4, sort_keys=True)

            # Proposals crops export as images
            if args.outputDirCrop is not None:

                if os.path.exists(args.outputDirCrop) is False:
                    os.makedirs(args.outputDirCrop)

                exportCrop(img, regions, args.outputDirCrop, imageNameWithoutExtension,
                           minSize=args.minSize, maxRegions=args.maxRegions)


if __name__ == "__main__":
    main()

