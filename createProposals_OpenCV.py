import cv2
import argparse
import glob
import json
import sys
import os

from tqdm import tqdm
import numpy as np

from common import ImageProposals, resizeImage

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), ".."))
from proposals.Proposal import Proposal


def do_parsing():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Script to export selective search proposals')
    parser.add_argument('--imagesDir', type=str, required=True, help='Folder containing input images')
    parser.add_argument('--maxSide', type=int, required=False, default=300, help='Max side of image preprocess resize')
    parser.add_argument('--outputDirCrop', type=str, required=False, help='Output dir where to put images crops')
    parser.add_argument('--quality', action="store_true", help='More quality (recall), but slower')
    # Index format, single file
    parser.add_argument('--outputIndexFile', type=str, required=False, help='Output index file with proposals')
    parser.add_argument('-r', '--relativeCoord', action='store_true',
                        help='Use relative coordinates instead of absolute pixels for index file export')
    # Object detection proposals format, one JSON for each image
    parser.add_argument('--outputDirJSON', type=str, required=False, help='Output dir where to put JSON files')
    # Over 1000 regions the crops seem to be very similar
    parser.add_argument('--maxRegions', type=int, required=False, default=1000,
                        help='Maximum number of regions to export')
    parser.add_argument('--minSize', type=int, required=False, default=20,
                        help='Minimum image side (pixels) dimension to save the file crop.'
                             'Pixels are computed on the resized image, refers to maxSide parameter.')
    return parser.parse_args()


if __name__ == '__main__':

    args = do_parsing()
    print(args)

    # speed-up using multithreads
    cv2.setUseOptimized(True)
    cv2.setNumThreads(4)

    # create Selective Search Segmentation Object using default parameters
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()

    if args.outputIndexFile:
        # TODO: Check existence and raise error, because we append
        pass

    # TODO: Iterate over one level of directories
    subset = ""

    #Files to analyze, ordered by name
    imgFormats = ["jpg", "png", "jpeg"]
    for imgFormat in imgFormats:
        files = sorted(glob.glob(args.imagesDir + "/*." + imgFormat), key=lambda s: s.lower())
        for imageFile in tqdm(files, unit="Image"):

            imageNameWithoutExtension = imageFile[imageFile.rindex('/') + 1:imageFile.rindex('.')]

            # HWC image loading
            img = cv2.imread(imageFile)
            width = img.shape[1]
            height = img.shape[0]

            # resize image, REALLY REALLY IMPORTANT FOR PERFORMANCES
            img_resized = resizeImage(img, width=width, height=height, maxSide=args.maxSide)

            resized_width = img_resized.shape[1]
            resized_height = img_resized.shape[0]

            # set input image on which we will run segmentation
            ss.setBaseImage(img_resized)

            # Quality switch must executed for each image
            if args.quality:
                # Switch to high recall but slow Selective Search method
                ss.switchToSelectiveSearchQuality()
            else:
                # Switch to fast but low recall Selective Search method
                ss.switchToSelectiveSearchFast()

            # run selective search segmentation on input image
            regions = ss.process()
            print('Total Number of Region Proposals: {}'.format(len(regions)))

            # TODO: Manage new outputs format, probably they are the same as before (array of cv2 rects as numpy array)

            if args.outputIndexFile:

                with open(args.outputIndexFile, 'a') as outputIndexFile:

                    rects_dict = []

                    for boundingBox in regions[:args.maxRegions]:

                        x, y, w, h = boundingBox

                        if w >= args.minSize and h >= args.minSize:

                            this_rect_dict = dict()

                            if args.relativeCoord:
                                XMinR = float(x) / float(resized_width - 1)
                                YMinR = float(y) / float(resized_height - 1)
                                widthR = float(w) / float(resized_width - 1)
                                heightR = float(h) / float(resized_height - 1)

                                this_rect_dict["x_rel"] = XMinR
                                this_rect_dict["y_rel"] = YMinR
                                this_rect_dict["width_rel"] = widthR
                                this_rect_dict["height_rel"] = heightR
                            else:
                                # Retrieve absolute positions on the original image
                                x_original = int(x * width / resized_width)
                                y_original = int(y * height / resized_height)
                                w_original = int(w * width / resized_width)
                                h_original = int(h * height / resized_height)

                                this_rect_dict["x"] = x_original
                                this_rect_dict["y"] = y_original
                                this_rect_dict["width"] = w_original
                                this_rect_dict["height"] = h_original

                            rects_dict.append(this_rect_dict)

                    rects_dict_string = json.dumps(rects_dict)

                    outputIndexFile.writelines([os.path.join(subset, os.path.basename(imageFile)),
                                                "\t", rects_dict_string, "\n"])

            # Proposals export in JSON for our object detection training pipeline
            if args.outputDirJSON is not None:

                if os.path.exists(args.outputDirJSON) is False:
                    os.makedirs(args.outputDirJSON)

                #proposals = prepareJSON(width, height, regions, maxRegions=args.maxRegions)
                # {
                # "image": "000001",
                # "proposals": [
                #    {
                #        "height": 1.0,
                #        "width": 0.725212454795837,
                #        "x": 0.0906515568494797,
                #        "y": 0.0
                #    },
                #imageProposal = ImageProposals(imageName=imageNameWithoutExtension, proposals=proposals)
                #with open(os.path.join(args.outputDirJSON, imageNameWithoutExtension + ".json"), 'w') as outfile:
                #    json.dump(imageProposal, outfile, default=lambda o: o.__dict__, indent=4, sort_keys=True)

            # Proposals crops export as images
            if args.outputDirCrop is not None:

                if os.path.exists(args.outputDirCrop) is False:
                    os.makedirs(args.outputDirCrop)

                #exportCrop(img, regions, args.outputDirCrop, imageNameWithoutExtension,
                #           minSize=args.minSize, maxRegions=args.maxRegions)
