import cv2
import argparse
import glob
from tqdm import tqdm

import skimage.data
import skimage.io
import json

import sys
import os

from common import ImageProposals, prepareJSON, exportCrop

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), ".."))
from proposals.Proposal import Proposal

def do_parsing():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Script to export selective search proposals')
    parser.add_argument('--imagesDir', type=str, required=True, help='Folder containing input images')
    parser.add_argument('--outputDirCrop', type=str, required=False, help='Output dir where to put images crops')
    parser.add_argument('--outputDirJSON', type=str, required=False, help='Output dir where to put JSON files')
    # Over 1000 regions the crops seem to be very similar
    parser.add_argument('--maxRegions', type=int, required=False, default=1000,
                        help='Maximum number of regions to export')
    parser.add_argument('--minSize', type=int, required=False, default=32,
                        help='Minimum image side dimension to save the file crop')
    return parser.parse_args()


if __name__ == '__main__':

    args = do_parsing()
    print(args)

    # speed-up using multithreads
    cv2.setUseOptimized(True);
    cv2.setNumThreads(4);

    #Files to analyze, ordered by name
    imgFormats = ["jpg", "png"]
    for imgFormat in imgFormats:
        files = sorted(glob.glob(args.imagesDir + "/*." + imgFormat), key=lambda s: s.lower())
        for imageFile in tqdm(files, unit="Image"):

            imageNameWithoutExtension = imageFile[imageFile.rindex('/') + 1:imageFile.rindex('.')]

            # HWC image loading
            img = cv2.imread(imageFile)
            width = img.shape[1]
            height = img.shape[0]

            # resize image, REALLY REALLY IMPORTANT FOR PERFORMANCES
            newHeight = 200
            newWidth = int(img.shape[1] * 200 / img.shape[0])
            img_resized = cv2.resize(img, (newWidth, newHeight))

            # create Selective Search Segmentation Object using default parameters
            ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()

            # set input image on which we will run segmentation
            ss.setBaseImage(img_resized)

            # Switch to fast but low recall Selective Search method
            #ss.switchToSelectiveSearchFast()

            # Switch to high recall but slow Selective Search method
            ss.switchToSelectiveSearchQuality()

            # run selective search segmentation on input image
            regions = ss.process()
            print('Total Number of Region Proposals: {}'.format(len(regions)))

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

# if __name__ == '__main__':
#
#     args = do_parsing()
#     print(args)
#
#     # If image path and f/q is not passed as command
#     # line arguments, quit and display help message
#     if len(sys.argv) < 3:
#         print(__doc__)
#         sys.exit(1)
#
#     # speed-up using multithreads
#     cv2.setUseOptimized(True)
#     cv2.setNumThreads(4)
#
#     # read image
#     im = cv2.imread(sys.argv[1])
#     # resize image
#     newHeight = 200
#     newWidth = int(im.shape[1]*200/im.shape[0])
#     im = cv2.resize(im, (newWidth, newHeight))
#
#     # create Selective Search Segmentation Object using default parameters
#     ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
#
#     # set input image on which we will run segmentation
#     ss.setBaseImage(im)
#
#     # Switch to fast but low recall Selective Search method
#     if (sys.argv[2] == 'f'):
#         ss.switchToSelectiveSearchFast()
#
#     # Switch to high recall but slow Selective Search method
#     elif (sys.argv[2] == 'q'):
#         ss.switchToSelectiveSearchQuality()
#     # if argument is neither f nor q print help message
#     else:
#         print(__doc__)
#         sys.exit(1)
#
#     # run selective search segmentation on input image
#     rects = ss.process()
#     print('Total Number of Region Proposals: {}'.format(len(rects)))
#
#     # number of region proposals to show
#     numShowRects = 100
#     # increment to increase/decrease total number
#     # of reason proposals to be shown
#     increment = 50
#
#     while True:
#         # create a copy of original image
#         imOut = im.copy()
#
#         # itereate over all the region proposals
#         for i, rect in enumerate(rects):
#             # draw rectangle for region proposal till numShowRects
#             if (i < numShowRects):
#                 x, y, w, h = rect
#                 cv2.rectangle(imOut, (x, y), (x+w, y+h), (0, 255, 0), 1, cv2.LINE_AA)
#             else:
#                 break
#
#         # show output
#         cv2.imshow("Output", imOut)
#
#         # record key press
#         k = cv2.waitKey(0) & 0xFF
#
#         # m is pressed
#         if k == 109:
#             # increase total number of rectangles to show by increment
#             numShowRects += increment
#         # l is pressed
#         elif k == 108 and numShowRects > increment:
#             # decrease total number of rectangles to show by increment
#             numShowRects -= increment
#         # q is pressed
#         elif k == 113:
#             break
#     # close image show window
#     cv2.destroyAllWindows()
