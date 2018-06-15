import cv2
import argparse
import glob
import json
import sys
import os

from tqdm import tqdm

from common import ImageProposals, resizeImage

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), ".."))
from proposals.Proposal import Proposal


def do_parsing():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Script to export selective search proposals')
    parser.add_argument('--imagesDir', type=str, required=True,
                        help='Folder containing input images, it can includes one level of subdirectories')
    parser.add_argument('--maxSide', type=int, required=False, default=300, help='Max side of image preprocess resize')
    parser.add_argument('--quality', action="store_true", help='More quality (recall), but slower')
    # Index format, single file
    parser.add_argument('--outputIndexFile', type=str, required=False, help='Output index file with proposals')
    parser.add_argument('-r', '--relativeCoordIndexFile', action='store_true',
                        help='Use relative coordinates instead of absolute pixels for index file export')
    # Object detection proposals format, one JSON for each image
    parser.add_argument('--outputDirJson', type=str, required=False,
                        help='Output dir where to put JSON files '
                             '(serialization of proposals objects with relative coordinates')
    parser.add_argument('--outputDirCrop', type=str, required=False, help='Output dir where to put images crops')
    # Over 1000 regions the crops seem to be very similar
    parser.add_argument('--maxRegions', type=int, required=False, default=1000,
                        help='Maximum number of regions to export')
    parser.add_argument('--minSize', type=int, required=False, default=20,
                        help='Minimum image side (pixels) dimension to save the file crop.'
                             'Pixels are computed on the resized image, refers to maxSide parameter.')
    return parser.parse_args()


def writeIndexFile(imageRelPath, outputIndexFilePath, rects_dict):

    with open(outputIndexFilePath, 'a') as outputIndexFile:

        rects_dict_string = json.dumps(rects_dict)

        outputIndexFile.writelines([imageRelPath, "\t", rects_dict_string, "\n"])


def prepareJSON(imageFileName, subdir, rects_dict, outputDirJson):

    proposals = []

    for rect_dict in tqdm(rects_dict, desc="JSON"):

        relX = rect_dict["x_rel"]
        relY = rect_dict["y_rel"]
        relW = rect_dict["width_rel"]
        relH = rect_dict["height_rel"]

        assert relW <= 1.0, relW >= 0.0
        assert relH <= 1.0, relH >= 0.0

        proposal = Proposal(relX, relY, relW, relH)
        proposals.append(proposal)

    imageNameWithoutExtension = imageFileName[:imageFileName.rindex('.')]
    imageProposal = ImageProposals(imageName=imageNameWithoutExtension, proposals=proposals)
    with open(os.path.join(outputDirJson, subdir, imageNameWithoutExtension + ".json"), 'w') as outfile:
        json.dump(imageProposal, outfile, default=lambda o: o.__dict__, indent=4, sort_keys=True)


def exportCrop(imageFileName, subdir, image, rects_dict, outputDirCrop):

    for index, rect_dict in tqdm(enumerate(rects_dict), desc="Crop"):

        x = rect_dict["x"]
        y = rect_dict["y"]
        w = rect_dict["width"]
        h = rect_dict["height"]

        crop = image[y:y+h, x:x+w]
        imageNameWithoutExtension = imageFileName[:imageFileName.rindex('.')]
        cv2.imwrite(os.path.join(outputDirCrop, subdir, imageNameWithoutExtension + "_" + str(index+1) + ".jpg"),
                    crop)


if __name__ == '__main__':

    args = do_parsing()
    print(args)

    # speed-up using multithreads
    cv2.setUseOptimized(True)
    cv2.setNumThreads(4)

    # create Selective Search Segmentation Object using default parameters
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()

    if args.outputIndexFile:
        if os.path.exists(args.outputIndexFile):
            raise Exception("Index file " + args.outputIndexFile + " already exists")

    if args.outputDirJson:
        os.makedirs(args.outputDirJson, exist_ok=False)

    if args.outputDirCrop:
        os.makedirs(args.outputDirCrop, exist_ok=False)

    # Retrieve subdirectories
    subdirs = next(os.walk(args.imagesDir))[1]
    if len(subdirs) == 0:
        subdirs = [""]

    # Files to analyze, ordered by name
    imgFormats = ["jpg", "png", "jpeg"]

    for subdir in tqdm(subdirs, desc="subdir"):

        if args.outputDirCrop:
            os.makedirs(os.path.join(args.outputDirCrop, subdir), exist_ok=True)

        if args.outputDirJson:
            os.makedirs(os.path.join(args.outputDirJson, subdir), exist_ok=True)

        for imgFormat in imgFormats:
            files = sorted(glob.glob(os.path.join(args.imagesDir, subdir) + "/*." + imgFormat), key=lambda s: s.lower())
            for imageFile in tqdm(files, desc=("Image in " + imgFormat + " format")):

                imageNameWithoutExtension = imageFile[imageFile.rindex('/') + 1:imageFile.rindex('.')]

                # HWC image loading
                image = cv2.imread(imageFile)
                width = image.shape[1]
                height = image.shape[0]

                # resize image, REALLY REALLY IMPORTANT FOR PERFORMANCES
                image_resized = resizeImage(image, width=width, height=height, maxSide=args.maxSide)

                resized_width = image_resized.shape[1]
                resized_height = image_resized.shape[0]

                # set input image on which we will run segmentation
                ss.setBaseImage(image_resized)

                # Quality switch must executed for each image
                if args.quality:
                    # Switch to high recall but slow Selective Search method
                    ss.switchToSelectiveSearchQuality()
                else:
                    # Switch to fast but low recall Selective Search method
                    ss.switchToSelectiveSearchFast()

                # run selective search segmentation on input image
                try:
                    regions = ss.process()
                    print('Total Number of Region Proposals: {}'.format(len(regions)))
                except cv2.error as e:
                    print(e)
                    print("Error processing image " + imageFile + ", skipping")
                    continue

                rects_dict_rel = []
                rects_dict_abs = []

                acceptable_bounding_boxes = 0

                for boundingBox in regions:

                    x, y, w, h = boundingBox

                    # Check size on resized image
                    if w >= args.minSize and h >= args.minSize:

                        # Pixels are in range [0, width] and [0, height].
                        # Don't know why last value is added, instead of [0, width - 1] and [0, height - 1]
                        XMinR = float(x) / float(resized_width)
                        YMinR = float(y) / float(resized_height)
                        widthR = float(w) / float(resized_width)
                        heightR = float(h) / float(resized_height)

                        # Retrieve absolute positions on the original image
                        x_original = int(round(x * width / resized_width))
                        y_original = int(round(y * height / resized_height))
                        w_original = int(round(w * width / resized_width))
                        h_original = int(round(h * height / resized_height))

                        this_rect_dict_rel = dict()
                        this_rect_dict_rel["x_rel"] = XMinR
                        this_rect_dict_rel["y_rel"] = YMinR
                        this_rect_dict_rel["width_rel"] = widthR
                        this_rect_dict_rel["height_rel"] = heightR

                        rects_dict_rel.append(this_rect_dict_rel)

                        this_rect_dict_abs = dict()
                        this_rect_dict_abs["x"] = x_original
                        this_rect_dict_abs["y"] = y_original
                        this_rect_dict_abs["width"] = w_original
                        this_rect_dict_abs["height"] = h_original

                        rects_dict_abs.append(this_rect_dict_abs)

                        acceptable_bounding_boxes += 1
                        if acceptable_bounding_boxes >= args.maxRegions:
                            break

                if args.outputIndexFile:
                    rects_dict = rects_dict_rel if args.relativeCoordIndexFile else rects_dict_abs
                    writeIndexFile(imageRelPath=os.path.join(subdir, os.path.basename(imageFile)),
                                   outputIndexFilePath=args.outputIndexFile, rects_dict=rects_dict)

                # Proposals export in JSON for our object detection training pipeline, subdirectories tree is maintained
                if args.outputDirJson:
                    # {
                    # "image": "000001",
                    # "proposals": [
                    #    {
                    #        "height": 1.0,
                    #        "width": 0.725212454795837,
                    #        "x": 0.0906515568494797,
                    #        "y": 0.0
                    #    },
                    prepareJSON(imageFileName=os.path.basename(imageFile),
                                subdir=subdir, rects_dict=rects_dict_rel, outputDirJson=args.outputDirJson)

                # Proposals crops export as images, subdirectories tree is maintained
                if args.outputDirCrop:
                    exportCrop(imageFileName=os.path.basename(imageFile),
                               subdir=subdir, image=image, rects_dict=rects_dict_abs,
                               outputDirCrop=args.outputDirCrop)
