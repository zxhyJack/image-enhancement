from __future__ import print_function
import cv2 as cv
import argparse
# parser = argparse.ArgumentParser(description='Code for Histogram Equalization tutorial.')
# parser.add_argument('--input', help='Path to input image.', default='lena.jpg')
# args = parser.parse_args()
# src = cv.imread(cv.samples.findFile(args.input))
src = cv.imread('./imageData/cameraman.tif')
# if src is None:
#     print('Could not open or find the image:', args.input)
#     exit(0)
src = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
dst = cv.equalizeHist(src)
cv.imshow('Source image', src)
cv.imshow('Equalized Image', dst)
cv.waitKey()