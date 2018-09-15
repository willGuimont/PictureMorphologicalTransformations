import cv2 as cv
import numpy as np
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--image', '-i', type=str, required=True, help='Image to morph')
parser.add_argument('--gray', '-g', action='store_true', help='Load image in grayscale')
parser.add_argument('--out', '-o', type=str, required=True, help='Output')
parser.add_argument('--trans', '-t', type=str, required=True, help='Transformation file')

args = parser.parse_args()

image = args.image
gray = args.gray
out = args.out
trans = args.trans

img = cv.imread(image)

if img is None:
    print(f'Can\'t load image {image}')
    exit(1)

if gray:
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

with open(trans, 'r') as file:
    rows = file.readlines()
    for r in rows:
        r.strip()
        if r.startswith('#') or r == '':
            continue

        operation, iterations, kx, ky, *kernel_data = r.split(',')
        iterations = int(iterations)
        size = int(kx), int(ky)
        kernel = [float(x) for x in kernel_data]
        kernel = np.array(kernel, dtype=img.dtype)
        kernel = np.resize(kernel, size)

        if operation == 'erode':
            img = cv.erode(img, kernel, iterations=iterations)
        elif operation == 'dilate':
            img = cv.dilate(img, kernel, iterations=iterations)
        elif operation == 'opening':
            img = cv.morphologyEx(img, cv.MORPH_OPEN, kernel, iterations=iterations)
        elif operation == 'closing':
            img = cv.morphologyEx(img, cv.MORPH_CLOSE, kernel, iterations=iterations)
        elif operation == 'gradient':
            img = cv.morphologyEx(img, cv.MORPH_GRADIENT, kernel, iterations=iterations)
        elif operation == 'tophat':
            img = cv.morphologyEx(img, cv.MORPH_TOPHAT, kernel, iterations=iterations)
        elif operation == 'blackhat':
            img = cv.morphologyEx(img, cv.MORPH_BLACKHAT, kernel, iterations=iterations)

show = cv.pyrDown(img)
cv.imshow('Output', show)
cv.waitKey()
cv.imwrite(out, img)
