from src.transform import interpolation
from src.transform import without_interpolation
from src.transform import warped
from src.transform import original_image
import numpy as np
import argparse
import cv2

# Parâmetros
# --image images/example.jpg --coords "[(311, 46), (633, 109), (634, 413), (342, 568)]"

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", help = "path to the image file")
ap.add_argument("-c", "--coords", help = "comma seperated list of source points")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])
pts = np.array(eval(args["coords"]), dtype = "float32")

warped = warped(image, pts)
interpolation = interpolation(image, pts)
without_interpolation = without_interpolation(image, pts)
original_image = original_image(image, pts)

cv2.imwrite('result/original_image.jpg',original_image)
cv2.imwrite('result/warped.jpg',warped)
cv2.imwrite('result/interpolation.jpg',interpolation)
cv2.imwrite('result/without_interpolation.jpg',without_interpolation)
