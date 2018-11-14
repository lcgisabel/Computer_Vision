import numpy as np
import cv2

def order_points(pts):

	rect = np.zeros((4, 2), dtype = "float32")

	s = pts.sum(axis = 1)
	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]

	diff = np.diff(pts, axis = 1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]

	return rect

def original_image(image, pts):

	rect = order_points(pts)
	(tl, tr, br, bl) = rect

	cv2.circle(image, (bl[0], bl[1]), 5, (0, 0, 255), -1)
	cv2.circle(image, (br[0], br[1]), 5, (0, 0, 255), -1)
	cv2.circle(image, (tl[0], tl[1]), 5, (0, 0, 255), -1)
	cv2.circle(image, (tr[0], tr[1]), 5, (0, 0, 255), -1)

	return image

def warped(image, pts):

	rect = order_points(pts)
	(tl, tr, br, bl) = rect

	widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
	widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
	maxWidth = max(int(widthA), int(widthB)) * 2

	heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
	heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
	maxHeight = max(int(heightA), int(heightB))

	dst = np.array([
		[0, 0],
		[maxWidth - 1, 0],
		[maxWidth - 1, maxHeight - 1],
		[0, maxHeight - 1]], dtype = "float32")

	M = cv2.getPerspectiveTransform(rect, dst)
	warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

	return warped

def without_interpolation(image, pts):

	rect = order_points(pts)
	(tl, tr, br, bl) = rect

	widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
	widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
	maxWidth = max(int(widthA), int(widthB)) * 2

	heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
	heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
	maxHeight = max(int(heightA), int(heightB))

	dst = np.array([
		[310, 370],
		[maxWidth + 310, 370],
		[maxWidth + 310, maxHeight + 370],
		[310, maxHeight + 370 ]], dtype = "float32")

	M = cv2.getPerspectiveTransform(rect, dst)
	without_interpolation = cv2.warpPerspective(image, M, (2120,1790))

	return without_interpolation


def interpolation(image, pts):

	rect = order_points(pts)
	(tl, tr, br, bl) = rect

	widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
	widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
	maxWidth = max(int(widthA), int(widthB)) * 2

	heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
	heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
	maxHeight = max(int(heightA), int(heightB))

	dst = np.array([
		[310, 370],
		[maxWidth + 310, 370],
		[maxWidth + 310, maxHeight + 370],
		[310, maxHeight + 370 ]], dtype = "float32")

	M = cv2.getPerspectiveTransform(rect, dst)
	interpolation = cv2.warpPerspective(image, M, (2120,1790))

	interpolation = cv2.resize(interpolation, (0, 0), fx=2, fy=2, interpolation=cv2.INTER_LINEAR)

	return interpolation