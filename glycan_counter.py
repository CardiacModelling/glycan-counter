import cv2
import numpy as np

def count_glycans(im_file, bottom_left=(None, None), top_right=(None, None)):
  # Read image in grayscale
  im = cv2.imread(im_file, cv2.IMREAD_GRAYSCALE)

  # Crop area of interest
  x0, y0 = bottom_left
  x1, y1 = top_right

  if not any([x0 is None, y0 is None, x1 is None, y1 is None]):
    im = im[y1:y0, x0:x1]

  # Denoise with Gaussian and min filters
  im = cv2.GaussianBlur(im, ksize=(5,5), sigmaX=0)
  
  kernel = cv2.getStructuringElement(shape=cv2.MORPH_RECT, ksize=(2, 2))
  im = cv2.erode(im, kernel)

  # Binarize
  thresh = max(80, np.quantile(im, 0.05))
  _, im = cv2.threshold(im, thresh=thresh, maxval=255, type=cv2.THRESH_BINARY_INV)

  # Find and count shapes
  contours, _ = cv2.findContours(im, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

  min_area = 100
  max_area = 1000

  glycan_count = 0
  for contour in contours:
    area = cv2.contourArea(contour)
    if area > max_area:
      glycan_count += area//max_area
          
    elif area > min_area:
      glycan_count += 1

  return glycan_count
