import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import math
from skimage.feature import graycomatrix, graycoprops
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.util import img_as_ubyte
import pandas as pd 
from processing import Processing

class Kmeans(Processing):
    def get_segmentation(self):
        image_gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        pixel_values = image_gray.reshape((-1, 1)) 
        pixel_values = np.float32(pixel_values)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        k = 3
        _, labels, centers = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        centers = np.uint8(centers)
        segmented_image = centers[labels.flatten()] 
        segmented_image = segmented_image.reshape(image_gray.shape) 
        return segmented_image