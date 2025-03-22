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

class Watershed_segmentation(Processing):
    
    def get_segmentation(self):
        image_gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(image_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        dist = cv2.distanceTransform(thresh, cv2.DIST_L2, 5) 
        ret, sure_fg = cv2.threshold(dist, 0.1 * dist.max(), 255, cv2.THRESH_BINARY) 
        sure_fg = sure_fg.astype(np.uint8)
        ret, markers = cv2.connectedComponents(sure_fg) 
        markers = cv2.watershed(self.image, markers)
        return markers