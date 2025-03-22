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

class Processing:
    
    def __init__(self, image_path):
        self.image = cv2.imread(image_path)
        if self.image is None:
            raise ValueError("Image could not be read. Check the path.")

    def get_texture(self):
        raise NotImplementedError("Subclasses must implement this method.")

    def get_segmentation(self):
        raise NotImplementedError("Subclasses must implement this method.")