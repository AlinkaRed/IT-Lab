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

class Texture(Processing):
    def get_texture(self):
        gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        mean = np.mean(gray_image)
        variance = np.var(gray_image)
        std_dev = np.std(gray_image)
        return mean, variance, std_dev