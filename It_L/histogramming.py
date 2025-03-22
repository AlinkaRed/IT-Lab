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

class Histogramming(Processing):

    def get_histogram(self):
        histSize = 256
        histRange = (0, 256)
        accumulate = False
        b = self.image[:, :, 0]
        b_hist = cv2.calcHist([b], [0], None, [histSize], histRange, accumulate=accumulate)
        return b_hist

    def get_cumulative_histogram(self):
        b_hist = self.get_histogram()
        b_hist_cum = b_hist.cumsum()
        return b_hist_cum

    def get_texture(self):
        hist = self.get_histogram()
        cum_hist = self.get_cumulative_histogram()
        return hist, cum_hist