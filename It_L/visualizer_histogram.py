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

class Visualizer_histogram:
    @staticmethod
    def plot_histogram(hist):
        plt.plot(hist)
        plt.title("Гистограмма яркости (B)")
        plt.xlabel("Яркость")
        plt.ylabel("Частота")
        plt.show()

    @staticmethod
    def plot_cumulative_histogram(cum_hist):
        plt.plot(cum_hist)
        plt.title("Кумулятивная гистограмма яркости (B)")
        plt.xlabel("Яркость")
        plt.ylabel("Кумулятивная частота")
        plt.show()