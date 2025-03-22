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

class Visualizer_glcm:

    @staticmethod
    def visualize_glcm(glcm):
        plt.figure(figsize=(8, 6))
        plt.imshow(glcm, cmap='gray', interpolation='nearest')
        plt.title("Матрица взаимной встречаемости пикселей")
        plt.xlabel("Уровень 1")
        plt.ylabel("Уровень 2")
        plt.colorbar(label="Частота")
        plt.show()