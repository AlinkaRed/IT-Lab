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

class Laws(Processing):

    @staticmethod
    def laws_kernels():
        L5 = np.array([1, 4, 6, 4, 1])
        E5 = np.array([-1, -2, 0, 2, 1])
        S5 = np.array([-1, 0, 2, 0, -1])
        R5 = np.array([1, -4, 6, -4, 1])
        W5 = np.array([-1, 2, 0, -2, 1])
        vectors = [L5, E5, S5, R5, W5]
        names = ['L5', 'E5', 'S5', 'R5', 'W5']
        filters = {}
        for i, vec1 in enumerate(vectors):
            for j, vec2 in enumerate(vectors):
                kernel = np.outer(vec1, vec2)
                filters[f"{names[i]}{names[j]}"] = kernel

        return filters

    @staticmethod
    def get_texture(image_path):
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Use stored image path
        if image is None:
            raise ValueError("Не удалось загрузить изображение")

        image = image.astype(np.float32)
        image -= cv2.blur(image, (15, 15))

        kernels = Laws.laws_kernels()

        feature_vector = []
        for name, kernel in kernels.items():
            filtered = cv2.filter2D(image, -1, kernel)
            energy = np.mean(np.abs(filtered))
            feature_vector.append(energy)

        return np.array(feature_vector)