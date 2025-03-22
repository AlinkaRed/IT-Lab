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

class GLCMAnalysis(Processing):

    def calculate_entropy(self, glcm):
        glcm_normalized = glcm / np.sum(glcm)
        glcm_nonzero = glcm_normalized[glcm_normalized > 0]
        return -np.sum(glcm_nonzero * np.log2(glcm_nonzero))

    def get_texture(self, distances=[1], angles=[0]):
        gray_image = rgb2gray(self.image)
        gray_image = img_as_ubyte(gray_image)
        
        glcm = graycomatrix(gray_image, distances=distances, angles=angles, levels=256, symmetric=True, normed=False)
        
        contrast = graycoprops(glcm, 'contrast')[0, 0]
        energy = graycoprops(glcm, 'energy')[0, 0]
        homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
        entropy = self.calculate_entropy(glcm[:, :, 0, 0])
        
        features = {
            'contrast': contrast,
            'energy': energy,
            'homogeneity': homogeneity,
            'entropy': entropy
        }
        
        return glcm[:, :, 0, 0], features