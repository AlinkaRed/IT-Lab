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

class Reg:

    @staticmethod
    def homo_average(img, mask, point, T):
        av_val = img[mask > 0].sum() / np.count_nonzero(img[mask > 0])
                                                                
        if abs(av_val - img[point]) <= T:
            return True
        
        return False

    @staticmethod
    def get_segmentation(image, seed_point, homo_fun, r, T):
        mask = np.zeros(image_gray.shape, np.uint8)
        mask[seed_point] = 1
        count = 1
        q = 0
        while count > 0 and q<=10:
            count = 0
            local_mask = np.zeros(image_gray.shape, np.uint8)
            for i in range(r, image.shape[0] - r):
                for j in range(r, image.shape[1] - r):
                    if mask[i,j]==0 and mask[i - r:i + r, j-r: j+r].sum() > 0:
                        if homo_fun(image, mask, (i,j), T):
                            local_mask[i,j] = 1
            count = np.count_nonzero(local_mask)
            #print(count)
            mask += local_mask
            q+=1
            
        return mask*255