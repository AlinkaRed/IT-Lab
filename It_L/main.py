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
from texture import Texture
from laws import Laws
from histogramming import Histogramming
from visualizer_histogram import Visualizer_histogram
from GLCMAnalysis import GLCMAnalysis
from visualizer_glcm import Visualizer_glcm
from kmeans import Kmeans
from watershed_segmentation import Watershed_segmentation
from reg import Reg

def main():
    data = []
    image_names = [f'img_{i}.jpg' for i in range(1, 101)]
    mask_names = [f'mask_{i}.jpg' for i in range(1, 101)]
    for i in range(0, 100):
        image_name = image_names[i]
        mask_name = mask_names[i]
        
        texture_processor = Texture(image_name)
        mean, variance, std_dev = texture_processor.get_texture()
        
        texture_processor1 = Texture(mask_name)
        mean1, variance1, std_dev1 = texture_processor1.get_texture()
    
        
        feature_vector = Laws.get_texture(image_name)
        
        feature_vector1 = Laws.get_texture(mask_name)
    
        
        glcm_analysis = GLCMAnalysis(image_name)
        glsm, features = glcm_analysis.get_texture()
        
        glcm_analysis1 = GLCMAnalysis(mask_name)
        glsm1, features1 = glcm_analysis1.get_texture()
    
        
        image = cv2.imread(image_name)
        mask = cv2.imread(mask_name)
        intersection = np.sum(np.minimum(image, mask))
        over = np.sum(np.maximum(image, mask))
        
        true_image = cv2.imread(mask_name, cv2.IMREAD_GRAYSCALE)
        predicted_image = cv2.imread(image_name, cv2.IMREAD_GRAYSCALE)
        true_image_bin = (true_image > 0).astype(np.uint8)
        predicted_image_bin = (predicted_image > 0).astype(np.uint8)
        true_positive = np.sum(true_image_bin & predicted_image_bin)
        false_positive = np.sum(~true_image_bin & predicted_image_bin)
        false_negative = np.sum(true_image_bin & ~predicted_image_bin)
        precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
        recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
        l1_metric = np.mean(np.abs(true_image.astype(np.float32) - predicted_image.astype(np.float32)))
    
        
        w = Watershed_segmentation(image_name)
        water = w.get_segmentation()
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        intersection_w = np.sum(np.minimum(water, mask))
        over_w = np.sum(np.maximum(water, mask))
    
        predicted_image = water
        predicted_image_bin = (predicted_image > 0).astype(np.uint8)
        true_positive = np.sum(true_image_bin & predicted_image_bin)
        false_positive = np.sum(~true_image_bin & predicted_image_bin)
        false_negative = np.sum(true_image_bin & ~predicted_image_bin)
        precision_w = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
        recall_w = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
        l1_metric_w = np.mean(np.abs(true_image.astype(np.float32) - predicted_image.astype(np.float32)))
        
        k = Kmeans(image_name)
        km = k.get_segmentation()
        intersection_k = np.sum(np.minimum(km, mask))
        over_k = np.sum(np.maximum(km, mask))
        
        predicted_image = km
        predicted_image_bin = (predicted_image > 0).astype(np.uint8)
        true_positive = np.sum(true_image_bin & predicted_image_bin)
        false_positive = np.sum(~true_image_bin & predicted_image_bin)
        false_negative = np.sum(true_image_bin & ~predicted_image_bin)
        precision_k = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
        recall_k = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
        l1_metric_k = np.mean(np.abs(true_image.astype(np.float32) - predicted_image.astype(np.float32)))
    
        
        data.append({'Имя изображения': image_name, 
                     'Имя маски': mask_name, 
                     'Среднее значение изображения': mean, 
                     'Среднее значение маски': mean1, 
                     'Дисперсия изображения': variance, 
                     'Дисперсия маски': variance1, 
                     'Стандартное отклонение изображения': std_dev, 
                     'Стандартное отклонение маски': std_dev1, 
                     'Текстурные характеристики Laws изображения': feature_vector,
                     'Текстурные характеристики Laws маски': feature_vector1,
                     'Признаки изображения': features,
                     'Признаки маски': features1,
                     'Матрица взаимной встречаемости пикселей изображения': glsm,
                     'Матрица взаимной встречаемости пикселей маски': glsm1,
                     'intersection изображения и маски': intersection,
                     'over изображения и маски': over, 
                     'precision изображения и маски': precision,
                     'recall изображения и маски': recall,
                     'l1_metric изображения и маски': l1_metric,
                     'intersection watershed_segmentation и маски': intersection_w,
                     'over watershed_segmentation и маски': over_w,
                     'precision watershed_segmentation и маски': precision_w,
                     'recall watershed_segmentation и маски': recall_w,
                     'l1_metric watershed_segmentation и маски': l1_metric_w,
                     'intersection kmeans и маски': intersection_k,
                     'over kmeans и маски': over_k, 
                     'precision kmeans и маски': precision_k,
                     'recall kmeans и маски': recall_k,
                     'l1_metric kmeans и маски': l1_metric_k})
    df = pd.DataFrame(data)
    df.to_excel('results.xlsx', index=False)

if __name__ == "__main__":
    main()