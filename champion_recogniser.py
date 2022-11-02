import numpy as np
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
import cv2

def mse(img1, img2):
    error = np.sum((img1.astype('float') - img2.astype('float')) ** 2)
    error /= float(img1.shape[0] * img1.shape[1])
    return error
