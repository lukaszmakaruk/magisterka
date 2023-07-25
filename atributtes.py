import pytesseract
from pytube import YouTube
from datetime import datetime
import os
import cv2
import numpy as np
import math
from colorthief import ColorThief
import pandas as pd
import pickle
from image_similarity_measures.quality_metrics import rmse, psnr, uiq, sam, sre
from skimage.metrics import structural_similarity as ssim
from PIL import Image

pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract'


# ZABAWA OBRAZEM
def get_text(image):
    config = r'--psm 7'
    text = pytesseract.image_to_string(image, config=config)
    return text


def get_gray_scale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def remove_noise(image):
    return cv2.medianBlur(image, 5)


def thresholding(image):
    return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]


def resize(image, scale_percent):
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    return resized


def show(image):
    cv2.imshow('1', image)
    cv2.waitKey()


def unsharp_mask(image, kernel_size=(5, 5), sigma=1.0, amount=1.0, threshold=0):
    """Return a sharpened version of the image, using an unsharp mask."""
    blurred = cv2.GaussianBlur(image, kernel_size, sigma)
    sharpened = float(amount + 1) * image - float(amount) * blurred
    sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
    sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
    sharpened = sharpened.round().astype(np.uint8)
    if threshold > 0:
        low_contrast_mask = np.absolute(image - blurred) < threshold
        np.copyto(sharpened, image, where=low_contrast_mask)
    return sharpened


def mse_calc(img1, img2):
    error = np.sum((img1.astype('float') - img2.astype('float')) ** 2)
    error /= float(img1.shape[0] * img1.shape[1])
    return error


custom_config = r'--oem 3 --psm 6'
custom_config2 = r'--psm 13 --oem 3 -c tessedit_char_whitelist=0123456789:'


# ATRYBUTY WYSOKOPOZIOMOWE
# gold
def get_gold_blue(image):
    img = image[:35, 540:590]
    img = resize(img, 150)
    img = get_gray_scale(img)
    img = unsharp_mask(img)
    text = pytesseract.image_to_string(img, config=r'--psm 7 -c tessedit_char_whitelist=0123456789k.')
    text = text.strip()
    try:
        text = int(float(text[:-1]) * 1000)
    except:
        text = None
    try:
        if text > 100000:
            raise
    except:
        img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        img = img.crop((547, 0, 590, 35))
        skala = 1000 / 100
        img = img.resize((round(img.size[0] * skala), round(img.size[1] * skala)))
        img = img.convert('L')
        text = pytesseract.image_to_string(img, config=r'--psm 7 -c tessedit_char_whitelist=0123456789k.')
        text = text.strip()
        if int(float(text[:-1]) * 1000) > 100000:
            text = int(text[:-1]) * 100
        else:
            text = int(float(text[:-1]) * 1000)
    return text


def get_gold_red(image):
    img = image[:35, 730:780]
    img = resize(img, 150)
    img = get_gray_scale(img)
    img = unsharp_mask(img)
    text = pytesseract.image_to_string(img, config=r'--psm 7 -c tessedit_char_whitelist=0123456789k.')
    text = text.strip()
    try:
        text = int(float(text[:-1]) * 1000)
    except:
        text = None
    try:
        if text > 100000:
            raise
    except:
        img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        img = img.crop((730, 0, 780, 35))
        skala = 600 / 100
        img = img.resize((round(img.size[0] * skala), round(img.size[1] * skala)))
        img = img.convert('L')
        text = pytesseract.image_to_string(img, config=r'--psm 7 -c tessedit_char_whitelist=0123456789k.')
        text = text.strip()
        if int(float(text[:-1]) * 1000) > 100000:
            text = int(text[:-1]) * 100
        else:
            text = int(float(text[:-1]) * 1000)
    return text


# towers


def get_tower_blue(image):
    img = image[:35, 485:510]
    img = resize(img, 120)
    img = get_gray_scale(img)
    img = unsharp_mask(img)
    text = pytesseract.image_to_string(img, config=r'--psm 7 -c tessedit_char_whitelist=0123456789')
    try:
        text = int(text)
    except:
        try:
            img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            img = img.crop((485, 0, 510, 35))
            skala = 340 / 100
            img = img.resize((round(img.size[0] * skala), round(img.size[1] * skala)))
            img = img.convert('L')
            text = pytesseract.image_to_string(img, config=r'--psm 7 -c tessedit_char_whitelist=0123456789/')
            text = int(text)
        except:
            text = None
    return text


def get_tower_red(image):
    img = image[:35, 795:810]
    img = resize(img, 120)
    img = get_gray_scale(img)
    img = unsharp_mask(img)
    text = pytesseract.image_to_string(img, config=r'--psm 7 -c tessedit_char_whitelist=0123456789')
    try:
        text = int(text)
    except:
        try:
            img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            img = img.crop((795, 0, 810, 35))
            skala = 340 / 100
            img = img.resize((round(img.size[0] * skala), round(img.size[1] * skala)))
            img = img.convert('L')
            text = pytesseract.image_to_string(img, config=r'--psm 7 -c tessedit_char_whitelist=0123456789')
            text = int(text)
        except:
            text = None
    return text


# kills


def get_kills_blue(image):
    img = image[:35, 590:630]
    resized = resize(img, 250)
    gray_scaled = get_gray_scale(resized)
    sharpen = unsharp_mask(gray_scaled)
    text = pytesseract.image_to_string(sharpen, config=r'--psm 7 -c tessedit_char_whitelist=0123456789')
    try:
        text = int(text)
    except:
        try:
            img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            img = img.crop((590, 0, 630, 35))
            skala = 120 / 100
            img = img.resize((round(img.size[0] * skala), round(img.size[1] * skala)))
            img = img.convert('L')
            text = pytesseract.image_to_string(img, config=r'--psm 7 -c tessedit_char_whitelist=0123456789')
            text = text.strip()
            text = int(text)
        except:
            text = None
    return text


def get_kills_red(image):
    img = image[:35, 650:690]
    resized = resize(img, 250)
    gray_scaled = get_gray_scale(resized)
    sharpen = unsharp_mask(gray_scaled)
    text = pytesseract.image_to_string(sharpen, config=r'--psm 7 -c tessedit_char_whitelist=0123456789')
    try:
        text = int(text)
    except:
        try:
            img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            img = img.crop((650, 0, 690, 35))
            skala = 120 / 100
            img = img.resize((round(img.size[0] * skala), round(img.size[1] * skala)))
            img = img.convert('L')
            text = pytesseract.image_to_string(img, config=r'--psm 7 -c tessedit_char_whitelist=0123456789')
            text = text.strip()
            text = int(text)
        except:
            text = None
    return text


# kda


def get_kda_top_blue(image):
    try:
        img = image[560:585, 540:580]
        img = resize(img, 120)
        img = get_gray_scale(img)
        img = unsharp_mask(img)
        text = pytesseract.image_to_string(img, config=r'--psm 7 -c tessedit_char_whitelist=0123456789/')
        text = text.strip().split('/')
        if text[2] > 20:
            raise
        if int(text[1]) != 0:
            text = (int(text[0]) + int(text[2])) / int(text[1])
        else:
            text = int(text[0]) + int(text[2])
    except:
        try:
            img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            img = img.crop((540, 565, 580, 590))
            skala = 340 / 100
            img = img.resize((round(img.size[0] * skala), round(img.size[1] * skala)))
            img = img.convert('L')
            text = pytesseract.image_to_string(img, config=r'--psm 7 -c tessedit_char_whitelist=0123456789/')
            text = text.strip().split('/')
            if int(text[1]) != 0:
                text = (int(text[0]) + int(text[2])) / int(text[1])
            else:
                text = int(text[0]) + int(text[2])
        except:
            try:
                img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                img = img.crop((540, 565, 580, 590))
                skala = 3000 / 100
                img = img.resize((round(img.size[0] * skala), round(img.size[1] * skala)))
                img = img.convert('L')
                text = pytesseract.image_to_string(img, config=r'--psm 7 -c tessedit_char_whitelist=0123456789/')
                text = text.strip().split('/')
                if int(text[1]) != 0:
                    text = (int(text[0]) + int(text[2])) / int(text[1])
                else:
                    text = int(text[0]) + int(text[2])
            except:
                text = None
    return text


def get_kda_top_red(image):
    try:
        img = image[565:585, 705:750]
        img = resize(img, 120)
        img = get_gray_scale(img)
        img = unsharp_mask(img)
        text = pytesseract.image_to_string(img, config=r'--psm 7 -c tessedit_char_whitelist=0123456789/')
        text = text.strip().split('/')
        if text[2] > 20:
            raise
        if int(text[1]) != 0:
            text = (int(text[0]) + int(text[2])) / int(text[1])
        else:
            text = int(text[0]) + int(text[2])
    except:
        try:
            img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            img = img.crop((705, 565, 750, 590))
            skala = 340 / 100
            img = img.resize((round(img.size[0] * skala), round(img.size[1] * skala)))
            img = img.convert('L')
            text = pytesseract.image_to_string(img, config=r'--psm 7 -c tessedit_char_whitelist=0123456789/')
            text = text.strip().split('/')
            if int(text[1]) != 0:
                text = (int(text[0]) + int(text[2])) / int(text[1])
            else:
                text = int(text[0]) + int(text[2])
        except:
            try:
                img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                img = img.crop((705, 565, 750, 590))
                skala = 3000 / 100
                img = img.resize((round(img.size[0] * skala), round(img.size[1] * skala)))
                img = img.convert('L')
                text = pytesseract.image_to_string(img, config=r'--psm 7 -c tessedit_char_whitelist=0123456789/')
                text = text.strip().split('/')
                if int(text[1]) != 0:
                    text = (int(text[0]) + int(text[2])) / int(text[1])
                else:
                    text = int(text[0]) + int(text[2])
            except:
                text = None
    return text


def get_kda_jungle_blue(image):
    try:
        img = image[590:620, 544:580]
        img = resize(img, 120)
        img = get_gray_scale(img)
        img = unsharp_mask(img)
        text = pytesseract.image_to_string(img, config=r'--psm 7 -c tessedit_char_whitelist=0123456789/')
        text = text.strip().split('/')
        if text[2] > 20:
            raise
        if int(text[1]) != 0:
            text = (int(text[0]) + int(text[2])) / int(text[1])
        else:
            text = int(text[0]) + int(text[2])
    except:
        try:
            img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            img = img.crop((544, 590, 580, 620))
            skala = 340 / 100
            img = img.resize((round(img.size[0] * skala), round(img.size[1] * skala)))
            img = img.convert('L')
            text = pytesseract.image_to_string(img, config=r'--psm 7 -c tessedit_char_whitelist=0123456789/')
            text = text.strip().split('/')
            if int(text[1]) != 0:
                text = (int(text[0]) + int(text[2])) / int(text[1])
            else:
                text = int(text[0]) + int(text[2])
        except:
            try:
                img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                img = img.crop((544, 590, 580, 590))
                skala = 3000 / 100
                img = img.resize((round(img.size[0] * skala), round(img.size[1] * skala)))
                img = img.convert('L')
                text = pytesseract.image_to_string(img, config=r'--psm 7 -c tessedit_char_whitelist=0123456789/')
                text = text.strip().split('/')
                if int(text[1]) != 0:
                    text = (int(text[0]) + int(text[2])) / int(text[1])
                else:
                    text = int(text[0]) + int(text[2])
            except:
                try:
                    img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                    img = img.crop((544, 590, 580, 590))
                    skala = 600 / 100
                    img = img.resize((round(img.size[0] * skala), round(img.size[1] * skala)))
                    img = img.convert('L')
                    text = pytesseract.image_to_string(img, config=r'--psm 7 -c tessedit_char_whitelist=0123456789/')
                    text = text.strip().split('/')
                    if int(text[1]) != 0:
                        text = (int(text[0]) + int(text[2])) / int(text[1])
                    else:
                        text = int(text[0]) + int(text[2])
                except:
                    text = None
    return text


def get_kda_jungle_red(image):
    try:
        img = image[595:615, 710:745]  # [590:620, 544:580]
        img = resize(img, 120)
        img = get_gray_scale(img)
        img = unsharp_mask(img)
        text = pytesseract.image_to_string(img, config=r'--psm 7 -c tessedit_char_whitelist=0123456789/')
        text = text.strip().split('/')
        if text[2] > 20:
            raise
        if int(text[1]) != 0:
            text = (int(text[0]) + int(text[2])) / int(text[1])
        else:
            text = int(text[0]) + int(text[2])
    except:
        try:
            img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            img = img.crop((710, 595, 745, 620))
            skala = 340 / 100
            img = img.resize((round(img.size[0] * skala), round(img.size[1] * skala)))
            img = img.convert('L')
            text = pytesseract.image_to_string(img, config=r'--psm 7 -c tessedit_char_whitelist=0123456789/')
            text = text.strip().split('/')
            if int(text[1]) != 0:
                text = (int(text[0]) + int(text[2])) / int(text[1])
            else:
                text = int(text[0]) + int(text[2])
        except:
            try:
                img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                img = img.crop((710, 595, 745, 620))
                skala = 3000 / 100
                img = img.resize((round(img.size[0] * skala), round(img.size[1] * skala)))
                img = img.convert('L')
                text = pytesseract.image_to_string(img, config=r'--psm 7 -c tessedit_char_whitelist=0123456789/')
                text = text.strip().split('/')
                if int(text[1]) != 0:
                    text = (int(text[0]) + int(text[2])) / int(text[1])
                else:
                    text = int(text[0]) + int(text[2])
            except:
                try:
                    img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                    img = img.crop((710, 595, 745, 620))
                    skala = 600 / 100
                    img = img.resize((round(img.size[0] * skala), round(img.size[1] * skala)))
                    img = img.convert('L')
                    text = pytesseract.image_to_string(img, config=r'--psm 7 -c tessedit_char_whitelist=0123456789/')
                    text = text.strip().split('/')
                    if int(text[1]) != 0:
                        text = (int(text[0]) + int(text[2])) / int(text[1])
                    else:
                        text = int(text[0]) + int(text[2])
                except:
                    text = None
    return text


def get_kda_mid_blue(image):
    try:
        img = image[635:650, 537:580]
        img = resize(img, 120)
        img = get_gray_scale(img)
        img = unsharp_mask(img)
        text = pytesseract.image_to_string(img, config=r'--psm 7 -c tessedit_char_whitelist=0123456789/')
        text = text.strip().split('/')
        if text[2] > 20:
            raise
        if int(text[1]) != 0:
            text = (int(text[0]) + int(text[2])) / int(text[1])
        else:
            text = int(text[0]) + int(text[2])
    except:
        try:
            img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            img = img.crop((537, 630, 580, 645))
            skala = 340 / 100
            img = img.resize((round(img.size[0] * skala), round(img.size[1] * skala)))
            img = img.convert('L')
            text = pytesseract.image_to_string(img, config=r'--psm 7 -c tessedit_char_whitelist=0123456789/')
            text = text.strip().split('/')
            if int(text[1]) != 0:
                text = (int(text[0]) + int(text[2])) / int(text[1])
            else:
                text = int(text[0]) + int(text[2])
        except:
            try:
                img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                img = img.crop((540, 630, 580, 645))
                skala = 3000 / 100
                img = img.resize((round(img.size[0] * skala), round(img.size[1] * skala)))
                img = img.convert('L')
                text = pytesseract.image_to_string(img, config=r'--psm 7 -c tessedit_char_whitelist=0123456789/')
                text = text.strip().split('/')
                if int(text[1]) != 0:
                    text = (int(text[0]) + int(text[2])) / int(text[1])
                else:
                    text = int(text[0]) + int(text[2])
            except:
                text = None
    return text


def get_kda_mid_red(image):
    try:
        img = image[635:650, 710:750]
        img = resize(img, 120)
        img = get_gray_scale(img)
        img = unsharp_mask(img)
        text = pytesseract.image_to_string(img, config=r'--psm 7 -c tessedit_char_whitelist=0123456789/')
        text = text.strip().split('/')
        if text[2] > 20:
            raise
        if int(text[1]) != 0:
            text = (int(text[0]) + int(text[2])) / int(text[1])
        else:
            text = int(text[0]) + int(text[2])
    except:
        try:
            img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            img = img.crop((710, 630, 750, 645))
            skala = 340 / 100
            img = img.resize((round(img.size[0] * skala), round(img.size[1] * skala)))
            img = img.convert('L')
            text = pytesseract.image_to_string(img, config=r'--psm 7 -c tessedit_char_whitelist=0123456789/')
            text = text.strip().split('/')
            if int(text[1]) != 0:
                text = (int(text[0]) + int(text[2])) / int(text[1])
            else:
                text = int(text[0]) + int(text[2])
        except:
            try:
                img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                img = img.crop((710, 630, 750, 645))
                skala = 3000 / 100
                img = img.resize((round(img.size[0] * skala), round(img.size[1] * skala)))
                img = img.convert('L')
                text = pytesseract.image_to_string(img, config=r'--psm 7 -c tessedit_char_whitelist=0123456789/')
                text = text.strip().split('/')
                if int(text[1]) != 0:
                    text = (int(text[0]) + int(text[2])) / int(text[1])
                else:
                    text = int(text[0]) + int(text[2])
            except:
                text = None
    return text


def get_kda_adc_blue(image):
    try:
        img = image[662:680, 535:580]
        img = resize(img, 120)
        img = get_gray_scale(img)
        img = unsharp_mask(img)
        text = pytesseract.image_to_string(img, config=r'--psm 7 -c tessedit_char_whitelist=0123456789/')
        text = text.strip().split('/')
        if text[2] > 20:
            raise
        if int(text[1]) != 0:
            text = (int(text[0]) + int(text[2])) / int(text[1])
        else:
            text = int(text[0]) + int(text[2])
    except:
        try:
            img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            img = img.crop((535, 662, 580, 680))
            skala = 340 / 100
            img = img.resize((round(img.size[0] * skala), round(img.size[1] * skala)))
            img = img.convert('L')
            text = pytesseract.image_to_string(img, config=r'--psm 7 -c tessedit_char_whitelist=0123456789/')
            text = text.strip().split('/')
            if int(text[1]) != 0:
                text = (int(text[0]) + int(text[2])) / int(text[1])
            else:
                text = int(text[0]) + int(text[2])
        except:
            try:
                img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                img = img.crop((535, 662, 580, 680))
                skala = 3000 / 100
                img = img.resize((round(img.size[0] * skala), round(img.size[1] * skala)))
                img = img.convert('L')
                text = pytesseract.image_to_string(img, config=r'--psm 7 -c tessedit_char_whitelist=0123456789/')
                text = text.strip().split('/')
                if int(text[1]) != 0:
                    text = (int(text[0]) + int(text[2])) / int(text[1])
                else:
                    text = int(text[0]) + int(text[2])
            except:
                text = None
    return text


def get_kda_adc_red(image):
    try:
        img = image[662:680, 710:750]
        img = resize(img, 120)
        img = get_gray_scale(img)
        img = unsharp_mask(img)
        text = pytesseract.image_to_string(img, config=r'--psm 7 -c tessedit_char_whitelist=0123456789/')
        text = text.strip().split('/')
        if text[2] > 20:
            raise
        if int(text[1]) != 0:
            text = (int(text[0]) + int(text[2])) / int(text[1])
        else:
            text = int(text[0]) + int(text[2])
    except:
        try:
            img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            img = img.crop((710, 662, 750, 680))
            skala = 340 / 100
            img = img.resize((round(img.size[0] * skala), round(img.size[1] * skala)))
            img = img.convert('L')
            text = pytesseract.image_to_string(img, config=r'--psm 7 -c tessedit_char_whitelist=0123456789/')
            text = text.strip().split('/')
            if int(text[1]) != 0:
                text = (int(text[0]) + int(text[2])) / int(text[1])
            else:
                text = int(text[0]) + int(text[2])
        except:
            try:
                img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                img = img.crop((710, 662, 750, 680))
                skala = 3000 / 100
                img = img.resize((round(img.size[0] * skala), round(img.size[1] * skala)))
                img = img.convert('L')
                text = pytesseract.image_to_string(img, config=r'--psm 7 -c tessedit_char_whitelist=0123456789/')
                text = text.strip().split('/')
                if int(text[1]) != 0:
                    text = (int(text[0]) + int(text[2])) / int(text[1])
                else:
                    text = int(text[0]) + int(text[2])
            except:
                text = None
    return text


def get_kda_supp_blue(image):
    try:
        img = image[690:710, 540:580]
        img = resize(img, 120)
        img = get_gray_scale(img)
        img = unsharp_mask(img)
        text = pytesseract.image_to_string(img, config=r'--psm 7 -c tessedit_char_whitelist=0123456789/')
        text = text.strip().split('/')
        if text[2] > 20:
            raise
        if int(text[1]) != 0:
            text = (int(text[0]) + int(text[2])) / int(text[1])
        else:
            text = int(text[0]) + int(text[2])
    except:
        try:
            img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            img = img.crop((540, 690, 580, 710))
            skala = 340 / 100
            img = img.resize((round(img.size[0] * skala), round(img.size[1] * skala)))
            img = img.convert('L')
            text = pytesseract.image_to_string(img, config=r'--psm 7 -c tessedit_char_whitelist=0123456789/')
            text = text.strip().split('/')
            if int(text[1]) != 0:
                text = (int(text[0]) + int(text[2])) / int(text[1])
            else:
                text = int(text[0]) + int(text[2])
        except:
            try:
                img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                img = img.crop((540, 690, 580, 710))
                skala = 3000 / 100
                img = img.resize((round(img.size[0] * skala), round(img.size[1] * skala)))
                img = img.convert('L')
                text = pytesseract.image_to_string(img, config=r'--psm 7 -c tessedit_char_whitelist=0123456789/')
                text = text.strip().split('/')
                if int(text[1]) != 0:
                    text = (int(text[0]) + int(text[2])) / int(text[1])
                else:
                    text = int(text[0]) + int(text[2])
            except:
                text = None
    return text


def get_kda_supp_red(image):
    try:
        img = image[690:710, 710:750]
        img = resize(img, 120)
        img = get_gray_scale(img)
        img = unsharp_mask(img)
        text = pytesseract.image_to_string(img, config=r'--psm 7 -c tessedit_char_whitelist=0123456789/')
        text = text.strip().split('/')
        if text[2] > 20:
            raise
        if int(text[1]) != 0:
            text = (int(text[0]) + int(text[2])) / int(text[1])
        else:
            text = int(text[0]) + int(text[2])
    except:
        try:
            img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            img = img.crop((710, 690, 750, 710))
            skala = 340 / 100
            img = img.resize((round(img.size[0] * skala), round(img.size[1] * skala)))
            img = img.convert('L')
            text = pytesseract.image_to_string(img, config=r'--psm 7 -c tessedit_char_whitelist=0123456789/')
            text = text.strip().split('/')
            if int(text[1]) != 0:
                text = (int(text[0]) + int(text[2])) / int(text[1])
            else:
                text = int(text[0]) + int(text[2])
        except:
            try:
                img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                img = img.crop((710, 690, 750, 710))
                skala = 3000 / 100
                img = img.resize((round(img.size[0] * skala), round(img.size[1] * skala)))
                img = img.convert('L')
                text = pytesseract.image_to_string(img, config=r'--psm 7 -c tessedit_char_whitelist=0123456789/')
                text = text.strip().split('/')
                if int(text[1]) != 0:
                    text = (int(text[0]) + int(text[2])) / int(text[1])
                else:
                    text = int(text[0]) + int(text[2])
            except:
                text = None
    return text


# minions


def get_minions_top_blue(image):
    img = image[568:585, 575:610]
    gray_scaled = get_gray_scale(img)
    sharpen = unsharp_mask(gray_scaled)
    text = pytesseract.image_to_string(sharpen, config=r'--psm 7 -c tessedit_char_whitelist=0123456789')
    try:
        text = int(text)
    except:
        text = None
    return text


def get_minions_top_red(image):
    img = image[565:585, 680:710]
    resized = resize(img, 130)
    gray_scaled = get_gray_scale(resized)
    sharpen = unsharp_mask(gray_scaled)
    text = pytesseract.image_to_string(sharpen, config=r'--psm 7 -c tessedit_char_whitelist=0123456789')
    try:
        text = int(text)
    except:
        text = None
    return text


def get_minions_jungle_blue(image):
    img = image[595:615, 575:610]
    resized = resize(img, 110)
    gray_scaled = get_gray_scale(resized)
    sharpen = unsharp_mask(gray_scaled)
    text = pytesseract.image_to_string(sharpen, config=r'--psm 7 -c tessedit_char_whitelist=0123456789')
    try:
        text = int(text)
    except:
        text = None
    return text


def get_minions_jungle_red(image):
    img = image[595:615, 680:710]
    resized = resize(img, 120)
    gray_scaled = get_gray_scale(resized)
    sharpen = unsharp_mask(gray_scaled)
    text = pytesseract.image_to_string(sharpen, config=r'--psm 7 -c tessedit_char_whitelist=0123456789')
    try:
        text = int(text)
    except:
        text = None
    return text


def get_minions_mid_blue(image):
    img = image[630:650, 575:610]
    resized = resize(img, 120)
    gray_scaled = get_gray_scale(resized)
    sharpen = unsharp_mask(gray_scaled)
    text = pytesseract.image_to_string(sharpen, config=r'--psm 7 -c tessedit_char_whitelist=0123456789')
    try:
        text = int(text)
    except:
        text = None
    return text


def get_minions_mid_red(image):
    img = image[630:650, 680:710]
    resized = resize(img, 120)
    gray_scaled = get_gray_scale(resized)
    sharpen = unsharp_mask(gray_scaled)
    text = pytesseract.image_to_string(sharpen, config=r'--psm 7 -c tessedit_char_whitelist=0123456789')
    try:
        text = int(text)
    except:
        text = None
    return text


def get_minions_adc_blue(image):
    img = image[662:680, 575:610]
    resized = resize(img, 120)
    gray_scaled = get_gray_scale(resized)
    sharpen = unsharp_mask(gray_scaled)
    text = pytesseract.image_to_string(sharpen, config=r'--psm 7 -c tessedit_char_whitelist=0123456789')
    try:
        text = int(text)
    except:
        text = None
    return text


def get_minions_adc_red(image):
    img = image[662:680, 680:710]
    resized = resize(img, 120)
    gray_scaled = get_gray_scale(resized)
    sharpen = unsharp_mask(gray_scaled)
    text = pytesseract.image_to_string(sharpen, config=r'--psm 7 -c tessedit_char_whitelist=0123456789')
    try:
        text = int(text)
    except:
        text = None
    return text


def get_minions_supp_blue(image):
    img = image[691:708, 575:610]
    resized = resize(img, 120)
    gray_scaled = get_gray_scale(resized)
    sharpen = unsharp_mask(gray_scaled)
    text = pytesseract.image_to_string(sharpen, config=r'--psm 7 -c tessedit_char_whitelist=0123456789')
    try:
        text = int(text)
    except:
        text = None
    return text


def get_minions_supp_red(image):
    img = image[691:710, 680:710]
    resized = resize(img, 120)
    gray_scaled = get_gray_scale(resized)
    sharpen = unsharp_mask(gray_scaled)
    text = pytesseract.image_to_string(sharpen, config=r'--psm 7 -c tessedit_char_whitelist=0123456789')
    try:
        text = int(text)
    except:
        text = None
    return text


# ATRYBUTY GŁÓWNE

# time
def get_time(image):
    config = r'--psm 7  -c tessedit_char_whitelist=0123456789:'
    img = image[32:56, 610:665]
    img = resize(img, 160)
    img = get_gray_scale(img)
    img = unsharp_mask(img)
    text = pytesseract.image_to_string(img, config=config)
    try:
        text = datetime.strptime(text.strip(), '%M:%S').time()
    except:
        img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        img = img.crop((610, 32, 665, 56))
        skala = 600 / 100
        img = img.resize((round(img.size[0] * skala), round(img.size[1] * skala)))
        img = img.convert('L')
        text = pytesseract.image_to_string(img, config=r'--psm 7 -c tessedit_char_whitelist=0123456789:')
        text = text.strip()
        try:
            text = datetime.strptime(text.strip(), '%M:%S').time()
        except:
            text = None
    return text


# gold

def get_gold_diff_blue(image):
    img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    img_red_basic = img.crop((730, 0, 770, 35))
    img_blue_basic = img.crop((547, 0, 590, 35))
    skala = 150 / 100
    img_red = img_red_basic.resize((round(img_red_basic.size[0] * skala), round(img_red_basic.size[1] * skala)))
    img_red = img_red.convert('L')
    img_blue = img_blue_basic.resize((round(img_blue_basic.size[0] * skala), round(img_blue_basic.size[1] * skala)))
    img_blue = img_blue.convert('L')
    text_red = pytesseract.image_to_string(img_red, config=r'--psm 7 -c tessedit_char_whitelist=0123456789k.')
    text_blue = pytesseract.image_to_string(img_blue, config=r'--psm 7 -c tessedit_char_whitelist=0123456789k.')
    text_red = text_red.strip()
    text_blue = text_blue.strip()
    try:
        text_red = int(float(text_red[:-1]) * 1000)
        if text_red > 80000:
            text_red = int(text_red / 10)
    except:
        skala = 600 / 100
        img_red = img_red_basic.resize((round(img_red_basic.size[0] * skala), round(img_red_basic.size[1] * skala)))
        img_red = img_red.convert('L')
        text_red = pytesseract.image_to_string(img_red, config=r'--psm 7 -c tessedit_char_whitelist=0123456789k.')
        text_red = text_red.strip()
        try:
            text_red = int(float(text_red[:-1]) * 1000)
            if text_red > 80000:
                text_red = int(text_red / 10)
        except:
            text_red = None
    try:
        text_blue = int(float(text_blue[:-1]) * 1000)
        if text_blue > 80000:
            text_blue = int(text_blue / 10)
    except:
        skala = 600 / 100
        img_blue = img_blue_basic.resize((round(img_blue_basic.size[0] * skala), round(img_blue_basic.size[1] * skala)))
        img_blue = img_blue.convert('L')
        text_blue = pytesseract.image_to_string(img_blue, config=r'--psm 7 -c tessedit_char_whitelist=0123456789k.')
        text_blue = text_blue.strip()
        try:
            text_blue = int(float(text_blue[:-1]) * 1000)
            if text_blue > 80000:
                text_blue = int(text_blue / 10)
        except:
            text_blue = None
    try:
        diff = int(text_blue - text_red)
        if abs(diff) >= 20000:
            diff = diff / 10
    except:
        diff = None
    return diff


# towers

def get_tower_diff_blue(image):
    try:
        diff = get_tower_blue(image) - get_tower_red(image)
        if diff > 14:
            diff = None
    except:
        diff = None
    return diff


# kills
def get_kills_diff_blue(image):
    try:
        diff = get_kills_blue(image) - get_kills_red(image)
        if diff > 14:
            diff = None
    except:
        diff = None
    return diff


# teams
def get_team_blue(image):
    img = image[:20, 417:450]
    resized = resize(img, 130)
    gray_scaled = get_gray_scale(resized)
    sharpen = unsharp_mask(gray_scaled)
    text = pytesseract.image_to_string(sharpen, config=r'--psm 7')
    text = text.strip().upper()
    if text == 'VIT':
        text = 'Team Vitality'
    elif text == 'MAD':
        text = 'Mad Lions'
    elif text == 'SK':
        text = 'SK Gaming'
    elif text == 'G2':
        text = 'G2 Esports'
    elif text == 'RGE':
        text = 'Rogue'
    elif text == 'FNC':
        text = 'Fnatic'
    elif text == 'MSF':
        text = 'Misfits Gaming'
    elif text == 'XL':
        text = 'Excel Esports'
    elif text == 'BDS':
        text = 'Team BDS'
    elif text == 'AST':
        text = 'Astralis'
    elif text == 'SO4' or text == 'S04':
        text = 'FC Schalke 04'
    else:
        text = None
    return text


def get_team_red(image):
    img = image[:20, 825:867]
    resized = resize(img, 125)
    gray_scaled = get_gray_scale(resized)
    sharpen = unsharp_mask(gray_scaled)
    text = pytesseract.image_to_string(sharpen, config=r'--psm 7')
    text = text.strip().upper()
    if text == 'VIT':
        text = 'Team Vitality'
    elif text == 'MAD':
        text = 'Mad Lions'
    elif text == 'SK':
        text = 'SK Gaming'
    elif text == 'G2':
        text = 'G2 Esports'
    elif text == 'RGE':
        text = 'Rogue'
    elif text == 'FNC':
        text = 'Fnatic'
    elif text == 'MSF':
        text = 'Misfits Gaming'
    elif text == 'XL':
        text = 'Excel Esports'
    elif text == 'BDS':
        text = 'Team BDS'
    elif text == 'AST':
        text = 'Astralis'
    elif text == 'SO4' or text == 'S04':
        text = 'FC Schalke 04'
    else:
        text = None
    return text


# kda
def get_kda_diff_top_blue(image):
    try:
        diff = get_kda_top_blue(image) - get_kda_top_red(image)
    except:
        diff = None
    return diff


def get_kda_diff_jungle_blue(image):
    try:
        diff = get_kda_jungle_blue(image) - get_kda_jungle_red(image)
    except:
        diff = None
    return diff


def get_kda_diff_mid_blue(image):
    try:
        diff = get_kda_mid_blue(image) - get_kda_mid_red(image)
    except:
        diff = None
    return diff


def get_kda_diff_adc_blue(image):
    try:
        diff = get_kda_adc_blue(image) - get_kda_adc_red(image)
    except:
        diff = None
    return diff


def get_kda_diff_supp_blue(image):
    try:
        diff = get_kda_supp_blue(image) - get_kda_supp_red(image)
    except:
        diff = None
    return diff


# minions


def get_minions_diff_top_blue(image):
    img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    img_red_basic = img.crop((680, 565, 710, 585))
    img_blue_basic = img.crop((575, 565, 610, 585))
    skala = 120 / 100
    img_red = img_red_basic.resize((round(img_red_basic.size[0] * skala), round(img_red_basic.size[1] * skala)))
    img_blue = img_blue_basic.resize((round(img_blue_basic.size[0] * skala), round(img_blue_basic.size[1] * skala)))
    img_red = img_red.convert('L')
    img_blue = img_blue.convert('L')
    text_red = pytesseract.image_to_string(img_red, config=r'--psm 7 -c tessedit_char_whitelist=0123456789')
    text_red = text_red.strip()
    text_blue = pytesseract.image_to_string(img_blue, config=r'--psm 7 -c tessedit_char_whitelist=0123456789')
    text_blue = text_blue.strip()
    try:
        text_red = int(text_red)
        if text_red > 400 or text_red < 5:
            raise
    except:
        skala = 150 / 100
        img_red = img_red_basic.resize((round(img_red_basic.size[0] * skala), round(img_red_basic.size[1] * skala)))
        img_red = img_red.convert('L')
        text_red = pytesseract.image_to_string(img_red, config=r'--psm 7 -c tessedit_char_whitelist=0123456789')
        text_red = text_red.strip()
        try:
            text_red = int(text_red)
        except:
            text_red = None
    try:
        text_blue = int(text_blue)
        if text_blue > 400 or text_blue < 5:
            raise
    except:
        skala = 150 / 100
        img_blue = img_blue_basic.resize((round(img_blue_basic.size[0] * skala), round(img_blue_basic.size[1] * skala)))
        img_blue = img_blue.convert('L')
        text_blue = pytesseract.image_to_string(img_blue, config=r'--psm 7 -c tessedit_char_whitelist=0123456789')
        text_blue = text_blue.strip()
        try:
            text_blue = int(text_blue)
        except:
            text_blue = None
    try:
        diff = text_blue - text_red
        if abs(diff) > 80:
            raise
    except:
        skala = 600 / 100
        img_red = img_red_basic.resize((round(img_red_basic.size[0] * skala), round(img_red_basic.size[1] * skala)))
        img_blue = img_blue_basic.resize((round(img_blue_basic.size[0] * skala), round(img_blue_basic.size[1] * skala)))
        img_red = img_red.convert('L')
        img_blue = img_blue.convert('L')
        text_red = pytesseract.image_to_string(img_red, config=r'--psm 7 -c tessedit_char_whitelist=0123456789')
        text_red = text_red.strip()
        text_blue = pytesseract.image_to_string(img_blue, config=r'--psm 7 -c tessedit_char_whitelist=0123456789')
        text_blue = text_blue.strip()
        try:
            diff = int(text_blue) - int(text_red)
            if diff > 1000:
                diff = None
        except:
            diff = None
    return diff


def get_minions_diff_jungle_blue(image):
    img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    img_red_basic = img.crop((680, 595, 710, 615))
    img_blue_basic = img.crop((575, 595, 610, 615))
    skala = 150 / 100
    img_red = img_red_basic.resize((round(img_red_basic.size[0] * skala), round(img_red_basic.size[1] * skala)))
    img_blue = img_blue_basic.resize((round(img_blue_basic.size[0] * skala), round(img_blue_basic.size[1] * skala)))
    img_red = img_red.convert('L')
    img_blue = img_blue.convert('L')
    text_red = pytesseract.image_to_string(img_red, config=r'--psm 7 -c tessedit_char_whitelist=0123456789')
    text_red = text_red.strip()
    text_blue = pytesseract.image_to_string(img_blue, config=r'--psm 7 -c tessedit_char_whitelist=0123456789')
    text_blue = text_blue.strip()
    try:
        text_red = int(text_red)
        if text_red > 400 or text_red < 5:
            raise
    except:
        skala = 200 / 100
        img_red = img_red_basic.resize((round(img_red_basic.size[0] * skala), round(img_red_basic.size[1] * skala)))
        img_red = img_red.convert('L')
        text_red = pytesseract.image_to_string(img_red, config=r'--psm 7 -c tessedit_char_whitelist=0123456789')
        text_red = text_red.strip()
        try:
            text_red = int(text_red)
        except:
            text_red = None
    try:
        text_blue = int(text_blue)
        if text_blue > 400 or text_blue < 5:
            raise
    except:
        skala = 200 / 100
        img_blue = img_blue_basic.resize((round(img_blue_basic.size[0] * skala), round(img_blue_basic.size[1] * skala)))
        img_blue = img_blue.convert('L')
        text_blue = pytesseract.image_to_string(img_blue, config=r'--psm 7 -c tessedit_char_whitelist=0123456789')
        text_blue = text_blue.strip()
        try:
            text_blue = int(text_blue)
        except:
            text_blue = None
    try:
        diff = text_blue - text_red
        if abs(diff) > 80:
            raise
    except:
        skala = 600 / 100
        img_red = img_red_basic.resize((round(img_red_basic.size[0] * skala), round(img_red_basic.size[1] * skala)))
        img_blue = img_blue_basic.resize((round(img_blue_basic.size[0] * skala), round(img_blue_basic.size[1] * skala)))
        img_red = img_red.convert('L')
        img_blue = img_blue.convert('L')
        text_red = pytesseract.image_to_string(img_red, config=r'--psm 7 -c tessedit_char_whitelist=0123456789')
        text_red = text_red.strip()
        text_blue = pytesseract.image_to_string(img_blue, config=r'--psm 7 -c tessedit_char_whitelist=0123456789')
        text_blue = text_blue.strip()
        try:
            diff = int(text_blue) - int(text_red)
            if diff > 1000:
                diff = None
        except:
            diff = None
    return diff


def get_minions_diff_mid_blue(image):
    img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    img_red_basic = img.crop((680, 630, 710, 650))
    img_blue_basic = img.crop((580, 630, 610, 650))
    skala = 120 / 100
    img_red = img_red_basic.resize((round(img_red_basic.size[0] * skala), round(img_red_basic.size[1] * skala)))
    img_blue = img_blue_basic.resize((round(img_blue_basic.size[0] * skala), round(img_blue_basic.size[1] * skala)))
    img_red = img_red.convert('L')
    img_blue = img_blue.convert('L')
    text_red = pytesseract.image_to_string(img_red, config=r'--psm 7 -c tessedit_char_whitelist=0123456789')
    text_red = text_red.strip()
    text_blue = pytesseract.image_to_string(img_blue, config=r'--psm 7 -c tessedit_char_whitelist=0123456789')
    text_blue = text_blue.strip()
    try:
        text_red = int(text_red)
        if text_red > 400:
            raise
    except:
        skala = 150 / 100
        img_red = img_red_basic.resize((round(img_red_basic.size[0] * skala), round(img_red_basic.size[1] * skala)))
        img_red = img_red.convert('L')
        text_red = pytesseract.image_to_string(img_red, config=r'--psm 7 -c tessedit_char_whitelist=0123456789')
        text_red = text_red.strip()
        try:
            text_red = int(text_red)
        except:
            text_red = None
    try:
        text_blue = int(text_blue)
        if text_blue > 400:
            raise
    except:
        skala = 150 / 100
        img_blue = img_blue_basic.resize((round(img_blue_basic.size[0] * skala), round(img_blue_basic.size[1] * skala)))
        img_blue = img_blue.convert('L')
        text_blue = pytesseract.image_to_string(img_blue, config=r'--psm 7 -c tessedit_char_whitelist=0123456789')
        text_blue = text_blue.strip()
        try:
            text_blue = int(text_blue)
        except:
            text_blue = None
    try:
        diff = text_blue - text_red
        if abs(diff) > 90:
            raise
    except:
        skala = 600 / 100
        img_red = img_red_basic.resize((round(img_red_basic.size[0] * skala), round(img_red_basic.size[1] * skala)))
        img_blue = img_blue_basic.resize((round(img_blue_basic.size[0] * skala), round(img_blue_basic.size[1] * skala)))
        img_red = img_red.convert('L')
        img_blue = img_blue.convert('L')
        text_red = pytesseract.image_to_string(img_red, config=r'--psm 7 -c tessedit_char_whitelist=0123456789')
        text_red = text_red.strip()
        text_blue = pytesseract.image_to_string(img_blue, config=r'--psm 7 -c tessedit_char_whitelist=0123456789')
        text_blue = text_blue.strip()
        try:
            diff = int(text_blue) - int(text_red)
            if diff > 1000:
                diff = None
        except:
            diff = None
    return diff


def get_minions_diff_adc_blue(image):
    img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    img_red_basic = img.crop((680, 662, 710, 680))
    img_blue_basic = img.crop((575, 662, 610, 680))
    skala = 120 / 100
    img_red = img_red_basic.resize((round(img_red_basic.size[0] * skala), round(img_red_basic.size[1] * skala)))
    img_blue = img_blue_basic.resize((round(img_blue_basic.size[0] * skala), round(img_blue_basic.size[1] * skala)))
    img_red = img_red.convert('L')
    img_blue = img_blue.convert('L')
    text_red = pytesseract.image_to_string(img_red, config=r'--psm 7 -c tessedit_char_whitelist=0123456789')
    text_red = text_red.strip()
    text_blue = pytesseract.image_to_string(img_blue, config=r'--psm 7 -c tessedit_char_whitelist=0123456789')
    text_blue = text_blue.strip()
    try:
        text_red = int(text_red)
        if text_red > 400:
            raise
    except:
        skala = 150 / 100
        img_red = img_red_basic.resize((round(img_red_basic.size[0] * skala), round(img_red_basic.size[1] * skala)))
        img_red = img_red.convert('L')
        text_red = pytesseract.image_to_string(img_red, config=r'--psm 7 -c tessedit_char_whitelist=0123456789')
        text_red = text_red.strip()
        try:
            text_red = int(text_red)
        except:
            text_red = None
    try:
        text_blue = int(text_blue)
        if text_blue > 400:
            raise
    except:
        skala = 150 / 100
        img_blue = img_blue_basic.resize((round(img_blue_basic.size[0] * skala), round(img_blue_basic.size[1] * skala)))
        img_blue = img_blue.convert('L')
        text_blue = pytesseract.image_to_string(img_blue, config=r'--psm 7 -c tessedit_char_whitelist=0123456789')
        text_blue = text_blue.strip()
        try:
            text_blue = int(text_blue)
        except:
            text_blue = None
    try:
        diff = text_blue - text_red
        if diff > 90:
            raise
    except:
        skala = 600 / 100
        img_red = img_red_basic.resize((round(img_red_basic.size[0] * skala), round(img_red_basic.size[1] * skala)))
        img_blue = img_blue_basic.resize((round(img_blue_basic.size[0] * skala), round(img_blue_basic.size[1] * skala)))
        img_red = img_red.convert('L')
        img_blue = img_blue.convert('L')
        text_red = pytesseract.image_to_string(img_red, config=r'--psm 7 -c tessedit_char_whitelist=0123456789')
        text_red = text_red.strip()
        text_blue = pytesseract.image_to_string(img_blue, config=r'--psm 7 -c tessedit_char_whitelist=0123456789')
        text_blue = text_blue.strip()
        try:
            diff = int(text_blue) - int(text_red)
        except:
            diff = None
    return diff


def get_minions_diff_supp_blue(image):
    img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    img_red_basic = img.crop((680, 691, 710, 710))
    img_blue_basic = img.crop((575, 691, 610, 710))
    skala = 120 / 100
    img_red = img_red_basic.resize((round(img_red_basic.size[0] * skala), round(img_red_basic.size[1] * skala)))
    img_blue = img_blue_basic.resize((round(img_blue_basic.size[0] * skala), round(img_blue_basic.size[1] * skala)))
    img_red = img_red.convert('L')
    img_blue = img_blue.convert('L')
    text_red = pytesseract.image_to_string(img_red, config=r'--psm 7 -c tessedit_char_whitelist=0123456789')
    text_red = text_red.strip()
    text_blue = pytesseract.image_to_string(img_blue, config=r'--psm 7 -c tessedit_char_whitelist=0123456789')
    text_blue = text_blue.strip()
    try:
        text_red = int(text_red)
        if text_red > 100:
            raise
    except:
        skala = 150 / 100
        img_red = img_red_basic.resize((round(img_red_basic.size[0] * skala), round(img_red_basic.size[1] * skala)))
        img_red = img_red.convert('L')
        text_red = pytesseract.image_to_string(img_red, config=r'--psm 7 -c tessedit_char_whitelist=0123456789')
        text_red = text_red.strip()
        try:
            text_red = int(text_red)
        except:
            text_red = None
    try:
        text_blue = int(text_blue)
        if text_blue > 100:
            raise
    except:
        skala = 150 / 100
        img_blue = img_blue_basic.resize((round(img_blue_basic.size[0] * skala), round(img_blue_basic.size[1] * skala)))
        img_blue = img_blue.convert('L')
        text_blue = pytesseract.image_to_string(img_blue, config=r'--psm 7 -c tessedit_char_whitelist=0123456789')
        text_blue = text_blue.strip()
        try:
            text_blue = int(text_blue)
        except:
            text_blue = None
    try:
        diff = text_blue - text_red
        if diff > 50:
            raise
    except:
        skala = 600 / 100
        img_red = img_red_basic.resize((round(img_red_basic.size[0] * skala), round(img_red_basic.size[1] * skala)))
        img_blue = img_blue_basic.resize((round(img_blue_basic.size[0] * skala), round(img_blue_basic.size[1] * skala)))
        img_red = img_red.convert('L')
        img_blue = img_blue.convert('L')
        text_red = pytesseract.image_to_string(img_red, config=r'--psm 7 -c tessedit_char_whitelist=0123456789')
        text_red = text_red.strip()
        text_blue = pytesseract.image_to_string(img_blue, config=r'--psm 7 -c tessedit_char_whitelist=0123456789')
        text_blue = text_blue.strip()
        try:
            diff = int(text_blue) - int(text_red)
        except:
            diff = None
    return diff


# dragons
def dragon_detector(frame):
    dragons = ['hextech', 'infernal', 'mountain', 'ocean', 'wind']
    dragons_counter = {}
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_blue = frame[:75, 400:650]
    frame_red = frame[:75, 660:900]
    frame_sites = [frame_blue, frame_red]
    for dragon in dragons:
        for frame in frame_sites:
            template = cv2.imread(f'dragons/{dragon}.jpg', cv2.IMREAD_GRAYSCALE)
            result = cv2.matchTemplate(template, frame, cv2.TM_CCOEFF_NORMED)
            # set threshold
            if dragon == 'infernal':
                threshold = 0.85
            else:
                threshold = 0.87
            # Get the location of all matches above the threshold
            locations = np.where(result >= threshold)
            locations = list(zip(*locations[::-1]))

            template_width = template.shape[1]
            template_height = template.shape[0]

            # Draw a rectangle around each detected object and count them
            count = 0
            for loc in locations:
                top_left = loc
                bottom_right = (top_left[0] + template_width, top_left[1] + template_height)
                cv2.rectangle(frame, top_left, bottom_right, (0, 0, 255), 2)
                count += 1
            if np.array_equal(frame, frame_blue):
                dragons_counter[f'{dragon}_blue'] = count
            elif np.array_equal(frame, frame_red):
                dragons_counter[f'{dragon}_red'] = count
    hextech_dragon_diff = dragons_counter['hextech_blue'] - dragons_counter['hextech_red']
    infernal_dragon_diff = dragons_counter['infernal_blue'] - dragons_counter['infernal_red']
    mountain_dragon_diff = dragons_counter['mountain_blue'] - dragons_counter['mountain_red']
    ocean_dragon_diff = dragons_counter['ocean_blue'] - dragons_counter['ocean_red']
    wind_dragon_diff = dragons_counter['wind_blue'] - dragons_counter['wind_red']
    dragon_diff = [hextech_dragon_diff, infernal_dragon_diff, mountain_dragon_diff, ocean_dragon_diff,
                   wind_dragon_diff]
    return dragon_diff


# baron
def baron_detector(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_blue = frame[:30, 375:410]
    frame_red = frame[:30, 825:860]
    baron_red = cv2.imread(f'baron/baron_red.jpg', cv2.IMREAD_GRAYSCALE)
    baron_blue = cv2.imread(f'baron/baron_blue.jpg', cv2.IMREAD_GRAYSCALE)
    result_red = cv2.matchTemplate(baron_red, frame_red, cv2.TM_CCOEFF_NORMED)
    result_blue = cv2.matchTemplate(baron_blue, frame_blue, cv2.TM_CCORR_NORMED)
    # set threshold
    threshold = 0.8
    # Get the location of all matches above the threshold
    # RED
    locations_red = np.where(result_red >= threshold)
    locations_red = list(zip(*locations_red[::-1]))
    # BLUE
    locations_blue = np.where(result_blue >= threshold)
    locations_blue = list(zip(*locations_blue[::-1]))

    baron_dict = {}

    if locations_red:
        baron_dict['red'] = 1
        baron_dict['blue'] = 0
    elif locations_blue:
        baron_dict['red'] = 0
        baron_dict['blue'] = 1
    else:
        baron_dict['red'] = 0
        baron_dict['blue'] = 0

    baron_diff = baron_dict['blue'] - baron_dict['red']

    return baron_diff


# elder
def elder_detector(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_blue = frame[:35, 310:350]
    frame_red = frame[:35, 910:940]
    elder_red = cv2.imread(f'elder/elder_red.jpg', cv2.IMREAD_GRAYSCALE)
    elder_blue = cv2.imread(f'elder/elder_blue.jpg', cv2.IMREAD_GRAYSCALE)
    result_red = cv2.matchTemplate(elder_red, frame_red, cv2.TM_CCOEFF_NORMED)
    result_blue = cv2.matchTemplate(elder_blue, frame_blue, cv2.TM_CCORR_NORMED)
    # set threshold
    threshold = 0.8
    # Get the location of all matches above the threshold
    # RED
    locations_red = np.where(result_red >= threshold)
    locations_red = list(zip(*locations_red[::-1]))
    # BLUE
    locations_blue = np.where(result_blue >= threshold)
    locations_blue = list(zip(*locations_blue[::-1]))

    elder_dict = {}

    if locations_red:
        elder_dict['red'] = 1
        elder_dict['blue'] = 0
    elif locations_blue:
        elder_dict['red'] = 0
        elder_dict['blue'] = 1
    else:
        elder_dict['red'] = 0
        elder_dict['blue'] = 0

    elder_diff = elder_dict['blue'] - elder_dict['red']

    return elder_diff


# herald
def herald_detector(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_blue = frame[520:800, 390:420]
    frame_red = frame[520:800, 850:900]
    herald = cv2.imread(f'herald/herald.png', cv2.IMREAD_GRAYSCALE)
    result_red = cv2.matchTemplate(herald, frame_red, cv2.TM_CCOEFF_NORMED)
    result_blue = cv2.matchTemplate(herald, frame_blue, cv2.TM_CCORR_NORMED)
    # set threshold
    threshold = 0.91
    # Get the location of all matches above the threshold
    # RED
    locations_red = np.where(result_red >= threshold)
    locations_red = list(zip(*locations_red[::-1]))
    # BLUE
    locations_blue = np.where(result_blue >= threshold)
    locations_blue = list(zip(*locations_blue[::-1]))

    herald_dict = {}

    if locations_red:
        herald_dict['red'] = 1
        herald_dict['blue'] = 0
    elif locations_blue:
        herald_dict['red'] = 0
        herald_dict['blue'] = 1
    else:
        herald_dict['red'] = 0
        herald_dict['blue'] = 0

    herald_diff = herald_dict['blue'] - herald_dict['red']

    return herald_diff


# champions
def champions_recogniser(image, position, site):
    top_champs = ['Aatrox', 'Akali', 'Camille', "Cho'Gath", 'Darius', 'Dr. Mundo', 'Fiora', 'Gangplank', 'Garen',
                  'Gnar', 'Gragas', 'Graves', 'Gwen', 'Illaoi', 'Irelia', 'Jarvan IV', 'Jax', 'Jayce',
                  'Kayle', 'Kennen', 'Kled', 'Lee Sin', 'Malphite', 'Maokai', 'Mega Gnar', 'Mordekaiser', 'Olaf',
                  'Ornn', 'Nasus', 'Pantheon', 'Poppy', 'Renekton', 'Riven', 'Rumble', 'Quinn', 'Ryze', 'Sejuani',
                  'Sett', 'Shen', 'Shyvana', 'Singed',
                  'Sion', 'Sylas', 'Teemo', 'Trundle', 'Urgot', 'Vladimir', 'Volibear', 'Warwick', 'Wukong', 'Yasuo',
                  'Yorick']
    jungle_champs = ['Zac', 'Xin Zhao', 'Wukong', 'Volibear', 'Warwick', 'Vi', 'Viego', 'Udyr', 'Trundle', 'Tryndamere',
                     'Talon', 'Taliyah', 'Skarner', 'Shyvana', 'Sejuani', 'Rhaast', 'Rengar', 'Rammus', "Rek'Sai",
                     'Qiyana', 'Pantheon', 'Poppy', 'Nidalee', 'Nocturne', 'Nunu & Willump', 'Olaf', 'Morgana',
                     'Master Yi', 'Shaco', 'Lillia', 'Lee Sin', "Kha'Zix", 'Kindred', 'Kayn', 'Karthus', 'Jarvan IV',
                     'Jax', 'Irelia', 'Ivern', 'Gragas',
                     'Graves', 'Gwen', 'Hecarim', 'Fizz', 'Fiddlesticks', 'Ekko', 'Elise', 'Evelynn', 'Diana',
                     'Dr. Mundo', 'Camille', 'Amumu']
    mid_champs = ['Ahri', 'Akali', 'Akshan', 'Anivia', 'Annie', 'Aurelion Sol', 'Azir', 'Brand', 'Cassiopeia', 'Corki',
                  'Diana', 'Ekko', 'Ezreal', 'Fizz', 'Galio', 'Irelia', 'Jayce', 'Karma', 'Kassadin', 'Katarina',
                  'Kayle', 'Kennen', "Kog'Maw", 'LeBlanc', 'Lissandra', 'Lucian', 'Lux', 'Malzahar', 'Morgana', 'Neeko',
                  'Orianna', 'Qiyana', 'Ryze', 'Swain', 'Sylas', 'Syndra', 'Talon', 'Tryndamere', 'Twisted Fate',
                  'Veigar',
                  "Vel'Koz", 'Vex', 'Viktor', 'Vladimir', 'Xerath', 'Yasuo', 'Zed', 'Zilean', 'Ziggs', 'Zoe']
    adc_champs = ['Akshan', 'Aphelios', 'Ashe', 'Caitlyn', 'Corki', 'Draven', 'Ezreal', 'Jhin', 'Jinx', "Kai'Sa",
                  'Kalista', "Kog'Maw", 'Lucian', 'MissFortune', 'Quinn', 'Senna', 'Sivir', 'Tristana', 'Twitch',
                  'Varus', 'Vayne',
                  'Xayah', 'Zeri', 'Ziggs']
    supp_champs = ['Alistar', 'Amumu', 'Bard', 'Blitzcrank', 'Brand', 'Braum', 'Galio', 'Heimerdinger', 'Janna',
                   'Karma', 'Leona', 'Lulu', 'Lux', 'Maokai', 'Morgana', 'Nami', 'Nautilus', 'Pyke', 'Rakan',
                   'Renata Glasc',
                   'Shen', 'Sona', 'Soraka', 'Swain', 'Tahm Kench', 'Taric', 'Thresh', 'Trundle', "Vel'Koz", 'Xerath',
                   'Yuumi', 'Zilean', 'Zyra']
    champions = {}
    if position.lower() == 'top':
        lista = top_champs
        if site.lower() == 'blue':
            image = image[105:135, 22:45]
        else:
            image = image[105:135, 1235:1260]
    elif position.lower() == 'jungle':
        lista = jungle_champs
        if site.lower() == 'blue':
            image = image[175:200, 22:45]
        else:
            image = image[175:200, 1235:1260]
    elif position.lower() == 'mid':
        lista = mid_champs
        if site.lower() == 'blue':
            image = image[245:272, 22:45]
        else:
            image = image[245:272, 1235:1260]
    elif position.lower() == 'adc':
        lista = adc_champs
        if site.lower() == 'blue':
            image = image[315:340, 22:45]
        else:
            image = image[315:340, 1235:1260]
    else:
        lista = supp_champs
        if site.lower() == 'blue':
            image = image[380:410, 22:45]
        else:
            image = image[380:410, 1235:1260]
    image = cv2.resize(image, (120, 120))
    with open('champs.pickle', 'rb') as f:
        data = pickle.load(f)
    for i in range(len(data)):
        img2 = data[i][1]
        img2 = cv2.resize(img2, (image.shape[1], image.shape[0]))
        img1 = image
        if data[i][0] in lista:
            mse = mse_calc(img1, img2)
            ssim_calc = ssim(img1, img2, multichannel=True)
            rmse_calc = rmse(img1, img2)
            psnr_calc = psnr(img1, img2)
            champions[data[i][0]] = [mse, ssim_calc, rmse_calc, psnr_calc]
    min_mse_champ = min(champions.items(), key=lambda x: x[1][0])[0]
    max_ssim_champ = max(champions.items(), key=lambda x: x[1][1])[0]
    min_rmse_champ = min(champions.items(), key=lambda x: x[1][2])[0]
    max_psnr_champ = max(champions.items(), key=lambda x: x[1][3])[0]
    if min_mse_champ == max_ssim_champ:
        champion = min_mse_champ
    elif min_mse_champ == 'Renata Glasc' or max_ssim_champ == "Renata Glasc" or min_rmse_champ == "Renata Glasc" or max_psnr_champ == 'Renata Glasc':
        champion = 'Renata Glasc'
    elif min_mse_champ == min_rmse_champ == max_psnr_champ == 'Taliyah' and max_ssim_champ == "Warwick":
        champion = 'Wukong'
    elif min_mse_champ == min_rmse_champ == max_psnr_champ == 'Lillia' and max_ssim_champ == "Diana":
        champion = 'Xin Zhao'
    elif min_mse_champ == min_rmse_champ == max_psnr_champ == 'Tahm Kench' and max_ssim_champ == "Morgana":
        champion = 'Renata Glasc'
    elif min_mse_champ == min_rmse_champ == max_psnr_champ == 'Akshan' and max_ssim_champ == "Orianna":
        champion = 'Viktor'
    elif min_mse_champ == min_rmse_champ == max_psnr_champ == 'Akali' and max_ssim_champ == "Orianna":
        champion = 'Akali'
    elif min_mse_champ == min_rmse_champ == max_psnr_champ and max_ssim_champ not in ('Gnar',
                                                                                      'MissFortune', 'Renata Glasc',
                                                                                      'Orianna'):
        champion = min_mse_champ
    elif min_mse_champ == min_rmse_champ == max_psnr_champ and max_ssim_champ in ('Gnar',
                                                                                  'MissFortune', 'Renata Glasc',
                                                                                  'Orianna'):
        champion = max_ssim_champ
    else:
        champion = None
    return champion


testujemy = True

if testujemy:
    print(f"Czas stratu: {datetime.now().strftime('%H:%M:%S')}")

    czaslista = []
    zlotolista = []
    wiezelista = []
    killelista = []
    team_bluelista = []
    team_redlista = []
    kda_toplista = []
    kda_junglelista = []
    kda_midlista = []
    kda_adclista = []
    kda_supplista = []
    minions_toplista = []
    minions_junglelista = []
    minions_midlista = []
    minions_adclista = []
    minions_supplista = []
    hextech_dragonlista = []
    infernal_dragonlista = []
    mountain_dragonlista = []
    ocean_dragonlista = []
    wind_dragonlista = []
    baronlista = []
    elderlista = []
    heraldlista = []
    champion_red_toplista = []
    champion_red_junglelista = []
    champion_red_midlista = []
    champion_red_adclista = []
    champion_red_supplista = []
    champion_blue_toplista = []
    champion_blue_junglelista = []
    champion_blue_midlista = []
    champion_blue_adclista = []
    champion_blue_supplista = []

    for filename in os.listdir('mecze'):
        f = os.path.join('mecze', filename)
        # checking if it is a file
        if os.path.isfile(f):
            frame = cv2.imread(f)
            czas = get_time(frame)
            czaslista.append(czas)
            zloto = get_gold_diff_blue(frame)
            zlotolista.append(zloto)
            wieze = get_tower_diff_blue(frame)
            wiezelista.append(wieze)
            kille = get_kills_diff_blue(frame)
            killelista.append(kille)
            team_blue = get_team_blue(frame)
            team_bluelista.append(team_blue)
            team_red = get_team_red(frame)
            team_redlista.append(team_red)
            kda_top = get_kda_diff_top_blue(frame)
            kda_toplista.append(kda_top)
            kda_jungle = get_kda_diff_jungle_blue(frame)
            kda_junglelista.append(kda_jungle)
            kda_mid = get_kda_diff_mid_blue(frame)
            kda_midlista.append(kda_mid)
            kda_adc = get_kda_diff_adc_blue(frame)
            kda_adclista.append(kda_adc)
            kda_supp = get_kda_diff_supp_blue(frame)
            kda_supplista.append(kda_supp)
            minions_top = get_minions_diff_top_blue(frame)
            minions_toplista.append(minions_top)
            minions_jungle = get_minions_diff_jungle_blue(frame)
            minions_junglelista.append(minions_jungle)
            minions_mid = get_minions_diff_mid_blue(frame)
            minions_midlista.append(minions_mid)
            minions_adc = get_minions_diff_adc_blue(frame)
            minions_adclista.append(minions_adc)
            minions_supp = get_minions_diff_supp_blue(frame)
            minions_supplista.append(minions_supp)
            dragon_diff = dragon_detector(frame)
            hextech_dragonlista.append(dragon_diff[0])
            infernal_dragonlista.append(dragon_diff[1])
            mountain_dragonlista.append(dragon_diff[2])
            ocean_dragonlista.append(dragon_diff[3])
            wind_dragonlista.append(dragon_diff[4])
            baronlista.append(baron_detector(frame))
            elderlista.append(elder_detector(frame))
            heraldlista.append(herald_detector(frame))
            champion_red_toplista.append(champions_recogniser(frame, 'top', 'red'))
            champion_red_junglelista.append(champions_recogniser(frame, 'jungle', 'red'))
            champion_red_midlista.append(champions_recogniser(frame, 'mid', 'red'))
            champion_red_adclista.append(champions_recogniser(frame, 'adc', 'red'))
            champion_red_supplista.append(champions_recogniser(frame, 'supp', 'red'))
            champion_blue_toplista.append(champions_recogniser(frame, 'top', 'blue'))
            champion_blue_junglelista.append(champions_recogniser(frame, 'jungle', 'blue'))
            champion_blue_midlista.append(champions_recogniser(frame, 'mid', 'blue'))
            champion_blue_adclista.append(champions_recogniser(frame, 'adc', 'blue'))
            champion_blue_supplista.append(champions_recogniser(frame, 'supp', 'blue'))

    df_data = {'time': czaslista,
               'gold': zlotolista,
               'towers': wiezelista,
               'kills': killelista,
               'blue_team': team_bluelista,
               'red_team': team_redlista,
               'kda_top': kda_toplista,
               'kda_jungle': kda_junglelista,
               'kda_mid': kda_midlista,
               'kda_adc': kda_adclista,
               'kda_supp': kda_supplista,
               'minions_top': minions_toplista,
               'minions_jungle': minions_junglelista,
               'minions_mid': minions_midlista,
               'minions_adc': minions_adclista,
               'minions_supp': minions_supplista,
               'hextech_dragon': hextech_dragonlista,
               'infernal_dragon': infernal_dragonlista,
               'mountain_dragon': mountain_dragonlista,
               'ocean_dragon': ocean_dragonlista,
               'wind_dragon': wind_dragonlista,
               'baron': baronlista,
               'elder': elderlista,
               'herald': heraldlista,
               'champion_red_top': champion_red_toplista,
               'champion_red_jungle': champion_red_junglelista,
               'champion_red_mid': champion_red_midlista,
               'champion_red_adc': champion_red_adclista,
               'champion_red_supp': champion_red_supplista,
               'champion_blue_top': champion_blue_toplista,
               'champion_blue_jungle': champion_blue_junglelista,
               'champion_blue_mid': champion_blue_midlista,
               'champion_blue_adc': champion_blue_adclista,
               'champion_blue_supp': champion_blue_supplista
               }

    df = pd.DataFrame(df_data)

    print(f"Czas końca: {datetime.now().strftime('%H:%M:%S')}")
    df.to_csv('baza.csv', index=False)
