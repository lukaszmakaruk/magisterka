import pytesseract
from pytube import YouTube
from datetime import datetime
import os
import cv2
import numpy as np
import math
from colorthief import ColorThief
import pandas as pd
from atributtes import resize,get_gray_scale,unsharp_mask

def get_kda_jungle_blue(image):
    img = image[595:618, 532:572]
    resized = resize(img, 135)
    gray_scaled = get_gray_scale(resized)
    sharpen = unsharp_mask(gray_scaled)
    text = pytesseract.image_to_string(sharpen, config=r'--psm 7 -c tessedit_char_whitelist=0123456789/')
    try:
        text = text.strip().split('/')
        if len(text) < 3 and int(text[0]) > 10:
            text = [text[0][0], text[0][1], text[1]]
        if len(text) < 3 and int(text[1]) > 10:
            text = [text[0], text[1][0], text[1][1:]]
        if len(text[1]) > 1 and text[1][0] == 0:
            text = [text[0], text[1][0], text[1][1]]
        if int(text[1]) != 0:
            text = (int(text[0]) + int(text[2])) / int(text[1])
        else:
            text = int(text[0]) + int(text[2])
    except:
        text = None
    return text


def get_kda_jungle_red(image):
    img = image[595:615, 715:760]
    resized = resize(img, 135)
    gray_scaled = get_gray_scale(resized)
    sharpen = unsharp_mask(gray_scaled)
    text = pytesseract.image_to_string(sharpen, config=r'--psm 7 -c tessedit_char_whitelist=0123456789/')
    try:
        text = text.strip().split('/')
        if len(text) < 3 and int(text[0]) > 10:
            text = [text[0][0], text[0][1], text[1]]
        if len(text) < 3 and int(text[1]) > 10:
            text = [text[0], text[1][0], text[1][1:]]
        if len(text[1]) > 1 and text[1][0] == 0:
            text = [text[0], text[1][0], text[1][1]]
        if int(text[1]) != 0:
            text = (int(text[0]) + int(text[2])) / int(text[1])
        else:
            text = int(text[0]) + int(text[2])
    except:
        text = None
    return text

def get_kda_diff_jungle_blue(image):
    try:
        diff = get_kda_jungle_blue(image) - get_kda_jungle_red(image)
    except:
        diff = None
    return diff

#https://gol.gg/tournament/tournament-stats/LEC%20Spring%202022/

pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract'
os.chdir('D:\\Python\\magisterka')

pierwszytest=True
if pierwszytest:
    flist=[]
    with open(r'klatki_red.txt', 'r') as fp:
        for line in fp:

            x = line[:-1]

            # add current item to the list
            flist.append(x)

obrazy_gotowe = True
if obrazy_gotowe:
    ile=0
    for f in flist:
        image = cv2.imread(f'mecze/{f}')
        img = image[595:615, 715:760]
        resized = resize(img, 135)
        gray_scaled = get_gray_scale(resized)
        sharpen = unsharp_mask(gray_scaled)
        text = pytesseract.image_to_string(sharpen, config=r'--psm 7 -c tessedit_char_whitelist=0123456789/')
        #cv2.imshow('img',img)
        #cv2.waitKey(0)
        try:
            text = text.strip().split('/')
            if len(text)<3 and int(text[0])>10:
                text=[text[0][0],text[0][1],text[1]]
            if len(text)<3 and int(text[1])>10:
                text=[text[0],text[1][0],text[1][1:]]
            if len(text[1])>1 and text[1][0]==0:
                text=[text[0],text[1][0],text[1][1]]
        except:
            text = None
        if text==None:
            ile+=1
        if text!=None and len(text)<3:
            print(f,text)
        if text==None:
            print(f, text)
    print(ile/len(flist))

drugitest=False
if drugitest:
    for filename in os.listdir('mecze'):
        f = os.path.join('mecze', filename)
        # checking if it is a file
        if os.path.isfile(f):
            frame = cv2.imread(f)
            text_red = get_kda_mid_red(frame)
            text_blue = get_kda_mid_blue(frame)
            text = get_kda_diff_mid_blue(frame)
            print(f'red:{text_red}, blue:{text_blue}, diff:{text}')

        image = cv2.imread(f'mecze/frame588000.jpg')
        img = image[595:618, 532:572]
        resized = resize(img, 135)
        gray_scaled = get_gray_scale(resized)
        sharpen = unsharp_mask(gray_scaled)
        text = pytesseract.image_to_string(sharpen, config=r'--psm 7 -c tessedit_char_whitelist=O0123456789/')
        cv2.imshow('img',sharpen)
        cv2.waitKey(0)
        print(text)