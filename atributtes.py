import pytesseract
from pytube import YouTube
from datetime import datetime
import os
import cv2
import numpy as np
import math
from colorthief import ColorThief
import pandas as pd

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


custom_config = r'--oem 3 --psm 6'
custom_config2 = r'--psm 13 --oem 3 -c tessedit_char_whitelist=0123456789:'


# ATTRIBUTES LOW
# gold
def get_gold_blue(image):
    img = image[:35, 547:583]
    resized = resize(img, 130)
    gray_scaled = get_gray_scale(resized)
    sharpen = unsharp_mask(gray_scaled)
    text = pytesseract.image_to_string(sharpen, config=r'--psm 7')
    text = text.strip()
    try:
        if text[-1] == 'k':
            text = int(float(text[:-1]) * 1000)
            if text > 100000:
                text = int(str(text)[:-1])
    except:
        text = None
    return text


def get_gold_red(image):
    img = image[:35, 727:770]
    resized = resize(img, 130)
    gray_scaled = get_gray_scale(resized)
    sharpen = unsharp_mask(gray_scaled)
    text = pytesseract.image_to_string(sharpen, config=r'--psm 7')
    text = text.strip()
    try:
        if text[-1] == 'k':
            text = int(float(text[:-1]) * 1000)
            if abs(get_gold_blue(image) - text) >= 20000:
                text = int(str(text)[:-1])
    except:
        text = None
    return text

# towers


def get_tower_blue(image):
    img = image[:35, 485:520]
    resized = resize(img, 130)
    gray_scaled = get_gray_scale(resized)
    sharpen = unsharp_mask(gray_scaled)
    text = pytesseract.image_to_string(sharpen, config=r'--psm 7')
    try:
        text = int(text)
    except:
        text = None
    return text


def get_tower_red(image):
    img = image[:35, 792:820]
    resized = resize(img, 130)
    gray_scaled = get_gray_scale(resized)
    sharpen = unsharp_mask(gray_scaled)
    text = pytesseract.image_to_string(sharpen, config=r'--psm 7')
    try:
        text = int(text)
    except:
        text = None
    return text

# kills


def get_kills_blue(image):
    img = image[:35, 590:630]
    resized = resize(img, 130)
    gray_scaled = get_gray_scale(resized)
    sharpen = unsharp_mask(gray_scaled)
    text = pytesseract.image_to_string(sharpen, config=r'--psm 7')
    try:
        text = int(text)
    except:
        text = None
    return text


def get_kills_red(image):
    img = image[:35, 650:690]
    resized = resize(img, 130)
    gray_scaled = get_gray_scale(resized)
    sharpen = unsharp_mask(gray_scaled)
    text = pytesseract.image_to_string(sharpen, config=r'--psm 7')
    try:
        text = int(text)
    except:
        text = None
    return text

# kda


def get_kda_top_blue(image):
    img = image[568:585, 530:575]
    resized = resize(img, 130)
    gray_scaled = get_gray_scale(resized)
    sharpen = unsharp_mask(gray_scaled)
    text = pytesseract.image_to_string(sharpen, config=r'--psm 7 -c tessedit_char_whitelist=0123456789/')
    try:
        text = text.strip().split('/')
        if int(text[1]) != 0:
            text = (int(text[0])+int(text[2]))/int(text[1])
        else:
            text = int(text[0]) + int(text[2])
    except:
        text = None
    return text


def get_kda_top_red(image):
    img = image[568:585, 715:755]
    resized = resize(img, 130)
    gray_scaled = get_gray_scale(resized)
    sharpen = unsharp_mask(gray_scaled)
    text = pytesseract.image_to_string(sharpen, config=r'--psm 7 -c tessedit_char_whitelist=0123456789/')
    try:
        text = text.strip().split('/')
        if int(text[1]) != 0:
            text = (int(text[0]) + int(text[2])) / int(text[1])
        else:
            text = int(text[0]) + int(text[2])
    except:
        text = None
    return text


def get_kda_jungle_blue(image):
    img = image[595:615, 530:575]
    resized = resize(img, 130)
    gray_scaled = get_gray_scale(resized)
    sharpen = unsharp_mask(gray_scaled)
    text = pytesseract.image_to_string(sharpen, config=r'--psm 7 -c tessedit_char_whitelist=0123456789/')
    try:
        text = text.strip().split('/')
        if int(text[1]) != 0:
            text = (int(text[0]) + int(text[2])) / int(text[1])
        else:
            text = int(text[0]) + int(text[2])
    except:
        text = None
    return text


def get_kda_jungle_red(image):
    img = image[595:615, 715:760]
    resized = resize(img, 130)
    gray_scaled = get_gray_scale(resized)
    sharpen = unsharp_mask(gray_scaled)
    text = pytesseract.image_to_string(sharpen, config=r'--psm 7 -c tessedit_char_whitelist=0123456789/')
    try:
        text = text.strip().split('/')
        if int(text[1]) != 0:
            text = (int(text[0]) + int(text[2])) / int(text[1])
        else:
            text = int(text[0]) + int(text[2])
    except:
        text = None
    return text


def get_kda_mid_blue(image):
    img = image[630:650, 530:575]
    resized = resize(img, 130)
    gray_scaled = get_gray_scale(resized)
    sharpen = unsharp_mask(gray_scaled)
    text = pytesseract.image_to_string(sharpen, config=r'--psm 7 -c tessedit_char_whitelist=0123456789/')
    try:
        text = text.strip().split('/')
        if int(text[1]) != 0:
            text = (int(text[0]) + int(text[2])) / int(text[1])
        else:
            text = int(text[0]) + int(text[2])
    except:
        text = None
    return text


def get_kda_mid_red(image):
    img = image[630:650, 715:760]
    resized = resize(img, 230)
    gray_scaled = get_gray_scale(resized)
    sharpen = unsharp_mask(gray_scaled)
    text = pytesseract.image_to_string(sharpen, config=r'--psm 7 -c tessedit_char_whitelist=0123456789/')
    try:
        text = text.strip().split('/')
        if int(text[1]) != 0:
            text = (int(text[0]) + int(text[2])) / int(text[1])
        else:
            text = int(text[0]) + int(text[2])
    except:
        text = None
    return text


def get_kda_adc_blue(image):
    img = image[662:680, 530:575]
    resized = resize(img, 140)
    gray_scaled = get_gray_scale(resized)
    sharpen = unsharp_mask(gray_scaled)
    text = pytesseract.image_to_string(sharpen, config=r'--psm 7 -c tessedit_char_whitelist=0123456789/')
    try:
        text = text.strip().split('/')
        if int(text[1]) != 0:
            text = (int(text[0]) + int(text[2])) / int(text[1])
        else:
            text = int(text[0]) + int(text[2])
    except:
        text = None
    return text


def get_kda_adc_red(image):
    img = image[662:680, 715:760]
    resized = resize(img, 140)
    gray_scaled = get_gray_scale(resized)
    sharpen = unsharp_mask(gray_scaled)
    text = pytesseract.image_to_string(sharpen, config=r'--psm 7 -c tessedit_char_whitelist=0123456789/')
    try:
        text = text.strip().split('/')
        if int(text[1]) != 0:
            text = (int(text[0]) + int(text[2])) / int(text[1])
        else:
            text = int(text[0]) + int(text[2])
    except:
        text = None
    return text


def get_kda_supp_blue(image):
    img = image[691:710, 530:575]
    resized = resize(img, 140)
    gray_scaled = get_gray_scale(resized)
    sharpen = unsharp_mask(gray_scaled)
    text = pytesseract.image_to_string(sharpen, config=r'--psm 7 -c tessedit_char_whitelist=0123456789/')
    try:
        text = text.strip().split('/')
        if int(text[1]) != 0:
            text = (int(text[0]) + int(text[2])) / int(text[1])
        else:
            text = int(text[0]) + int(text[2])
    except:
        text = None
    return text


def get_kda_supp_red(image):
    img = image[691:710, 715:760]
    resized = resize(img, 120)
    gray_scaled = get_gray_scale(resized)
    sharpen = unsharp_mask(gray_scaled)
    text = pytesseract.image_to_string(sharpen, config=r'--psm 7 -c tessedit_char_whitelist=0123456789/')
    try:
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
        text=int(text)
    except:
        text=None
    return text


def get_minions_top_red(image):
    img = image[565:585, 680:710]
    resized = resize(img, 130)
    gray_scaled = get_gray_scale(resized)
    sharpen = unsharp_mask(gray_scaled)
    text = pytesseract.image_to_string(sharpen, config=r'--psm 7 -c tessedit_char_whitelist=0123456789')
    try:
        text=int(text)
    except:
        text=None
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


# ATTRIBUTES MAIN

# time
def get_time(image):
    config = r'--psm 7'
    img = image[32:56, 610:665]
    img1 = resize(img, 150)
    img2 = get_gray_scale(img1)
    img3 = unsharp_mask(img2)
    text = pytesseract.image_to_string(img3, config=config)
    try:
        text2 = datetime.strptime(text.strip(), '%M:%S').time()
    except:
        text2 = None
    return text2

# gold


def get_gold_diff_blue(image):
    try:
        diff = get_gold_blue(image) - get_gold_red(image)
    except:
        diff = None
    return diff

# towers

def get_tower_diff_blue(image):
    try:
        diff = get_tower_blue(image) - get_tower_red(image)
        if diff>14:
            diff=None
    except:
        diff = None
    return diff

# kills


def get_kills_diff_blue(image):
    try:
        diff = get_kills_blue(image) - get_kills_red(image)
        if diff>14:
            diff=None
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
    try:
        diff = get_minions_top_blue(image) - get_minions_top_red(image)
    except:
        diff = None
    return diff


def get_minions_diff_jungle_blue(image):
    try:
        diff = get_minions_jungle_blue(image) - get_minions_jungle_red(image)
    except:
        diff = None
    return diff


def get_minions_diff_mid_blue(image):
    try:
        diff = get_minions_mid_blue(image) - get_minions_mid_red(image)
    except:
        diff = None
    return diff


def get_minions_diff_adc_blue(image):
    try:
        diff = get_minions_adc_blue(image) - get_minions_adc_red(image)
    except:
        diff = None
    return diff


def get_minions_diff_supp_blue(image):
    try:
        diff = get_minions_supp_blue(image) - get_minions_supp_red(image)
    except:
        diff = None
    return diff

testujemy= False

if testujemy:
    print(f"Czas stratu: {datetime.now().strftime('%H:%M:%S')}")

    czaslista=[]
    zlotolista=[]
    wiezelista=[]
    killelista=[]
    team_bluelista=[]
    team_redlista=[]
    kda_toplista=[]
    kda_junglelista=[]
    kda_midlista=[]
    kda_adclista=[]
    kda_supplista=[]
    minions_toplista=[]
    minions_junglelista=[]
    minions_midlista=[]
    minions_adclista=[]
    minions_supplista=[]

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

    df_data={'time': czaslista,
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
             'minions_supp': minions_supplista}

    df= pd.DataFrame(df_data)

    print(f"Czas końca: {datetime.now().strftime('%H:%M:%S')}")
    df.to_csv('test.csv',index=False)