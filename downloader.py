import pytesseract
from pytube import YouTube
from datetime import datetime
import os
import cv2
import numpy as np
import math

save_path = 'D:/Python/magisterka/filmy'
link = 'https://www.youtube.com/watch?v=O7mNMvMT4Cs'
link2= 'https://www.youtube.com/watch?v=gU0R6i02QOE'

# POBIERANIE
def downloader(link,save_path):
    yt = YouTube(link)
    ys = yt.streams.filter(file_extension="mp4").get_by_itag(22)
    print(f'Start: {datetime.now().strftime("%H:%M:%S")}')
    ys.download(save_path)
    print(f'Koniec: {datetime.now().strftime("%H:%M:%S")}')

# ZMIANA NA KLATKI
def video_to_images(video_path, frames_per_second=0.02):
    cam = cv2.VideoCapture(video_path)
    frame_list = []
    frame_rate = cam.get(cv2.CAP_PROP_FPS)  # video frame rate

    # frame
    current_frame = 0

    # create directory if it does not exist
    images_path = f'./klatki'
    if not os.path.exists(images_path):
        os.makedirs(images_path)

    if frames_per_second > frame_rate or frames_per_second == -1:
        frames_per_second = frame_rate

    while (True):

        # reading from frame
        ret, frame = cam.read()

        if ret:

            # if video is still left continue creating images
            file_name = f'{images_path}/frame' + str(current_frame) + '.jpg'
            print('Creating...' + file_name)
            # print('frame rate', frame_rate)
            if current_frame % (math.floor(frame_rate / frames_per_second)) == 0:
                    # adding frame to list
                frame_list.append(frame)

                    # writing selected frames to images_path
                cv2.imwrite(file_name, frame)

                # increasing counter so that it will
                # show how many frames are created
            current_frame += 1
        else:
            break

    # Release all space and windows once done
    cam.release()
    cv2.destroyAllWindows()

    return frame_list

# ZABAWA OBRAZEM
def ocr_core(image):
    text = pytesseract.image_to_string(image)
    return text
def get_gray_scale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
def remove_noise(image):
    return cv2.medianBlur(image,5)
def thresholding(image):
    return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
def resize(image,scale_percent):
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

# ATRYBUTY
def get_gold_blue(image):
    img = image[:35, 550:580]
    resized = resize(img,130)
    gray_scaled = get_gray_scale(resized)
    sharpen = unsharp_mask(gray_scaled)
    text=ocr_core(sharpen)
    if len(text)>0:
        return text
    else:
        return None

pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract'

#test=False
test=True
#test='kot'

#Zmienianie filmow na klatki
if test == 'kot':
    for filename in os.listdir('filmy'):
        f = os.path.join('filmy', filename)
        # checking if it is a file
        if os.path.isfile(f):
            video_to_images(f)
            pass

#Wybieranie filmów których klatki są z meczu

if test==True:
    for filename in os.listdir('klatki'):
        f = os.path.join('klatki', filename)
        # checking if it is a file
        if os.path.isfile(f):
            frame = cv2.imread(f)
            if get_gold_blue(frame)!=None:
                os.chdir('D:\Python\magisterka\mecze')
                cv2.imwrite(filename, frame)
                os.chdir('D:\Python\magisterka')
            elif get_gold_blue(frame)==None:
                os.chdir(r'D:\Python\magisterka\niemecze')
                cv2.imwrite(filename, frame)
                os.chdir('D:\Python\magisterka')
            os.remove(f)


