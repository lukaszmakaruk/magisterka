import numpy as np
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
import cv2
import os
import pickle

def resize(image, scale_percent):
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    return resized

def mse_calc(img1, img2):
    error = np.sum((img1.astype('float') - img2.astype('float')) ** 2)
    error /= float(img1.shape[0] * img1.shape[1])
    return error

create_list_champs=False
if create_list_champs==True:
    nazwy=[]
    obrazy=[]
    model=[]
    for filename in os.listdir('champions'):
            f = os.path.join('champions', filename)
            # checking if it is a file
            if os.path.isfile(f):
                nazwy.append(filename.replace('Square.png',''))
                #obrazy.append(cv2.cvtColor(cv2.imread(f),cv2.COLOR_BGR2GRAY))
                obrazy.append(cv2.imread(f))

    for i in range(len(nazwy)):
        data=(nazwy[i],obrazy[i])
        model.append(data)

    with open('champs.pickle', 'wb') as f:
        pickle.dump(model, f)
                                    #(champion_both[-1],champion_mse[-1],champion_ssim[-1])
#top b    [105:135,22:45] msin          ('ORNN', champion_mse[-1] ,'Aphelios')
#jungle b [175:200,22:45] mse           ('Aatrox', 'VI', 'SAKayn')
#mid b    [245:272,22:45] msin+mse      ('AZIR', 'Diana', 'Ryze')
#adc b    [315:340,22:45]               ('Akali', 'KALISTA', 'Elise')
#supp b   [380:410,22:45]               nie zgadło

# W kolorze                         #(champion_both[-1],champion_mse[-1],champion_ssim[-1])
#top b    [105:135,22:45] msin          ('ORNN', champion_mse[-1] ,'Aphelios')
#jungle b [175:200,22:45] mse           ('Aatrox', 'VI', 'Xayah')
#mid b    [245:272,22:45] msin+mse      ('AZIR', 'Amumu', 'Ornn')
#adc b    [315:340,22:45]               ('Akali', 'KALISTA', 'Elise')
#supp b   [380:410,22:45]               ('Akali', 'Tahm Kench', 'RENATA GLASC')
def champions_recogniser(image,position,site):
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
                   'Karma', 'Leona', 'Lulu', 'Lux', 'Maokai', 'Morgana', 'Nami', 'Nautilus', 'Neeko', 'Pyke', 'Rakan',
                   'Renata Glasc',
                   'Shen', 'Sona', 'Soraka', 'Swain', 'Tahm Kench', 'Taric', 'Thresh', 'Trundle', "Vel'Koz", 'Xerath',
                   'Yuumi', 'Zilean', 'Zyra']
    if position.lower() == 'top':
        lista=top_champs
        if site.lower()== 'blue':
            image=image[105:135,22:45]
        else:
            image = image[105:135, 1235:1260]
    elif position.lower() == 'jungle':
        lista=jungle_champs
        if site.lower() == 'blue':
            image=image[175:200,22:45]
        else:
            image = image[175:200, 1235:1260]
    elif position.lower() == 'mid':
        lista=mid_champs
        if site.lower() == 'blue':
            image=image[245:272,22:45]
        else:
            image = image[245:272, 1235:1260]
    elif position.lower() == 'adc':
        lista=adc_champs
        if site.lower() == 'blue':
            image=image[315:340,22:45]
        else:
            image = image[315:340, 1235:1260]
    else:
        lista=supp_champs
        if site.lower() == 'blue':
            image=image[380:410,22:45]
        else:
            image = image[380:410, 1235:1260]
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (120, 120))
    min_mse=1000000
    min_ssim=0
    champion_both=[]
    champion_mse=[]
    champion_ssim=[]
    with open('champs.pickle', 'rb') as f:
        data = pickle.load(f)
        for i in range(len(data)):
            img2 = data[i][1]
            img2 = cv2.resize(img2, (image.shape[1],image.shape[0]))
            img1 = image
            mse=mse_calc(img1,img2)
            ssim_calc=ssim(img1,img2,multichannel=True )
            if mse<min_mse  and ssim_calc>min_ssim and data[i][0] in lista:
                min_mse=mse
                min_ssim=ssim_calc
                champion_both.append(data[i][0])
            elif mse<min_mse and ssim_calc<min_ssim and data[i][0] in lista:
                min_mse=mse
                champion_mse.append(data[i][0])
            elif mse>min_mse and ssim_calc>min_ssim and data[i][0] in lista:
                min_ssim=ssim_calc
                champion_ssim.append(data[i][0])
            else:
                pass
    if len(champion_both)>0 and len(champion_mse)>0 and len(champion_ssim)>0:
        #print(len(champion_both),len(champion_mse),len(champion_ssim))
        #print(champion_both[-1], champion_mse[-1], champion_ssim[-1])
        if champion_both[-1] in lista and len(champion_both)>=len(champion_mse) and len(champion_both)>=len(champion_ssim):
            return champion_both[-1]
        elif champion_mse[-1] in lista and len(champion_mse)>=len(champion_both) and len(champion_mse)>=len(champion_ssim):
            return champion_mse[-1]
        elif champion_ssim[-1] in lista and len(champion_ssim)>=len(champion_both) and len(champion_ssim)>=len(champion_mse):
            return champion_ssim[-1]
        elif champion_ssim[-1] in lista and champion_mse[-1] in lista and len(champion_ssim)<=len(champion_mse):
            return champion_mse[-1]
        elif champion_ssim[-1] in lista and champion_mse[-1] in lista and len(champion_ssim)>=len(champion_mse):
            return champion_ssim[-1]
        elif champion_both[-1] in lista and champion_mse[-1] in lista and len(champion_both)>=len(champion_mse):
            return champion_both[-1]
        elif champion_both[-1] in lista and champion_mse[-1] in lista and len(champion_both)<=len(champion_mse):
            return champion_mse[-1]
        elif champion_both[-1] in lista and champion_ssim[-1] in lista and len(champion_both)>=len(champion_ssim):
            return champion_both[-1]
        elif champion_both[-1] in lista and champion_ssim[-1] in lista and len(champion_both)<=len(champion_ssim):
            return champion_ssim[-1]
        elif champion_both[-1] in lista and champion_mse[-1] not in lista and len(champion_both)<=len(champion_mse):
            return champion_both[-1]
    else:
        #print(len(champion_both),len(champion_ssim))
        #print(champion_both[-1], champion_ssim[-1])
        if champion_both[-1] in lista and len(champion_both)>=len(champion_mse) and len(champion_both)>=len(champion_ssim):
            return champion_both[-1]
        elif champion_ssim[-1] in lista and len(champion_ssim)>=len(champion_both) and len(champion_ssim)>=len(champion_mse):
            return champion_ssim[-1]
        elif champion_both[-1] in lista and champion_ssim[-1] in lista and len(champion_both)>=len(champion_ssim):
            return champion_both[-1]
        elif champion_both[-1] in lista and champion_ssim[-1] in lista and len(champion_both)<=len(champion_ssim):
            return champion_ssim[-1]

image=cv2.imread('mecze/frame579000.jpg')
#print(image.shape)
image = image[105:135, 1235:1260]
#image=resize(image,150)
#cv2.imshow('img2',image)
#cv2.waitKey(0)

image = cv2.resize(image, (120, 120))
min_mse = 1000000
min_ssim = 0
champion_both = {}
champion_mse = {}
champion_ssim = {}
lista = ['Aatrox', 'Akali', 'Camille', "Cho'Gath", 'Darius', 'Dr. Mundo', 'Fiora', 'Gangplank', 'Garen',
              'Gnar', 'Gragas', 'Graves', 'Gwen', 'Illaoi', 'Irelia', 'Jarvan IV', 'Jax', 'Jayce',
              'Kayle', 'Kennen', 'Kled', 'Lee Sin', 'Malphite', 'Maokai', 'Mega Gnar', 'Mordekaiser', 'Olaf',
              'Ornn', 'Nasus', 'Pantheon', 'Poppy', 'Renekton', 'Riven', 'Rumble', 'Quinn', 'Ryze', 'Sejuani',
              'Sett', 'Shen', 'Shyvana', 'Singed',
              'Sion', 'Sylas', 'Teemo', 'Trundle', 'Urgot', 'Vladimir', 'Volibear', 'Warwick', 'Wukong', 'Yasuo',
              'Yorick']
with open('champs.pickle', 'rb') as f:
    data = pickle.load(f)
    for i in range(len(data)):
        img2 = data[i][1]
        img2 = cv2.resize(img2, (image.shape[1], image.shape[0]))
        img1 = image
        mse = mse_calc(img1, img2)
        ssim_calc = ssim(img1, img2, multichannel=True)
        if mse < min_mse and ssim_calc > min_ssim and data[i][0] in lista:
            min_mse = mse
            min_ssim = ssim_calc
            champion_both[data[i][0]]=[mse,ssim_calc]
        elif mse < min_mse and ssim_calc < min_ssim and data[i][0] in lista:
            min_mse = mse
            champion_mse[data[i][0]]=[mse,ssim_calc]
        elif mse > min_mse and ssim_calc > min_ssim and data[i][0] in lista:
            min_ssim = ssim_calc
            champion_ssim[data[i][0]]=[mse,ssim_calc]
        else:
            pass

#print(len(champion_both),len(champion_mse),len(champion_ssim))
#print(champion_both)
print(champion_mse)
print(champion_ssim)
print(champion_both )
#print(champions_recogniser(image,'top','blue1'))

#mecze
#                   RGE - FNC
#             blue         red
#top            T           N
#jungle         T           T
#mid            T           T
#adc            T           N
#supp           N           N
#                   VIT - MAD
#             blue         red
#top            N           T
#jungle         T           N
#mid            N           N
#adc            T           N
#supp           T           N
#                   SK - BDS
#             blue         red
#top            T           T
#jungle         T           N
#mid            T           T
#adc            N           N
#supp           T           N
#                   XL - MSF
#             blue         red
#top            T           T
#jungle         T           N
#mid            T           N
#adc            T           N
#supp           T           N
#                   G2 - AST
#             blue         red
#top            T           N
#jungle         T           T
#mid            N           N
#adc            T           N
#supp           N           N
