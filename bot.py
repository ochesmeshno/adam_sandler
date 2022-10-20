from ctypes import resize
import telebot
import os.path
import json
import random
from PIL import Image, ImageFile, ImageDraw, ImageFont, ImageChops, ImageEnhance, ImageOps
from random import randint, shuffle
import rusyllab
import time
from datetime import datetime
import numpy as np
import cv2
import torch
from rembg.bg import remove
import io
ImageFile.LOAD_TRUNCATED_IMAGES = True
import requests
from requests_html import HTMLSession
from bs4 import BeautifulSoup
from matplotlib import cm


token = open("token.txt", "r")
bot = telebot.TeleBot(str(token.read()))
token.close()
st_trigger = "********"
bold = "fonts/2223.ttf"
thin = "fonts/2221.ttf"
hl2_font = "fonts/Headhumper.ttf"
img = Image.open('nums/street.png').convert('RGBA')
draw = ImageDraw.Draw(img)
del_sticker = False
new_price = 0
tumenec_date = 0


ekat = "-1001575990977"
test = "-1001214662606"


#создание класса слов для рисования на картинке (генератор адресной таблички улицы)
class Words:
    color = (255, 255, 255, 255)
    font_size = 105

    def __init__(self, type_street, text):
        #определение параметров надписей
        self.text = text

        if type_street == "name":
            self.max_size = 620
            self.font_type = bold
            self.font_size = 95
            self.font = ImageFont.truetype(bold, self.font_size)
            w, h = draw.textsize(self.text, self.font)
            #если ширина текста больше разрешенной области, уменьшить шрифт на единицу
            if w > self.max_size:
                while w > self.max_size:
                    self.font_size -= 1
                    self.font = ImageFont.truetype(bold, self.font_size)
                    w, h = draw.textsize(self.text, self.font)
                self.width = w
                self.height = h
            else:
                self.font = ImageFont.truetype(bold, self.font_size)
                w, h = draw.textsize(self.text, self.font)
                self.width = w
                self.height = h
            self.left = (830 - self.width) / 2
            self.top = (img.height - self.height - 30) / 2
            self.font = ImageFont.truetype(self.font_type, self.font_size)
        # другие надписи
        elif type_street == "number":
            self.max_size = 210
            self.font_size = 105
            self.font = ImageFont.truetype(bold, self.font_size)
            self.font_type = bold
            w, h = draw.textsize(self.text, self.font)
            if h > self.max_size:
                while h > self.max_size:
                    self.font_size -= 1
                    self.font = ImageFont.truetype(bold, self.font_size)
                    w, h = draw.textsize(self.text, self.font)
            else:
                self.font = ImageFont.truetype(bold, self.font_size)
                w, h = draw.textsize(self.text, self.font)
            self.left = 810 + (img.width - 810 - w) / 2
            self.top = (img.height - h) / 2 - 30
            self.font = ImageFont.truetype(self.font_type, self.font_size)
        elif type_street == "street_type":
            self.font_type = thin
            self.font_size = 45
            self.font = ImageFont.truetype(self.font_type, self.font_size)
            w, h = draw.textsize(self.text, self.font)
            self.top = 220
            self.left = (830 - w) / 2
        elif type_street == "building":
            self.top = 220
            self.left = 850
            self.font_type = thin
            self.font_size = 45
            self.font = ImageFont.truetype(self.font_type, self.font_size)
            w, h = draw.textsize(self.text, self.font)
            self.top = 220
            self.left = 850

#список в кортеж
def convert_to_tuple(list):
    return tuple(i for i in list)

def nat_geo(raw):
    name_pic = randint(10,99)
    path = f"natgeo/{name_pic}.jpg"
    file_info = bot.get_file(raw)
    downloaded_file = bot.download_file(file_info.file_path)
    with open(path, 'wb') as new_file:
        new_file.write(downloaded_file)
    img = Image.open(path).convert("RGBA")
    natgeo = Image.open("img/natgeo.png").convert("RGBA")
    name = f"natgeo/{int(datetime.today().timestamp())}.png"
    w = natgeo.width
    h = natgeo.height
    point_x = int(0.025 * w)
    point_y = int(0.05 * h)
    while natgeo.width > img.width * 0.25:
        w = int(w * 0.99)
        h = int(h * 0.99)
        natgeo.thumbnail((w, h))
    background = Image.new('RGBA', (img.width, img.height), color=(255, 255, 255, 255))
    background.paste(img, (0, 0), img)
    background.paste(natgeo, (point_x, point_y), natgeo)
    background.save(name, format='png', subsampling=0, quality=100)
    return name


def believe(path):
    img = Image.open(path).convert("RGBA")
    believe = Image.open("img/believe.png").convert("RGBA")
    if img.width > img.height:
        return False
    else:
        background = Image.new('RGBA', (20, 20), color=(255, 255, 255, 255))
        if img.height > believe.height:
            print("тут")
            w = img.width
            h = img.height
            while h > believe.height and w > believe.width:
                w = int(w * 0.99)
                h = int(h * 0.99)
                img.thumbnail((w, h))
        else:
            print("или тут")
            w = img.width
            h = img.height
            while believe.height > h and  believe.width > w:
                w = int(w * 0.99)
                h = int(h * 0.99)
                believe.thumbnail((w, h))
        background = Image.new('RGBA', (believe.width, believe.height), color=(255, 255, 255, 255))
        background.paste(img, (0, 0), img)
        background.paste(believe, (0, 0), believe)
        name = f"believe/{int(datetime.today().timestamp())}.png"
        background.save(name, format='png', subsampling=0, quality=100)
        return name


def gallery(path):
    img = Image.open(path).convert("RGBA")
    gallery = Image.open("img/gallery.png").convert("RGBA")
    if img.width > img.height:
        background = Image.new('RGBA', (gallery.width, gallery.height), color=(255, 255, 255, 255))
        w = img.width
        h = img.height
        if w < 950:
            return False
        while w > 950:
            w = int(0.95 * w)
            h = int(0.95 * h)
        img.thumbnail((w, h))
        background.paste(img, (35, 141), img)
        background.paste(gallery, (0, 0), gallery)
        name = f"gallery/{int(datetime.today().timestamp())}.png"
        background.save(name, format='png', subsampling=0, quality=100)
        return name
    else:
        return False


def four_colors_poster(picture, text):
    mode = [4, 5, 6]
    pre = Image.open(picture).convert('L')
    res = []
    for m in mode:

        pre_w = pre.width
        pre_h = pre.height
        picture = np.array(pre)
        n = m   # Number of levels of quantization
        indices = np.arange(0,256)   # List of all colors
        divider = np.linspace(0,255,n+1)[1] # we get a divider
        quantiz = np.int0(np.linspace(0,255,n)) # we get quantization colors
        color_levels = np.clip(np.int0(indices/divider),0,n-1) # color levels 0,1,2..
        palette = quantiz[color_levels] # Creating the palette
        ksize = (5, 5)
        picture = cv2.blur(picture, ksize)
        im2 = palette[picture]  # Applying palette on image
        im2 = cv2.convertScaleAbs(im2) # Converting image back to uint8
        name_bw = randint(0,9)
        filename = f"{name_bw}.png"
        cv2.imwrite(filename, im2)
        image = cv2.imread(filename)
        os.remove(filename)
        rgb=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)

        # Define lower and uppper limits of what we call "brown"
        grey0 = np.array([0, 0, 0])
        grey1 = np.array([0, 0, 60])
        grey2 = np.array([0, 0, 61])
        grey3 = np.array([0, 0, 110])
        grey4 = np.array([0, 0, 121])
        grey5 = np.array([0, 0, 190])
        grey6 = np.array([0, 0, 191])
        grey7 = np.array([0, 0, 255])
        mask0 = cv2.inRange(rgb, grey0, grey1)
        mask1 = cv2.inRange(rgb, grey2, grey3)
        mask2 = cv2.inRange(rgb, grey4, grey5)
        mask3 = cv2.inRange(rgb, grey6, grey7)
        colors = [(102, 255, 255), (204, 51, 51), (153, 204, 0), (255, 204, 0)]
        ims = []
        i = 0
        for color in colors:
            image[mask0 > 0] = (0, 0, 0)
            image[mask1 > 0] = (51, 51, 51)
            image[mask2 > 0] = color
            image[mask3 > 0] = (255, 255, 255)
            im = Image.fromarray(image)
            if im.width > im.height:
                delta = int((im.width - im.height) / 2)
                crop = im.height
                point = delta
                area = (point, 0, crop+point, crop)
            else:
                delta = int((im.height - im.width) / 2)
                crop = im.width
                point = delta
                area = (point, 0, crop + point, crop)
            im = im.crop(area)
            ims.append(im)
            i = i + 1
        p1 = ims[0].height
        p2 = ims[0].width
        background = np.zeros((p1 * 2, p2 * 2, 3), np.uint8)
        background[0:p1, 0:p2] = ims[0]
        background[p1:, 0:p2] = ims[1]
        background[0:p1, p2:] = ims[2]
        background[p1:, p2:] = ims[3]
        background = Image.fromarray(background)
        background.thumbnail((int(pre_w * 0.8), int(pre_h * 0.8)))
        draw_background = ImageDraw.Draw(background)
        font_size = 2000
        font = ImageFont.truetype(bold, font_size)
        x_t, y_t = draw_background.textsize(text, font)
        stripe = int(0.05 * background.width)
        while x_t > 0.99 * background.width:
            font_size -= 10
            font = ImageFont.truetype(bold, font_size)
            x_t, y_t = draw_background.textsize(text, font)
        new_width = 2 * stripe + background.width
        new_height = stripe + background.height + stripe + y_t
        background_new = Image.new('RGB', (new_width, new_height), color=(255, 255, 200))
        draw_background_new = ImageDraw.Draw(background_new)
        c = int((new_width - x_t) / 2)
        d = stripe + background.height
        #print(c, d)
        draw_background_new.text(
            (c, d),
            text=text,
            fill=(0, 0, 0),
            font=font)
        background_new.paste(background, (stripe, stripe))
        name = f"posters/{int(datetime.today().timestamp())}.png"
        background_new.save(name, format='png', subsampling=0, quality=100)
    return name


#удаление фона
def remove_background(raw, mode):
    name_pic = randint(10,99)
    path = f"rmbg/{name_pic}.jpg"
    file_info = bot.get_file(raw)
    downloaded_file = bot.download_file(file_info.file_path)
    with open(path, 'wb') as new_file:
        new_file.write(downloaded_file)

    name = f"rmbg/{int(datetime.today().timestamp())}_no_bg.png"

    input = cv2.imread(path)
    output = remove(input)
    cv2.imwrite(name, output)

    if True:
        img = Image.open(name)
        img = img.convert("RGBA")
        img_arr = np.array(img)
        i = 0
        j = 0
        empty_index = 0
        summ = 0
        stop = False
        for row in img_arr:
            i += 1
            if stop:
                break
            for point in row:
                j += 1
                summ = 0
                for n in point:
                    summ += n
                if summ == 0:
                    empty_index += 1
                else:
                    stop = True
                    break
            summ = 0
        up = i

        i = 0
        j = 0
        stop = False
        summ = 0
        while i < img.width - 1:
            i += 1
            if stop:
                break
            while j < img.height - 1:
                j += 1
                for p in img_arr[j][i]:
                    summ += p
                if summ > 60:
                    stop = True
                    break
            j = 0
        left = i

        i = img.width - 1
        j = img.height - 1
        stop = False
        summ = 0
        right = 0
        while i > 0:
            if stop:
                break
            while j > 0:
                j -= 1
                for p in img_arr[j][i]:
                    summ += p
                if summ > 60:
                    right = i
                    stop = True
                    break
            i -= 1
            j = img.height

        i = img.width - 1
        j = img.height - 1
        stop = False
        summ = 0
        down = 0
        while j > 0:
            if stop:
                break
            while i > 0:
                i -= 1
                for p in img_arr[j][i]:
                    summ += p
                if summ > 60:
                    down = j
                    stop = True
                    break
            j -= 1
            i = img.width

        area = (left, up, right, down)
        img = img.crop(area)
        # if img.width > img.height:
            # delta = int((img.width - img.height) / 2)
            # crop = img.height
            # point = delta
            # area = (point, 0, crop+point, crop)
        # else:
            # delta = int((img.height - img.width) / 2)
            # crop = img.width
            # point = delta
            # area = (point, 0, crop + point, crop)
        # img = img.crop(area)
        img.save(name)
    if mode == "popart":
        return
        # Не работает
        img = Image.open(io.BytesIO(result)).convert("LA")
        img = img.convert("RGBA")
        if img.width > img.height:
            delta = int((img.width - img.height) / 2)
            crop = img.height
            point = delta
            area = (point, 0, crop+point, crop)
        else:
            delta = int((img.height - img.width) / 2)
            crop = img.width
            point = delta
            area = (point, 0, crop + point, crop)
        img = img.crop(area)
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(4)
        img_bigger = Image.new('RGBA', (int(img.width * 1.5), int(img.height * 1.5)), color=(255, 255, 255, 0))
        w = int((img_bigger.width - img.width) / 2)
        z = int((img_bigger.height - img.height) / 2)
        img_bigger.paste(img, (w, z), img)
        img = img_bigger
        img.save(name)
    return name


def four_color(img):
    img = Image.open(img)
    colors = {
        "pallete0" : [(255, 68, 85, 0), (250, 100, 65, 0), (25, 78, 114, 0), (33, 67, 66, 0)],
        "pallete1" : [(0, 78, 114, 0), (33, 67, 66, 0), (254, 170, 186, 0), (50, 118, 117, 0)],
        "pallete2" : [(254, 170, 186, 0), (50, 118, 117, 0), (147, 34, 34, 0), (118, 152, 152, 0)],
        "pallete3" : [(83, 204, 185, 0), (221, 102, 34, 0), (0, 102, 83, 0), (102, 34, 34, 0)],
        "pallete4" : [(255, 255, 187, 0), (255, 85, 85, 0), (169, 154, 135, 0), (153, 85, 85, 0)],
        "pallete5" :[(250, 30, 120, 0), (250, 95, 30, 0), (28, 150, 90, 0), (251, 28, 96, 0)]
    }
    output_path = f"fc/{int(datetime.today().timestamp())}_fc.png"
    levels = [255, 90]
    pallete = "pallete" + str(randint(0, len(colors) - 1))
    #print(pallete)
    pallete_colors = colors[pallete]
    new_color = []
    new_palette = []
    layers = []
    for level in levels:
        for color in pallete_colors:
            for point in color:
                if color.index(point) == 3:
                    point = level
                new_color.append(point)
            new_palette.append(new_color)
            new_color = convert_to_tuple(new_color)
            overlay_img = Image.new('RGBA', (img.width, img.height), color=new_color)
            layers.append(overlay_img)
            new_color = []
    new_image_width = img.width * 2
    new_image_height = img.height * 2
    new_image = Image.new('RGBA', (new_image_width, new_image_height), color=(255, 255, 255, 0))
    new_image_overlay = Image.new('RGBA', (new_image_width, new_image_height), color=(255, 255, 255, 255))
    picture = np.array(new_image)
    p1 = img.height
    p2 = img.width
    picture[0:p1, 0:p2] = img
    picture[p1:, 0:p2] = img
    picture[0:p1, p2:] = img
    picture[p1:, p2:] = img

    picture_background = np.array(new_image_overlay)
    picture_background[0:p1, 0:p2] = layers[0]
    picture_background[p1:, 0:p2] = layers[1]
    picture_background[0:p1, p2:] = layers[2]
    picture_background[p1:, p2:] = layers[3]
    picture_background = Image.fromarray(picture_background)
    picture = Image.fromarray(picture)
    picture_background.paste(picture, (0, 0), picture)

    picture_overlay = np.array(new_image_overlay)
    picture_overlay[0:p1, 0:p2] = layers[4]
    picture_overlay[p1:, 0:p2] = layers[5]
    picture_overlay[0:p1, p2:] = layers[6]
    picture_overlay[p1:, p2:] = layers[7]


    picture_overlay = Image.fromarray(picture_overlay)
    picture_background.paste(picture_overlay, (0, 0), picture_overlay)

    picture_background.save(output_path)
    return output_path


#функция отправки сообщений, доделать так, чтобы работало в слоумоде
def send_it(cid, content, type_of_msg):
    if type_of_msg == "text":
        bot.send_message(cid, f"{content}")
    elif type_of_msg == "sticker":
        bot.send_sticker(cid, content)
    elif type_of_msg == "video_note":
        bot.send_video_note(cid, content)
    elif type_of_msg == "pic":
        photo = open(content, 'rb')
        bot.send_photo(cid, photo)


#функция переноса слов
def split_it(word, max_len):
    slogs = rusyllab.split_words(word.split())
    word1 = ""
    word2 = ""
    while len(word1) < max_len:
        word1 += slogs[0]
        slogs.pop(0)
    word2 = ''.join(slogs)
    if word2 != "":
        slovo = word1 + "-\n" + word2
        #print(slovo)
        return word1 + "-\n" + word2
    else:
        return word1


def get_menu_from_hl(raw):
    name_pic = randint(10,99)
    path = f"hl2/{name_pic}.png"
    file_info = bot.get_file(raw)
    downloaded_file = bot.download_file(file_info.file_path)
    with open(path, 'wb') as new_file:
        new_file.write(downloaded_file)
    image = Image.open(path)
    # enhancer = ImageEnhance.Contrast(image)
    # image = enhancer.enhance(0.9)
    menu = Image.open("hl2/hl2.png")
    i_w = image.width
    i_h = image.height
    m_w = menu.width
    m_h = menu.height
    if True:
        while m_w >= 0.98 * i_w:
            m_w = 0.9 * m_w
            m_h = 0.9 * m_h
        menu.thumbnail((int(m_w), int(m_h)))
        image.paste(menu, (0, 0),  menu)
        image.save(path, format='PNG', subsampling=0, quality=100)
        return path
    else:
        return "None"


def change_img(picture, text):
    #im = picture
    height, width = picture.shape
    picture = Image.fromarray(picture)
    enhancer = ImageEnhance.Contrast(picture)
    picture = enhancer.enhance(1.2)
    picture = np.array(picture)
    n = 5    # Number of levels of quantization
    indices = np.arange(0,256)   # List of all colors
    divider = np.linspace(0,255,n+1)[1] # we get a divider
    quantiz = np.int0(np.linspace(0,255,n)) # we get quantization colors
    color_levels = np.clip(np.int0(indices/divider),0,n-1) # color levels 0,1,2..
    palette = quantiz[color_levels] # Creating the palette
    ksize = (7, 7)
    picture = cv2.blur(picture, ksize)
    im2 = palette[picture]  # Applying palette on image
    im2 = cv2.convertScaleAbs(im2) # Converting image back to uint8
    name_bw = randint(0,9)
    filename = f"chnged/{name_bw}.png"
    cv2.imwrite(filename, im2)
    image = cv2.imread(filename)
    rgb=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)

    # Define lower and uppper limits of what we call "brown"
    grey=np.array([0,0,0])
    grey0=np.array([0,0,40])

    grey1=np.array([0,0,41])
    grey2=np.array([0,0,101])

    grey3=np.array([0,0,102])
    grey4=np.array([0,0,181])

    #grey5=np.array([0,0,152])
    #grey6=np.array([0,0,201])

    grey7 = np.array([0, 0, 182])
    grey8 = np.array([0, 0, 255])

    # Mask image to only select browns
    mask0 = cv2.inRange(rgb, grey, grey0)
    mask1 = cv2.inRange(rgb, grey1, grey2)
    mask2 = cv2.inRange(rgb, grey3, grey4)
    #mask3 = cv2.inRange(rgb, grey5, grey6)
    mask4 = cv2.inRange(rgb, grey7, grey8)

    # Change image to red where we found brown
    image[mask0 > 0] = (77, 32, 0)
    image[mask1 > 0] = (154, 105, 57)
    image[mask2 > 0] = (32, 21, 187)
    #image[mask3 > 0] = (158, 149, 24)
    image[mask4 > 0] = (169, 229, 253)
    name = 'chnged/' + str(randint(1000,1999)) + ".png"
    img_pil = Image.fromarray(image)
    draw1 = ImageDraw.Draw(img_pil)
    font_size = 2000
    font = ImageFont.truetype(bold, font_size)
    x_t, y_t = draw1.textsize(text, font)
    max_height = 0.3 * img_pil.height
    while x_t > 0.8 * img_pil.width or y_t > max_height:
        font_size -= 10
        font = ImageFont.truetype(bold, font_size)
        x_t, y_t = draw1.textsize(text, font)
    h = height + y_t
    w = width
    background = np.zeros((h + 10, w, 3), np.uint8)
    background[0:,0:] = (77, 32, 0)
    background[0:image.shape[0], 0:image.shape[1]] = image
    img_pil = Image.fromarray(background)
    draw1 = ImageDraw.Draw(img_pil)
    x = int(width - x_t) / 2
    y = int(height)
    draw1.text(
        (x, y-10),
        text=text,
        fill=(154, 105, 57),
        font=font)
    background = np.array(img_pil)
    offset = int(0.05 * background.shape[1]) + 1
    y_offset = int(0.05 * background.shape[1]) + 1
    h = int(2*offset + background.shape[0])
    w = int(2*offset + background.shape[1])
    background2 = np.zeros((h, w, 3), np.uint8)
    background2[0:,0:] = (169, 229, 253)
    background2[offset:offset + background.shape[0], offset:offset + background.shape[1]] = background
    cv2.imwrite(name, background2)
    return name


def transliterate(name):
   # Слоаврь с заменами
   slovar = {'а':'a','б':'b','в':'v','г':'g','д':'d','е':'e','ё':'e',
      'ж':'zh','з':'z','и':'i','й':'i','к':'k','л':'l','м':'m','н':'n',
      'о':'o','п':'p','р':'r','с':'s','т':'t','у':'u','ф':'f','х':'h',
      'ц':'c','ч':'cz','ш':'sh','щ':'scz','ъ':'','ы':'y','ь':'','э':'e',
      'ю':'u','я':'ja', 'А':'A','Б':'B','В':'V','Г':'G','Д':'D','Е':'E','Ё':'E',
      'Ж':'ZH','З':'Z','И':'I','Й':'I','К':'K','Л':'L','М':'M','Н':'N',
      'О':'O','П':'P','Р':'R','С':'S','Т':'T','У':'U','Ф':'F','Х':'H',
      'Ц':'C','Ч':'CZ','Ш':'SH','Щ':'SCH','Ъ':'','Ы':'y','Ь':'','Э':'E',
      'Ю':'U','Я':'YA',',':'','?':'',' ':'_','~':'','!':'','@':'','#':'',
      '$':'','%':'','^':'','&':'','*':'','(':'',')':'','-':'','=':'','+':'',
      ':':'',';':'','<':'','>':'','\'':'','"':'','\\':'','/':'','№':'',
      '[':'',']':'','{':'','}':'','ґ':'','ї':'', 'є':'','Ґ':'g','Ї':'i',
      'Є':'e', '—':''}

   for key in slovar:
      name = name.replace(key, slovar[key])
   return name


@bot.message_handler(content_types=["new_chat_members"])
def handler_new_member(message):
    cid = message.chat.id
    hello_pack = ["CAACAgIAAx0CSGZHzgACIOhiBnkoVndDqA90mHN09HzNCFbB8AACFwEAAtLdaQXeYl54aAbKpiME", 
    "CAACAgIAAx0CSGZHzgACIOpiBnml2CUc7XAI3bIsPcJYob98fgACggADv4vTGIo1o3McvzNZIwQ",
    "CAACAgIAAx0CSGZHzgACIOxiBnoE3AQGVCKTvjDaNS8MXjhaWwAC3QADQcv8I4FMVmKHuEALIwQ",
    "CAACAgIAAx0CSGZHzgACIO5iBnoRx5EYOm9QuDfpv2-D-4g4qwACxMoAAmOLRgzqJ2yzxBcP9CME"
    ]
    index = random.randint(0, 4)
    #print(index)
    sticker = hello_pack[index]
    bot.send_sticker(cid, sticker)


@bot.message_handler(content_types=["left_chat_member"])
def handler_left_member(message):
    cid = message.chat.id
    hello_pack = [
        "CAACAgIAAx0CSGZHzgACIPBiBno1WUlxGsvFbEYEvbs8Qda2hwACrAADdY2aHLh-CNd8mGWyIwQ",
        "CAACAgIAAx0CSGZHzgACIPxiBnxmLkP5YAEQRbpXUlUMttQkSAACoQADaJpdDA-1tNwdM_RKIwQ",
        "CAACAgIAAx0CSGZHzgACIPpiBnxO8OxhyptaEkOMSj4IsQtazQACYgADaJpdDJ9PJNArZRlvIwQ"
    ]
    sticker = hello_pack[random.randint(0, 2)]
    bot.send_sticker(cid, sticker)
    with open(f"chats/{cid}_members.json", 'r', encoding='utf-8') as cid_mebers:
        dump_from_file_1 = json.load(cid_mebers)
    all_members = list(dump_from_file_1[str(cid)]['members'].keys())
    #Удалить человека из списка участников
    #all_members.remove()


@bot.message_handler(commands=['tumen'])
def tmn(message):
    cid = message.chat.id
    global tumenec_date

    last_tumenec = datetime.now().strftime("%d")

    with open(f"chats/{cid}_members.json", 'r', encoding='utf-8') as cid_mebers:
        dump_from_file_1 = json.load(cid_mebers)

    all_members = list(dump_from_file_1[str(cid)]['members'].keys())
    selected_tumenec = all_members[randint(0, len(all_members)-1)]

    tmn_date = 0
    try:
        tmn_date = dump_from_file_1[str(cid)]["tmn_date"]
    except:
        pass

    if tmn_date != last_tumenec:
        dump_from_file_1[str(cid)]['members'][str(selected_tumenec)] = int(dump_from_file_1[str(cid)]['members'][str(selected_tumenec)]) + 1
        dump_from_file_1[str(cid)]["tmn_date"] = last_tumenec
        with open(f"chats/{cid}_members.json", 'w+', encoding='utf-8') as cid_mebers:
            json.dump(dump_from_file_1, cid_mebers, ensure_ascii=False)

        tumenec_pic = ["img/tmn.png", "img/tumen_2.jpg"]
        tumenec_pic = tumenec_pic[randint(0,1)]
        ff = open(tumenec_pic, 'rb')
        try:
            user = bot.get_chat_member(cid, selected_tumenec)
            name = user.user.first_name
            mention = f"[{name}](tg://user?id={str(selected_tumenec)})"
            msg = f"Встречайте тюменца дня! Сегодня это {mention}."
            bot.send_photo(cid, ff, caption=msg, parse_mode="Markdown")
            ff.close()
        except:
            bot.send_message(cid, f"Ошибка user_id: {selected_tumenec}")
    else:
        bot.send_message(cid, "Сегодня уже выбирали тюменца")


@bot.message_handler(commands=['tumen_stat'])
def tmn(message):
    cid = message.chat.id
    with open(f"chats/{cid}_members.json", 'r', encoding='utf-8') as cid_mebers:
        dump_from_file_1 = json.load(cid_mebers)
    
    members = list(dump_from_file_1[str(cid)]["members"].keys())
    msg = "Статистика тюменцев дня:\n"
    for m in members:
        try:
            user = bot.get_chat_member(cid, m).user.first_name
            count = dump_from_file_1[str(cid)]["members"][str(m)]
            #print(f"TMN: {user}")
            if int(count) != 0:
                msg += f"{user}: *{count}*\n"
        except:
            pass
    bot.send_message(cid, msg, parse_mode="Markdown")
 

@bot.message_handler(commands=['whereiam'])
def get_chat(message):
    cid = str(message.chat.id)
    user_id = message.from_user.id
    with open("chats/countries.json", 'r', encoding='utf-8') as cs:
        countries_and_cities = json.load(cs)
    #print(countries_and_cities)
    countries = countries_and_cities.keys()
    countries_list = list(countries)
    country = countries_list[randint(0, len(countries) - 1)]
    cities = countries_and_cities[country]
    city = cities[randint(0, len(cities) - 1)]
    day = randint(1,28)
    month = randint(1,12)
    year = randint(1800,2200)
    mention = f"[{message.from_user.first_name}](tg://user?id={str(user_id)})"
    message = f"{mention} сейчас в *{city}, {country}* ({year} год)"
    bot.send_message(cid, message, parse_mode="Markdown")




urls = ["https://www.perekrestok.ru/cat/366/p/luk-repcatyj-krasnyj-kg-79012",
        "https://www.perekrestok.ru/cat/371/p/syr-syrobogatov-gauda-45-200g-3411013",
        "https://www.perekrestok.ru/cat/390/p/makarony-makfa-spirali-450g-62190",
        "https://www.perekrestok.ru/cat/423/p/napitok-gazirovannyj-coca-cola-zero-900ml-3660913",
        "https://www.perekrestok.ru/cat/381/p/file-indejki-perekrestok-kg-89015",
        "https://www.perekrestok.ru/cat/383/p/sosiski-cerkizovo-slivocnye-po-cerkizovski-650g-3438500",
        "https://www.perekrestok.ru/cat/391/p/caj-richard-royal-ceylon-cernyj-listovoj-180g-3503942",
        "https://www.perekrestok.ru/cat/465/p/lapsa-dosirak-bystrogo-prigotovlenia-so-vkusom-govadiny-90g-53019"
        ]


def get_latest_prices(cid):
    global urls
    last_json_data = {}
    msg_0 = ""
    for url in urls:
        bot.send_chat_action(cid, "typing")
        response_0 = requests.get(url)
        soup_0 = BeautifulSoup(response_0.text, 'lxml')
        try:
            for EachPart in soup_0.select('div[class="price-card-unit-value"]'):
                new_price = EachPart.get_text()
                #print(f"price: {new_price}")
            for EachPart in soup_0.select('h1[class="sc-fubCfw ejmUOF product__title"]'):
                name_card = EachPart.get_text()
                #print(f"Name card: {name_card}")
            
            msg_0 += f"📈 {name_card}: *{new_price}* \n"
            new_price = new_price.split("₽")
            new_price = new_price[0].lstrip().rstrip().replace(",",".")
            new_price = float(new_price)
        except:
            new_price = "N/A"
            name_card = "N/A"
            msg_0 += f"Ошибка при получении цены: {url}\n"
        if name_card != "N/A":
            last_json_data[str(name_card)] = {"url": url, "price": new_price, "delta": 0}
        
    return last_json_data
    

def get_prices(cid, path, old_prices_check=False):
    global urls
    msg_0 = ""
    json_data = {}
    old_prices = {}
    msg = ""
    new_prices = get_latest_prices(cid)
    if old_prices_check:
        with open(path, 'r', encoding='utf-8') as prices:
            old_prices = json.load(prices)

        for item in old_prices.keys():
            try:
                if item != "N/A":
                    old_price = old_prices[item]["price"]
                    new_price = new_prices[item]["price"]
                    delta = (new_price / old_price - 1) * 100
                    delta_2 = float(new_price) - float(old_price)
                    delta = round(delta, 2)
                    delta_2 = round(delta_2, 2)
                    part_1 = ""
                    part_2 = ""
                    part_3 = ""
                    part_2 += f"{item}: *{new_price}* ₽ "
                    if delta < 0:
                        part_1 = "🔻"
                    elif delta > 0:
                        part_1 = "🔺"      
                    elif delta == 0:
                        part_1 = "📈"
                    if delta != 0:
                        if delta_2 > 0:
                            delta_2 = "+" + str(delta_2)
                            delta = "+" + str(delta)
                        part_3 = f"(" + str(delta) + f" %, {delta_2} ₽)"
                    part_3 += "\n\n"
                    msg += part_1 + part_2 + part_3
            except:
                msg += "Ошибка запроса цены\n"
    else:
        for item in new_prices.keys():
            try:
                if item != "N/A":
                    new_price = new_prices[item]["price"]
                    #item = (item[:30] + '..') if len(item) > 35 else item
                    item += ": "
                    msg += f"📈 {item} *{new_price}* ₽\n\n"   
            except:
                msg += "\nОшибка\n"
    with open(path, 'w+', encoding='utf-8') as prices:
        json.dump(new_prices, prices, ensure_ascii=False)
    bot.send_message(cid, msg, parse_mode="Markdown")

         


@bot.message_handler(commands=['price'])
def luk_price(message):
    global new_price
    cid = message.chat.id
    bot.send_message(cid, "Out of service", parse_mode="Markdown")
    return
    if True:
        pricespath = "chats/" + str(cid) + "_prices.json"
        check_file = os.path.exists(pricespath)
        if check_file == False:
            prices = get_prices(cid, pricespath)
        else:
            prices = get_prices(cid, pricespath, True)
    return
    
    if message.text == "/price" or message.text == "/price@realrealtalk_bot":
        url = urls[random.randint(0, len(urls) - 1)]
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'lxml')
        rewiews = []
        details = ""
        

        try:
            for EachPart in soup.select('div[class="price-card-unit-value"]'):
                new_price = EachPart.get_text()
            for EachPart in soup.select('h1[class="product__title"]'):
                name_card = EachPart.get_text()
            for EachPart in soup.select('a[class="product__review"]'):
                details = EachPart.get_text()
            for EachPart in soup.select('a[class="product__comment"]'):
                details += f", {EachPart.get_text()}"
                
            for EachPart in soup.select('div[class=review-header-wrapper]'):
                reviews = EachPart.get_text()
                
            msg = f"📈 {name_card}: *{new_price}*\n{details}"
        except:
            msg = f"Ошибка при получении цены {url}" 
            new_price = "N/A"
            name_card = "N/A"
        int_price = "N/A"


        try:
            int_price = new_price.split("₽")
            int_price = int_price[0].lstrip().rstrip().replace(",",".")
            int_price = float(int_price)
        except:
            pass

        with open('chats/prices.json', 'r', encoding='utf-8') as prices:
            dump_from_file = json.load(prices)
        
        try:
            old_price = dump_from_file[str(name_card)]
            old_price = float(old_price)
        except:
            old_price = "N/A"
        
        
        if old_price != "N/A" and int_price != "N/A":
            delta = (int_price / old_price - 1) * 100
            delta = round(delta, 2)
            msg += f"\n Изменение цены: "
            if delta < 0:
                msg += "🔻 " + str(delta)
            elif delta > 0:
                msg += "🔺 +" + str(delta)
            elif delta == 0:
                msg += "не зафиксировано"
            
            #print(f"Старая цена: {old_price}\nНовая цена: {int_price}\nРазница: {delta}")
            
        # | (a — b) / [ (a + b) / 2 ] | * 100 %

        dump_from_file.update({str(name_card): int_price})
        #dump_from_file["prices"][str(name_card)]["price"] = str()

        with open('chats/prices.json', 'w+', encoding='utf-8') as prices:
            json.dump(dump_from_file, prices, ensure_ascii=False)

        bot.send_message(cid, msg, parse_mode="Markdown")
    else:
        pass
    
   

@bot.message_handler(commands=['stock'])
def get_stock_images(message):
    cid = message.chat.id
    text = ""
    if message.text == "/stock@realrealtalk_bot" or message.text == "/stock":
        bot.send_message(cid, "Пример: /stock Слово или фраза для поиска")
    else:
        text = message.text.split("/stock")[1].lstrip()
        url = f"https://stocksnap.io/search/{text}"
        mydivs = []
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'lxml')
        all_images = soup.find_all("a", class_="photo-grid-item sponsored")
        images = []
        for img in all_images:
            for i in img:
                if "img" in str(i):
                    image = str(i['src'])
                    images.append(image)
        bot.send_photo(cid, images[randint(0, len(images) - 1)])



@bot.message_handler(commands=['today'])
def today_is(message):
    cid = message.chat.id
    first_date = "2022-01-07"
    first_date = first_date.split('-')
    aa = datetime(int(first_date[0]),int(first_date[1]),int(first_date[2]))
    bb = datetime.today()
    cc = bb - aa
    dd = str(cc)
    delta = dd.split()[0]
    page = 6415 + int(delta)
    url = f'https://www.calend.ru/narod/{page}'
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'lxml')
    head = f"Сегодня по народному календарю - *{soup.h1.get_text()}*\n"
    try:
        for EachPart in soup.select('div[class*="maintext"]'):
            text_today = EachPart.get_text()
            sym = 0
            new_text = ""
            while sym < len(text_today):
                new_text += text_today[sym]
                if text_today[sym] == "." and text_today[sym + 1] != " ":
                    new_text += "\n\n"
                sym += 1
            sym = 0
            new_text = new_text.split(". ")
            del new_text[0]
            new_text = ". ".join(new_text)
        msg = f"{head}\n\n{new_text}\nИсточник: {url}"
        bot.send_message(cid, msg, parse_mode="Markdown")
    except:
        bot.send_message(cid, "Ошибка получения содержимого", parse_mode="Markdown")


@bot.message_handler(commands=['quote'])
def new_quote(message):
    cid = message.chat.id
    test = bot.get_chat(cid)
    pack = "_"
    trim = 0
    bot_id = 1814550471
    if len(message.text) >= 7:
        try:
            trim = int(message.text.split("quote")[1].lstrip())
        except:
            trim = 0
    start_num = 1
    #проверка на реплай
    try:
        text = message.reply_to_message.text
    except:
        text = "___"
    ab = ""
    try:
        type_msg = message.reply_to_message.forward_from_chat.type
    except:
        type_msg = "notchannel"
    lastname = ""
    can_create_sticker = False
    if message.text == "/quote" or trim > 0 or message.text == "/quote@realrealtalk_bot":
        can_create_sticker = True
    if can_create_sticker and text != "___" and type_msg != "channel":
        forwarded = False
        #проверка на пересланное сообщение
        if str(message.reply_to_message.forward_date) != "None":
            forwarded = True
        else:
            forwarded = False
        if forwarded:
            try:
                name = str(message.reply_to_message.forward_from.first_name)
            except:
                name = str(message.reply_to_message.forward_sender_name)
            try:
                lastname = str(message.reply_to_message.forward_from.last_name)
            except:
                lastname = "None"
            ab = str(name[0])
            if str(lastname) != "None":
                name += f" {lastname}"
                ab += str(lastname[0])
        else:
            name = message.reply_to_message.from_user.first_name
            ab = name[0]
            try:
                lastname = str(message.reply_to_message.from_user.last_name)
            except:
                lastname = ""
            if lastname != "None":
                    name += f" {lastname}"
                    ab += lastname[0]
        user_id = "None"
        if forwarded:
            try:
                user_id = message.reply_to_message.forward_from.id
            except:
                user_id = "None"
        else:
            user_id = message.reply_to_message.from_user.id
        ab = ab.rstrip()
        path = f"users/userpic{randint(0,9)}.png"
        userpic = ""
        mask = Image.open('img/usermask.png').convert('L')
        #разобраться с закрытыми профилями
        photo_founded = False
        try:
            user_photo = bot.get_user_profile_photos(user_id).photos[0][0].file_id
            file_info = bot.get_file(user_photo)
            downloaded_file = bot.download_file(file_info.file_path)
            with open(path, 'wb') as new_file:
                new_file.write(downloaded_file)
            photo = Image.open(path)
            out = ImageOps.fit(photo, mask.size)
            out.putalpha(mask)
            userpic = f"users/userpic{randint(0,9)}.png"
            out.save(userpic)
            userpic = Image.open(userpic)
        except:
            user_colors = [(250, 167, 116), (238, 122, 74), (110, 201, 203)]
            photo = Image.new("RGB", (mask.width, mask.height), user_colors[randint(0, len(user_colors)-1)] )
            out = ImageOps.fit(photo, mask.size)
            out.putalpha(mask)
            userpic = f"users/userpic{randint(0,9)}.png"
            font_size = 30
            font = ImageFont.truetype(thin, font_size)
            draw2 = ImageDraw.Draw(out)
            w, h = draw2.textsize(ab, font)
            x = (out.width - w) /2
            y = (out.height - h) / 2 - 2
            draw2.text(
                (x, y),
                text=ab,
                fill=(255, 255, 255),
                font=font)
            out.save(userpic)
            userpic = Image.open(userpic)
        if str(text) != "None":
            text_arr = text.split(" ")
            text_len = len(text)
            words_counts = len(text_arr)
            #new_text = split_it(text, 30)
            new_text  = ""
            m = 30
            n = 1
            for i in text_arr:
                new_text += i + " "
                if len(new_text) >= n * 30:
                    new_text += "\n"
                    n += 1
            font_size = 35
            new_text = (new_text[:350] + '...') if len(new_text) > 350 else new_text
            new_text = (new_text[:trim]) if trim > 0 else new_text
            font_name = ImageFont.truetype(bold, 26)
            font = ImageFont.truetype(thin, font_size)
            img = Image.open('img/t.png').convert('RGBA')
            draw1 = ImageDraw.Draw(img)
            w, h = draw1.textsize(new_text, font)
            if w < img.height / 2:
                w = int(img.height / 2)
            while w >= 400:
                font_size -= 1
                font = ImageFont.truetype(thin, font_size)
                w, h = draw1.textsize(new_text, font)
            x = 80
            y = (img.height - h) / 2
            a = x
            b = y-20
            c = x + w + 30
            d = y + h + 30
            if c <= 260:
                c = 300
            draw1.rounded_rectangle((a, b, c, d), radius=20, fill=(41, 58, 76))
            draw1.text((x+15,y-15),text=name,fill=(169, 229, 253),font=font_name)
            draw1.text(
                (x + 15, y + 16),
                text=new_text,
                fill=(255, 255, 255),
                font=font)
            point = int(datetime.today().timestamp())
            name = f"users/{point}.png"
            img.convert('RGBA')
            e = (img.height - h) / 2 + h - userpic.height/2
            img.paste(userpic, (10, int(e)))
            img.save(name, format='PNG', subsampling=0, quality=100)
            ff = open(name, 'rb')
            #chat_id = -1001214662606
            #здесь нужен транслит
            start_name = transliterate(f"{message.chat.title}_fav{start_num}_by_@realrealtalk_bot")
            emojis = ["😂", "🤦🏻‍", "😅", "😜", "🤜", "😳", "🙅‍", "😏", "🙈", "🙌", "👍", "👻"]
            title = f"{message.chat.title} Избранное #{start_num}"
            files = -569667681
            file_id = bot.send_document(files, ff).document.file_id
            found_stickers = False
            packs = []
            try:
                while found_stickers == False:
                    start_name = transliterate(f"{message.chat.title}_fav{start_num}_by_@realrealtalk_bot")
                    title = f"{message.chat.title} Избранное #{start_num}"
                    pack = bot.get_sticker_set(start_name)
                    packs.append(pack)
                    len_of_pack = len(pack.stickers)
                    if len_of_pack > 80:
                        start_num += 1
                    else:
                        found_stickers = True
            except:
                pack = bot.create_new_sticker_set(1249881249, name=start_name, title=title, png_sticker=file_id, tgs_sticker="", emojis=emojis[randint(0, len(emojis)-1)])
                len_of_pack = len(bot.get_sticker_set(start_name).stickers) - 1
                sticker = bot.get_sticker_set(start_name).stickers[len_of_pack].file_id
                send_it(cid, sticker, "sticker")
            else:
                bot.add_sticker_to_set(1249881249, name=start_name, png_sticker=file_id, tgs_sticker="", emojis=emojis[randint(0, len(emojis)-1)])
                len_of_pack = len(bot.get_sticker_set(start_name).stickers) - 1
                sticker = bot.get_sticker_set(start_name).stickers[len_of_pack].file_id
                send_it(cid, sticker, "sticker")
    #самый первый if. Если боту отправили просто "/quote"
    else:
        if message.text == "/quote del":
            global del_sticker
            msg = "Чтобы удалить стикер отправьте его в следующем сообщении"
            send_it(cid, msg, "text")
            del_sticker = True
        else:
            text = "Команда для создания стикера из сообщения. Чтобы создать стикер необходимо ответить на нужное сообщение с этой командой"
            send_it(cid, text, "text")


@bot.message_handler(content_types=["photo"])
def handle_photo(message):
    cid = message.chat.id
    global photach_frequency
    

    if message.chat.type == "private":
        bot.send_message(cid, "Этот бот не работает в личных сообщениях. Нужно пригласить его в чат")
    try:
        raw = message.photo[3].file_id
    except IndexError:
        try:
            raw = message.photo[2].file_id
        except:
            print("Ошибка, слишком маленькое изображение")
    try:
        caption = message.caption.lower()
    except AttributeError:
        caption = "NoneNone"
    else:
        if message.chat.type != "private" and (caption == "gallery"):
            name_pic = randint(10,99)
            path = f"gallery/{name_pic}.jpg"
            file_info = bot.get_file(raw)
            downloaded_file = bot.download_file(file_info.file_path)
            with open(path, 'wb') as new_file:
                new_file.write(downloaded_file)
            res = gallery(path)
            if res != False:
                ff = open(res, 'rb')
                bot.send_document(cid, ff)
                return
            else:
                bot.send_message(cid, "Вертикальное либо слишком маленькое изображение")
                return
        if message.chat.type != "private" and (caption == "believe"):
            name_pic = randint(10,99)
            path = f"believe/{name_pic}.jpg"
            file_info = bot.get_file(raw)
            downloaded_file = bot.download_file(file_info.file_path)
            with open(path, 'wb') as new_file:
                new_file.write(downloaded_file)
            res = believe(path)
            if res != False:
                ff = open(res, 'rb')
                bot.send_document(cid, ff)
                return
            else:
                bot.send_message(cid, "Горизонтальное либо слишком маленькое изображение")
                return
        if message.chat.type != "private" and (caption == "natgeo"):
            res = nat_geo(raw)
            if res != False:
                ff = open(res, 'rb')
                bot.send_photo(cid, ff)
                return
            else:
                bot.send_message(cid, "Слишком маленькое изображение")
                return
        elif message.chat.type != "private" and caption == "rmbg":
            bot.send_chat_action(cid, "upload_photo")
            pic_1 = remove_background(raw, "classic")
            bot.send_chat_action(cid, "upload_photo")
            ff = open(pic_1, 'rb')
            bot.send_sticker(cid, ff)
            ff.close()
            ff = open(pic_1, 'rb')
            bot.send_chat_action(cid, "upload_photo")
            bot.send_document(cid, ff)
            ff.close()
            os.remove(pic_1)
            return
        elif message.chat.type != "private" and (caption == "obey" or caption[0:4] == "obey"):
            words = caption.split(" ")
            if len(words) <=3 and len(caption) < 18:
                name_pic = randint(10,99)
                path = f"chnged/{name_pic}.jpg"
                file_info = bot.get_file(raw)
                downloaded_file = bot.download_file(file_info.file_path)
                with open(path, 'wb') as new_file:
                    new_file.write(downloaded_file)
                im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                img_text = caption.split("obey")[1]
                if len(img_text) < 2:
                    img_text = "OBEY"
                else:
                    img_text = img_text.lstrip().rstrip().upper()
                pic = change_img(im, img_text)
                ff = open(pic, 'rb')
                bot.send_document(cid, ff)
                ready = True
                return
            else:
                send_it(cid, "Слишком много символов",  "text")
                return
        elif message.chat.type != "private" and (caption == "hl2" or caption[0:3] == "hl2"):
            pic1 = get_menu_from_hl(raw)
            if pic1 != "None":
                ff = open(pic1, 'rb')
                bot.send_document(cid, ff)
                os.remove(pic1)
                ready  = True
                return
            else:
                send_it(cid, "Это рботает только с горизонтальными изображениями", "text")
                return
        elif message.chat.type != "private" and caption == "4":
            name_pic = randint(5000,9999)
            path = f"rmbg/{name_pic}.png"
            file_info = bot.get_file(raw)
            downloaded_file = bot.download_file(file_info.file_path)
            with open(path, 'wb') as new_file:
                new_file.write(downloaded_file)
            pic_1 = remove_background(path, "popart")
            pic_2 = four_color(pic_1)
            ff = open(pic_2, 'rb')
            bot.send_sticker(cid, ff)
            print("удаление фона")
            return
        elif message.chat.type != "private" and (caption == "poster" or caption[0:6] == "poster"):
            words = caption.split(" ")
            if len(words) <=3 and len(caption) < 18:
                name_pic = randint(100,999)
                path = f"posters/{name_pic}.jpg"
                file_info = bot.get_file(raw)
                downloaded_file = bot.download_file(file_info.file_path)
                with open(path, 'wb') as new_file:
                    new_file.write(downloaded_file)
                img_text = caption.split("poster")[1]
                if len(img_text) < 2:
                    img_text = "POSTER"
                else:
                    img_text = img_text.lstrip().rstrip().upper()
                pic = four_colors_poster(path, img_text)
                ff = open(pic, 'rb')
                bot.send_document(cid, ff)
                ready = True
                return
            else:
                send_it(cid, "Слишком много символов",  "text")
                return
    case_caption = "none" not in caption.lower()
    case_chat = str(cid) == photach_chat or str(cid) == test
    chance = 20
    if caption.lower() != "nonenone":
        chance = 30
    case_random = randint(50,80) == 0
    case0 = case_chat and case_random
    if case_chat:
        print(f"CASE 0: {case0}, CASE RANDOM: {case_random}, CAPTION: {caption.lower()}")
    if case0:
        return
        photach_text = photach(caption.lower())
        bot.send_chat_action(cid, "typing")
        time.sleep(5)
        photach_frequency = randint(120,150)
        bot.send_message(cid, photach_text, parse_mode="HTML", reply_to_message_id=message.message_id)
        return


@bot.message_handler(commands=['street'])
def gen_image(message):
    cid = message.chat.id
    if message.chat.type == "private":
        bot.send_message(cid, "Этот бот не работает в личных сообщениях. Нужно пригласить его в чат")
    else:
        if message.text == "/street@realrealtalk_bot" or message.text == "/street":
            send_it(cid, "Сгенерировать табличку - /street Название улицы", "text")
        else:
            original_text = message.text.split('/street', 1)[1].lstrip().upper()
            text_arr = original_text.lstrip().rstrip().split(' ')
            words_counts = len(text_arr)
            street_type = ""
            zh_rod = ["улица", "аллея", "линия", "набережная", "магистраль", "площадь"]
            m_rod = ["переулок", "проезд", "проспект", "бульвар", "вал", "канал", "тракт", "тупик"]
            all_r = ["улица", "аллея", "линия", "набережная", "магистраль", "площадь", "переулок", "проезд",
                     "проспект", "бульвар", "вал", "канал", "тракт", "тупик", "шоссе"]
            # Проверка слова и определение рода
            if original_text[-2:] == "ОЙ" or original_text[-2:] == "ИЙ" or original_text[-2:] == "ЫЙ":
                street_type = m_rod[randint(0, len(m_rod) - 1)]
            elif original_text[-2:] == "АЯ" or original_text[-2:] == "ЯЯ":
                street_type = zh_rod[randint(0, len(zh_rod) - 1)]
            elif original_text[-2:] == "ОЕ" or original_text[-2:] == "ЕЕ":
                street_type = "шоссе"
            else:
                street_type = all_r[randint(0, len(all_r) - 1)]
            num = 1
            if street_type == "площадь" or street_type == "переулок" or street_type == "проезд" or street_type == "тупик":
                num = str(randint(1, 5))
            else:
                num = str(randint(1, 250))
            res = ""
            #original_text = "Отдыха от чата"
            if len(original_text) <= 33:
                if len(original_text) >= 16 and words_counts == 1:
                    res = split_it(original_text, 12)
                elif len(original_text) < 16:
                    res = original_text
                elif len(original_text) >= 16 and words_counts >= 2:
                    new_text = original_text.split(' ')
                    half = len(new_text) // 2
                    i = 0
                    word1 = ""
                    word2 = ""
                    while i != half:
                        word1 += new_text[0] + " "
                        new_text.pop(0)
                        i += 1
                    word2 = ' '.join(new_text)
                    res = word1 + "\n" + word2
                building = ["строение 1", "строение 2", "строение 3"]
                b = building[randint(0, 2)]
                street_name = Words("name", res)
                street_num = Words("number", num)
                street_building = Words("building", b)
                street_t = Words("street_type", street_type)

                data_arr = [street_name, street_num, street_building, street_t]
                img = Image.open('nums/street.png').convert('RGBA')
                draw = ImageDraw.Draw(img)
                for d in data_arr:
                    draw.text(
                        (d.left, d.top),
                        text=d.text,
                        fill=d.color,
                        font=d.font
                    )
                name = f"streets/{int(datetime.today().timestamp())}.png"
                img.convert('RGBA')
                img.save(name, format='PNG', subsampling=0, quality=100)
                photo = open(name, 'rb')
                bot.send_sticker(cid, photo)
            else:
                send_it(cid, "Слишком много символов. Максимальная длина - 32", "text")


@bot.message_handler(commands=['start'])
def start_bot(message):
    chat_properties = {}
    cid = message.chat.id
    if message.chat.type == "private":
        bot.send_message(cid, "Этот бот не работает в личных сообщениях. Нужно пригласить его в чат")
    else:
        filepath = "chats/" + str(cid) + ".json"
        check_file = os.path.exists(filepath)
        if check_file:
            bot.send_message(cid, f"Бот был запущен ранее")
        else:
            chat_properties['chat'] = {"status": "on", "mode": "default", "id": cid}
            chat_properties['triggers'] = {}
            chat_properties['sticker_replies'] = {}
            bot.send_message(cid, f"Бот запущен")
            with open(filepath, 'w+', encoding='utf-8') as fh0:
                json.dump(chat_properties, fh0, ensure_ascii=False)


@bot.message_handler(commands=['st_reply'])
def handle_text(message):
    cid = message.chat.id
    if message.chat.type == "private":
        bot.send_message(cid, "Этот бот не работает в личных сообщениях. Нужно пригласить его в чат")
    else:
        global st_trigger
        if message.text == "/st_reply@realrealtalk_bot" or message.text == "/st_reply":
            bot.send_message(cid, f"Чтобы добавить триггер со стикером отправьте: \n/st_reply слово-триггер и"
                                  f" стикер в следующем сообщении")
        else:
            st_trigger = message.text[9:].rstrip().lstrip().lower()
            bot.send_message(cid, "Отправьте стикер в следующем сообщении")


#  обработка стикеров на команду /st_reply
@bot.message_handler(content_types=["sticker"])
def handle_text(message):
    cid = message.chat.id
    filepath = "chats/" + str(cid) + ".json"
    global st_trigger
    if message.chat.type == "private":
        bot.send_message(cid, "Этот бот не работает в личных сообщениях. Нужно пригласить его в чат")
    else:
        if st_trigger != "********":
            sticker_id = message.sticker.file_id
            with open(filepath, 'r', encoding='utf-8') as fh3:
                dump_from_file = json.load(fh3)
            dump_from_file['sticker_replies'].update({st_trigger: sticker_id})
            with open(filepath, 'w', encoding='utf-8') as fh4:
                json.dump(dump_from_file, fh4, ensure_ascii=False)
            st_trigger = "********"
            bot.send_message(cid, "Ответ добавлен")
        global del_sticker
        if del_sticker == True:
            fileid = message.sticker.file_id
            try:
                try_delete = bot.delete_sticker_from_set(fileid)
            except:
                send_it(cid, "Ошибка. Нельзя удалять чужие стикеры", "text")
            else:
                send_it(cid, "Стикер удален", "text")
            del_sticker = False


@bot.message_handler(commands=['addtrigger'])
#  Добавление триггеров
def create_trigger(message):
    cid = message.chat.id
    filepath = "chats/" + str(cid) + ".json"
    check_file = os.path.exists(filepath)
    if message.chat.type == "private":
        bot.send_message(cid, "Этот бот не работает в личных сообщениях. Нужно пригласить его в чат")
    else:
        if check_file:
            if message.text == "/addtrigger@realrealtalk_bot" or message.text == "/addtrigger":
                bot.send_message(cid, f"Добавить триггер: \n/addtrigger слово1;слово2 = ответ")
            # если отправлена правильная команда
            else:
                if "=" in message.text:
                    trigger = message.text.split('=', 1)[0][12:].rstrip().lower()
                    phrase = message.text.split('=', 1)[1].lstrip()
                    if len(trigger) <= 1 or len(phrase) <= 1:
                        bot.send_message(cid, f"Ошибка. Минимальное количество символов - два")
                    else:
                        if ";" in trigger:
                            message = "Добавлено:\n"
                            trig_list = trigger.split(';')
                            with open(filepath, 'r', encoding='utf-8') as fh1:
                                dump_from_file = json.load(fh1)
                            for i in trig_list:
                                if len(i) > 1:
                                    dump_from_file['triggers'].update({i: phrase})
                                    message += f"{i} = {phrase}\n"
                                else:
                                    bot.send_message(cid, f"Слишком короткая фраза")
                            with open(filepath, 'w', encoding='utf-8') as fh2:
                                json.dump(dump_from_file, fh2, ensure_ascii=False)
                            bot.send_message(cid, message)
                        # если слово одно
                        else:
                            with open(filepath, 'r', encoding='utf-8') as fh3:
                                dump_from_file = json.load(fh3)
                            dump_from_file['triggers'].update({trigger: phrase})
                            with open(filepath, 'w', encoding='utf-8') as fh4:
                                json.dump(dump_from_file, fh4, ensure_ascii=False)
                            bot.send_message(cid, f"Добавлено: {trigger} = {phrase}")
                else:
                    bot.send_message(cid, f"/addtrigger слово1;слово2 = ответ")
        else:
            bot.send_message(cid, f"Отправьте команду /start для инициализации бота")


@bot.message_handler(commands=['showtriggers'])
def show_triggers(message):
    cid = message.chat.id
    filepath = "chats/" + str(cid) + ".json"
    check_file = os.path.exists(filepath)
    if message.chat.type == "private":
        bot.send_message(cid, "Этот бот не работает в личных сообщениях. Нужно пригласить его в чат")
    else:
        message = "Список активных триггеров:\n\n"
        if check_file:
            with open(filepath, 'r', encoding='utf-8') as fh5:
                dump_from_file = json.load(fh5)
            for i in dump_from_file['triggers']:
                message = message + f"{i} = {dump_from_file['triggers'][i]}\n"
            message = message + "\nСписок триггеров со стикерами:\n\n"
            try:
                for i in dump_from_file['sticker_replies']:
                    message = message + f"{i}\n"
            except KeyError:
                message += ""
            bot.send_message(cid, message)
        else:
            bot.send_message(cid, f"Отправьте команду /start для инициализации бота")


@bot.message_handler(commands=['deltrigger'])
def delete_trigger(message):
    cid = message.chat.id
    filepath = "chats/" + str(cid) + ".json"
    check_file = os.path.exists(filepath)
    deleted = "fdsjhfkdsghjythrdswq"
    if check_file:
        if message.text == "/deltrigger@realrealtalk_bot" or message.text == "/deltrigger":
            bot.send_message(cid, "Удалить триггер: /deltrigger слово")
        else:
            deleted = f"{message.text[11:].lstrip().rstrip().lower()}"
            with open(filepath, 'r', encoding='utf-8') as fh6:
                dump_from_file = json.load(fh6)
            try:
                del dump_from_file['triggers'][deleted]
                deleted = "******/*****"
                with open(filepath, 'w', encoding='utf-8') as fh8:
                    json.dump(dump_from_file, fh8, ensure_ascii=False)
            except KeyError:
                msg = "Слово не найдено"
            try:
                del dump_from_file['sticker_replies'][deleted]
                deleted = "******/*****"
                with open(filepath, 'w', encoding='utf-8') as fh8:
                    json.dump(dump_from_file, fh8, ensure_ascii=False)
            except KeyError:
                msg = "Слово не найдено"
        if deleted == "******/*****":
            msg = "Удалено"
        bot.send_message(cid, msg)
    else:
        bot.send_message(cid, f"Отправьте команду /start для инициализации бота")


comments = []
last_day_update = int(datetime.today().timestamp())


def photach(msg_text):
    global comments
    global last_day_update
    today = int(datetime.today().timestamp())
    message_words = msg_text.lower().split(" ")
    delta = today - last_day_update
    print(F"DELTA: {delta}")
    if len(comments) == 0 or  delta > 600800: 
        print("PHOTACH REINIT COMMENTS")
        comments = []
        try:
            message_words.remove("фотач")
        except:
            pass
        url_threads = f"https://2ch.hk/p/index.json"
        thread_request = requests.get(url_threads)
        threads = thread_request.json()["threads"]
        thread_num = 0
        threads_nums_arr = []
        for thread in threads:
            thread_num = thread["thread_num"]
            link = f"https://2ch.hk/p/res/{thread_num}.json"
            threads_nums_arr.append(link)

        for url in threads_nums_arr:
            r = requests.get(url).json()
            posts = r["threads"][0]["posts"]
            for post in posts:
                comment = post["comment"]
                comment = comment.replace("<br>", "\n")
                comment = BeautifulSoup(comment, 'lxml')
                comment = comment.get_text() + "\n"
                if "предыдущий тред" in comment.lower() or len(comment) > 150 or len(comment) < 9:
                    pass
                else:
                    comments.append(comment)
    print(f"LEN OF COMMENTS ARR: {len(comments)}")
    comment = comments[randint(0, len(comments) - 1)]

    try:
        message_words.remove("фотач")
    except:
        pass
    for word in message_words:
        if len(word) < 5:
            message_words.remove(word)
    if len(message_words) > 0:
        keyword = message_words[randint(0, len(message_words) - 1)]
        comment_with_keyword = ""
        for comment_ in comments:
            if keyword in comment_:
                print(f"COMMENT WITH KEYWORD: {keyword}")
                comment_with_keyword = comment_
                comments.remove(comment_)
                print(f"COMMENT: {comment_}")
                return comment_
        if comment_with_keyword == "":
            print("RANDOM COMMENT: KEYWORD NOT FOUND")
            comments.remove(comment)
            return comment
    else:
        print("RANDOM COMMENT: TOO SHORT MESSAGE")
        comments.remove(comment)
        return comment

    

photach_frequency = 150

@bot.message_handler(content_types=["text"])
def handle_text(message):
    global update
    global photach_frequency
    global photach_chat, test, ekat
    cid = message.chat.id
    filepath = "chats/" + str(cid) + ".json"
    current_date = datetime.now().strftime("%d")
    check_file = os.path.exists(filepath)
    user_id = message.from_user.id
    sent = False
    check_member = bot.get_chat_member(cid, user_id).status
    reply_id = 0
    
    if message.chat.type == "private":
        bot.send_message(cid, "Этот бот не работает в личных сообщениях. Нужно пригласить его в чат")
        return
    try:
        reply_id = message.reply_to_message.from_user.id
    except:
        reply_id = 0
    if str(reply_id)=="1814550471" and (message.text.lower() == "иди нахуй" or message.text.lower() == "иди лесом" or message.text.lower() == "иди воруй"):
        try:
            bot.delete_message(cid, message.reply_to_message.message_id)
        except:
            print(f"Delete error")

    if message.from_user.first_name != "Channel" and random.randint(0,3) == 0:
        try:
            with open(f"chats/{cid}_members.json", 'r', encoding='utf-8') as cid_mebers:
                dump_from_file_1 = json.load(cid_mebers)
        except:
            data_json = {cid: {"members": {user_id: 0}}}
            with open(f"chats/{cid}_members.json", 'w+', encoding='utf-8') as cid_mebers:
                json.dump(data_json, cid_mebers, ensure_ascii=False)

        with open(f"chats/{cid}_members.json", 'r', encoding='utf-8') as cid_mebers:
            dump_from_file_1 = json.load(cid_mebers)
        all_members = dump_from_file_1[str(cid)]['members'].keys()

        if str(user_id) in all_members:
            pass
        else:
            dump_from_file_1[str(cid)]["members"][str(user_id)] = 0
            with open(f"chats/{cid}_members.json", 'w+', encoding='utf-8') as cid_mebers:
                json.dump(dump_from_file_1, cid_mebers, ensure_ascii=False)
            
    videonotes = {
            "кринж":    {"file": "DQACAgIAAx0ESGZHzgACOstibsEH_zbAobLHm4TgjQFb94bgBAACLRgAAqdaeEuPtiLFauXi9yQE", "type": "part"},
            "граффити": {"file": "DQACAgIAAx0ESGZHzgACPOtieRAQBDHJpEVM5f_qLn2CYcaK_gACeBcAApZcyEvPrrEzs-th7yQE", "type": "part"},
            "зачем?":   {"file": "DQACAgIAAx0ESGZHzgACOwlib4LIhX39kiM3ypDeEuUMvQ7ZGgACDRoAAqdaeEvWzS8BwUpJwCQE", "type": "full"},
            "срать":    {"file": "DQACAgIAAx0ESGZHzgACO3xib9H7C99pRuK8Qv1LyeVlZNMNmQACJhUAAmtUgUuFyhnBycHb2SQE", "type": "part"},
            "ссать":    {"file": "DQACAgIAAx0ESGZHzgACO3xib9H7C99pRuK8Qv1LyeVlZNMNmQACJhUAAmtUgUuFyhnBycHb2SQE", "type": "part"},
            "кархарт":  {"file": "DQACAgIAAxkBAAETlj1ib4oWFSXkFNMZS8l9TcxD_OgIzAACLRoAAqdaeEtQiyOGoZMsoyQE", "type": "part"},
            "аирмакс":  {"file": "DQACAgIAAxkBAAETlj1ib4oWFSXkFNMZS8l9TcxD_OgIzAACLRoAAqdaeEtQiyOGoZMsoyQE", "type": "part"},
            "остался один": {"file": "DQACAgIAAx0ESGZHzgACOo1ibr5gSaGsfsh8ta5idFYUb-UzFwACDhgAAqdaeEtHQz1d4f1qPiQE", "type": "part"},
            "отобрали":     {"file": "DQACAgIAAx0ESGZHzgACOo1ibr5gSaGsfsh8ta5idFYUb-UzFwACDhgAAqdaeEtHQz1d4f1qPiQE", "type": "part"},
            "забрали":      {"file": "DQACAgIAAx0ESGZHzgACOo1ibr5gSaGsfsh8ta5idFYUb-UzFwACDhgAAqdaeEtHQz1d4f1qPiQE", "type": "part"},
            "не оставили":      {"file": "DQACAgIAAx0ESGZHzgACOo1ibr5gSaGsfsh8ta5idFYUb-UzFwACDhgAAqdaeEtHQz1d4f1qPiQE", "type": "part"},
            "не осталось":      {"file": "DQACAgIAAx0ESGZHzgACOo1ibr5gSaGsfsh8ta5idFYUb-UzFwACDhgAAqdaeEtHQz1d4f1qPiQE", "type": "part"}
            #"доброе утро":  {"file": "DQACAgIAAx0ESGZHzgACOoNibr37ptjbJKzYXeePWNSo2pjKbAACChgAAqdaeEseIl6McDWwtSQE", "type": "part"}
        }

    elif check_member != "left":
        #  ответ стикером
        if check_file:
            with open(filepath, 'r', encoding='utf-8') as fh7:
                dump_from_file = json.load(fh7)
            for a in dump_from_file['sticker_replies']:
                nado_li = random.randint(0,3)
                if a in message.text.lower() and nado_li == 1:  # and check_member != "left":
                    bot.send_sticker(cid, dump_from_file['sticker_replies'][a])
                    sent = True
                    break
        
        
        if message.text.lower() == "natgeo":
            try:
                all_photos = message.reply_to_message.json["photo"]
                photo = all_photos[-1]["file_id"]
            except:
                print("Get photo fail")
            res = nat_geo(photo)
            if res != False:
                ff = open(res, 'rb')
                bot.send_photo(cid, ff)
                return
            else:
                bot.send_message(cid, "Слишком маленькое изображение")
                return
        if message.text.lower() == "rmbg":
            try:
                all_photos = message.reply_to_message.json["photo"]
                photo = all_photos[-1]["file_id"]
            except:
                print("Get photo fail")
            bot.send_chat_action(cid, "upload_photo")
            pic_1 = remove_background(photo, "classic")
            bot.send_chat_action(cid, "upload_photo")
            ff = open(pic_1, 'rb')
            bot.send_sticker(cid, ff)
            ff.close()
            ff = open(pic_1, 'rb')
            bot.send_chat_action(cid, "upload_photo")
            bot.send_document(cid, ff)
            ff.close()
            os.remove(pic_1)
            return
        if message.text.lower() == "hl2":
            try:
                all_photos = message.reply_to_message.json["photo"]
                photo = all_photos[-1]["file_id"]
            except:
                print("Get photo fail")
            pic1 = get_menu_from_hl(photo)
            if pic1 != "None":
                ff = open(pic1, 'rb')
                bot.send_document(cid, ff)
                #print(pic1)
                os.remove(pic1)
                ready  = True
                return
            else:
                send_it(cid, "Это рботает только с горизонтальными изображениями", "text")
                return
    
    
        videonotes_keys = list(videonotes.keys())

        for key in videonotes_keys:
            trigger_type = videonotes[key]["type"]
            content = videonotes[key]["file"]
            chance = randint(0,2)
            if message.text.lower() == key and trigger_type == "full" and chance == 0:
                print(f"{cid}: {key}")
                bot.send_chat_action(cid, "record_video_note")
                time.sleep(5)
                bot.send_video_note(cid, content)
                return
            if key in message.text.lower() and trigger_type == "part" and chance == 0:
                print(f"{cid}: {key}")
                bot.send_chat_action(cid, "record_video_note")
                time.sleep(5)
                bot.send_video_note(cid, content)
                return

        hitler = f"img/hitler_ ({randint(1,11)}).jpg"
        photo_replies = {
            "гитлер": {"file": hitler, "type": "part", "chance": 0},
            "а знаете кто ещ": {"file": hitler, "type": "part", "chance": 0},
            "а знаете, кто ещ": {"file": hitler, "type": "part", "chance": 0}
        }
        photo_replies_keys = list(photo_replies.keys())
        for key in photo_replies_keys:
            trigger_type = photo_replies[key]["type"]
            content = photo_replies[key]["file"]
            chance = randint(0,2)
            try:
                chance = photo_replies[key]["chance"]
            except:
                chance = randint(0,2)
            image = open(hitler, 'rb')
            if message.text.lower() == key and trigger_type == "full" and chance == 0:
                print(f"{cid}: {key}")
                bot.send_sticker(cid, image)
                return
            if key in message.text.lower() and trigger_type == "part" and chance == 0:
                print(f"{cid}: {key}")
                bot.send_sticker(cid, image)
                return
            image.close()
        mention = f"[{message.from_user.first_name}](tg://user?id={str(user_id)})"

        text = message.text.lower()
        answer = random.randint(0, 3)
        print(f"TEXT ANSWER: {answer}")
        if answer == 0 and sent == False:
            if check_file:
                with open(filepath, 'r', encoding='utf-8') as fh7:
                    dump_from_file = json.load(fh7)
                for i in dump_from_file['triggers']:
                    if i in text and check_member != "left":
                        bot.send_message(cid, f"{mention}, {dump_from_file['triggers'][i]}", parse_mode="Markdown")
                        break
            else:
                bot.send_message(cid, f"Отправьте команду /start для инициализации бота")
        text_replies = {
            "да.": {"text": "*Манда* 😂😂😂", "type": "full", "chance": 1},
            "да!": {"text": "*Манда* 😂😂😂", "type": "full", "chance": 1},
            "да?": {"text": "*Манда* 😂😂😂", "type": "full", "chance": 1},
            "да...": {"text": "*Манда* 😂😂😂", "type": "full", "chance": 1},
            "куда?": {"text": "*В жопу резать провода* 😂😂😂", "type": "full", "chance": 1}
        }
        text_replies_keys = list(text_replies.keys())
        for key in text_replies_keys:
            trigger_type = text_replies[key]["type"]
            content = text_replies[key]["text"]
            chance = randint(0,2)
            try:
                chance = text_replies[key]["chance"]
                chance = randint(0, chance)
            except:
                chance = randint(0,2)

            if message.text.lower() == key and trigger_type == "full" and chance == 0:
                print(f"{cid}: {key}")
                bot.send_message(cid, content, parse_mode="Markdown")
                return
            if key in message.text.lower() and trigger_type == "part" and chance == 0:
                print(f"{cid}: {key}")
                bot.send_message(cid, content, parse_mode="Markdown")
                return
        if (str(cid) == photach_chat or str(cid) == test) and (message.text.lower() == "нет!" or message.text.lower() == "нет?" or message.text.lower() == "нет." or message.text.lower() == "нет...") and sent == False:
            answers = ["*Фудживода ответ* 😂😂😂", "*Гиродрочера ответ* 😂😂😂", "*Да!*"]
            bot.send_message(cid, answers[randint(0, len(answers)-1)], parse_mode="Markdown")
            return
        
        case1 = (str(cid) == photach_chat or str(cid) == test)
        print(f"CASE1: {case1}")
        case2 = message.message_id % photach_frequency == 0 or "фотач" in  message.text.lower() 
        if case1:
            print(f"CASE1: {case1}, CASE2: {case2}, FREQUENCY: {photach_frequency}")
        
        if case1 and case2 and sent == False:
            photach_text = photach(message.text.lower())
            bot.send_message(cid, photach_text, parse_mode="HTML", reply_to_message_id=message.message_id)
            if message.message_id % photach_frequency == 0:
                photach_frequency = randint(120, 180)
                print(f"NEW PHOTACH_FREQUENCY: {photach_frequency}")
            return
        if ("обедать" in message.text.lower() or "завтрак" in message.text.lower()
            or "ужин" in message.text.lower() or "хавать" in message.text.lower()
            or "кушать" in message.text.lower()) and check_member != "left" and sent == False:
            nado_li = random.randint(0,3)
            if nado_li == 1:
                bot.send_sticker(cid, 'CAACAgIAAxkBAAEK3uBgzbWPQZxmCEyXMXfViPlCX-0r5wACSwEAApNA9g7iKdrXmmxgcB8E')
                return
        
        print(f"SENT: {sent}")

        
        


bot.polling(none_stop=True, interval=0)