# -*-coding:utf-8-*-
'''
文档主要集成了处理噪音的方法

'''

import os
from os import listdir
from os.path import isfile, join

import cv2
import imutils
import numpy as np
from PIL import Image
from PIL import ImageDraw, ImageFont
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import map_coordinates

from get_random import get_distribution

# import scipy.stats

######File2List
'''
Read file and return the list of characters

params:
path：input file path

returns：
return the list of characters
'''
def File2List(path):
    f = open(path, 'r')
    line = f.readline()
    str = ''
    while line:
        line = f.readline()
        str += line
    f.close()
    str = str.split('\n')
    str = ''.join(str)
    list1 = list(set(str)) #remove the repeated elements
    return list1

# print(File2List('./dictionary1.txt'))

#####plot_image
'''
show the image 
'''
def plot_image(img):
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#Get_Edges
'''
let the character reach the 4 edges of the image
note that the background of the input should be black

params
img: the input image

returns
return the results
'''
def get_edges(img1):
    H,W = img1.shape
    # img1 = cv2.Canny(img,50,150)
    img_H, img_W = img1.shape
    cut_r,cut_l,cut_t,cut_b = [0,0,0,0]
    #cut right
    for i in range(0,img_W,2):
        pixel_sum = np.sum(img1[0:img_H, img_W-i-2:img_W-i])
        if pixel_sum >500:
            cut_r = i
            break
    #cut left
    for i in range(0, img_W, 2):
        pixel_sum = np.sum(img1[0:img_H, i:i+2])
        if pixel_sum > 500:
            cut_l = i
            break
    #cut top
    for i in range(0, img_H, 2):
        pixel_sum = np.sum(img1[i:i + 2,0:img_W])
        if pixel_sum > 500:
            cut_t = i
            break
    #cut bottom
    for i in range(0, img_H, 2):
        pixel_sum = np.sum(img1[ img_H-i-2:img_H-i,0:img_W])
        if pixel_sum > 500:
            cut_b = i
            break

    img_result = img1[cut_t:H-cut_b,cut_l:img_W - cut_r]

    return img_result


####GetRawPic
'''
get and save the raw character image of expected size by drawing

params
list: input the list of characters to be drawn
font_name: the font that expected to have
font_size: the size of the font that expected to have
path: output path

'''
def GetRawPic(list,font_name,font_size,path):
    #get the font
    font = ImageFont.FreeTypeFont(font_name,font_size)
    #drawing
    for char in list:
        img = Image.new('RGB', (2*font_size, 2*font_size), (255, 255, 255))
        draw = ImageDraw.Draw(img)
        draw.text((0,0),char,(0),font=font)
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
        # img = get_edges(img)
        cv2.imwrite(path + '/'+char + '.png',img)
    print ('done!')

def get_one_pic(char, font_name='/Library/Fonts/华文中宋.ttf', font_size=50):
    font = ImageFont.FreeTypeFont(font_name, font_size)
    img = Image.new('RGB', (int(1.5 * font_size), int(1.5 * font_size)), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((0, 0), char, (0), font=font)
    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
    return img


#clean char
def clean_char(img):
    H, W = img.shape
    # img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    #                             cv2.THRESH_BINARY,21, 5) #27 3
    img = 255 - img
    img_sum = img.sum(1)
    start = -1
    end = -1
    for pos, val in enumerate(img_sum):
        if val > 0:
            if start < 0:
                start = pos
        if img_sum[-pos] > 0:
            if end < 0:
                end = H - pos

    return 255 - img[start:end]




#img_pad
'''
pad the image of expected size
 params
 padding the input image to the expected size
 
 :returns
 padding image

'''
def pad_image(img, H, W):
    """ white padding an image to expected size
    Args:
        (np.array) img: image
        (int) H: expected height
        (int) W: expected width
    Returns:
        (np.array) all the blocks stacked in a np array
    """
    # img = clean_char(img)
    img_H, img_W = img.shape[:2]
    padded_img = np.zeros((H, W), dtype=np.uint8)
    if img_H >= img_W:
        img =imutils.resize(img, height=H)
        w_pos = (W - img.shape[1]) // 2
        padded_img[:, w_pos:w_pos+img.shape[1]] = 255 - img

    else:
        print (img.shape)
        img =imutils.resize(img, width=W)
        h_pos = (H - img.shape[0]) // 2
        padded_img[h_pos:h_pos+img.shape[0], :] = 255 - img

    # plot_image(padded_img)
    return padded_img


#resizing
'''
xCompress_Fetch and yCompress_Fetch are used to resize the image in x and y direction respectively
by fetching or compressing the image

params
img: the input image
level: the intensity of resizing 

return 
the output image
'''
#Compress and Fetch for x
def xCompress_Fetch(img,level):
    H,W = img.shape
    img = cv2.resize(img,(H,round(W * level)),interpolation=cv2.INTER_AREA)
    return img

#Compress and Fetch for y
def yCompress_Fetch(img,level):
    H,W = img.shape
    img = cv2.resize(img,(round(H*level),W ),interpolation=cv2.INTER_AREA)
    return img


'''
dilating and eroding are used to dilate or erode the picture as ways to adding noise.
params
img: the input image
level: the size of structure matrix 

:returns
output image

'''
#dilate
def Dilate(img,level):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (round(level), round(level)))
    img = cv2.dilate(255 - img, kernel)
    return img


#erode
def Erode(img,level):
    kernel1 = cv2.getStructuringElement(cv2.MORPH_CROSS, (round(level), round(level)))
    img = cv2.erode(img, kernel1)
    return img

#process_img
def process_img(img,compress1,compress2,dilate,erode):
    # print (compress1)
    img = xCompress_Fetch(img,compress1)
    # plot_image(img)
    # print (compress1)
    img = yCompress_Fetch(img,compress2)
    # plot_image(img)
    img = Dilate(img, dilate)
    # plot_image(img)
    img = Erode(img, erode)
    # plot_image(img)
    return img

#RotatePic
'''
Rotate the picture with expected angles
params
img: input picture 
angle:angles

returns
 the output image

'''
def RotatePic(img, angle):
    H,W = img.shape
    rotated = imutils.rotate(img, angle)
    return rotated


'''
get  distorted image with different intensity
params:
image: the input image
alpha: the intensity of distortion
sigma: the distribution of the pixel. if sigma is small, the picture will be much more dotted
'''
def elastic_transform(image, alpha, sigma, random_state=None):
    """Elastic deformation of images as described in [Simard2003]_.
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
       Convolutional Neural Networks applied to Visual Document Analysis", in
       Proc. of the International Conference on Document Analysis and
       Recognition, 2003.
    """
    assert len(image.shape) == 2

    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape

    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha

    x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
    indices = np.reshape(x + dx, (-1, 1)), np.reshape(y + dy, (-1, 1))

    return map_coordinates(image, indices, order=1,mode='reflect').reshape(shape)

#adding_single noise
'''
Adding noise to get a single picture
elastic distort + blur
params:
img: the input image
sigma: variance
alpha: the cross rate
blur: the blur kernel

:returns
return the processed image
'''

def Adding_single_noise(img,s,a,b):

    img1 = img.copy()
    if(s != '' and a != ''):
        img1 = 255 - elastic_transform(img1, a, s, random_state=None)
    img1 = cv2.GaussianBlur(img1, (b, b), 0)
    return img1


'''
get the skeleton structure for image
 params
  img: the input image
  
 :return
 processed image

'''
def get_skeleton(img):
    size = np.size(img)
    skel = np.zeros(img.shape, np.uint8)
    #
    # ret,img = cv2.threshold(img,127,255,0)
    # img = 255 - cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    #                                      cv2.THRESH_BINARY, 11, 5)
    # img = cv2.resize(img,(55,55),interpolation=cv2.INTER_AREA)
    ret, img = cv2.threshold(img, 127, 255, 0)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    done = False

    while (not done):
        eroded = cv2.erode(img, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(img, temp)
        skel = cv2.bitwise_or(skel, temp)
        img = eroded.copy()

        zeros = size - cv2.countNonZero(img)
        if zeros == size:
            done = True
    return skel


'''
processing the training data
path: input directory
dest: output directory
angles: the rotation range. if angles = 3, the range will be (-3,3)
compress: the compress range
kernel:the structure of dilation and erosion
sigma:
sigma1:useless param

before using this function, you have to set all the params above
'''

def ProcessingForSkeletons(path,dest,angles=1,compress=[1],dilates=[1],erodes=[1],sigma=[1],alpha=[0],blur=[1],skel=False,adapt = True):
    #find directory and list all pictures
    dir = path
    onlyfiles = [f for f in listdir(dir) if isfile(join(dir, f))]
    #deal with each pic
    for file in onlyfiles:
        #read the image in the file
        if '.png' in file:
            directory_name = file[:len(file)-4]
            dir_tmp = dir + '/' + file
            directory_path = dest+ '/' +directory_name   #output address
            if not os.path.exists(directory_path):
                os.makedirs(directory_path)
            img = cv2.imread(dir_tmp,0)

        #processing the image with different combination

            print  (directory_name)   #print the processing word
            for x in compress:
                for y in compress:
                    for dilate in dilates:
                        for erode in erodes:
                            imgtmp = img.copy()
                            imgtmp = process_img(imgtmp, x, y, dilate, erode)
                            # angle loop
                            for angle in range(-angles, angles+1, 1):
                                filename = directory_name +str(x) + str(y)  + str(dilate) + str(erode) + str(angle)
                                filepath = directory_path + '/' + filename + '.png'

                                imgtmp1 = RotatePic(255 - imgtmp, angle)
                                # add noise
                                imgtmp2 = get_edges(imgtmp1)
                                imgtmp2 = pad_image(255 - imgtmp2, 28, 28)
                                #see if doing the adaption
                                if adapt:
                                    imgtmp2 = 255 - cv2.adaptiveThreshold(255 - imgtmp2, 255,
                                                                          cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                                          cv2.THRESH_BINARY, 17, 5)
                                    # see if getting a skeleton
                                    if skel:
                                        imgtmp2 = get_skeleton(imgtmp2)
                                cv2.imwrite(filepath, imgtmp2)
                                #add distortion and blur
                                for a in alpha:
                                            imgtmpx = imgtmp1.copy()
                                            for s in sigma:
                                                for b in blur:
                                                    filenamex = directory_name + str(x) + str(y) + str(dilate) + str(
                                                        erode) + str(angle)+str(a)+str(s)+str(b)
                                                    filepathx = directory_path + '/' + filenamex + '.png'
                                                    imgtmpx = Adding_single_noise(imgtmpx,s,a,b)

                                                    imgtmpx = get_edges(255 - imgtmpx)

                                                    imgtmpx = pad_image(255 - imgtmpx, 28, 28)

                                                    # if using the adaptive method
                                                    if adapt:
                                                        imgtmpx = 255 - cv2.adaptiveThreshold(255 - imgtmpx, 255,
                                                                                              cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                                                              cv2.THRESH_BINARY, 17, 5)

                                                        plot_image(imgtmpx)
                                                        # see if getting a skeleton
                                                        if skel:
                                                            imgtmpx = get_skeleton(imgtmpx)
                                                    cv2.imwrite(filepathx, imgtmpx)


'''
processing the training data
path: input directory
dest: output directory
angles: the rotation range. if angles = 3, the range will be (-3,3)
compress: the compress range
kernel:the structure of dilation and erosion
sigma:
sigma1:useless param

before using this function, you have to set all the params above
'''

def process_data_random(path,dest,numbers,angles=1,compress=[1],dilates=[1],erodes=[1],sigma=[1],alpha=[0],blur=[1],skel=False,adapt = True):
    # find directory and list all pictures

    dir = path
    onlyfiles = [f for f in listdir(dir) if isfile(join(dir, f))]
    #deal with each pic
    for file in onlyfiles:
        #read the image in the file
        if '.png' in file:
            directory_name = file[:len(file)-4]
            dir_tmp = dir + '/' + file
            directory_path = dest+ '/' +directory_name   #output address
            if not os.path.exists(directory_path):
                os.makedirs(directory_path)
            img = cv2.imread(dir_tmp,0)


'''
produce random number according to standard normalization with mu 0 and sigma 1.0

return 
random number
'''

def stan_gaussian_random_num():
    num = np.random.normal(0,1)
    if(num>1.5):
        num = 1.5
    elif(num<-1.5):
        num = -1.5
    else:
        num = num
    return num


'''
According to givern condition, fit the random number into corresponding interval
params:
lists: the list containing labels where the random number would fit in
returns:
randomx: the output number
'''
def ran_distribution(lists):
    randomx = stan_gaussian_random_num()

    angle_cate = len(lists)
    if angle_cate == 0:
        return 1
    elif angle_cate == 1:
        return lists[0]
    else:
        interval = 3 / angle_cate
        labels = [-1.5 + interval * (i + 1) for i in range(0, angle_cate - 1)]
        if (randomx < labels[0]):
            randomx = lists[0]
        if (randomx >= labels[len(labels) - 1]):
            randomx = lists[len(lists) - 1]
        for i in range(len(labels) - 1):

            if (randomx >= labels[i] and randomx < labels[i + 1]):
                randomx = lists[i + 1]
        return randomx


'''
Processing the data through several process including compressing, dilating, distorting, bluring. 
params:
imgtmp: the input img
char: the name of the img
path: saving path
angles: list of angles
compress: list of compress
dilates: list of dilates
sigmas: list of sigmas
distorts: list of alphas
blurs: list of blurs
'''
def process_random_single_word(i,imgtmp,char,path,angles,compress,dilates,sigmas,distorts,blurs,skel):
    directory_path = path + char  # output address
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
    angle = ran_distribution(angles)
    compressx = ran_distribution(compress)
    compressy = ran_distribution(compress)
    dilate = ran_distribution(dilates)
    sigma = ran_distribution(sigmas)
    distort = ran_distribution(distorts)
    blur = ran_distribution(blurs)
    imgtmp = process_img(imgtmp, compressx, compressy, dilate, 1)
    imgtmp1 = RotatePic(imgtmp, angle)
    imgtmpx = Adding_single_noise(imgtmp1, sigma, distort, blur)
    imgtmpx = get_edges(255 - imgtmpx)
    H, W = imgtmpx.shape
    if (H < 10 or W < 10):
        print('bad img!', 'blur', blur, 'distort', distort, 'sigma', sigma)
        plot_image(imgtmpx)
    imgtmpx = pad_image(255 - imgtmpx, 64, 64)
    # plot_image(imgtmpx)
    imgtmpx = cv2.adaptiveThreshold(255 - imgtmpx, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY, 21, 5)  # 27 3
    if skel :
        imgtmpx = get_skeleton(255 - imgtmpx)
    else:
        imgtmpx = 255 - imgtmpx
    cv2.imwrite(directory_path + '/' + char + str(blur) + str(distort) + str(sigma) + str(compressy) + str(compressx) + str(
        dilate) + str(angle) + str(i) + '.png', imgtmpx)



'''
processing several same words under different conditions
params:
char: the name of the img
path: saving path
numbers: the number of the words
angles: list of angles
compress: list of compress
dilates: list of dilates
sigmas: list of sigmas
distorts: list of alphas
blurs: list of blurs

'''

def single_gaussian_processing(char, path, numbers, anglex=1,compress=[1],dilates=[1],erodes=[1],sigmas=[1],distorts=[0],blurs=[1],skel=False,adapt = True):
    angles = [angle for angle in range(-anglex,anglex+1)]
    img = get_one_pic(char)
    imgtmp = img.copy()
    i = 0
    while(i < numbers):
        process_random_single_word(i, imgtmp, char, path, angles, compress, dilates, sigmas, distorts, blurs,skel)
        i += 1


s = single_gaussian_processing('1', './', 100, skel=False)

