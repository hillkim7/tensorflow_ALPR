"""
image loader utility functions
"""
import tensorflow as tf
import matplotlib.pyplot as plt
import gzip
import numpy as np
import sys
import cv2
from os import listdir
import PIL
from PIL import Image
import random
import numpy as np

def convertToOneHot(vector, num_classes=None):
    """
    Converts an input 1-D vector of integers into an output
    2-D array of one-hot vectors, where an i'th input value
    of j will set a '1' in the i'th row, j'th column of the
    output array.

    Example:
        v = np.array((1, 0, 4))
        one_hot_v = convertToOneHot(v)
        print one_hot_v

        [[0 1 0 0 0]
         [1 0 0 0 0]
         [0 0 0 0 1]]
    """

    assert isinstance(vector, np.ndarray)
    assert len(vector) > 0

    if num_classes is None:
        num_classes = np.max(vector)+1
    else:
        assert num_classes > 0
        assert num_classes >= np.max(vector)

    result = np.zeros(shape=(len(vector), num_classes))
    result[np.arange(len(vector)), vector] = 1
    return result.astype(int)

# # 데이터를 바이트단위로 불러오는 함수
def batch1(data, index, batch_size,y_size,x_size):
    # print('data.shape: ', data.shape)
    # print('type(data) :', type(data))    #538,y_size*x_size
    offset =  batch_size * index
    cdata = data[offset:offset + batch_size] 
    cdata = cdata.reshape(-1,y_size*x_size)
    # print('cdata.shape:', cdata.shape) #,batch_size,y_size*x_size
    # print('type(cdata) :', type(cdata))       
    return cdata

def batch2(labels, index, batch_size):
    # print('labels.shape: ', labels.shape) #538
    # print('type(labels) :', type(labels))    
    offset =  batch_size * index
    clabels = labels[offset:offset + batch_size] 
    clabels = convertToOneHot(clabels, num_classes=10)
    # print('clabels.shape:', clabels.shape)  #batch_size, 10
    # print('type(clabels) :', type(clabels))       
    return clabels

def read_data_from_img(path,x_size,y_size):
    trainingFileList = listdir(path)           #load the training set
    m = len(trainingFileList)
    data_data = np.zeros((m,y_size*x_size))  # 28*25 = 875  y*x
    data_labels = np.zeros((m))
    # print(data_data.shape)  #n,875  y*x
    # print(data_labels.shape)  #n
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]     #take off .jpg or png
        classNumStr = int(fileStr.split('_')[0])
        data_labels[i] = classNumStr
        img = cv2.imread(path+'/%s' % fileNameStr)
        # img = cv2.imread(path+'\\%s' % fileNameStr)
        # print(img.shape)
        img = cv2.resize(img, (x_size, y_size)) #x, y = 875
        # print(img.shape) #x, y = 875
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # print(fileNameStr)
        # cv2.imshow(str(m),gray)
        # cv2.waitKey(0)
        # print(gray.shape)
        data_data[i,:] = gray.reshape(y_size*x_size,)
    data_data = data_data.reshape(-1,y_size,x_size)
    data_labels = data_labels.astype(np.int32)    
    # print(data_data.shape)  #n,y_size,x_size
    # print(data_labels.shape)  #n
    return data_data, data_labels

def read_data_from_img_one(fname,x_size,y_size):
    data_data = np.zeros((y_size*x_size))  # 35*25 = 875  y*x
    img = cv2.imread(fname)
    # print(img.shape)
    img = cv2.resize(img, (x_size, y_size)) #x, y = 875
    # print(img.shape) #x, y = 875
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # print(fileNameStr)
    # cv2.imshow(str(m),gray)
    # cv2.waitKey(0)
    # print(gray.shape)
    data_data[:] = gray.reshape(y_size*x_size,)
    data_data = data_data.reshape(-1,y_size*x_size)
    return data_data

def make_background(filename,x_size,y_size):
    new_img = Image.new("RGB",(x_size,y_size),"white")
    im = Image.open(filename)
    #------------------------------------------------#
    if im.size[0] / im.size[1] > 2.5 : #긴 번호판
        basewidth = x_size
        wpercent = (basewidth / float(im.size[0]))
        hsize = int((float(im.size[1]) * float(wpercent)))
        im = im.resize((basewidth, hsize), PIL.Image.ANTIALIAS)
    else:
        baseheight = y_size
        hpercent = (baseheight / float(im.size[1]))
        wsize = int((float(im.size[0]) * float(hpercent)))
        im = im.resize((wsize, baseheight), PIL.Image.ANTIALIAS)
    #------------------------------------------------#
    im.thumbnail((x_size,y_size),Image.ANTIALIAS)
    load_img = im.load()
    load_newimg = new_img.load()
    i_offset = (x_size-im.size[0])/2
    j_offset = (y_size-im.size[1])/2
    for i in range(0, im.size[0]) :
        for j in range(0, im.size[1]) :
            load_newimg[i+i_offset, j+j_offset] = load_img[i,j]

    new_img = np.array(new_img)
    new_img = new_img[:, :, ::-1].copy()
    # new_img.save(outfile, "JPEG")
    # new_img.show()
    return new_img

def read_data_from_img_exp(path,x_size,y_size):

    trainingFileList = listdir(path)           #load the training set
    m = len(trainingFileList)
    data_data = np.zeros((m,y_size*x_size))  # 35*25 = 875  y*x
    data_labels = np.zeros((m))
    # print(data_data.shape)  #n,875  y*x
    # print(data_labels.shape)  #n
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]     #take off .jpg or png
        classNumStr = int(fileStr.split('_')[0])
        data_labels[i] = classNumStr
        # img = cv2.imread(path+'\\%s' % fileNameStr)
        filename = path+'/'+fileNameStr
        # print(i, filename)
        img = make_background(filename,x_size,y_size)
        # print(img.shape)
        # cv2.imshow(str(m),img)
        # cv2.waitKey(0)
        # sys.exit()
        # img = cv2.resize(img, (x_size, y_size)) #x, y = 875
        # print(img.shape) #x, y = 875
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # print(fileNameStr)
        # cv2.imshow(str(m),gray)
        # cv2.waitKey(0)
        # print(gray.shape)
        data_data[i,:] = gray.reshape(y_size*x_size,)
    data_data = data_data.reshape(-1,y_size,x_size)
    data_labels = data_labels.astype(np.int32)
    # print(data_data.shape)  #n,y_size,x_size
    # print(data_labels.shape)  #n
    return data_data, data_labels
"""
data_data, data_labels = read_data_from_img('c:\\turbo\python\split\\number')
# m = len(data_data)
# for r in range(m):
#     plt.imshow(data_data[r:r + 1].reshape(35, 25), cmap='Greys', interpolation='nearest')
#     plt.show()

data_labels = convertToOneHot(data_labels, num_classes=10)
print(data_labels.shape)
data_data = data_data.reshape(-1,875)
print(data_data.shape)

# Get one and predict
# r = random.randint(0, len(data_data) - 1)
# plt.imshow(data_data[r:r + 1].
#         reshape(35, 25), cmap='Greys', interpolation='nearest')
# plt.show()    
# r=1
# cv2.imshow('a',data_data[r:r + 1].reshape(35, 25))
# cv2.waitKey(0)
"""