# -*- coding: utf-8 -*-
"""
Created on Tue Aug 31 14:37:06 2021

@author: WEBER_LIN
v. 0.1
20230410  nozzle consistence
20230428  add folder process
          adjust L/R and A/B radio
20230525  fix block compution , add quickly block rmse
20230615 1. fix redundant print text.  2. Fix the compution range for folder, 3. add pass or fail when check threshold 
20230616  Add setting threshold on GUI
"""

from skimage import feature
import cv2
#import mediapipe
import numpy as np
import math
from imutils import face_utils
import PySimpleGUI as sg
import platform as pf
import os
import sys

import tkinter
from matplotlib import pyplot as plt

import random

import time
import csv

from collections import OrderedDict

from PIL import Image
from sklearn.metrics import mean_squared_error, r2_score
from matplotlib.pyplot import MultipleLocator
from skimage.metrics import structural_similarity as compare_ssim

#### define threhold
# RMSE_thr = 2.59
# Mean_error_thr = 1.1
# Std_error_thr = 0.2
# key program
RGB_Liquid_color_table = OrderedDict([
    ("F101", (249, 225, 183)),
    ("F102", (255, 218, 182)),
    ("F103", (232, 198, 169)),
    ("F104", (242, 198, 162)),
    ("F105", (236, 186, 147)),
    ("F106", (224, 179, 132)),
    ("F107", (223, 168, 128)),
    ("F108", (195, 164, 146)),
    ("F109", (142, 110, 92)),
    ("F110", (119, 92, 80)),
    ("B301", (255, 184, 202)),
    ("B302", (230, 181, 201)),
    ("B303", (244, 195, 203)),
    ("B304", (255, 183, 174)),
    ("B305", (235, 181, 165)),
    ("B306", (247, 185, 132)),
    ("H401", (253, 208, 134)),
    ("H402", (245, 229, 234)),
    ("Y501", (255, 235, 104)),
    ("Y502", (213, 162, 134)),
    ("Y503", (164, 82, 72)),
    ("Y504", (105, 97, 95)),
    ("Y505", (236, 186, 168)),
    ("Y506", (243, 162, 147)),
    ("Y507", (222, 163, 156)),
    ("Y508", (175, 105, 91)),
    ("Y509", (196, 132, 144)),
    ("Y510", (92, 136, 218)),
    ("Y511", (112, 118, 149)),
    ("Y512", (62, 142, 139)),
    ("Y513", (90, 104, 93)),
    ("Y514", (140, 120, 133)),
    ("Y515", (201, 220, 172)),
    ("Y516", (75, 136, 172)),
    ("Y517", (193, 167, 226)),
    ("Y518", (212, 213, 212)),
])


def recursionOTSU(img, n):

    array = np.ravel(img)  # 將 二維數組img 降維爲 一維數組array

    for i in range(n):

        # 對 array 同時進行 閾值化爲0 和 OTSU算法
        retval, array = cv2.threshold(
            array, 0, 0, cv2.THRESH_TOZERO + cv2.THRESH_OTSU)

        array = array[array > 0]  # 將 array 中等於0的元素去除掉

        retval, img = cv2.threshold(
            img, retval, 0, cv2.THRESH_TOZERO)  # 用新的閾值對圖像進行 閾值化爲0 的操作

    # 對遞歸後的圖像進行二進制閾值化操作，讓圖像更清晰
    retval, img = cv2.threshold(img, retval, 255, cv2.THRESH_BINARY)

    return img


def cal_p2p_rmse(crop_image1, crop_image2, crop_gray1, crop_gray2,mode):
    # mean1R, stddv1R = cv2.meanStdDev(crop_image1[:, :, 2])
    # mean1G, stddv1G = cv2.meanStdDev(crop_image1[:, :, 1])
    # mean1B, stddv1B = cv2.meanStdDev(crop_image1[:, :, 0])
    # mean2R, stddv2R = cv2.meanStdDev(crop_image2[:, :, 2])
    # mean2G, stddv2G = cv2.meanStdDev(crop_image2[:, :, 1])
    # mean2B, stddv2B = cv2.meanStdDev(crop_image2[:, :, 0])
    mean1,stddv1 =  cv2.meanStdDev(crop_image1)
    mean2,stddv2 =  cv2.meanStdDev(crop_image2)
    gray_mean1, gray_std1 = cv2.meanStdDev(crop_gray1)
    gray_mean2, gray_std2 = cv2.meanStdDev(crop_gray2)
    diff = crop_gray1- crop_gray2#cv2.subtract(crop_gray1, crop_gray2)
    square = np.square(diff)
    MSE = square.mean()
    RMSE = np.sqrt(MSE)
    min_d = np.min(diff)
    max_d = np.max(diff)
    if mode == 1 :
        print("--1R:", np.round(mean1[2][0],2), np.round(stddv1[2][0],2))
        print("--1G:", np.round(mean1[1][0],2), np.round(stddv1[1][0],2))
        print("--1B:", np.round(mean1[0][0],2), np.round(stddv1[0][0],2))
        print("--2R:", np.round(mean2[2][0],2), np.round(stddv2[2][0],2))
        print("--2G:", np.round(mean2[1][0],2), np.round(stddv2[1][0],2))
        print("--2B:", np.round(mean2[0][0],2), np.round(stddv2[0][0],2))
        print("--1Gray:", np.round(gray_mean1[0][0],2), np.round(gray_std1[0][0],2))
        print("--2Gray:", np.round(gray_mean2[0][0],2), np.round(gray_std2[0][0],2))
        print("--p2p RMSE:", np.round(RMSE,2))
         
        print("diff min,max:",min_d,max_d)
    return RMSE,gray_mean1[0][0],gray_std1[0][0],gray_mean2[0][0],gray_std2[0][0]

def cal_5blocks_rmse(image1, image2, gray1, gray2,image3, p1x, p1y, p2x, p2y):
    output = image3.copy()
    mask3 = np.zeros((image1.shape[0], image2.shape[1], 1), np.uint8)
    mask3[:] = 0
    bk1_p1x = bk1_p2x = bk1_p2x = bk1_p2y = 0
    bk2_p1x = bk2_p2x = bk2_p2x = bk2_p2y = 0
    bk3_p1x = bk3_p2x = bk3_p2x = bk3_p2y = 0
    bk4_p1x = bk4_p2x = bk4_p2x = bk4_p2y = 0
    bk5_p1x = bk5_p2x = bk5_p2x = bk5_p2y = 0
    block_width = 500
    block_height = 500
    bk1_p1x = p1x
    bk1_p2x = p1x + block_width
    bk1_p1y = p1y
    bk1_p2y = p1y + block_height
    bk2_p1x = p2x - block_width
    bk2_p2x = p2x
    bk2_p1y = p1y
    bk2_p2y = p1y + block_height
    bk4_p1x = p1x
    bk4_p2x = p1x + block_width
    bk4_p1y = p2y - block_height
    bk4_p2y = p2y
    bk5_p1x = p2x - block_width
    bk5_p2x = p2x
    bk5_p1y = p2y - block_height
    bk5_p2y = p2y
    bk3_p1x = (p2x + p1x)//2 - block_width//2
    bk3_p2x = (p2x + p1x)//2 + block_width//2
    bk3_p1y = (p2y + p1y)//2 - block_height//2
    bk3_p2y = (p2y + p1y)//2 + block_height//2
    block_avg1 = []
    block_avg2 = []
    block_diff = []
    mask3[:] = 0
    cv2.rectangle(mask3, (bk1_p1x, bk1_p1y),
                  (bk1_p2x,  bk1_p2y), (255, 255, 255),  -1)
    block1_avg1, block1_std1 = cv2.meanStdDev(gray1, mask=mask3)
    block1_avg2, block1_std2 = cv2.meanStdDev(gray2, mask=mask3)
    block_avg1.append(block1_avg1[0][0])
    block_avg2.append(block1_avg2[0][0])
    block_diff.append(block1_avg1[0][0]-block1_avg2[0][0])
    mask3[:] = 0
    cv2.rectangle(mask3, (bk2_p1x, bk2_p1y),
                  (bk2_p2x,  bk2_p2y), (255, 255, 255),  -1)
    block2_avg1, block2_std1 = cv2.meanStdDev(gray1, mask=mask3)
    block2_avg2, block2_std2 = cv2.meanStdDev(gray2, mask=mask3)
    block_avg1.append(block2_avg1[0][0])
    block_avg2.append(block2_avg2[0][0])
    block_diff.append(block2_avg1[0][0]-block2_avg2[0][0])
    mask3[:] = 0
    cv2.rectangle(mask3, (bk3_p1x, bk3_p1y),
                  (bk3_p2x,  bk3_p2y), (255, 255, 255),  -1)
    block3_avg1, block3_std1 = cv2.meanStdDev(gray1, mask=mask3)
    block3_avg2, block3_std2 = cv2.meanStdDev(gray2, mask=mask3)
    block_avg1.append(block3_avg1[0][0])
    block_avg2.append(block3_avg2[0][0])
    block_diff.append(block3_avg1[0][0]-block3_avg2[0][0])
    mask3[:] = 0
    cv2.rectangle(mask3, (bk4_p1x, bk4_p1y),
                  (bk4_p2x,  bk4_p2y), (255, 255, 255),  -1)
    block4_avg1, block4_std1 = cv2.meanStdDev(gray1, mask=mask3)
    block4_avg2, block4_std2 = cv2.meanStdDev(gray2, mask=mask3)
    block_avg1.append(block4_avg1[0][0])
    block_avg2.append(block4_avg2[0][0])
    block_diff.append(block4_avg1[0][0]-block4_avg2[0][0])
    mask3[:] = 0
    cv2.rectangle(mask3, (bk5_p1x, bk5_p1y),
                  (bk5_p2x,  bk5_p2y), (255, 255, 255),  -1)
    block5_avg1, block5_std1 = cv2.meanStdDev(gray1, mask=mask3)
    block5_avg2, block5_std2 = cv2.meanStdDev(gray2, mask=mask3)
    block_avg1.append(block5_avg1[0][0])
    block_avg2.append(block5_avg2[0][0])
    block_diff.append(block5_avg1[0][0]-block5_avg2[0][0])
    RMSE_gray_by_5blocks = np.sqrt(mean_squared_error(block_avg1, block_avg2))
    cv2.rectangle(output, (bk1_p1x, bk1_p1y),
                  (bk1_p2x,  bk1_p2y), (255, 0, 0),  10)
    cv2.rectangle(output, (bk2_p1x, bk2_p1y),
                  (bk2_p2x,  bk2_p2y), (255, 0, 0),  10)
    cv2.rectangle(output, (bk3_p1x, bk3_p1y),
                  (bk3_p2x,  bk3_p2y), (255, 0, 0),  10)
    cv2.rectangle(output, (bk4_p1x, bk4_p1y),
                  (bk4_p2x,  bk4_p2y), (255, 0, 0),  10)
    cv2.rectangle(output, (bk5_p1x, bk5_p1y),
                  (bk5_p2x,  bk5_p2y), (255, 0, 0),  10)
    if showimage == 1:
        cv2.imshow('all',  cv2.resize(output, (600, 800)))
        cv2.waitKey(0)
    print("block1_avg1,block1_std1:", block1_avg1[0][0], block1_std1[0][0])
    print("block2_avg1,block2_std1:", block2_avg1[0][0], block2_std1[0][0])
    print("block3_avg1,block3_std1:", block3_avg1[0][0], block3_std1[0][0])
    print("block4_avg1,block4_std1:", block4_avg1[0][0], block4_std1[0][0])
    print("block5_avg1,block5_std1:", block5_avg1[0][0], block5_std1[0][0])
    print("block1_avg2,block1_std2:", block1_avg2[0][0], block1_std2[0][0])
    print("block2_avg2,block2_std2:", block2_avg2[0][0], block2_std2[0][0])
    print("block3_avg2,block3_std2:", block3_avg2[0][0], block3_std2[0][0])
    print("block4_avg2,block4_std2:", block4_avg2[0][0], block4_std2[0][0])
    print("block5_avg2,block5_std2:", block5_avg2[0][0], block5_std2[0][0])

    print("-5 blocks RMSE:", RMSE_gray_by_5blocks)
    if showplt == 1 :
        fig, ax = plt.subplots()
        maker_s = ['o', '*', '^', '8', 's', 'v', '^', '>', '<', '1', '2', '3', '4']
        lable = "block =" + str(500) + "x"+str(500)
        ax.plot(range(0,  len(block_diff), 1), block_diff,
                marker=maker_s[1], markersize=4, color='red', label=lable)
        ax.legend(loc="upper left", fontsize=6)
        ax.set_title("Block gray image differentail chart")
        ax.xaxis.set_label_text("block no.")
        ax.yaxis.set_label_text("error")
        plt.show()


def cal_row_rmse(image1, image2, gray1, gray2,mode):  # input crop image
    #test = np.zeros((gray1.shape[0], gray1.shape[1], 1), np.uint8)
    #test1 = np.zeros((gray1.shape[0], gray1.shape[1], 1), np.uint8)
    # test1[:] = 220 
    # test[:] = 220 
    # for i in range(0,gray1.shape[1]//2 ):
    #     for j in range(0,gray1.shape[0] ):
    #         #test[j][i] = 221
    #         if j % 2 == 0 :
    #             test[j][i] = 221
    #         else:
    #             test[j][i] = 220
    #cv2.imshow('test',  cv2.resize(test, (800, 1200)))
    #cv2.imshow('test1',  cv2.resize(test1, (800, 1200)))
    #cv2.waitKey(0)            
    # gray1 = test
    # gray2 = test1
    img1_mean = np.mean(gray1, axis=1)
    img2_mean = np.mean(gray2, axis=1)
    diff = (img1_mean- img2_mean)#cv2.subtract(img1_mean, img2_mean)
    #square = np.square(diff)
    #MSE = square.mean()
    #RMSE = np.sqrt(MSE)
    RMSE  = np.sqrt(mean_squared_error(img1_mean.flatten(), img2_mean.flatten()))
    if mode == 1:
        print("--row RMSE:", RMSE)
        if showplt == 1:
            fig, ax = plt.subplots()
            lable = "rows error"
            maker_s = ['o', '*', '^', '8', 's', 'v', '^', '>', '<', '1', '2', '3', '4']
            ax.plot(range(0,  len(diff), 1), diff,
                    marker=maker_s[1], markersize=4, color='red', label=lable)
            ax.legend(loc="upper right", fontsize=6)
            ax.set_title("row gray image differentail chart")
            ax.xaxis.set_label_text("row no.")
            ax.yaxis.set_label_text("error")
            plt.show()
    return RMSE

def cal_col_rmse(image1, image2, gray1, gray2,mode):  # input crop image

    img1_mean = np.mean(gray1, axis=0)
    img2_mean = np.mean(gray2, axis=0)
    diff = img1_mean- img2_mean#cv2.subtract(img1_mean, img2_mean)
    square = np.square(diff)
    MSE = square.mean()
    RMSE = np.sqrt(MSE)


    if mode == 1 :
        print("--column RMSE:", RMSE)
        if showplt == 1:
            fig, ax = plt.subplots()
            lable = "columns error"
            maker_s = ['o', '*', '^', '8', 's', 'v', '^', '>', '<', '1', '2', '3', '4']
            ax.plot(range(0,  len(diff), 1), diff,
                    marker=maker_s[1], markersize=4, color='red', label=lable)
            ax.legend(loc="upper right", fontsize=6)
            ax.set_title("column gray image differentail chart")
            ax.xaxis.set_label_text("column no.")
            ax.yaxis.set_label_text("error")
            plt.show()
    return RMSE 
def pearson_correlation(image1, image2):
    image1_flat = image1.flatten()
    image2_flat = image2.flatten()
    correlation = np.corrcoef(image1_flat, image2_flat)[0, 1]
    return correlation
def fft_comare(image1, image2):
    fft_image1 = np.fft.fftshift(np.fft.fft2(image1))
    fft_image2 = np.fft.fftshift(np.fft.fft2(image2))
    # 计算频谱图的幅度谱
    magnitude_spectrum1 = 20 * np.log(np.abs(fft_image1))
    magnitude_spectrum2 = 20 * np.log(np.abs(fft_image2))
    # 显示频谱图
    plt.subplot(121), plt.imshow(magnitude_spectrum1, cmap='gray')
    plt.title('Magnitude Spectrum 1'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(magnitude_spectrum2, cmap='gray')
    plt.title('Magnitude Spectrum 2'), plt.xticks([]), plt.yticks([])
    plt.show()
    # 计算频谱图的差异
    difference = 100*np.abs(magnitude_spectrum1 - magnitude_spectrum2)
    
    # 显示差异图
    plt.imshow(difference, cmap='gray')
    plt.title('Difference Spectrum'), plt.xticks([]), plt.yticks([])
    plt.show()
def hist_compare(image1, image2):
    # 计算直方图
    histogram1 = cv2.calcHist([image1], [0], None, [256], [0, 256])
    histogram2 = cv2.calcHist([image2], [0], None, [256], [0, 256])
    
    # 归一化直方图
    histogram1 /= histogram1.sum()
    histogram2 /= histogram2.sum()
    
    # 比较直方图
    similarity = cv2.compareHist(histogram1, histogram2, cv2.HISTCMP_CORREL)
    print("Similarity:", similarity)
def split_image_into_blocks(image, block_size):
    height, width = image.shape[:2]
    block_height, block_width = block_size
    num_blocks_h = height // block_height
    num_blocks_w = width // block_width

    blocks = np.empty((num_blocks_h, num_blocks_w, block_height, block_width))

    for i in range(num_blocks_h):
        for j in range(num_blocks_w):
            block = image[i * block_height:(i + 1) * block_height, j * block_width:(j + 1) * block_width]
            blocks[i, j] = block

    return blocks
def calculate_rmse(blocks1, blocks2):
    num_blocks_h, num_blocks_w = blocks1.shape[:2]
    #flattened_blocks1 = blocks1.reshape(num_blocks_h * num_blocks_w, -1)
    #flattened_blocks2 = blocks2.reshape(num_blocks_h * num_blocks_w, -1)
    #diff = flattened_blocks1 - flattened_blocks2
    block1_means = np.mean(blocks1, axis=(2, 3))
    block2_means = np.mean(blocks2, axis=(2, 3))
    block1_std = np.std(block1_means)
    block2_std = np.std(block2_means)
    # diff = blocks1 - blocks2
    # mse = np.mean(diff ** 2)#, axis=(2, 3))
    #mse = np.mean(diff ** 2)
    # rmse = np.sqrt(mse)
    #print(block1_means[0])
    #print(len(block1_means[0]),len(block1_means))
    rmse_gray_rec = np.sqrt(mean_squared_error(block1_means.flatten(), block2_means.flatten()))
    
    #print(rmse_gray_rec)
    return rmse_gray_rec,block1_std,block2_std
def block_test(gray1): 
    height, width  = gray1.shape
    k = 0 
    for i in range(0,height ,29):
        if i + 29 > height : break
        for j in range(0,width ,29):
            if j+29 > width : break
            k = k+1
            for m in range(i,i+29):
                for n in range(j,j+29):
                    gray1[m][n] = min(k,255)    
def cal_byblocks_rmse(image1, image2, gray1, gray2, image3,rx, ry):  # input crop image
    height, width, _ = image1.shape
    output = image2.copy()
    mask3 = np.zeros((image1.shape[0], image2.shape[1], 1), np.uint8)    
    gray_avg_rec_1 = []
    gray_avg_rec_2 = []
    diff_rec = []
    mask3[:] = 0
    bignumber =0
    smallnumber = 0
    for i in range(0, height, ry ):
        if i+ry  > height:
            break
        for j in range(0, width, rx ):
            if j+rx > width:
                break
            mean_b = 0
            mean_a = 0
            ic = 0
            # cv2.rectangle(output, (j, i), (min(width, j+rx-1 ),
            #              min(height, i+ry-1 )), (10, 10, 10),  5) 
            for m in range(i,i+29):
                for n in range(j,j+29):
                    mean_b += gray1[m][n] 
                    mean_a += gray2[m][n] 
                    ic=ic+1
            gray_avg_rec_1.append(mean_b/ic)
            gray_avg_rec_2.append(mean_a/ic)
            # cv2.rectangle(mask3, (j, i), (min(width, j+rx-1 ),
            #               min(height, i+ry-1 )), (255, 255, 255),  1)  

            #grayrec_mean1, grayrec_std1 = cv2.meanStdDev(gray1, mask=mask3)
            #grayrec_mean2, grayrec_std2 = cv2.meanStdDev(gray2, mask=mask3)
            #gray_avg_rec_1.append(grayrec_mean1[0][0])
            #gray_avg_rec_2.append(grayrec_mean2[0][0])
            b2aDiff = abs(mean_b/ic-mean_a/ic)
            if (b2aDiff> 2): bignumber +=1
            else :  smallnumber +=1
            diff_rec.append(b2aDiff)
            ### show the difference area, it can disable
            if (abs(b2aDiff) >= 7):
                cv2.rectangle(output, (j, i), (min(width, j+rx-1),
                              min(height, i+ry-1)), (0, 0, 255),  5)
            elif (abs(b2aDiff) >= 6):
                cv2.rectangle(output, (j, i), (min(width, j+rx-1),
                              min(height, i+ry-1)), (255, 0, 0),  5)
            elif (abs(b2aDiff) >= 5):
                cv2.rectangle(output, (j, i), (min(width, j+rx-1),
                              min(height, i+ry-1)), (0, 255, 0),  5)
            elif (abs(b2aDiff) >= 4):
                cv2.rectangle(output, (j, i), (min(width, j+rx-1),
                              min(height, i+ry-1)), (0, 255, 255), 5)
            elif (abs(b2aDiff) >= 3):
                cv2.rectangle(output, (j, i), (min(width, j+rx-1),
                              min(height, i+ry-1)), (255, 255, 0), 5)
 
            mask3[:] = 0
    rmse_gray_rec = np.sqrt(mean_squared_error(
        gray_avg_rec_1, gray_avg_rec_2))  # blocks
    gray1_std = np.std(gray_avg_rec_1)
    gray2_std = np.std(gray_avg_rec_2)
    print("bignm:",bignumber,",smallnm:",smallnumber)
    print("-RMSE:"+str(rx)+"x"+str(ry), rmse_gray_rec)
    
    fig, ax = plt.subplots()
    lable = "block =" + str(rx) + "x"+str(ry)
    maker_s = ['o', '*', '^', '8', 's', 'v', '^', '>', '<', '1', '2', '3', '4']
    ax.plot(range(0,  len(diff_rec), 1), diff_rec,
            marker=maker_s[1], markersize=4, color='red', label=lable)
    ax.legend(loc="upper right", fontsize=6)
    ax.set_title("Block gray image differentail chart")
    ax.xaxis.set_label_text("block no.")
    ax.yaxis.set_label_text("error")
    # Get the plot figure
    figu = plt.gcf()
    # Update the plot figure
    figu.canvas.draw()

    # Convert the updated plot figure to an image array
    image = np.array(figu.canvas.renderer.buffer_rgba())
    
    # Convert the image to PNG format
    img_encoded = cv2.imencode(".png", cv2.resize(image, (300, 200)))[1].tobytes()
    window['-IMAGE_H-'].update(data=img_encoded)    
    if showplt == 1:    
        plt.show()

    # Update the image in the GUI window
    if width > 2100 :
        # Convert the image to PNG format
        img_encoded = cv2.imencode(".png", cv2.resize(output, (300, 400)))[1].tobytes()
        window['-IMAGE_G-'].update(data=img_encoded)    
    
    if showimage == 1:
        cv2.imshow('blocks'+str(rx)+"x"+str(ry),  cv2.resize(output, (600, 800)))
        cv2.waitKey(0)
    return rmse_gray_rec,  gray1_std,  gray2_std
def cal_foundation_rmse_by_folder(image_fname):
    filenamelist = []
    imagelist=[]
    newfilenamelist = []
    listgrainsfeature = []
    bgr_imagelist = []
    for file_name in os.listdir( image_folder):
        
        if not(file_name.split('.')[-1] == 'jpg' or file_name.split('.')[-1] == 'bmp' or file_name.split('.')[-1] == 'JPG' or file_name.split('.')[-1] == 'BMP'):
            continue
        fname=image_folder + "/" + file_name
       
        filenamelist.append(fname) 
        
    if len(filenamelist)< 1 : return
    # p1x = 1000#1300
    # p1y = 1200  # 1279
    # p2x = 4100#3900
    # p2y = 4500#3700
    p1x = 1260 #1000 #1300
    p1y = 1400   # 1279
    p2x = 3600 #4100 #3900
    p2y = 4000 #4500 #3700  
    block_size = (29 , 29 )  # 替换為自定義的塊大小
    Get_data = []
    data_fields = []
 
    for i in range(0, len(filenamelist)* (len(filenamelist)-1) ):
        Get_data.append([])
    loc_img = ['TL','TR','DL','DR']
    rec_no = 0
    for i in range(0,len(filenamelist)): 
        ori_img1 = cv2.imread(filenamelist[i])
        height1, width1, _ = ori_img1.shape
        image1 =  ori_img1[1500:height1-2000, 500:width1-500]       
        gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
        crop_1 = image1[p1y:p2y, p1x:p2x]
        crop_1_gray = cv2.cvtColor(crop_1, cv2.COLOR_BGR2GRAY)
        file1_name = os.path.basename(filenamelist[i]).split('.')[0]
        # output=image1.copy()
        # cv2.rectangle(output, (p1x,p1y), (  p2x ,   p2y ), (255, 0, 0),  10)
        # cv2.imshow('in',  cv2.resize(output, (600, 800)))
        # cv2.waitKey(0)
        for j in range(i+1,len(filenamelist))  :
            ori_img2 = cv2.imread(filenamelist[j])
            height2, width2, _ = ori_img2.shape
            image2 =  ori_img2[1500:height2-2000, 500:width2-500]             
             
            gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
            crop_2 = image2[p1y:p2y, p1x:p2x] 
            crop_2_gray = cv2.cvtColor(crop_2, cv2.COLOR_BGR2GRAY)
            RMSE_P = RMSE_R = RMSE_C = RMSE_B = 0
            gray1_mean  =  np.mean(crop_1_gray)
            gray2_mean  =  np.mean(crop_2_gray)
            gray1_std  =  np.std(crop_1_gray)
            gray2_std  =  np.std(crop_2_gray)
          
            if rmsebypixel == 1:
                RMSE_P,gray1_mean,gray1_std,gray2_mean,gray2_std = cal_p2p_rmse(crop_1,crop_2,crop_1_gray ,crop_2_gray,2 )
            if rmsebyrow == 1:
                RMSE_R = cal_row_rmse(crop_1,crop_2,crop_1_gray ,crop_2_gray,2 )
            if rmsebycol == 1:
                RMSE_C = cal_col_rmse(crop_1,crop_2,crop_1_gray ,crop_2_gray,2 )
            #RMSE_B, crop1_std1,crop2_std = cal_byblocks_rmse(crop_1, crop_2, crop_1_gray, crop_2_gray, image2,29, 29)
            if rmsebyblock == 1:
               RMSE_B, crop1_std,crop2_std = quick_cal_byblocks_rmse(crop_1, crop_2, crop_1_gray, crop_2_gray,crop_2,block_size) 
              
            file2_name = os.path.basename(filenamelist[j]).split('.')[0]
            string1 = file1_name +" avg gray : "+ str(np.round(gray1_mean,2))+" std:"+ str(np.round(gray1_std,2)) + ", Gray:" + str(np.round(crop1_std ,4) )
            string2 = file2_name +" avg gray : "+ str(np.round(gray2_mean,2))+" std:"+ str(np.round(gray2_std,2)) + ", Gray:" + str(np.round(crop2_std,4) )
            string3 = file1_name +" to " + file2_name 
            # string3 = string3 + " ,RMSE_P: " + str(np.round(RMSE_P,5))+" RMSE_R: "+ str(np.round(RMSE_R,5)) + \
            #     " ,RMSE_C:"+ str(np.round(RMSE_C,5)) 
            string3 = string3 + " ,gray: "+str(np.round(RMSE_B,5))
            #string4 = "file "+file1_name + " std=" + str(crop1_std) + ", "+ "file "+file2_name + " std=" + str(crop2_std)
            print(file1_name +".jpg avg gray : ",np.round(gray1_mean,2)," std:",np.round(gray1_std,2), ", block std:" , str(np.round(crop1_std,4 ) ))
            print(file2_name +",jpg avg gray : ",np.round(gray2_mean,2)," std:",np.round(gray2_std,2), ", block std:" , str(np.round(crop2_std,4 ) ))
            print(file1_name +" to " +file2_name + " ,RMSE_B: ",np.round(RMSE_B,5))#," ,RMSE_R: ",np.round(RMSE_R,5)," ,RMSE_C: ",np.round(RMSE_C,5)," ,RMSE_P: ",np.round(RMSE_P),5)
            Get_data[rec_no].append(file1_name)
            Get_data[rec_no].append(file2_name)
            Get_data[rec_no].append(gray1_mean)
            Get_data[rec_no].append(gray1_std)
            Get_data[rec_no].append(gray2_mean)
            Get_data[rec_no].append(gray2_std)
            Get_data[rec_no].append(RMSE_B)
            #print(string4)
            window['-MTLINE-'].print(string1)
            window['-MTLINE-'].print(string2)
            window['-MTLINE-'].print(string3)
            divx_num = 2
            divy_num = 2
            cnt_no = 0
            c_ht,c_wt = crop_1_gray.shape
            pass_cnt = 0
            for m in range (0,c_ht,c_ht//divy_num):
                for n in range(0,c_wt,c_wt//divx_num):
                    Rmse ,_,_= quick_cal_byblocks_rmse(crop_1[m:m+c_ht//divy_num, n:n+c_wt//divx_num], crop_2[m:m+c_ht//divy_num, n:n+c_wt//divx_num], crop_1_gray[m:m+c_ht//divy_num, n:n+c_wt//divx_num], crop_2_gray[m:m+c_ht//divy_num, n:n+c_wt//divx_num],crop_2, block_size)  
                    Get_data[rec_no].append(np.round(Rmse,4))
                    data_fields.append("Qua "+ loc_img[cnt_no]+" RMSE_29x29 ")            
                    crop_1_gray_quarter_mean = np.round(np.mean(crop_1_gray[m:m+c_ht//divy_num, n:n+c_wt//divx_num]),4)
                    crop_1_gray_quarter_std  = np.round(np.std(crop_1_gray[m:m+c_ht//divy_num, n:n+c_wt//divx_num]),4)
                    crop_2_gray_quarter_mean = np.round(np.mean(crop_2_gray[m:m+c_ht//divy_num, n:n+c_wt//divx_num]),4)
                    crop_2_gray_quarter_std  = np.round(np.std(crop_2_gray[m:m+c_ht//divy_num, n:n+c_wt//divx_num]),4)
                    data_fields.append("Qua "+ loc_img[cnt_no]+ " fst_img mean ")
                    data_fields.append("Qua "+ loc_img[cnt_no]+ " fst_img std ")
                    data_fields.append("Qua "+ loc_img[cnt_no]+ " sec_img mean ")
                    data_fields.append("Qua "+ loc_img[cnt_no]+ " sec_img std ")
                    data_fields.append("Qua "+ loc_img[cnt_no]+ " diff_mean ")
                    data_fields.append("Qua "+ loc_img[cnt_no]+ " diff_std ")
                    Get_data[rec_no].append(crop_1_gray_quarter_mean)
                    Get_data[rec_no].append(crop_1_gray_quarter_std)
                    Get_data[rec_no].append(crop_2_gray_quarter_mean)
                    Get_data[rec_no].append(crop_2_gray_quarter_std)
                    Get_data[rec_no].append(abs(crop_2_gray_quarter_mean-crop_1_gray_quarter_mean))
                    Get_data[rec_no].append(abs(crop_2_gray_quarter_std-crop_1_gray_quarter_std))
                    cnt_no += 1
 
                    if abs(crop_1_gray_quarter_mean-crop_2_gray_quarter_mean) <= Mean_error_thr \
                        and abs(crop_1_gray_quarter_std-crop_2_gray_quarter_std) <=Std_error_thr :
                        if Rmse < RMSE_thr :
                            pass_cnt += 1
                            
            data_fields.append("PorF")
            if pass_cnt >= 4 :  ## all pass
                Get_data[rec_no].append("Pass")
                window['-MTLINE-'].print("Pass")
            else:
                Get_data[rec_no].append("Fail") 
                window['-MTLINE-'].print("Fail")
            rec_no += 1 
    if save_to_csv == 1:  ## write to csv file
        timestr = time.strftime("%Y%m%d-%H%M")
        csvfile= save_folder+ "/"+timestr+".csv"
        print(csvfile)
        with open(csvfile,'w',newline='') as csvfile:         
              writer = csv.writer(csvfile)
              writer.writerow(["filename1","filename2","gray1_mean",'gray1_std', "gray2_mean","gray2_std","Global_RMSE_B",\
                               "TL RMSE","fst_img mean","fst_img std","sec_img mean","sec_img std","diff_mean","diff_std",\
                               "TR RMSE","fst_img mean","fst_img std","sec_img mean","sec_img std","diff_mean","diff_std",\
                               "DL RMSE","fst_img mean","fst_img std","sec_img mean","sec_img std","diff_mean","diff_std",\
                               "DR RMSE","fst_img mean","fst_img std","sec_img mean","sec_img std","diff_mean","diff_std","P.orF"])
              for i in range(0, rec_no):
                  writer.writerow(Get_data[i])       
def cal_foundation_rmse_by_folder3(image_fname):
    filenamelist = []
    imagelist=[]
    newfilenamelist = []
    listgrainsfeature = []
    bgr_imagelist = []
    for file_name in os.listdir( image_folder):
        
        if not(file_name.split('.')[-1] == 'jpg' or file_name.split('.')[-1] == 'bmp' or file_name.split('.')[-1] == 'JPG' or file_name.split('.')[-1] == 'BMP'):
            continue
        fname=image_folder + "/" + file_name
       
        filenamelist.append(fname) 
        
    if len(filenamelist)< 1 : return
    p1x = 1000#1300
    p1y = 1200  # 1279
    p2x = 4100#3900
    p2y = 4500#3700
    block_size = (29 , 29 )  # 替换為自定義的塊大小
    Get_data=[]
    for i in range(0,len(filenamelist)): 
        ori_img1 = cv2.imread(filenamelist[i])
        height1, width1, _ = ori_img1.shape
        image1 =  ori_img1[1500:height1-2000, 500:width1-500]       
        gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
        crop_1 = image1[p1y:p2y, p1x:p2x]
        crop_1_gray = cv2.cvtColor(crop_1, cv2.COLOR_BGR2GRAY)
        file1_name = os.path.basename(filenamelist[i]).split('.')[0]
        # output=image1.copy()
        # cv2.rectangle(output, (p1x,p1y), (  p2x ,   p2y ), (255, 0, 0),  10)
        # cv2.imshow('in',  cv2.resize(output, (600, 800)))
        # cv2.waitKey(0)
        for j in range(i+1,len(filenamelist))  :
            ori_img2 = cv2.imread(filenamelist[j])
            height2, width2, _ = ori_img2.shape
            image2 =  ori_img2[1500:height2-2000, 500:width2-500]             
             
            gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
            crop_2 = image2[p1y:p2y, p1x:p2x] 
            crop_2_gray = cv2.cvtColor(crop_2, cv2.COLOR_BGR2GRAY)
            RMSE_P = RMSE_R = RMSE_C = RMSE_B = 0
            gray1_mean  =  np.mean(crop_1_gray)
            gray2_mean  =  np.mean(crop_2_gray)
            gray1_std  =  np.std(crop_1_gray)
            gray2_std  =  np.std(crop_2_gray)
       
            if rmsebypixel == 1:
                RMSE_P,gray1_mean,gray1_std,gray2_mean,gray2_std = cal_p2p_rmse(crop_1,crop_2,crop_1_gray ,crop_2_gray,2 )
            if rmsebyrow == 1:
                RMSE_R = cal_row_rmse(crop_1,crop_2,crop_1_gray ,crop_2_gray,2 )
            if rmsebycol == 1:
                RMSE_C = cal_col_rmse(crop_1,crop_2,crop_1_gray ,crop_2_gray,2 )
            #RMSE_B, crop1_std1,crop2_std = cal_byblocks_rmse(crop_1, crop_2, crop_1_gray, crop_2_gray, image2,29, 29)
            if rmsebyblock == 1:
               RMSE_B, crop1_std,crop2_std = quick_cal_byblocks_rmse(crop_1, crop_2, crop_1_gray, crop_2_gray,crop_2,block_size) 
               RMSE_B_R, crop1_std_R,crop2_std_R = quick_cal_byblocks_rmse(crop_1, crop_2, crop_1[:, :, 0], crop_2[:, :, 0],crop_2,block_size)
               RMSE_B_G, crop1_std_G,crop2_std_G = quick_cal_byblocks_rmse(crop_1, crop_2,  crop_1[:, :, 1], crop_2[:, :, 1],crop_2,block_size)
               RMSE_B_B, crop1_std_B,crop2_std_B = quick_cal_byblocks_rmse(crop_1, crop_2,  crop_1[:, :, 2], crop_2[:, :, 2],crop_2,block_size)
            file2_name = os.path.basename(filenamelist[j]).split('.')[0]
            string1 = file1_name +" avg gray : "+ str(np.round(gray1_mean,2))+" std:"+ str(np.round(gray1_std,2)) + ", block R:" + str(np.round(crop1_std_R,4))+ ", G:" + str(np.round(crop1_std_G,4))+ ", B:" + str(crop1_std_B)+ ", Gray:" + str(np.round(crop1_std ,4) )
            string2 = file2_name +" avg gray : "+ str(np.round(gray2_mean,2))+" std:"+ str(np.round(gray2_std,2)) + ", block R:" + str(np.round(crop2_std_R,4))+ ", G:" + str(np.round(crop2_std_G,4))+ ", B:" + str(crop2_std_B)+ ", Gray:" + str(np.round(crop2_std,4) )
            string3 = file1_name +" to " + file2_name 
            # string3 = string3 + " ,RMSE_P: " + str(np.round(RMSE_P,5))+" RMSE_R: "+ str(np.round(RMSE_R,5)) + \
            #     " ,RMSE_C:"+ str(np.round(RMSE_C,5)) 
            string3 = string3 +" ,RMSE_B: "+ str(np.round(RMSE_B_R,5))+ "  "+ str(np.round(RMSE_B_G,5))+ "  "+str(np.round(RMSE_B_B,5))+ " ,gray: "+str(np.round(RMSE_B,5))
            #string4 = "file "+file1_name + " std=" + str(crop1_std) + ", "+ "file "+file2_name + " std=" + str(crop2_std)
            print(file1_name +".jpg avg gray : ",np.round(gray1_mean,2)," std:",np.round(gray1_std,2), ", block std(R,G,B,gray):" , str(np.round(crop1_std_R,4)), str(np.round(crop1_std_G,4) ), str(np.round(crop1_std_B,4)), str(np.round(crop1_std,4 ) ))
            print(file2_name +",jpg avg gray : ",np.round(gray2_mean,2)," std:",np.round(gray2_std,2), ", block std(R,G,B,gray):" , str(np.round(crop2_std_R,4)), str(np.round(crop2_std_G,4) ), str(np.round(crop2_std_B,4) ), str(np.round(crop2_std,4 ) ))
            print(file1_name +" to " +file2_name + " ,RMSE_B(R,G,B,Gray): ",np.round(RMSE_B_R,5),np.round(RMSE_B_G,5),np.round(RMSE_B_B,5),np.round(RMSE_B,5))#," ,RMSE_R: ",np.round(RMSE_R,5)," ,RMSE_C: ",np.round(RMSE_C,5)," ,RMSE_P: ",np.round(RMSE_P),5)
            Get_data.append(file1_name)
            Get_data.append(file2_name)
            Get_data.append(gray1_mean)
            Get_data.append(gray1_std)
            Get_data.append(gray2_mean)
            Get_data.append(gray2_std)
            Get_data.append(RMSE_B)
            #print(string4)
            window['-MTLINE-'].print(string1)
            window['-MTLINE-'].print(string2)
            window['-MTLINE-'].print(string3)
def cal_foundation_rmse_by_folder2(image_fname):
    filenamelist = []
    imagelist=[]
    newfilenamelist = []
    listgrainsfeature = []
    bgr_imagelist = []
    for file_name in os.listdir( image_folder):
        
        if not(file_name.split('.')[-1] == 'jpg' or file_name.split('.')[-1] == 'bmp' or file_name.split('.')[-1] == 'JPG' or file_name.split('.')[-1] == 'BMP'):
            continue
        fname=image_folder + "/" + file_name
       
        filenamelist.append(fname) 
        
    if len(filenamelist)< 1 : return
    p1x = 1000#1300
    p1y = 1200  # 1279
    p2x = 4100#3900
    p2y = 4500#3700
    block_size = (29 , 29 )  # 替换為自定義的塊大小
    for i in range(0,len(filenamelist)): 
        ori_img1 = cv2.imread(filenamelist[i])
        height1, width1, _ = ori_img1.shape
        image1 =  ori_img1[1500:height1-2000, 500:width1-500] 
        gray1 = image1.astype(np.uint32)
        gray1 = (gray1[:, :, 0] << 16) + (gray1[:, :, 1] << 8) + gray1[:, :, 2]
        #gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
        crop_1 = image1[p1y:p2y, p1x:p2x]
        crop_1_gray =  crop_1.astype(np.uint32)#cv2.cvtColor(crop_1, cv2.COLOR_BGR2GRAY)
        crop_1_gray = (crop_1_gray[:, :, 0] << 16) + (crop_1_gray[:, :, 1] << 8) + crop_1_gray[:, :, 2]
        file1_name = os.path.basename(filenamelist[i]).split('.')[0]
        # output=image1.copy()
        # cv2.rectangle(output, (p1x,p1y), (  p2x ,   p2y ), (255, 0, 0),  10)
        # cv2.imshow('in',  cv2.resize(output, (600, 800)))
        # cv2.waitKey(0)
        for j in range(i+1,len(filenamelist))  :
            ori_img2 = cv2.imread(filenamelist[j])
            height2, width2, _ = ori_img2.shape
            image2 =  ori_img2[1500:height2-2000, 500:width2-500]             
            gray2 = image2.astype(np.uint32) 
            gray2 = (gray2[:, :, 0] << 16) + (gray2[:, :, 1] << 8) + gray2[:, :, 2]
            #gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
            crop_2 = image2[p1y:p2y, p1x:p2x] 
            crop_2_gray = crop_2.astype(np.uint32)
            crop_2_gray = (crop_2_gray[:, :, 0] << 16) + (crop_2_gray[:, :, 1] << 8) + crop_2_gray[:, :, 2]
            #crop_2_gray = cv2.cvtColor(crop_2, cv2.COLOR_BGR2GRAY)
            RMSE_P = RMSE_R = RMSE_C = RMSE_B = 0
            gray1_mean  =  np.mean(crop_1_gray)
            gray2_mean  =  np.mean(crop_2_gray)
            gray1_std  =  np.std(crop_1_gray)
            gray2_std  =  np.std(crop_2_gray)
            if rmsebypixel == 1:
                RMSE_P,gray1_mean,gray1_std,gray2_mean,gray2_std = cal_p2p_rmse(crop_1,crop_2,crop_1_gray ,crop_2_gray,2 )
            if rmsebyrow == 1:
                RMSE_R = cal_row_rmse(crop_1,crop_2,crop_1_gray ,crop_2_gray,2 )
            if rmsebycol == 1:
                RMSE_C = cal_col_rmse(crop_1,crop_2,crop_1_gray ,crop_2_gray,2 )
            #RMSE_B, crop1_std1,crop2_std = cal_byblocks_rmse(crop_1, crop_2, crop_1_gray, crop_2_gray, image2,29, 29)
            if rmsebyblock == 1:
               RMSE_B, crop1_std,crop2_std = quick_cal_byblocks_rmse(crop_1, crop_2, crop_1_gray, crop_2_gray,crop_2,block_size)
            file2_name = os.path.basename(filenamelist[j]).split('.')[0]
            string1 = file1_name +" avg gray : "+ str(np.round(gray1_mean,2))+" std:"+ str(np.round(gray1_std,2)) + ", block std:" + str(np.round(crop1_std),4)
            string2 = file2_name +" avg gray : "+ str(np.round(gray2_mean,2))+" std:"+ str(np.round(gray2_std,2)) + ", block std:" + str(np.round(crop2_std),4)
            string3 = file1_name +" to " +file2_name 
            # string3 = string3 + " ,RMSE_P: " + str(np.round(RMSE_P,5))+" RMSE_R: "+ str(np.round(RMSE_R,5)) + \
            #     " ,RMSE_C:"+ str(np.round(RMSE_C,5)) 
            string3 = string3 +" ,RMSE_B: "+ str(np.round(RMSE_B,5))
            #string4 = "file "+file1_name + " std=" + str(crop1_std) + ", "+ "file "+file2_name + " std=" + str(crop2_std)
            print(file1_name +".jpg avg gray : ",np.round(gray1_mean,2)," std:",np.round(gray1_std,2), ", block std:" , str(crop1_std))
            print(file2_name +",jpg avg gray : ",np.round(gray2_mean,2)," std:",np.round(gray1_std,2), ", block std:" , str(crop2_std))
            print(file1_name +" to " +file2_name + " ,RMSE_B: ",np.round(RMSE_B,5))#," ,RMSE_R: ",np.round(RMSE_R,5)," ,RMSE_C: ",np.round(RMSE_C,5)," ,RMSE_P: ",np.round(RMSE_P),5)
            #print(string4)
            window['-MTLINE-'].print(string1)
            window['-MTLINE-'].print(string2)
            window['-MTLINE-'].print(string3)


def quick_cal_byblocks_rmse(crop_1, crop_2, crop_1_gray, crop_2_gray,crop_3,block_size):
    blocks1 = split_image_into_blocks(crop_1_gray, block_size)
    blocks2 = split_image_into_blocks(crop_2_gray, block_size)
    # 計算整張圖像的均方根誤差（RMSE）
    rmse,std_blk, std_blk2 = calculate_rmse(blocks1, blocks2)
    #ssim_score = compare_ssim(crop_1_gray,crop_2_gray)
    #correlation = pearson_correlation(crop_1, crop_2)
    #print("pearson=",correlation,", ssim=",ssim_score)
    print("block:",rmse)
    return rmse,std_blk, std_blk2
def plot_gray_hist(image1,image2,title):
    # 計算灰階值分佈
    histogram1 = cv2.calcHist([image1], [0], None, [256], [0, 256])
    histogram2 = cv2.calcHist([image2], [0], None, [256], [0, 256])
    
       # 繪製灰階值分佈圖
    plt.figure(figsize=(10, 6))
    plt.title( "Quarter " + title + ' Grayscale Histogram')
    plt.xlabel('Bins')
    plt.ylabel('# of Pixels')
    # 繪製第一個灰階分佈
    plt.plot(histogram1, color='b', label= full_file_name_b )
    
    # 繪製第二個灰階分佈
    plt.plot(histogram2, color='r', label= full_file_name_a)
    #plt.xticks(np.arange(100, 251, 5))
    plt.xlim([0, 256])
    plt.legend()
    
    plt.tight_layout()
    
    # Get the plot figure
    fig = plt.gcf()
    # Update the plot figure
    fig.canvas.draw()

    # Convert the updated plot figure to an image array
    image = np.array(fig.canvas.renderer.buffer_rgba())
    
    # Convert the image to PNG format
    img_encoded = cv2.imencode(".png", cv2.resize(image, (300, 200)))[1].tobytes()

    # Update the image in the GUI window
    if title == "TL" :
        window['-IMAGE_C-'].update(data=img_encoded)
    elif  title == "TR" :
        window['-IMAGE_D-'].update(data=img_encoded)
    elif  title == "DL" :
        window['-IMAGE_E-'].update(data=img_encoded)
    elif  title == "DR" :
        window['-IMAGE_F-'].update(data=img_encoded)
    if showplt == 1:   
        plt.show()
def cal_foundation_rmse(image1, image2,image3):
    #from SSIM_PIL import compare_ssim
    height, width, _ = image1.shape
    p1x = 1260 #1000 #1300
    p1y = 1400   # 1279
    p2x = 3600 #4100 #3900
    p2y = 4000 #4500 #3700  
    y_center = (p2y-p1y)//2
    x_center = (p2x-p1x)//2
    output = image1.copy()
    cv2.rectangle(output, (p1x,p1y), (  p2x ,   p2y ), (255, 0, 0),  20)
    #cv2.line(output, (2640,p1y), (  2640 ,   p2y ), (255, 255, 0),  5)
    cv2.line(output, (2430 ,p1y), (  2430  ,   p2y ), (0, 255, 0),  10)  ## x center
    cv2.line(output, (1260  ,p1y+1300), (  3600   ,   p1y+1300 ), (255, 0, 0),  20) #y center
    #cv2.rectangle(output, (2640-1000,p1y), (  2640+1000 ,   p2y ), (255, 0, 255),  10)
    #cv2.rectangle(output, (1260,p1y), (  3600 ,   p2y ), (255, 0, 255),  10)
    #cv2.rectangle(output, (p1x, p1y), (p2x,   p1y+500), (255, 0, 0),  10)
    #cv2.rectangle(output, (p1x+1500, p1y), (p2x,   p2y), (255, 0, 0),  10)
    # cv2.imshow('in',  cv2.resize(output, (600, 800)))
    # cv2.waitKey(0)
    # newimage = Image.fromarray(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
    # newimage.save("output_4.jpg", quality=100, dpi=(720.0, 720.0))
    # Convert the image to PNG format
    img_encoded = cv2.imencode(".png", cv2.resize(output, (140, 200)))[1].tobytes()
    # Update the image in the GUI window
    window["-IMAGE_B-"].update(data=img_encoded)
    output = image2.copy()
    cv2.rectangle(output, (p1x,p1y), (  p2x ,   p2y ), (255, 0, 0),  20)
    cv2.line(output, (2430,p1y), (  2430 ,   p2y ), (0, 255, 0),  10)  ## x center
    cv2.line(output, (1260 ,p1y+1300), (  3600  ,   p1y+1300 ), (255, 0, 0),  20) #y center
    img_encoded = cv2.imencode(".png", cv2.resize(output, (140, 200)))[1].tobytes()
    # Update the image in the GUI window
    window["-IMAGE_A-"].update(data=img_encoded)

    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    gray3 = cv2.cvtColor(image3, cv2.COLOR_BGR2GRAY)
    crop_1 = image1[p1y:p2y, p1x:p2x]
    crop_2 = image2[p1y:p2y, p1x:p2x]
    crop_3 = image3[p1y:p2y, p1x:p2x]
    crop_1_gray = gray1[p1y:p2y, p1x:p2x]#cv2.cvtColor(crop_1, cv2.COLOR_BGR2GRAY)
    crop_2_gray = gray2[p1y:p2y, p1x:p2x]#cv2.cvtColor(crop_2, cv2.COLOR_BGR2GRAY) 
    if rmsebypixel == 1 :
        cal_p2p_rmse(crop_1,crop_2,crop_1_gray ,crop_2_gray,1 )
    if rmsebyrow == 1 :
        cal_row_rmse(crop_1,crop_2,crop_1_gray ,crop_2_gray,1 )
    if rmsebycol == 1:
        cal_col_rmse(crop_1,crop_2,crop_1_gray ,crop_2_gray,1 )
    # value = compare_ssim(Image.fromarray(crop_1_gray),Image.fromarray(crop_2_gray), GPU=False)
    # print("ssim:",value)
    Get_data=[]
    data_fields=[]
    #cal_5blocks_rmse(image1,image2,gray1 ,gray2 ,image3,p1x,p1y,p2x,p2y)
    #cal_byblocks_rmse(crop_1, crop_2, crop_1_gray, crop_2_gray,crop_3, 200, 200)
    #cal_byblocks_rmse(crop_1, crop_2, crop_1_gray, crop_2_gray,crop_1, 100, 100)
    #cal_byblocks_rmse(crop_1, crop_2, crop_1_gray, crop_2_gray,crop_3, 50, 50)
    #rmse_gray_rec,  gray1_std,  gray2_std =cal_byblocks_rmse(crop_1, crop_2, crop_1_gray, crop_2_gray,crop_1, 29, 29)  
    #Get_data.append(np.round(rmse_gray_rec,4))
    #Get_data.append(gray1_std)
    #Get_data.append(gray2_std)
    #data_fields.append("Global RMSE_29x29 ")
     
    divx_num = 2
    divy_num = 2
    cnt_no=0
    loc_img =['TL','TR','DL','DR']
    if 0 :###divide into 4 subblocks
        c_ht,c_wt = crop_1_gray.shape
        for i in range (0,c_ht,c_ht//divy_num):
            for j in range(0,c_wt,c_wt//divx_num):
                rmse_gray_rec,  gray1_std,  gray2_std =cal_byblocks_rmse(crop_1[i:i+c_ht//divy_num, j:j+c_wt//divx_num], crop_2[i:i+c_ht//divy_num, j:j+c_wt//divx_num], crop_1_gray[i:i+c_ht//divy_num, j:j+c_wt//divx_num], \
                                  crop_2_gray[i:i+c_ht//divy_num, j:j+c_wt//divx_num],gray3[i:i+c_ht//divy_num, j:j+c_wt//divx_num], 29, 29)  
                Get_data.append(np.round(rmse_gray_rec,4))
                #Get_data.append(gray1_std)
                #Get_data.append(gray2_std)
                data_fields.append("Qua "+ loc_img[cnt_no]+" RMSE_29x29 ")            
                crop_1_gray_quarter_mean = np.round(np.mean(crop_1_gray[i:i+c_ht//divy_num, j:j+c_wt//divx_num]),4)
                crop_1_gray_quarter_std  = np.round(np.std(crop_1_gray[i:i+c_ht//divy_num, j:j+c_wt//divx_num]),4)
                crop_2_gray_quarter_mean = np.round(np.mean(crop_2_gray[i:i+c_ht//divy_num, j:j+c_wt//divx_num]),4)
                crop_2_gray_quarter_std  = np.round(np.std(crop_2_gray[i:i+c_ht//divy_num, j:j+c_wt//divx_num]),4)
                data_fields.append("Qua "+ loc_img[cnt_no]+ " " + full_file_name_b +" mean ")
                data_fields.append("Qua "+ loc_img[cnt_no]+ " " + full_file_name_b + " std ")
                data_fields.append("Qua "+ loc_img[cnt_no]+ " " + full_file_name_a +" mean ")
                data_fields.append("Qua "+ loc_img[cnt_no]+ " " + full_file_name_a + " std ")
                Get_data.append(crop_1_gray_quarter_mean)
                Get_data.append(crop_1_gray_quarter_std)
                Get_data.append(crop_2_gray_quarter_mean)
                Get_data.append(crop_2_gray_quarter_std)
                #print(cv2.meanStdDev(crop_1_gray[i:i+c_ht//divy_num, j:j+c_wt//divx_num] ))
                #print(cv2.meanStdDev(crop_2_gray[i:i+c_ht//divy_num, j:j+c_wt//divx_num] ))
                plot_gray_hist(crop_1_gray[i:i+c_ht//divy_num, j:j+c_wt//divx_num],crop_2_gray[i:i+c_ht//divy_num, j:j+c_wt//divx_num], loc_img[cnt_no])
                cnt_no += 1

      
        # print(cv2.meanStdDev(gray1[p1y:p1y+y_center, p1x:p1x+x_center] ))
        # print(cv2.meanStdDev(gray2[p1y:p1y+y_center, p1x:p1x+x_center] ))
        # cal_byblocks_rmse(image1[p1y:p1y+y_center, p1x:p1x+x_center], image2[p1y:p1y+y_center, p1x:p1x+x_center], gray1[p1y:p1y+y_center, p1x:p1x+x_center], gray2[p1y:p1y+y_center, p1x:p1x+x_center],gray3[p1y:p1y+y_center, p1x:p1x+x_center], 29, 29)  
        # cal_byblocks_rmse(image1[p1y:p1y+y_center, p1x+x_center:p2x], image2[p1y:p1y+y_center, p1x+x_center:p2x], gray1[p1y:p1y+y_center, p1x+x_center:p2x], gray2[p1y:p1y+y_center, p1x+x_center:p2x],gray3[p1y:p1y+y_center, p1x+x_center:p2x], 29, 29) 
        # print(cv2.meanStdDev(gray1[p1y:p1y+y_center, p1x+x_center:p2x] ))
        # print(cv2.meanStdDev(gray2[p1y:p1y+y_center, p1x+x_center:p2x] ))
        # cal_byblocks_rmse(image1[p1y+y_center:p2y , p1x:p1x+x_center], image2[p1y+y_center:p2y , p1x:p1x+x_center], gray1[p1y+y_center:p2y , p1x:p1x+x_center], gray2[p1y+y_center:p2y , p1x:p1x+x_center],gray3[p1y+y_center:p2y , p1x:p1x+x_center], 29, 29)  
        # print(cv2.meanStdDev(gray1[p1y+y_center:p2y, p1x:p1x+x_center] ))
        # print(cv2.meanStdDev(gray2[p1y+y_center:p2y, p1x:p1x+x_center] ))
        # cal_byblocks_rmse(image1[p1y+y_center:p2y, p1x+x_center:p2x], image2[p1y+y_center:p2y, p1x+x_center:p2x], gray1[p1y+y_center:p2y, p1x+x_center:p2x], gray2[p1y+y_center:p2y, p1x+x_center:p2x],gray3[p1y+y_center:p2y, p1x+x_center:p2x], 29, 29) 
        # print(cv2.meanStdDev(gray1[p1y+y_center:p2y, p1x+x_center:p2x] ))
        # print(cv2.meanStdDev(gray2[p1y+y_center:p2y, p1x+x_center:p2x] ))
    if rmsebyblock == 1  :
        block_size = (29, 29)  # 替换為自定義的塊大小
        Rmse,_,_ = quick_cal_byblocks_rmse(crop_1, crop_2, crop_1_gray, crop_2_gray,crop_3,block_size)
        Get_data.append(np.round(Rmse,4))
        data_fields.append("Global RMSE_29x29 ")
        divx_num = 2
        divy_num = 2
        c_ht,c_wt = crop_1_gray.shape
        pass_cnt = 0
        for i in range (0,c_ht,c_ht//divy_num):
            for j in range(0,c_wt,c_wt//divx_num):
                Rmse,_,_ = quick_cal_byblocks_rmse(crop_1[i:i+c_ht//divy_num, j:j+c_wt//divx_num], crop_2[i:i+c_ht//divy_num, j:j+c_wt//divx_num], crop_1_gray[i:i+c_ht//divy_num, j:j+c_wt//divx_num], crop_2_gray[i:i+c_ht//divy_num, j:j+c_wt//divx_num],gray3[i:i+c_ht//divy_num, j:j+c_wt//divx_num], block_size)  
                Get_data.append(np.round(Rmse,4))
                data_fields.append("Qua "+ loc_img[cnt_no]+" RMSE_29x29 ")            
                crop_1_gray_quarter_mean = np.round(np.mean(crop_1_gray[i:i+c_ht//divy_num, j:j+c_wt//divx_num]),4)
                crop_1_gray_quarter_std  = np.round(np.std(crop_1_gray[i:i+c_ht//divy_num, j:j+c_wt//divx_num]),4)
                crop_2_gray_quarter_mean = np.round(np.mean(crop_2_gray[i:i+c_ht//divy_num, j:j+c_wt//divx_num]),4)
                crop_2_gray_quarter_std  = np.round(np.std(crop_2_gray[i:i+c_ht//divy_num, j:j+c_wt//divx_num]),4)
                data_fields.append("Qua "+ loc_img[cnt_no]+ " " + full_file_name_b + " mean ")
                data_fields.append("Qua "+ loc_img[cnt_no]+ " " + full_file_name_b+ " std ")
                data_fields.append("Qua "+ loc_img[cnt_no]+ " " + full_file_name_a +" mean ")
                data_fields.append("Qua "+ loc_img[cnt_no]+ " " + full_file_name_a + " std ")
                Get_data.append(crop_1_gray_quarter_mean)
                Get_data.append(crop_1_gray_quarter_std)
                Get_data.append(crop_2_gray_quarter_mean)
                Get_data.append(crop_2_gray_quarter_std)
      
                if abs(crop_1_gray_quarter_mean-crop_2_gray_quarter_mean) <= Mean_error_thr \
                    and abs(crop_1_gray_quarter_std-crop_2_gray_quarter_std) <=Std_error_thr :
                    if Rmse < RMSE_thr :
                        pass_cnt += 1
                plot_gray_hist(crop_1_gray[i:i+c_ht//divy_num, j:j+c_wt//divx_num],crop_2_gray[i:i+c_ht//divy_num, j:j+c_wt//divx_num], loc_img[cnt_no])
                cnt_no += 1
        #         print(cv2.meanStdDev(crop_1_gray[i:i+c_ht//divy_num, j:j+c_wt//divx_num] ))
        #         print(cv2.meanStdDev(crop_2_gray[i:i+c_ht//divy_num, j:j+c_wt//divx_num] ))
        data_fields.append("PorF")
        if pass_cnt == 4 :  ## all pass
            Get_data.append("Pass")
        else:
            Get_data.append("Fail")
    for i in range(len(Get_data)):
        string1 = data_fields[i] + " : " + str(Get_data[i])
        window['-MTLINE-'].print(string1)
         
def cal_eyeline(image):
    height, width, _ = image.shape
    output = image.copy()
    x1 = 520 #500#420
    y1 = 270 #270     
    y2 = 5860 
    x2 = x1 + 950#1050
     
    x3 = 2410 #x2 + 940
    x4 = x3 + 950 #1050 #1000  
    
    x5 = 4300 #x4 + 880
    x6 = x5 + 950#1050#940 
   
    y3 = 490 #y1 +  220
    
    y4 = 580
    y5 = y4 + 170
    
    ##gap 90
    y6 = 840
    y7 = y6 + 170
    
    y8 = 1130
    y9 = y8 + 170
    
    # cv2.line(output,(x1,y1),(x1,y2),(0,0,255),thickness=5)
    # cv2.line(output,(x2,y1),(x2,y2),(0,0,255),thickness=5)
    # cv2.line(output,(x3,y1),(x3,y2),(0,0,255),thickness=5)
    # cv2.line(output,(x4,y1),(x4,y2),(0,0,255),thickness=5)
    # cv2.line(output,(x5,y1),(x5,y2),(0,0,255),thickness=5)
    # cv2.line(output,(x6,y1),(x6,y2),(0,0,255),thickness=5)
    # y1 = 580
    # y3 = 580 + 170
    # cv2.line(output,(x1,y1),(x6,y1),(0,255,255),thickness=10)
    # cv2.line(output,(x1,y3),(x6,y3),(0,255,255),thickness=10)
    # cv2.line(output,(x1,y4),(x6,y4),(255,255,0),thickness=5)
    # cv2.line(output,(x1,y5),(x6,y5),(255,255,0),thickness=5)
    # cv2.line(output,(x1,y6),(x6,y6),(255,255,0),thickness=5)
    # cv2.line(output,(x1,y7),(x6,y7),(255,255,0),thickness=5)
    # cv2.line(output,(x1,y8),(x6,y8),(255,255,0),thickness=5)
    # cv2.line(output,(x1,y9),(x6,y9),(255,255,0),thickness=5)
    y1 = 580 #355 #+280
    j = 0 
    line_color = (0,255,0)
    b=0
    y_lst = 720+4670-320#int(y1 + 28.8*180)
    x1 = 480
    x3 = 5290
    w = 960
    for i in range (0,10):
        #y_loc = int(320 + i * 20 * 28.167)
        y_loc = int(596 + i * 20 * 28.235)
        #cv2.rectangle(output,(x1,y_loc ),(x6,y_loc+282 ),(255,0,0),thickness=10)
        cv2.rectangle(output,(x1,y_loc ),(x1+w  ,y_loc+283 ),(255,0,0),thickness=10)
        cv2.rectangle(output,(x1+w*2,y_loc ),(x1+w*3  ,y_loc+283 ),(255,0,0),thickness=10)
        cv2.rectangle(output,(x1+w*4,y_loc ),(x1+w*5,y_loc+283 ),(255,0,0),thickness=10)
        # cv2.line(output,(x1,y_loc+b-85),(x6,y_loc+b-85),(255,0,0),thickness=10)
        # cv2.line(output,(x1,y_loc+b),(x6,y_loc+b),line_color,thickness=10)
        # cv2.line(output,(x1,y_loc+b+85),(x6,y_loc+b+85),(255,0,0),thickness=10)
    # for i in range (y1,5860,560): #570
    #     cv2.line(output,(x1,i+b-85),(x6,i+b-85),(255,0,0),thickness=10)
    #     cv2.line(output,(x1,i+b),(x6,i+b),line_color,thickness=10)
    #     cv2.line(output,(x1,i+b+85),(x6,i+b+85),(255,0,0),thickness=10)
    #     b=b+ 10#10
    #     #cv2.line(output,(x1,i+170),(x6,i+170),line_color,thickness=10)
    #     print(i)
     
    # y1 = 580 -280
    # j = 0
    # line_color = (0,255,0)
    # for i in range (y1,5800,280): #570
    #     cv2.line(output,(x1,i),(x6,i),line_color,thickness=10)
    #     cv2.line(output,(x1,i+170),(x6,i+170),line_color,thickness=10)
    #     j = j+1
    #     if (j % 2 == 0) :
    #         line_color = (0,255,0)
    #     else :
    #         line_color = (255,0,255)
    # y1 = 580  
    # for i in range (y1,6000,570): #570
    #     cv2.line(output,(x1,i),(x6,i),(0,255,255),thickness=10)
    #     cv2.line(output,(x1,i+170),(x6,i+170),(0,255,255),thickness=10)   
    cv2.imshow('in',  cv2.resize(output, (800, 1000)))

    cv2.waitKey(0)
def test2(image):
    height, width, _ = image.shape
    output = image.copy()
    #output[:] = 0
    showimg = image.copy()
    p1x = 1300
    p1y = 1279
    p2x = 3900
    p2y = 3700
    # p1x = 1200
    # p1y = 1279
    # p2x = 3600
    # p2y = 3700
    cv2.rectangle(output, (p1x, p1y), (p2x,   p2y), (255, 0, 0),  10)
    #cv2.line(output, (p1x,p1y), ( p2x ,   p2y ), (255, 0, 0), 10)
    cv2.imshow('in',  cv2.resize(output, (600, 800)))

    cv2.waitKey(0)


def test(image):
    height, width, _ = image.shape
    output = image.copy()
    output[:] = 0
    showimg = image.copy()
    p1x = 1000
    p1y = 1550
    p2x = 4800
    p2y = 1550
    #cv2.rectangle(output, (1000,1550), (  4800 ,   5350 ), (255, 255, 255), -1)
    cv2.line(output, (p1x, p1y), (p2x,   p2y), (255, 255, 255), 10)
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # maskimg = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
    # print(cv2.meanStdDev(gray, mask=maskimg) )
    cv2.imshow('in',  cv2.resize(output, (600, 800)))

    cv2.waitKey(0)

def test3(image):
    height, width, _ = image.shape
    output = image.copy()
    if 0 :
        x = 824 
        y = 720  
        w = 319
        h = 319
        wh = 159 
        fx = 478.5
        #cv2.circle(output, (480, 320),5, (0, 0, 255), -1)
        cv2.circle(output, (998, y+244),4, (0, 0, 255), -1)
        cv2.circle(output, (4995, y+244),4, (0, 0, 255), -1)
        #cv2.circle(output, (2051+998, y+245+2304),5, (0, 0, 255), -1)
        cv2.circle(output, (480, y+4670),4, (0, 0, 255), -1)
        cv2.circle(output, (5310, y+4670),4, (0, 0, 255), -1)
    else :
        x = 850 
        y = 710  
        w = 319
        h = 319
        wh = 159 
        fx = 478.5
        cv2.circle(output, (480, 276),5, (0, 255, 0), -1)
        cv2.circle(output, (480, 596),5, (0, 0, 255), -1)
        cv2.circle(output, (1020, y+220),1, (0, 0, 255), -1)
        cv2.circle(output, (5010, y+206),1, (0, 0, 255), -1)
        cv2.circle(output, (5280, 300),5, (0, 0, 255), -1)  #10
        cv2.circle(output, (5280, 580),5, (0, 0, 255), -1) #11
        cv2.circle(output, (480,  4860),5, (0, 0, 255), -1) #3
        cv2.circle(output, (480,  int(276+28.235*180)),15, (0, 255, 0), -1) #5  #5415
        cv2.circle(output, (480,  int(28.235*180+596)),5, (0, 0, 255), -1) #6  #5415+280
        cv2.circle(output, (5290,  int(300+28.235*180)),5, (0, 0, 255), -1) #7  # 4860+560+281
        cv2.circle(output, (5290,  int(28.235*180+580)),5, (0, 0, 255), -1) #8  #4860+560
    k = output[4800:6400,300:600]
    cv2.imshow('in', cv2.resize(output, (600, 800)))
    #x:1:28.14788  / 28.41176
    #y:1 : 28.1847133
    cv2.waitKey(0)
def AddMark(image):
    x = 824
    y = 720
    w = 319
    h = 319
    wh = 159
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 0), 1)
    cv2.rectangle(image, (x+wh, y), (x+w, y+wh), (0, 0, 0), -1)
    cv2.rectangle(image, (x, y+wh), (x+wh, y+h), (0, 0, 0), -1)

    x = 4936
    y = 720
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 0), 1)
    cv2.rectangle(image, (x+wh, y), (x+w, y+wh), (0, 0, 0), -1)
    cv2.rectangle(image, (x, y+wh), (x+wh, y+h), (0, 0, 0), -1)

    x = 824
    y = 7438
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 0), 1)
    cv2.rectangle(image, (x+wh, y), (x+w, y+wh), (0, 0, 0), -1)
    cv2.rectangle(image, (x, y+wh), (x+wh, y+h), (0, 0, 0), -1)

    x = 4936
    y = 7438
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 0), 1)
    cv2.rectangle(image, (x+wh, y), (x+w, y+wh), (0, 0, 0), -1)
    cv2.rectangle(image, (x, y+wh), (x+wh, y+h), (0, 0, 0), -1)
    cv2.imshow('in',  cv2.resize(image, (600, 800)))
    cv2.waitKey(0)
    #cv2.imwrite('output.png', image, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    #cv2.imwrite('output.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 100])
    newimage = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    newimage.save("output.jpg", quality=100, dpi=(720.0, 720.0))

def image_rot(image,angle):
    height, width, _ = image.shape
    center = [ width//2,height//2]
    # 計算旋轉矩陣
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # 旋轉圖片
    rotated = cv2.warpAffine(image, M, (width, height),
            flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT)
    return rotated
def menu_fun():
    consistency_parameter =[[sg.Text("Q_RMSE", size=(7, 1), justification='left'),  sg.InputText('2.59',size=(5, 1)  ,text_color='black', background_color='white',key='-in_QRMSE-' ,tooltip="0.1~10", pad=(0, 0) ),
                        sg.Text("Mean_err", size=(8, 1), justification='right'),  sg.InputText('1.1',size=(5, 1)  ,text_color='black', background_color='white',key='-in_MEAN_ERR-' ,tooltip="0.1~10", pad=(0, 0) ),
                        sg.Text("Std_err", size=(8, 1), justification='right'),  sg.InputText('0.2',size=(5, 1)  ,text_color='black', background_color='white',key='-in_STD_ERR-' ,tooltip="0.1~10", pad=(0, 0) ),
               
                        ]]
    menu_list_column = [
        [sg.Frame('Types of test', key='-Select_test-', layout=[
            [sg.Radio('A/B Compare', "comp_SEL", key='-AB_compare-', default=True, size=(10, 1), enable_events=True, pad=(0, 0)),

                  sg.Radio('L/R Compare', "comp_SEL", key='-LR_compare-',
                           default=False, size=(10, 1), enable_events=True, pad=(0, 0)),
                  sg.Radio('All_IMAGE',"comp_SEL",key='-All_image-',default=False,size=(0, 1), enable_events=True,pad=(0,0)),\
                  ]])],
        [sg.Frame('Makeup type', key='BAcompareframe-', layout=[
            [
                sg.Radio('Eyeshadow', "2_FUNCTION_SEL", key='-Eyeshadow_function-',
                         default=False, size=(10, 1), enable_events=True, pad=(0, 0)),
                sg.Radio('Blush', "2_FUNCTION_SEL", key='-Blush_function-',
                         default=False, size=(6, 1), enable_events=True, pad=(0, 0)),
                sg.Radio('LRBlushtri', "2_FUNCTION_SEL", key='-LRBlush_function-',
                         default=False, size=(6, 1), disabled=True, enable_events=True, pad=(0, 0)),
                sg.Radio('Fountri', "2_FUNCTION_SEL", key='-Fountri_function-', default=False, size=(6, 1), disabled=True, enable_events=True, pad=(0, 0))],
            [sg.Radio('L_Foundation', "2_FUNCTION_SEL", key='-L_fundation-', default=False, size=(10, 1), enable_events=True, pad=(0, 0)),
             sg.Radio('R_Foundation', "2_FUNCTION_SEL", key='-R_fundation-',
                      default=False, size=(10, 1), enable_events=True, pad=(0, 0)),
             sg.Radio('Foundation', "2_FUNCTION_SEL", key='-LR_fundation-', disabled=True,
                      default=False, size=(15, 1), enable_events=True, pad=(0, 0)),
             ]])],
        [sg.Frame('ColorSpace', key='-colorspaceframe-', layout=[
            [sg.Radio('BGR', "COLORSPACE", key='-BGR_function-', default=False, size=(5, 1), enable_events=True, pad=(0, 0)),
             sg.Radio('HSV', "COLORSPACE", key='-HSV_function-',
                      default=True, size=(5, 1), enable_events=True, pad=(0, 0)),
             sg.Radio('CIELAB', "COLORSPACE", key='-LAB_function-',
                      default=False, size=(6, 1), enable_events=True, pad=(0, 0)),
             sg.Radio('YCbCr', "COLORSPACE", key='-Ycbcr_function-',
                      default=False, size=(5, 1), enable_events=True, pad=(0, 0)),
             sg.Radio('CMYK', "COLORSPACE", key='-CMYK_function-', default=False, size=(5, 1), enable_events=True, pad=(0, 0)),\
             # sg.Checkbox('To Gray',key='-chk_2gray-', default=False,size=(6, 1)   ), \
             ]
        ])],
        # [sg.Frame('Channel compare', key='-comparetframe-', layout=[
        #     [sg.Checkbox('All ch', key='-chk_channel-', default=False, size=(10, 1)), \
        #      sg.Checkbox('Y value', key='-chk_yalue-',
        #                  default=False, size=(10, 1)),
        #      sg.Checkbox('binarization', key='-chk_binarization-',
        #                  default=True, size=(10, 1)),

        #      ]
        # ])],
        [  sg.Frame('Threshold',consistency_parameter  )],
        [sg.Frame('RMSE methods', key='-RMSEframe-', layout=[
            [
             sg.Checkbox('PIXEL', key='-chk_PIXEL-',default=False, size=(5, 1), enable_events=True),
             sg.Checkbox('ROW', key='-chk_ROW-', default=False, size=(4, 1) ,enable_events=True), \
             sg.Checkbox('COLUMN', key='-chk_COLUMN-', default=False, size=(7, 1), enable_events=True),
             sg.Checkbox('BLOCK', key='-chk_BLOCK-', default=True, size=(6, 1), enable_events=True),
             sg.Checkbox('ALL', key='-chk_ALL-', default=False, size=(5, 1), enable_events=True),
             ]
        ])],
        [sg.Frame('Output', key='-outputframe-', layout=[
            [sg.Checkbox('Save result', key='-chk_save-', default=False, size=(8, 1)), \
             sg.Checkbox('Show image', key='-chk_show-', default=False, size=(9, 1)),\
             sg.Checkbox('Show plot', key='-chk_plot-', default=False, size=(7, 1)),              
             sg.Checkbox('Save to csv', key='-csv_save-', default=False, size=(9, 1)),
             ]
        ])],
        [sg.Frame('Source : Two / Single', key='-mainframe-', layout=[
            [
                  sg.Text('Select first markup image', size=(22, 1), auto_size_text=False, justification='left')],\
            [sg.Input(key='-imgfile_b-', size=(42, 1), pad=(0, 0)), sg.FileBrowse(file_types=(("JPG Files",
                                                                   "*.jpg"), ("JPEG Files", "*.jpeg"), ("bmp Files", "*.bmp"), ("png Files", "*.png"),), key='-first_filebrowse-')],
            [
                sg.Text('Select second markup image', size=(22, 1), auto_size_text=False, justification='left')],\
            [sg.Input(key='-imgfile_a-', size=(42, 1), pad=(0, 0)), sg.FileBrowse(file_types=(("JPG Files", "*.jpg"),
                                                                  ("JPEG Files", "*.jpeg"), ("bmp Files", "*.bmp"), ("png Files", "*.png"),), key='-sec_filebrowse-')],
            [   sg.Text('Please enter image folder',size=(20,1),auto_size_text=False,justification='left')] ,\
                [sg.InputText('d:/',key='-LOAD_SOU-', size=(42, 1),pad=(0,0)),sg.FolderBrowse(key='-folder_filebrowse-',disabled=True)]  
              
        ])],
        [sg.Frame('Result', key='-resultframe-', layout=[
            [sg.Text('Save destination directory', size=(25, 1), auto_size_text=False, justification='left', pad=(0, 0))],\
            [sg.InputText('d:/', key='-SAVE_DST-', justification='left', size=(42, 1), pad=(0, 0)), sg.FolderBrowse()] \
        ])],
        [sg.Button('Submit'), sg.Button('Cancel')]

    ]

    image_viewer_column1 = [
        [sg.Text('Quarter TL image', size=(18, 1),
                 auto_size_text=False, justification='left')],
        [sg.Image(key="-IMAGE_C-")],
        [sg.Text('Quarter DL image', size=(18, 1),
                 auto_size_text=False, justification='left')],
        [sg.Image(key="-IMAGE_E-")],
        [sg.Text('First & second image', size=(20, 1),
                 auto_size_text=False, justification='left')],
        [sg.Image(key="-IMAGE_B-"),sg.Image(key="-IMAGE_A-")],
    

    ]
    image_viewer_column2 = [

        [sg.Text('Quarter TR image', size=(18, 1),
                 auto_size_text=False, justification='left')],
        [sg.Image(key="-IMAGE_D-")],
        [sg.Text('Quarter DR image', size=(18, 1),
                 auto_size_text=False, justification='left')],
        [sg.Image(key="-IMAGE_F-")],
        # [sg.Text('Second image', size=(12, 1),
        #          auto_size_text=False, justification='left')],
        # [sg.Image(key="-IMAGE_A-")],
        [sg.Multiline(size=(30,10), font='Tahoma 13', key='-MTLINE-', autoscroll=True,auto_size_text=True,do_not_clear = False,)]    
    ]
    image_viewer_column3 = [

        [sg.Text('Diff image', size=(18, 1),
                 auto_size_text=False, justification='left')],
        [sg.Image(key="-IMAGE_G-")],

        [sg.Image(key="-IMAGE_H-")],
       
    ]
    # image_viewer_column3 = [

    #     [sg.Text('Difference image', key='-detailimage1-',
    #              size=(20, 1), auto_size_text=False, justification='left')],
    #     [sg.Image(key="-IMAGE_D-")],
    #     [sg.Text('Left image', key='-detailimage2-', size=(20, 1),
    #              auto_size_text=False, justification='left')],
    #     [sg.Image(key="-IMAGE_E-")],
    #     [sg.Text('Right image', key='-detailimage3-', size=(20, 1),
    #              auto_size_text=False, justification='left')],
    #     [sg.Image(key="-IMAGE_F-")],
    # ]
    
    layout = [
        [
            sg.Column(menu_list_column),
            sg.VSeperator(),
            sg.Column(image_viewer_column1),
            sg.VSeperator(),
            sg.Column(image_viewer_column2),
            sg.VSeperator(),
            #sg.Column(image_viewer_column3),
            sg.Column(image_viewer_column3)   

        ]


    ]
    # Create the Window
    global window
    window = sg.Window('makeup analysis tool', layout,
                       element_justification='l', resizable=True)

    while True:
        event, values = window.read()
        if event == sg.WIN_CLOSED or event == 'Cancel':  # if user closes window or clicks Close
            break
        image_fname_b = values["-imgfile_b-"]
        image_fname_a = values["-imgfile_a-"]
        global save_folder, show_results, save_files, file_name_b, file_name_a, foldername_b, foldername_a, image_folder, save_to_csv, sel_colorspace, Histogram_alignment
        global rmsebypixel,rmsebyrow,rmsebycol,rmsebyblock ,full_file_name_b ,full_file_name_a
        global showplt,showimage
        global RMSE_thr,Mean_error_thr,Std_error_thr 
        showplt = 0
        showimage = 0
        foldername_b = os.path.split(image_fname_b)[0] + "/"
        foldername_a = os.path.split(image_fname_a)[0] + "/"
        save_folder = values['-SAVE_DST-'] + "/" #+ \
            #os.path.basename(image_fname_b).split('.')[0]+"_"
        image_folder =  values['-LOAD_SOU-']
        file_name_b = os.path.basename(image_fname_b).split('.')[0]
        file_name_a = os.path.basename(image_fname_a).split('.')[0]
        full_file_name_b = os.path.basename(image_fname_b)
        full_file_name_a = os.path.basename(image_fname_a)
        show_results = 0
        save_files = 0
        Histogram_alignment = 0
        save_to_csv = 0
        sel_colorspace = 2
        func = 0
        compare_mode = 0
        togray = 0
        use_3_channels = 0
        diff_binarization = 0
        rmsebypixel = rmsebyrow = rmsebycol = rmsebyblock = 0
        if event == '-chk_ALL-' and  values['-chk_ALL-'] == True:
            rmsebypixel = rmsebyrow = rmsebycol = rmsebyblock = 1
            window['-chk_PIXEL-'].Update( True)
            window['-chk_ROW-'].Update( True)
            window['-chk_COLUMN-'].Update( True)
            window['-chk_BLOCK-'].Update( True)
        elif event == '-chk_ALL-' and  values['-chk_ALL-'] == False:
            rmsebypixel = rmsebyrow = rmsebycol = rmsebyblock = 0
            window['-chk_PIXEL-'].Update( False)
            window['-chk_ROW-'].Update( False)
            window['-chk_COLUMN-'].Update( False)
            window['-chk_BLOCK-'].Update( False) 
        if values['-chk_PIXEL-'] == True:
             rmsebypixel = 1
        if values['-chk_ROW-'] == True:
             rmsebyrow = 1
        if values['-chk_COLUMN-'] == True:
             rmsebycol = 1
        if values['-chk_BLOCK-'] == True:
             rmsebyblock = 1
        # if values['-chk_yalue-'] == True:  # gray value uses Ycbcr
        #     togray = 1
        # if values['-chk_channel-'] == True:
        #     use_3_channels = 1
        # if values['-chk_binarization-'] == True:
        #     diff_binarization = 1
        RMSE_thr = 2.59
        Mean_error_thr = 1.1
        Std_error_thr = 0.2

        if values['-BGR_function-'] == True:
            sel_colorspace = 1
        elif values['-HSV_function-'] == True:
            sel_colorspace = 2
        elif values['-LAB_function-'] == True:
            sel_colorspace = 3
        elif values['-Ycbcr_function-'] == True:
            sel_colorspace = 4
        elif values['-CMYK_function-'] == True:
            sel_colorspace = 5
        if values['-All_image-'] == True:
            compare_mode = 3
            window['-folder_filebrowse-'].Update(disabled=False)
            window['-LOAD_SOU-'].Update(disabled=False)
            window['-first_filebrowse-'].Update(disabled=True)
            window['-imgfile_b-'].Update(disabled=True)
            window['-sec_filebrowse-'].Update(disabled=True)
            window['-imgfile_a-'].Update(disabled=True)
        elif values['-LR_compare-'] == True:
            compare_mode = 2
            window['-L_fundation-'].Update(disabled=True)
            window['-R_fundation-'].Update(disabled=True)
            window['-LR_fundation-'].Update(disabled=False)
            window['-LRBlush_function-'].Update(disabled=False)
            window['-Fountri_function-'].Update(disabled=False)
            window['-first_filebrowse-'].Update(disabled=False)
            window['-imgfile_b-'].Update(disabled=False)
            window['-sec_filebrowse-'].Update(disabled=True)
            window['-imgfile_a-'].Update(disabled=True)
            window['-folder_filebrowse-'].Update(disabled=True)
            window['-LOAD_SOU-'].Update(disabled=True)
            if values['-Eyeshadow_function-'] == True:
                window['-sec_filebrowse-'].Update(disabled=True)
                window['-imgfile_a-'].Update(disabled=True)
            elif values['-Blush_function-'] == True:
                window['-sec_filebrowse-'].Update(disabled=True)
                window['-imgfile_a-'].Update(disabled=True)
            elif values['-LR_fundation-'] == True:
                window['-sec_filebrowse-'].Update(disabled=False)
                window['-imgfile_a-'].Update(disabled=False)
            elif values['-LRBlush_function-'] == True:
                window['-sec_filebrowse-'].Update(disabled=False)
                window['-imgfile_a-'].Update(disabled=False)
            elif values['-Fountri_function-'] == True:
                window['-sec_filebrowse-'].Update(disabled=False)
                window['-imgfile_a-'].Update(disabled=True)
        else:
            compare_mode = 1
            window['-first_filebrowse-'].Update(disabled=False)
            window['-imgfile_b-'].Update(disabled=False)
            window['-R_fundation-'].Update(disabled=False)
            window['-L_fundation-'].Update(disabled=False)
            window['-LR_fundation-'].Update(disabled=True)
            window['-imgfile_a-'].Update(disabled=False)
            window['-sec_filebrowse-'].Update(disabled=False)
            window['-folder_filebrowse-'].Update(disabled=True)
            window['-LOAD_SOU-'].Update(disabled=True)
            #window.FindElement('-after_image-').Update(disabled = False)
        if values['-chk_save-'] == True:
            save_files = 1
        if values['-chk_show-'] == True:
            showimage = 1
        if values['-chk_plot-'] == True:
            showplt = 1
        if values['-csv_save-'] == True:
            save_to_csv = 1
        if event == 'Submit':           
            
            try :
                RMSE_thr = float(values["-in_QRMSE-"] )
                if RMSE_thr <=0 : RMSE_thr = 2.59
            except ValueError:
                RMSE_thr = 2.59
            try :
                Mean_error_thr = float(values["-in_MEAN_ERR-"] )
                if Mean_error_thr <=0 : Mean_error_thr = 1.1
            except ValueError:
                Mean_error_thr = 1.1
        
            try :
                Std_error_thr = float(values["-in_STD_ERR-"] )
                if Std_error_thr <=0 : Std_error_thr = 0.2
            except ValueError:
                Std_error_thr = 0.2           
            
            err_flag = 1
            if compare_mode == 2:
                if (os.path.isfile(image_fname_b)):
                    err_flag = 0
                    first_image = cv2.imread(image_fname_b)
                    #cal_eyeline(first_image)
                    test3(first_image)
            elif compare_mode == 1:
                window['-MTLINE-'].update(value='')
                if (os.path.isfile(image_fname_b) and os.path.isfile(image_fname_a)):
                    err_flag = 0
                    first_image = cv2.imread(image_fname_b)
                    second_image = cv2.imread(image_fname_a)
                    image_fname_a = 'd:\\glory\\pynoise_rec\\nozzle consistence\\test\\img577.jpg'
                    if os.path.isfile(image_fname_a) :  ## debug use
                        tri_image = cv2.imread( image_fname_a )
                    else:  tri_image = second_image
                    #cal_foundation_rmse(first_image,second_image,tri_image)
                    height1, width1, _ = first_image.shape
                    height2, width2, _ = second_image.shape
                    cal_foundation_rmse(first_image[1500:height1-2000, 500:width1-500], second_image[1500:height2 - \
                                        2000, 500:width2-500], tri_image[1500:height2-2000, 500:width2-500])
       
            elif compare_mode == 3:
                    if os.path.isdir(image_folder):
                       
                        cal_foundation_rmse_by_folder(image_folder)


            # if err_flag == 0:

                # height1, width1, _ = first_image.shape
                # height2, width2, _ = second_image.shape
                
                # cal_foundation_rmse(first_image[1500:height1-2000, 500:width1-500], second_image[1500:height2 - \
                #                     2000, 500:width2-500], tri_image[1500:height2-2000, 500:width2-500])

    window.close()


if __name__ == "__main__":
    #drawingModule = mediapipe.solutions.drawing_utils
    #faceModule = mediapipe.solutions.face_mesh

    # circleDrawingSpec = drawingModule.DrawingSpec(
    #     thickness=1, circle_radius=1, color=(0, 255, 0))
    # lineDrawingSpec = drawingModule.DrawingSpec(thickness=1, color=(0, 255, 0))
    menu_fun()
