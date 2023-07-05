import numpy as np
import cv2
import matplotlib.pyplot as plt
 
from scipy import ndimage
import random
import PySimpleGUI as sg
 
import platform as pf
import os,sys

import tkinter
import tkinter.messagebox #這個是訊息框，對話方塊的關鍵
from sklearn.preprocessing import normalize
import gc
 
from skimage import feature
from scipy.stats import norm
#import scipy
 
import math
 
from sklearn.datasets import make_regression
 
from sklearn.metrics import mean_squared_error, r2_score
#import scipy.stats as st
import pandas as pd
 
#import scipy.stats
import csv
import time


newstates=[]
SEG_LEFT = 5
SEG_TOP = 6
SEG_WIDTH = 7
SEG_HEIGHT = 8
SEG_NEW_LABEL = 9
SEG_IN_WIDHT = 10
SEG_IN_HEIGHT = 11
SEG_AVG_GRAY = 12
Ratio_x = 10
Ratio_y = 10
CIR_PERCENT = 0
CIR_OUT_AREA = 1
CIR_DOT_COUNT = 2
CIR_MEAN = 3
CIR_STD = 4
CIR_CENTER_X = 5
CIR_CENTER_Y = 6
CIR_WIDTH = 7
CIR_HEIGHT = 8
CIR_MAX_BLOB_WIDTH = 9
CIR_MAX_BLOB_HEIGHT = 10
def compute_newMSE(data1,data2,data1_std ,data2_std ,minlen,sminlen,cidx1,cidx2):
 
    sdata1=[]
    sdata2=[]
    sdata1_std=[]
    sdata2_std=[]
    msexy = []
    L=255
    K1=0.01
    K2=0.03
    C1=(K1*L)*(K1*L)
    C2=(K2*L)*(K2*L)
    C3=C2/2
    SSIMxy=[]  
    #print(len(data1),len(data2),data1[0],data2[0])
    for i in range(0,sminlen ,slice_num):
        for j in range(i,i+slice_num):
 
            sdata1.append(data1[j ])
            sdata2.append(data2[j ])
 
      
        mse =   ((np.var(sdata1)+np.var(sdata2)))    - np.cov(sdata1,sdata2)[0][1]-np.cov(sdata1,sdata2)[1][0]+(np.mean(sdata1)-np.mean(sdata2))**2
        lxy=(2*np.mean(sdata1)*np.mean(sdata2)+C1)/(np.mean(sdata1)**2+np.mean(sdata2)**2+C1)
        cxy=(2*np.std(sdata1)*np.std(sdata2)+C2)/(np.var(sdata1)+np.var(sdata2)+C2)
        sxy=((np.cov(sdata1, sdata2)[0][1]+np.cov(sdata1, sdata2)[1][0])/2+C3)/(np.std(sdata1)*np.std(sdata2)+C3)
        msexy.append(mse) 
        ssim = max(0.000001,(min(1,lxy*cxy*sxy)))
        SSIMxy.append(mse/ ssim)
        sdata1.clear()
        sdata2.clear()
        sdata1_std.clear()
        sdata2_std.clear()
        if i+slice_num >= len(data1) or  i+slice_num >= len(data2): break
    
    return  np.mean(msexy) ,np.mean(SSIMxy)
def find_contour(img ):
    img_height, img_width= img.shape
    output = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    thresh=recursionOTSU(img, 1)
    img_convert=cv2.bitwise_not(thresh)
    outimg=img.copy()
    contours,hierarchy=cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    find_cir_num=0
    fit_x=0
    fit_y=0
    fit_w=0
    fit_h=0
 
    for cnt in contours:
       
        x, y, w, h = cv2.boundingRect(cnt)
        if w < 100 or h < 100: continue
        if len(cnt ) <5 : continue
 
        
        ellipse = cv2.fitEllipse(cnt)
        fit_x = max(0,int(ellipse[0][0]-ellipse[1][0]/2)) 
        fit_y = max(0,int(ellipse[0][1]-ellipse[1][1]/2))
        fit_w = int(ellipse[1][0])
        fit_h = int(ellipse[1][1]) 
        if ellipse[0][0] + ellipse[1][0] / 2 > img_width or ellipse[0][0] - ellipse[1][0] / 2 <=1 : continue
        if ellipse[0][1] + ellipse[1][1] / 2 > img_width or ellipse[0][1] - ellipse[1][1] / 2 <=1 : continue
        #cv2.ellipse(output, ellipse, (255, 0, 0), 2)
        find_cir_num +=1
    if  find_cir_num == 1:
         x = fit_x
         y = fit_y
         w = fit_w 
         h = fit_h 
    else:
        for c in contours:
            rect = cv2.boundingRect(c)
            if rect[2] < 100 or rect[3] < 100: continue
            x,y,w,h = rect
    #cv2.rectangle(outimg,(x,y),(x+w,y+h),(0,0,255),3)
    cv2.ellipse(output, (int(x+w/2),int(y+h/2)), (int(w/2), int(h/2)), 0,0, 360, (0,0, 255), 2)
    #cv2.ellipse(output, (int(x+w/2),int(y+h/2)), (int(1), int(2)), 0,0, 360, (255,0, 255), 5)
    #cv2.namedWindow("contour", cv2.WINDOW_NORMAL) 
    #cv2.imshow("contour",outimg)
    smallest = img[y:y+h,x:x+w].min(axis=0).min(axis=0)
     
    r_x =int(w/2)
    r_y = int(y/2)
    cx = int(x+w/2)
    cy = int(y+h/2)
    sec_r_width =  int (w/2+w/4  )
    sec_r_height = int (h/2+h/4 )
    ellipse_mask = np.zeros((img.shape[0], img.shape[1], 1), np.uint8)
    mask_img = np.zeros((img.shape[0], img.shape[1], 1), np.uint8)
    inner_mask = np.zeros((img.shape[0], img.shape[1], 1), np.uint8)
    inner_mask[:] = 255    
    for i in range(int(w/2),w) :       # exten r = w/2, h/2
         y_axis = i
         ellipse_mask[:] = 0
         if (x_equal_y == 0): 
            y_axis = int(i* h/w) 
            if  cy-y_axis <= 0 or cy+ y_axis >= img_height-1 :
                y_axis = min(cy+1,img_height-cy-1)
         if not (cx-i>0 and cx+i <img_width and( cy-y_axis>0 and cy+y_axis< img_height)):
            cv2.ellipse(ellipse_mask, (int( img_width/2)-1 , int( img_height/2)-1 ), (1, 1), 0, 0, 360, (255,255, 255), 1)
            #cv2.ellipse(output, (x1, y1),  (int( img_width/2)-1 , int( img_height/2)-1 ), 0, 0, 360, (0,0, 255), 2)
            break;        
 
         else:
            cv2.ellipse(ellipse_mask, (cx, cy), (min(img_width/2,max(1,int(i)-1)), int(y_axis -1)   ), 0, 0, 360, (255,255, 255), 1)
            #cv2.ellipse(output, (x1, y1), (int(i), int(i)), 0, 0, 360, (0,0, 255), 2)
             
         mask_img=cv2.bitwise_and(ellipse_mask,img_convert)
         out_cir_area=cv2.countNonZero(ellipse_mask)
       
         #cv2.imshow('in',mask_img)   
         #cv2.waitKey(0)
         nzCount = cv2.countNonZero(mask_img)
         stop_count=0.001
         if i == int(w/2) :
              stop_count = (nzCount/out_cir_area)/100
        # print(nzCount/out_cir_area)
         if nzCount/out_cir_area < stop_count:
             cv2.ellipse(thresh, (int(cx),int(cy)), (int(i), int(y_axis)), 0,0, 360, (0,0, 255), 5)
             sec_r_width =  int (i )
             sec_r_height = int (y_axis)
            
             break;
    cv2.ellipse(thresh, (int(cx),int(cy)), (int( sec_r_width ), int(sec_r_height )), 0,0, 360, (0,0, 255), 2)         
 
     
    cv2.namedWindow("contour", cv2.WINDOW_NORMAL) 
    cv2.imshow("contour",output)
    return int(x+w/2),int(y+h/2),w,h,sec_r_width,sec_r_height

def  cal_cir_hybrid_avg(img,img_idx):
    img_height, img_width= img.shape
  
    x1,y1,max_width,max_height,sec_r_width,sec_r_height=find_contour(img)
 
   
    max_cir_radius_in =int( max(max_width/2,max_height/2))
    max_cir_radius =int( max(sec_r_width,sec_r_height))
    if max_cir_radius < 10: return 
     
    output = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    ellipse_mask = np.zeros((img.shape[0], img.shape[1], 1), np.uint8)
    mask_img = np.zeros((img.shape[0], img.shape[1], 1), np.uint8)
    inner_mask = np.zeros((img.shape[0], img.shape[1], 1), np.uint8)
   
    inner_mask[:] = 255    
 
    n=int(360/slice_num)
    slice_cir_avg=[]
    slice_cir_std=[]
    slice_cir_bin_avg=[]
    slice_cir_bin_std=[]
    slice_cir_avg_all=[]
    slice_cir_std_all=[]
    for i in range(1,int(n+1)):  # from 1 to 3
        slice_cir_avg.append([])
        slice_cir_std.append([])
        slice_cir_bin_avg.append([])
        slice_cir_bin_std.append([])
    idx=0
  
    
    for i in range(20-1, max_cir_radius ,thickness_num ):    
        
        idx += 1
        
        #print(img_idx )
        for cs in range(0,360,n):
            ce = cs+n
           
            ellipse_mask[:] = 0
             
            y_axis = i
            if (x_equal_y == 0):
                # if  max_heigth > max_width :
                y_axis = int(i*  max_height/max_width )
                # else :
                #     y_axis = int(i* max_width/ max_height)
                if  y1-y_axis <= 0 or y1+ y_axis >= img_height-1 :
                    y_axis = min(y1+1,img_height-y1-1)
            if i == 20-1:
                cir_thickness = -1
            else:
                cir_thickness = thickness_num
            if not (x1-i>0 and x1+i <img_width and( y1-y_axis>0 and y1+y_axis< img_height)):
                cv2.ellipse(ellipse_mask, (int( img_width/2)-1 , int( img_height/2)-1 ), (1, 1), 0, cs, ce, (255,255, 255), cir_thickness)
                cv2.ellipse(output, (x1, y1),  (int( img_width/2)-1 , int( img_height/2)-1 ), 0, cs, ce, (0,0, 255),  cir_thickness)
                break;
            if i>=  max_cir_radius_in-1:
                cv2.ellipse(ellipse_mask, (x1, y1), (int(i)-1, int(y_axis)-1), 0, cs, ce, (255,255, 255), cir_thickness)
                cv2.ellipse(output, (x1, y1), (int(i), int(i)), 0, cs, ce, (0,0, 255), cir_thickness)
     
            else:
                cv2.ellipse(ellipse_mask, (x1, y1), (i-1, int(y_axis)-1), 0, cs, ce, (255,255, 255),  cir_thickness)
                cv2.ellipse(output, (x1, y1), (int(i), int(y_axis)), 0, cs, ce, (0,0, 255),  cir_thickness)
                
            mask_img= cv2.bitwise_and(ellipse_mask,inner_mask )
            #maskb_img= cv2.bitwise_and(ellipse_mask,img_convert  )  # only have binarization point
            out_cir_area=cv2.countNonZero(mask_img)
           
            mean,stddv = cv2.meanStdDev(img, mask=mask_img)
            #meanb,stddvb = cv2.meanStdDev(img, mask=maskb_img)
            inner_mask=cv2.bitwise_not(ellipse_mask) 
            # slice_cir_avg[idx].append(np.round(mean[0][0],2))
            # slice_cir_std[idx].append(np.round(stddv[0][0],4))
            slice_cir_avg_all.append(np.round(mean[0][0],2))
            slice_cir_std_all.append(np.round(stddv[0][0],4))
            #slice_cir_bin_avg[idx].append(np.round(meanb[0][0],2))
            #slice_cir_bin_std[idx].append(np.round(stddvb[0][0],4))
            # avg_data.append(np.round(mean[0][0],2))
            # cl_data.append("c"+str(cs))
 
        
    ### compute Perimeter
    cir_avg =[]
    cir_std =[]
    cir_bin_avg=[]
    cir_bin_std=[]
    skip_idx = 20 #20210729 test using 2
    inner_mask[:] = 255    
    for i in range(skip_idx -1, max_cir_radius ,thickness_num ):      
        
        ellipse_mask[:] = 0
        y_axis = i

        if (x_equal_y == 0):
            # if  max_heigth > max_width :
            y_axis = int(i*  max_height/max_width )
            # else :
            #     y_axis = int(i* max_width/ max_heigth)
            if  y1-y_axis <= 0 or y1+ y_axis >= img_height-1 :
                y_axis = min(y1+1,img_height-y1-1)
        if i == 1 :
            cv2.line(ellipse_mask, (x1, y1), (x1, y1) , (255,255, 255), 1)
            #print("line2",cv2.countNonZero(ellipse_mask),img[y1][x1])
        elif i == 2:
            cv2.ellipse(ellipse_mask, (x1, y1), (int(i-1), int(i)-1), 0, 0, 360, (255,255, 255), -1 )
            cv2.ellipse(output, (x1, y1), (int(i), int(i)), 0, 0, 360, (0,0, 255), 2)
        elif not (x1-i>0 and x1+i <img_width and( y1-y_axis>0 and y1+y_axis< img_height)):
            cv2.ellipse(ellipse_mask, (int( img_width/2)-1 , int( img_height/2)-1 ), (1, 1), 0, 0, 360, (255,255, 255),-1)
            cv2.ellipse(output, (x1, y1),  (int( img_width/2)-1 , int( img_height/2)-1 ), 0, 0, 360, (0,0, 255), 2)
          
            break 
        elif i>=  max_cir_radius-1:
            cv2.ellipse(ellipse_mask, (x1, y1), (int(i)-1, int(y_axis)-1), 0, 0, 360, (255,255, 255), -1)
            cv2.ellipse(output, (x1, y1), (int(i), int(y_axis)), 0, 0, 360, (0,0, 255), 2)
 
        else:
            #cv2.ellipse(ellipse_mask, (x1, y1), (min(img_width/2,max(1,int(i)-1)), int(y_axis)-1), 0, 0, 360, (255,255, 255), -1 )
            cv2.ellipse(ellipse_mask, (x1, y1), (int(i-1), int(y_axis)-1), 0, 0, 360, (255,255, 255), -1 )
            cv2.ellipse(output, (x1, y1), (int(i), int(y_axis)), 0, 0, 360, (0,0, 255), 2)

        mask_img= cv2.bitwise_and(ellipse_mask,inner_mask )
        #maskb_img= cv2.bitwise_and(ellipse_mask,img_convert  )  # only have binarization point
        #out_cir_area=cv2.countNonZero(mask_img)
        mean,stddv = cv2.meanStdDev(img, mask=mask_img)
        #meanb,stddvb = cv2.meanStdDev(img, mask=maskb_img)
        inner_mask=cv2.bitwise_not(ellipse_mask) 
        cir_avg.append(np.round(mean[0][0],2))
        cir_std.append(np.round(stddv[0][0],4))

    if 0:
        fig, ax = plt.subplots()
        # print(len(slice_cir_avg ),len(slice_cir_avg[1] ))
        lx = 0
        data_len=int(3*len(slice_cir_avg[1])/4)
        for p in range(1,len(slice_cir_avg )):
            
            r = random.random()
            b = random.random()
            g = random.random()
            line_color = (r, g, b)
            #plot_f= ax.figure()
          
            ax.plot(range(0, len(slice_cir_avg[p]),1  ),slice_cir_avg[p] ,marker='o', markersize=1,color=line_color, label=str(p)+":"+str(img_idx))
            #ax.plot(range(0, len(slice_cir_std[p]),1  ),slice_cir_std[p] ,marker='o', markersize=1,color= line_color, label=str(p)+":"+str(img_idx))
            if p == len(slice_cir_avg ) - 1 : 
                ax.text(data_len, int(100+(lx)*6 ), "H test : p>0.05"  , ha='center', va='bottom', fontsize=10)
                    
            for q in range(p+1,len(slice_cir_avg )):
               
                # h,per=scipy.stats.kruskal( slice_cir_avg[p],slice_cir_avg[q])
                txt_color = 'b'
                SAE=np.round(sum(abs(x - y) for x, y in zip(sorted(slice_cir_avg[p]), sorted(slice_cir_avg[q]))),4)
                rmse = np.sqrt(mean_squared_error(slice_cir_avg[p],slice_cir_avg[q]))
                # if per< 0.05 or SAE > len(slice_cir_avg[1])*2.5:
                #     txt_color = 'r'
                
                ax.text(data_len, int(100+lx*6),  str(p)+":"+str(q)+" SAE :"+str(np.round(SAE,2))+",RMSE:"+str(np.round(rmse,4))  , color = txt_color, ha='center', va='bottom', fontsize=10)
                lx +=1
                 
            r = random.random()
            b = random.random()
            g = random.random()
            line_color = (r, g, b)
             
        ax.legend()
        ax.set_title( file_name+':Avg gray value curv of circles:'+str(img_idx))
        #ax.set_title('Avg gray value and Standard deviation of circle')
        ax.xaxis.set_label_text("circle no. (from inside to outside (radius +"+str(thickness_num)+") )")
        ax.yaxis.set_label_text("Avg gray value ")
 
    if show_results == 1:  
        plt.show()  
        cv2.namedWindow('circles', cv2.WINDOW_NORMAL) 
        cv2.imshow("circles",output)
    if save_files == 1:
        filename= save_folder + "slice_circle_no_plot"+ str(img_idx)+"_"+str(p)+".jpg"
    
        fig.savefig(filename )
    
    del ellipse_mask,output ,inner_mask, mask_img
    gc.collect()
    return cir_avg,cir_std , slice_cir_avg_all, slice_cir_std_all
    
def recursionOTSU(img, n):
    
    array = np.ravel(img) # 將 二維數組img 降維爲 一維數組array 
 
    for i in range(n):
        
        retval,array = cv2.threshold(array,0,0,cv2.THRESH_TOZERO + cv2.THRESH_OTSU) # 對 array 同時進行 閾值化爲0 和 OTSU算法
        
        array = array[array > 0] # 將 array 中等於0的元素去除掉
        
        retval,img = cv2.threshold(img,retval,0,cv2.THRESH_TOZERO) # 用新的閾值對圖像進行 閾值化爲0 的操作
        
    retval,img = cv2.threshold(img,retval,255,cv2.THRESH_BINARY) # 對遞歸後的圖像進行二進制閾值化操作，讓圖像更清晰
    
    return img


def image_segment(crop_img ):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    img_bin=recursionOTSU(crop_img, 1)
    #(T,img_bin) = cv2.threshold(img,180,255, cv2.THRESH_BINARY);
    img_convert = cv2.bitwise_not(img_bin)
    img_convert = cv2.erode(img_convert, kernel, iterations = 3)
    img_height, img_width= crop_img.shape
     #CCL
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img_convert, connectivity=8)
    # 查看各个返回值
    # 连通域数量
    #print('num_labels = ',num_labels)
    # 连通域的信息：对应各个轮廓的x、y、width、height和面积
    #print('stats = ',stats)
    # 连通域的中心点
    #print('centroids = ',centroids)
    # 每一个像素的标签1、2、3.。。，同一个连通域的标签是一致的
    #print('labels = ',labels)
    cv2.namedWindow('bin_result', cv2.WINDOW_NORMAL) 
    cv2.imshow('bin_result', img_convert)
    indexes_group = np.argsort(-stats[:, cv2.CC_STAT_AREA])
    sort_stats = stats[indexes_group]
    max_c_w = sort_stats[1, cv2.CC_STAT_WIDTH]   ##find the maximun blob
    max_c_h =  sort_stats[1, cv2.cv2.CC_STAT_HEIGHT ]
   
    # 不同的连通域赋予不同的颜色
    #output = np.zeros((crop_img.shape[0], crop_img.shape[1], 3), np.uint8)
    output = cv2.cvtColor(crop_img, cv2.COLOR_GRAY2BGR)
    global newstates 
    for i in range(num_labels):
        newstates.append([])
   
    #for i in range(num_labels+1):    
    #    for j in range(9):      
    #        newstates[i].append(0)
    # ellipse_mask=img_convert
    # mask_img=inner_mask=ellipse_mask
    # inner_mask[:] = 255
    idx = 0
    # avg_inc = 0
    # in_width=0
    # in_height=0
    for i in range(1, num_labels):
         minX=0
         MaxX=0
         minY=0
         MaxY=0
         x1, y1, width, height, count = stats[i]
         x0, y0 = centroids[i]
         idx += 1
         mask = labels == i
         output[:, :, 0][mask] = np.random.randint(0, 255)
         output[:, :, 1][mask] = np.random.randint(0, 255)
         output[:, :, 2][mask] = np.random.randint(0, 255)
    #      cv2.rectangle(output, (x1,y1), (x1 + width,  y1 + height), (0, 255, 0), 1)
         
         #cv2.circle(output,( int(x0 )  , int(y0-height/2)), 1, (0, 0, 255), 1)
         #text=str(i)
         #cv2.putText(output, text, (int(x0),int(y0)), cv2.FONT_HERSHEY_SIMPLEX,  2, (0, 255, 255), 1, cv2.LINE_AA)
         
         
         ## cal avg1
         #if 0:
             # for cc in range(10,min(int(width/2-1),int(height/2-1)),1 ):      
             #    #print(i)
             #    ellipse_mask[:] = 0
             #    if not (x0-cc>0 and x0+cc <img_width and( y0-cc>0 and y0+cc< img_height)):
             #        cv2.ellipse(ellipse_mask, (int( width/2)-1 , int( height/2)-1 ), (1, 1), 0, 0, 360, (255,255, 255), -1)
             #        #cv2.ellipse(output, (x1, y1),  (int( img_width/2)-1 , int( img_height/2)-1 ), 0, 0, 360, (0,0, 255), 2)
             #        break;
             #    if cc>= min(width,height):
             #        cv2.ellipse(ellipse_mask, (int(x0), int(y0)), (int(cc)-1, int(cc)-1), 0, 0, 360, (255,255, 255), -1)
             #        #cv2.ellipse(output, (x1, y1), (int(i), int(i)), 0, 0, 360, (0,0, 255), 2)
         
             #    else:
             #        cv2.ellipse(ellipse_mask,(int(x0), int(y0)), (min(width/2,max(1,int(cc)-1)), int(cc)-1), 0, 0, 360, (255,255, 255), -1)
             #        #cv2.ellipse(output, (x0, y0), (int(cc), int(cc)), 0, 0, 360, (0,0, 255), 2)
                     
             #    mask_img=ellipse_mask#cv2.bitwise_and(ellipse_mask,inner_mask)
             #    out_cir_area=cv2.countNonZero(mask_img)
               
             #    #cv2.imshow('in',mask_img)   
             #    #cv2.waitKey(0)
             #    nzCount = cv2.countNonZero(cv2.bitwise_and(img_convert, mask_img))
             #    mean,stddv = cv2.meanStdDev(crop_img, mask=mask_img)
             #    if ( cc >= min(int(width/2-1),int(height/2-1))-1   or np.round(nzCount/(out_cir_area ),2)<0.90):
             #        avg_inc = np.round(mean[0][0],0)
             #        in_width=cc
             #        in_height=cc
             #        break
                #print(cc,np.round(nzCount/(out_cir_area ),2),avg_inc ,in_width,in_height)
            ####

         if 1:
             ####cal the average value
             avg_c=crop_img[int(y0),int(x0)]
             avg_x=0
             inx=0
             if width > 10 :
                 for j in range(int(x0 -  1*width/5) ,int(x0 + 1*width/5) ):
                     avg_x+=int(crop_img[int(y0-1),j])
                     avg_x+=int(crop_img[int(y0),j])
                     avg_x+=int(crop_img[int(y0+1),j])
                     inx+=1   
                 if inx >=1 : 
                     avg_x=avg_x/(inx*3)
             
                
             avg_y=crop_img[int(y0),int(x0)]            
             inx=0   
             if x0 >0 and height > 10 :
                 for j in range(int(y0 - 1*height/5) ,int(y0 + 1*height/5 )):
                     avg_y+=int(crop_img[j,int(x0-1)])
                     avg_y+=int(crop_img[j,int(x0)])
                     avg_y+=int(crop_img[j,int(x0+1)])
                     inx+=1 
                 if inx>=1:
                     avg_y=avg_y/(inx*3)
             avg_c = min(avg_x,avg_y)
         #####################
         
         ### extend the width and height to segment this object
         inx=1 
         sx = max(int(x0-width/2 )  ,1)
         minX= max(int(x0-width*2-1  )  ,1)
         ex_num=10
         for j in range(1,width*ex_num ):
             ax = max(sx - j,1)
             if (labels[int(y0), int(ax)] != i and labels[int(y0), int(ax)] >0):
                 minX = int((sx + ax) /2 )
                 break;
             if j >= width*ex_num-1:
                minX =  max(sx - width*3+1,1)
         ex = min(int(x0+width/2 )  ,crop_img.shape[1]-1)  
         MaxX = min(int(ex+width*2-1)  ,crop_img.shape[1]-2)  
         for j in range(1,width*ex_num ):
            ax = min(ex + j,crop_img.shape[1]-1)

            if (labels[int(y0), int(ax)] != i and labels[int(y0), int(ax)] >0):
                MaxX = int((ex + ax) /2) 
                break;
            if j >= width*ex_num-1:
                MaxX =  min(ex + width*3-1,crop_img.shape[1]-1)
                
         sy = max(int(y0-height/2 ) , 1)
         minY= max(int(y0-height*2-1 ) , 1)
         for j in range(1,height*ex_num ):
            ay = max(sy - j,1)
            
            if (labels[int(ay), int(x0)] != i and labels[int(ay), int(x0)] >0):
                minY = int((sy + ay) /2 )
                break;
            if  j>=height*ex_num-1 :
                minY = max(sy - height*1+1,1)
                
         ey = min(int(y0+height/2 ) ,crop_img.shape[0]-1)
         MaxY = min(int(ey+height*2-1) ,crop_img.shape[0]-2)
         for j in range(1,height*ex_num):
            ay = min(ey + j,crop_img.shape[0]-1)
            if ay == crop_img.shape[0]-1:
                MaxY= int(crop_img.shape[0]-1)
                break;
            if (labels[int(ay), int(x0)] != i and labels[int(ay), int(x0)] >0):
                MaxY =int((ey + ay) /2 )
                break;   
            if  j>=height*ex_num-1 :
                MaxY =  min(ey + height*1-1,crop_img.shape[0]-1)
        ###########################
         newstates[idx].append(stats[i, cv2.CC_STAT_LEFT])
         newstates[idx].append(stats[i, cv2.CC_STAT_TOP])
         newstates[idx].append(stats[i, cv2.CC_STAT_WIDTH])
         newstates[idx].append(stats[i, cv2.CC_STAT_HEIGHT])
         newstates[idx].append( stats[i, cv2.CC_STAT_AREA])
         newstates[idx].append( minX) #5
         newstates[idx].append( minY)
         newstates[idx].append( MaxX-minX)
         newstates[idx].append( MaxY-minY)
         newstates[idx].append(0) #[9]
         # newstates[idx].append(in_width) #10
         # newstates[idx].append(in_height) #11
         # newstates[idx].append(avg_inc)#12
         
         if 1:
             dx1=0
             dx2=0
             dy1=0
             dy2=0
             c=0
             thresh_w = max(10,int(width/100))
             thresh_h = max(10,int(height/100))
             if width > 100:
                 for j in range(int(x0)-width+1,int(x0) ):
                     if crop_img[int(y0),j] < avg_c :
                         dx1=j
                         c+=1
                         if (c>thresh_w):
                            break 
                 c=0
                 for j in range(int(x0)+width,int(x0),-1 ):
                  
                     if crop_img[int(y0),j] < avg_c :
                         dx2=j
                         c+=1
                         if (c>thresh_w):
                            break 
                 newstates[idx].append(int(max(min(int(x0-dx1),int(dx2-x0) ),1))*2) #10
             else :              
                  newstates[idx].append(width-2) #10
             if height > 100:
                 c=0
                 for j in range(int(y0)-height+1,int(y0) ):
                     if crop_img[j, int(x0)] < avg_c :
                         dy1=j
                         c+=1
                         if (c>thresh_h):
                            break 
                 c=0
                 for j in range(int(y0)+height,int(y0),-1 ):
                      
                     if crop_img[j,int(x0)] < avg_c :
                         dy2=j
                         c+=1
                         if (c>thresh_h):
                            break        
                 newstates[idx].append(int(max(min(int(y0-dy1),int(dy2-y0) ),1))*2) #11
             else :
                 newstates[idx].append(height-2) #11
             #print(avg_c)
             newstates[idx].append(int(avg_c))#12
    
             #print(minX,minY,MaxX,MaxY)
         #cv2.circle(crop_img,( int(x0 )  , int(y0 )), int(min(int(x0-dx1),int(dx2-x0) )), (0, 0, 255), 1)
         #cv2.rectangle(output, (minX,minY), (MaxX,  MaxY), (128, 128, 0), 1)
    
    ###### sort the label from top to bottom and from left to right
    ###### segment the binaration map 
    hist_r = np.zeros(img_height)
    hist_c = np.zeros(img_width)
    for i in range(0,img_height) :
        for j in range(0,img_width):
            if (img_convert [i][j]==255):
                hist_r[i] += 1
                
    for j in range(0,img_width):
        for i in range(0,img_height) : 
            if (img_convert [i][j]==255):
                hist_c[j] += 1
   
    seg_x=[]
    seg_y=[]
    first_seg_y = 0
    for i in range(0,img_height-2) :
        if i==1 or ((hist_r[i]>0 and hist_r[i+1]==0)):
            for j in range(i+2, img_height-2):
                if hist_r[j]==0 and hist_r[j+1]>0   :
                    if i==1:
                        #seg_y.append(max(10,int(j-3*(j-i)/4)) )
                        seg_y.append(max(5,int(j-max_c_h*1.3)  ))
                        first_seg_y = j
                       
                    else :
                        seg_y.append(int((i+1+j+1)/2))                    
                    i = j+1
                    break;
                elif j >= img_height-3:
                    #seg_y.append(min(img_height-3,int(2*(i+1+j+1)/3)))
                    #seg_y.append( min(img_height-3,i+int(5*( img_height-i )/6)))    #20210806 stop to use
                    gap = 0
                    for m in range(i-1,1,-1):
                        if hist_r[m]>0 and hist_r[m-1]==0:
                            for n in range(m-1,1,-1):
                                if hist_r[n]>0 and hist_r[n+1]==0 :
                                    gap =( m - n )/2  
                                     
                                    break;
                        if gap > 0 : break
                    if gap <= 5:
                        seg_y.append( min(img_height-3,i+max_c_h)) 
                    else :
                         seg_y.append( min(img_height-3,i+gap+2)) 
                         
                         seg_y[0] = max(5,int(first_seg_y-(gap+10))  )
                         
                    i = img_height
                   
                    break;
    for i in range(0,img_width-2) :
        if  i==1 or ((hist_c[i]>0 and hist_c[i+1]==0  )):
            for j in range(i+2, img_width-2):
                if hist_c[j]==0 and hist_c[j+1]>0  :
                    if i==1:
                        #seg_x.append(max(2,int(j-4*(j-i)/5)))   ##??????
                        #seg_x.append(2)
                        seg_x.append(max(2,int(j-max_c_w*1.1 )))
                    else: 
                        seg_x.append(int((i+1+j+1)/2))
                    
                    i = j+1
                    break;
                elif j >= img_width-3:
                    #seg_x.append(min(img_width-2,int(2*(i+1+j+1)/3)))
                    #seg_x.append( min(img_width-3,i+int(5*( img_width-i )/6))) 
                    seg_x.append( min(img_width-3,i+int(max_c_w*1.1 )))
                    i = img_width 
                    break;   
    ###set the sorting label number using overlapping
    print("num",num_labels)
    # if num_labels == 4:   
    #     newlabel=1
    # elif num_labels <= 10:        
    #     newlabel=1
    # else:
    #     newlabel= 0 
    newlabel= 1
    changeflag=0
    icnt=0
    start_seg_x = 0
    
    for i in range(0, len(seg_y)-1):   ## Check how many circles are in the first column to decide to start newlabel
        for m in range(1, num_labels):
                x0, y0 = centroids[m]
                if x0>seg_x[0] and x0< seg_x[0+1] and y0>seg_y[i] and y0 < seg_y[i+1]:
                    icnt+=1
  
    if  icnt==1 : start_seg_x = 1
 
    for j in range(start_seg_x, len(seg_x)-1):
        for i in range(0, len(seg_y)-1 ):
            changeflag = 0
            for m in range(1, num_labels):
                x0, y0 = centroids[m]
                if x0>seg_x[j] and x0< seg_x[j+1] and y0>seg_y[i] and y0 < seg_y[i+1]:
     
                    newstates[m][SEG_NEW_LABEL] = newlabel           #modify  
                    newstates[m][SEG_LEFT] = int(seg_x[j])
                    newstates[m][SEG_HEIGHT] = int(seg_y[i+1]-seg_y[i])
                    newstates[m][SEG_TOP] = int(seg_y[i])
                    newstates[m][SEG_WIDTH] = int(seg_x[j+1] - seg_x[j])
                    
                    #cv2.rectangle(crop_img, (int(newstates[m][SEG_LEFT]),int(newstates[m][SEG_TOP])), (int(newstates[m][SEG_WIDTH]+newstates[m][SEG_LEFT]),int( newstates[m][SEG_HEIGHT]+newstates[m][SEG_TOP] )), (128, 128, 128), 1) 
                    #cv2.putText(crop_img, str(newlabel),(int(x0),int(y0)), cv2.FONT_HERSHEY_SIMPLEX,  1, (128, 128, 128), 1, cv2.LINE_AA)
                    newlabel += 1
                    changeflag = 1
                    break;
 
               
    rows, cols = (num_labels, len( newstates[1]))
    sort_label= [[0]*cols]*rows  #np.zeros(num_labels,len( newstates[1]))               
    for i in range(1,  newlabel):
        for m in range(1, num_labels):
            if  newstates[m][SEG_NEW_LABEL] == i:
                #print(i,m)
                #for j in range(0,len( newstates[1])):
                sort_label[i] = newstates[m]
                break;
    
    ####exchange data
    for m in range(1, num_labels):
        #for j in range(0,len( newstates[1])):
            newstates[m]  =  sort_label[m]   
        #print(newstates[m][SEG_NEW_LABEL])
    #print( newstates)
    
    if show_results == 1 :
        for m in range(1, num_labels):   
            #x0, y0 = centroids[m]
            cv2.rectangle(crop_img, (int(newstates[m][SEG_LEFT]),int(newstates[m][SEG_TOP])), (int(newstates[m][SEG_WIDTH]+newstates[m][SEG_LEFT]),int(newstates[m][SEG_HEIGHT]+newstates[m][SEG_TOP])), (128, 128, 128), 1)
            cv2.putText(crop_img, str(newstates[m][SEG_NEW_LABEL]  ),(int(newstates[m][SEG_LEFT]),int(newstates[m][SEG_TOP]+newstates[m][SEG_HEIGHT]/2)), cv2.FONT_HERSHEY_SIMPLEX,  1, (128, 128, 128), 1, cv2.LINE_AA)
        cv2.namedWindow('seg_result', cv2.WINDOW_NORMAL) 
        cv2.imshow('seg_result', crop_img)
    sort_label.clear  
    if save_files ==  1:
        if show_results == 0 :
            for m in range(1, num_labels):   
                #x0, y0 = centroids[m]
                cv2.rectangle(crop_img, (int(newstates[m][SEG_LEFT]),int(newstates[m][SEG_TOP])), (int(newstates[m][SEG_WIDTH]+newstates[m][SEG_LEFT]),int(newstates[m][SEG_HEIGHT]+newstates[m][SEG_TOP])), (128, 128, 128), 1)
                cv2.putText(crop_img, str(newstates[m][SEG_NEW_LABEL]  ),(int(newstates[m][SEG_LEFT]),int(newstates[m][SEG_TOP]+newstates[m][SEG_HEIGHT]/2)), cv2.FONT_HERSHEY_SIMPLEX,  1, (128, 128, 128), 1, cv2.LINE_AA)
        filename= save_folder + "segment.jpg"
         
        cv2.imwrite(filename,  crop_img) 
  
def image_analy_folder(image_fname,cir_no):
    filenamelist = []
    imagelist=[]
    newfilenamelist = []
    for file_name in os.listdir( image_folder):
        
        if not(file_name.split('.')[-1] == 'jpg' or file_name.split('.')[-1] == 'bmp' or file_name.split('.')[-1] == 'JPG' or file_name.split('.')[-1] == 'BMP'):
            continue
        fname=image_folder + "/" + file_name
       
        filenamelist.append(fname) 
        
    if len(filenamelist)< 1 : return
    for i in range(0,len(filenamelist)):
 
        img = cv2.imread(filenamelist[i])
        img_gray =  cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ori_img_height, ori_img_width= img_gray.shape
        
        x = 0#10
        y = 0#10
        
        # 裁切區域的長度與寬度
        w = ori_img_width #5800
        h = ori_img_height
        # if ori_img_width<=4000:
        #     h =2000 
        # else: 
        #     h =5000
        
        # 裁切圖片
        crop_img = img_gray[y:y+h, x:x+w]
        resize_image = cv2.resize(crop_img, None, fx=1./Ratio_x , fy=1./Ratio_y , interpolation=cv2.INTER_AREA)
   
        image_segment(resize_image )    
     
        for j in range(1,len(newstates)  ):
          
            if (int(cir_no) > 0 and int(cir_no) == j) or cir_no==0 : 
                if newstates[j][SEG_WIDTH]<1 or newstates[j][SEG_HEIGHT]<1:             
                    continue
                
                crop_object = crop_img[newstates[j][SEG_TOP]*Ratio_y:(newstates[j][SEG_TOP]+newstates[j][SEG_HEIGHT])*Ratio_y,\
                        newstates[j][SEG_LEFT]*Ratio_x:(newstates[j][SEG_LEFT]+newstates[j][SEG_WIDTH])*Ratio_x]
                newfilenamelist.append(os.path.basename(filenamelist[i]).split('.')[0]+"_"+str(j))
                imagelist.append(crop_object)
        del img, img_gray,crop_img,resize_image
        newstates.clear()
        gc.collect()
        
    filenum = len(imagelist)
    
    acc_count=[]
    acc_label=[]
    acc_label.append(0) # skip index 0
    acc_count.append(0)  # skip index 0
    if filenum < 1  :
        tkinter.messagebox.showerror('Error','Cannot find circle')
        return
    
    cir_avg_all=[]
    cir_std_all=[]
    slice_cir_avg_all = []
    slice_cir_std_all = []   
    for i in range(0,filenum+1):
        cir_avg_all.append([])
        cir_std_all.append([])
        slice_cir_avg_all.append([])
        slice_cir_std_all.append([])
     
    for i in range(1,len(imagelist)+1)   :
         
            
            cir_avg,cir_std,slice_cir_avg,slice_cir_std = cal_cir_hybrid_avg(imagelist[i-1] ,i)
            cir_avg_all[i] =  cir_avg 
            cir_std_all[i] = cir_std
            slice_cir_avg_all[i] = slice_cir_avg[:]
            slice_cir_std_all[i] = slice_cir_std[:]
  
    if 1  :
        fig, ax = plt.subplots()
        
        maker_s=['o','*','^','8','s','v','^','>','<','1','2','3','4']
        avg_data = []
        cl_data = []
        minlen=len(cir_avg_all[1])   ## index 0 is empty
        sminlen=len(slice_cir_avg_all[1]) ## index 0 is empty
        for i in range(1,filenum+1 ):
            minlen=min(minlen,len(cir_avg_all[i]))
            sminlen=min(sminlen,len(slice_cir_avg_all[i]))
      
        lx = 0
        data_len=int(3*minlen/4)
        timestr = time.strftime("%Y%m%d-%H%M")
        csvfile=image_folder+ "/"+timestr+".csv"
        with open(csvfile,'w',newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['p1','p2','SAE','RMSE',"RMSE_STD","SSAE","SRMSE","nMSE","SSIMSE"])
            for k in range (1, filenum+1  ):
              
                r = random.random()
                b = random.random()
                g = random.random()
                line_color = (r, g, b)
                labelname= newfilenamelist[k-1]  ## index of the newfilenamelist from 0 to k-1
                ax.plot(range(0, minlen ,1  ),cir_avg_all[k][0:minlen] ,marker=maker_s[1], markersize=1,color=line_color, label=labelname)   
               
                 
                #plt.plot(range(0, len(cir_avg_all[k]),1  ),cir_std_all[k] ,marker=maker_s[k], markersize=1,color=line_color, label=str(k))   
                #print(k,":",scipy.stats.shapiro(cir_avg_all[k]))
                # if k == filenum/2 - 1 : 
                #     ax.text(data_len, int(100+(lx)*6 ), "H test : p>0.05"  , ha='center', va='bottom', fontsize=10)
                        
                for q in range(k+1,  filenum+1 ):
                    #if k >= q: break
                    SAE=np.round(sum(abs(x - y) for x, y in zip(sorted(cir_avg_all[k][0: minlen]), sorted(cir_avg_all[q][0: minlen]))),4)
                    rmse = np.sqrt(mean_squared_error(cir_avg_all[k][0: minlen],cir_avg_all[q][0: minlen]))
                    rmsestd = np.sqrt(mean_squared_error(cir_std_all[k][0: minlen],cir_std_all[q][0: minlen]))
                    SSAE = np.round(sum(abs(x - y) for x, y in zip(sorted(slice_cir_avg_all[k][0: sminlen]), sorted(slice_cir_avg_all[q][0: sminlen]))),4)
                    SRMSE =  np.sqrt(mean_squared_error(slice_cir_avg_all[k][0: sminlen],slice_cir_avg_all[q][0: sminlen]))
                    nMSE,ssim=compute_newMSE(slice_cir_avg_all[k][0: sminlen],slice_cir_avg_all[q][0: sminlen],slice_cir_std_all[k][0: sminlen],slice_cir_std_all[q][0: sminlen],minlen,sminlen,k,q)
                    # h,per=scipy.stats.kruskal( cir_avg_all[k],cir_avg_all[q])
                    # X=np.vstack([cir_avg_all[k][0: minlen],cir_avg_all[q][0: minlen]])
                    # d2=np.corrcoef(X)[0][1]
                    # d4=scipy.stats.spearmanr(cir_avg_all[k][0: minlen],cir_avg_all[q][0: minlen])[0]
                    # d5=scipy.stats.kendalltau(cir_avg_all[k][0: minlen],cir_avg_all[q][0: minlen])[0]
                    txt_color = 'b'
                    # if per< 0.05:
                    #     txt_color = 'r'
                    #show_text =  "SAE " +str(k)+"-"+str(q)+":"+str(SAE)+", RMSE:"+ str(np.round(rmse,3)) +", rmsestd:" +str(np.round(rmsestd,4))  
                    #show_text =  "SAE " +str(k)+"-"+str(q)+":"+str(SAE)+", RMSE:"+ str(np.round(rmse,3))+", rmsestd:" +str(np.round(rmsestd,4))+", SSAE="+str(np.round(SSAE,4))+", SRMSE="+str(np.round(SRMSE,4))+", nMSE="+str(np.round(nMSE,4))
                    show_text =  "SAE " +str(k)+"-"+str(q)+":"+str(SAE)+", RMSE:"  +str(np.round(rmse,4))+", sSAE="+str(np.round(SSAE,4))+", sRMSE="+str(np.round(SRMSE,4))+", nmse="+str(np.round(nMSE,4))+", ssim="+str(np.round(ssim,4))
                    writer.writerow([newfilenamelist[k-1],newfilenamelist[q-1], SAE ,rmse,rmsestd,SSAE,SRMSE,nMSE,ssim])  
                    print(show_text)
                    #ax.text(data_len, int(100+lx*6), show_text , color = txt_color, ha='center', va='bottom', fontsize=10)
                    lx +=1
                # for j in range(0,minlen) :           
                #     avg_data.append(cir_avg_all[k][j])
                #     cl_data.append("C"+str(k))
        r = random.random()
        b = random.random()
        g = random.random()
        line_color = (r, g, b)    
        ax.legend()
        
        title_plot= file_name #+":GaussianBlur3x3" #+':ATP 18000'
        ax.set_title( title_plot)
        ax.xaxis.set_label_text("circle no. (from inside to outside (x radius + "+str(thickness_num)+" ) )")
        ax.yaxis.set_label_text("avg gray value ")
        
        plt.show()  
        if save_files == 1:
            filename= save_folder + "circle_no_avg_gray.jpg"
        
            fig.savefig(filename )
    filenamelist.clear()
    imagelist.clear()
    newfilenamelist.clear()
 
    tkinter.messagebox.showinfo('Info','All done')

    #cv2.waitKey(0)
    cv2.destroyAllWindows()   
    newstates.clear()
    del  cir_avg_all,cir_std_all,slice_cir_avg_all,slice_cir_std_all
 
    return 1
def image_analy_files(image_fname,cir_no):
     
    filenum = 4
    acc_count=[]
    acc_label=[]
    acc_label.append(0) # skip index 0
    acc_count.append(0)  # skip index 0
    if int(cir_no) > filenum  :
        tkinter.messagebox.showerror('Error','Specified circle does not exist')
        return
    
    cir_avg_all=[]
    cir_std_all=[]
    slice_cir_avg_all = []
    slice_cir_std_all = [] 
    for i in range(0,filenum+1):
        cir_avg_all.append([])
        cir_std_all.append([])
        slice_cir_avg_all.append([])
        slice_cir_std_all.append([])
     
    for i in range(1,filenum+1)   :
      
            if i<=4:
                filenames=foldername+str(i)+".bmp"
                readimg=cv2.imread(filenames)
                crop_object =cv2.cvtColor(readimg, cv2.COLOR_BGR2GRAY)
            # elif  i<=7:
            #     filenames=foldername+str(i-4)+".bmp"
            #     readimg=cv2.imread(filenames)
            #     crimg =cv2.cvtColor(readimg, cv2.COLOR_BGR2GRAY)
            #     crop_object = cv2.flip(crimg,1)
            # elif i<=11:
            #     filenames=foldername+str(i-8)+".bmp"
            #     readimg=cv2.imread(filenames)
            #     crimg =cv2.cvtColor(readimg, cv2.COLOR_BGR2GRAY)
            #     crop_object = cv2.flip(crimg,0)
            # elif  i<=15:
            #     filenames=foldername+str(i-12)+".bmp"
            #     readimg=cv2.imread(filenames)
            #     crimg =cv2.cvtColor(readimg, cv2.COLOR_BGR2GRAY)
            #     crop_object = cv2.flip(crimg,-1)
            # elif i<=19:
            #     filenames=foldername+str(i-16)+".bmp"
            #     readimg=cv2.imread(filenames)
            #     crimg =cv2.cvtColor(readimg, cv2.COLOR_BGR2GRAY)
            #     crop_object = cv2.rotate(crimg, cv2.ROTATE_90_CLOCKWISE)
            # else:
            #     filenames=foldername+str(i-20)+".bmp"
            #     readimg=cv2.imread(filenames)
            #     crimg =cv2.cvtColor(readimg, cv2.COLOR_BGR2GRAY)
            #     crop_object = cv2.rotate(crimg, cv2.ROTATE_180)
            
            
            cir_avg,cir_std,slice_cir_avg,slice_cir_std = cal_cir_hybrid_avg(crop_object ,i)
            cir_avg_all[i] =  cir_avg 
            cir_std_all[i] = cir_std
            slice_cir_avg_all[i] = slice_cir_avg[:]
            slice_cir_std_all[i] = slice_cir_std[:]
            
            r = random.random()
            b = random.random()
            g = random.random()
            line_color = (r, g, b)
  
            #circle_info.clear()
            del crop_object   ,readimg , cir_avg,cir_std,slice_cir_avg,slice_cir_std
 
   
 
    if 1  :
        fig, ax = plt.subplots()
        
        maker_s=['o','*','^','8','s','v','^','>','<','1','2','3','4']
        avg_data = []
        cl_data = []
        minlen=len(cir_avg_all[1]) 
        sminlen=len(slice_cir_avg_all[1]) 
        for i in range(1,filenum+1 ):
            minlen=min(minlen,len(cir_avg_all[i]))
            sminlen=min(sminlen,len(slice_cir_avg_all[i]))
      
        lx = 0
        data_len=int(3*minlen/4)
        for k in range (1, filenum+1  ):
          
            r = random.random()
            b = random.random()
            g = random.random()
            line_color = (r, g, b)
            if k <= 4:
                ax.plot(range(0, minlen ,1  ),cir_avg_all[k][0:minlen] ,marker=maker_s[1], markersize=1,color=line_color, label="li_18000_same_2_"+str(k))   
            elif k> 4 and k<=7:
                ax.plot(range(0, minlen ,1  ),cir_avg_all[k][0:minlen] ,marker=maker_s[1], markersize=1,color=line_color, label="li_18000_same_2MH_"+str(k)) 
            elif k> 8 and k<=11:
                ax.plot(range(0, minlen ,1  ),cir_avg_all[k][0:minlen] ,marker=maker_s[1], markersize=1,color=line_color, label="li_18000_same_2MV_"+str(k)) 
            elif k> 12 and k<=15:
                ax.plot(range(0, minlen ,1  ),cir_avg_all[k][0:minlen] ,marker=maker_s[1], markersize=1,color=line_color, label="li_18000_same_2MHV_"+str(k)) 
            elif k> 16 and k<=19:
                ax.plot(range(0, minlen ,1  ),cir_avg_all[k][0:minlen] ,marker=maker_s[1], markersize=1,color=line_color, label="li_18000_same_2R90_"+str(k)) 
            elif k> 20 and k<=23:
                ax.plot(range(0, minlen ,1  ),cir_avg_all[k][0:minlen] ,marker=maker_s[1], markersize=1,color=line_color, label="li_18000_same_2R180_"+str(k))
            
            for q in range(k+1,  filenum+1 ):
                #if k >= q: break
                SAE=np.round(sum(abs(x - y) for x, y in zip(sorted(cir_avg_all[k][0: minlen]), sorted(cir_avg_all[q][0: minlen]))),4)
                rmse = np.sqrt(mean_squared_error(cir_avg_all[k][0: minlen],cir_avg_all[q][0: minlen]))
                rmsestd = np.sqrt(mean_squared_error(cir_std_all[k][0: minlen],cir_std_all[q][0: minlen]))
                SSAE = np.round(sum(abs(x - y) for x, y in zip(sorted(slice_cir_avg_all[k][0: sminlen]), sorted(slice_cir_avg_all[q][0: sminlen]))),4)
                SRMSE =  np.sqrt(mean_squared_error(slice_cir_avg_all[k][0: sminlen],slice_cir_avg_all[q][0: sminlen]))
                nMSE,ssim=compute_newMSE(slice_cir_avg_all[k][0: sminlen],slice_cir_avg_all[q][0: sminlen],slice_cir_std_all[k][0: sminlen],slice_cir_std_all[q][0: sminlen],minlen,sminlen,k,q)
                # h,per=scipy.stats.kruskal( cir_avg_all[k],cir_avg_all[q])
                # X=np.vstack([cir_avg_all[k][0: minlen],cir_avg_all[q][0: minlen]])
                # d2=np.corrcoef(X)[0][1]
                # d4=scipy.stats.spearmanr(cir_avg_all[k][0: minlen],cir_avg_all[q][0: minlen])[0]
                # d5=scipy.stats.kendalltau(cir_avg_all[k][0: minlen],cir_avg_all[q][0: minlen])[0]
                txt_color = 'b'
                # if per< 0.05:
                #     txt_color = 'r'
                #show_text =  "SAE " +str(k)+"-"+str(q)+":"+str(SAE)+", RMSE:"+ str(np.round(rmse,3))+", k:" +str(np.round(d5,4)) + ", p:" +str( d2 )+", s:" +str(np.round(d4,4))
                #show_text =  "SAE " +str(k)+"-"+str(q)+":"+str(SAE)+", RMSE:"+ str(np.round(rmse,3)) +", rmsestd:" +str(np.round(rmsestd,4)) 
                #show_text =  "SAE " +str(k)+"-"+str(q)+":"+str(SAE)+", RMSE:"+ str(np.round(rmse,3))+", rmsestd:" +str(np.round(rmsestd,4))+", SSAE="+str(np.round(SSAE,4))+", SRMSE="+str(np.round(SRMSE,4))+", nMSE="+str(np.round(nMSE,4))
                show_text =  "SAE " +str(k)+"-"+str(q)+":"+str(SAE)+", RMSE:"  +str(np.round(rmse,4))+", sSAE="+str(np.round(SSAE,4))+", sRMSE="+str(np.round(SRMSE,4))+", nmse="+str(np.round(nMSE,4))+", ssim="+str(np.round(ssim,4))
                print(show_text)
                #ax.text(data_len, int(100+lx*6), show_text , color = txt_color, ha='center', va='bottom', fontsize=10)
                lx +=1
            # for j in range(0,minlen) :           
            #     avg_data.append(cir_avg_all[k][j])
            #     cl_data.append("C"+str(k))
        r = random.random()
        b = random.random()
        g = random.random()
        line_color = (r, g, b)    
        ax.legend()
        
        title_plot= file_name #+":GaussianBlur3x3" #+':ATP 18000'
        ax.set_title( title_plot)
        ax.xaxis.set_label_text("circle no. (from inside to outside (x radius + "+str(thickness_num)+" ) )")
        ax.yaxis.set_label_text("avg gray value ")
        
        plt.show()  
        if save_files == 1:
            filename= save_folder + "circle_no_avg_gray.jpg"
        
            fig.savefig(filename )
    
    tkinter.messagebox.showinfo('Info','All done')

    #cv2.waitKey(0)
    cv2.destroyAllWindows()   
   
    newstates.clear()
    del  cir_avg_all,cir_std_all,slice_cir_avg_all,slice_cir_std_all
    return 1
def image_analy(image_fname,cir_no):
     
    img = cv2.imread(image_fname)
    # fimg= cv2.flip(img,1)
    # filename=  image_fname+"_fh"+".bmp"
    # cv2.imwrite(filename,   fimg) 
    # return
    img_gray =  cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ori_img_height, ori_img_width= img_gray.shape
    
    # 裁切區域的 x 與 y 座標（左上角）
    x = 0#10
    y = 0#10
    
    # 裁切區域的長度與寬度
    w = ori_img_width #5800
    h = ori_img_height
    # if ori_img_width<=4000:
    #     h =2000 
    # else: 
    #     h =5000
   
    # 裁切圖片
    crop_img = img_gray.copy()#[y:y+h, x:x+w]
    resize_image = cv2.resize(crop_img, None, fx=1./Ratio_x , fy=1./Ratio_y , interpolation=cv2.INTER_AREA)
   
    image_segment(resize_image )
   
    acc_count=[]
    acc_label=[]
    acc_label.append(0) # skip index 0
    acc_count.append(0)  # skip index 0
    if int(cir_no) > len(newstates)  :
        tkinter.messagebox.showerror('Error','Specified circle does not exist')
        return
    
    cir_avg_all=[]
    cir_std_all=[]
    slice_cir_avg_all = []
    slice_cir_std_all = [] 
    for i in range(len(newstates)+1):
        cir_avg_all.append([])
        cir_std_all.append([])
        slice_cir_avg_all.append([])
        slice_cir_std_all.append([])
    print(len(newstates))
    circle_img_cnt=0
    for i in range(1,len(newstates)  ):
        if newstates[i][SEG_WIDTH]<1 or newstates[i][SEG_HEIGHT]<1:             
            continue
        
        if (int(cir_no) > 0 and int(cir_no) == i) or cir_no==0 : 
       
            crop_object = crop_img[newstates[i][SEG_TOP]*Ratio_y:(newstates[i][SEG_TOP]+newstates[i][SEG_HEIGHT])*Ratio_y,\
                                    newstates[i][SEG_LEFT]*Ratio_x:(newstates[i][SEG_LEFT]+newstates[i][SEG_WIDTH])*Ratio_x]
            if crop_object.shape[0] <100 or crop_object.shape[1] <100 :
                continue
            # filename= save_folder + str(i+8)+".bmp"
            # cv2.imwrite(filename,   crop_object) 
            # continue
            cir_avg,cir_std,slice_cir_avg,slice_cir_std = cal_cir_hybrid_avg(crop_object ,i)
          
            cir_avg_all[i] = cir_avg
            cir_std_all[i] = cir_std
            slice_cir_avg_all[i] = slice_cir_avg[:]
            slice_cir_std_all[i] = slice_cir_std[:]
            out_cir_area_data=[]
            circle_img_cnt += 1
 
 
 
            del crop_object,cir_avg,cir_std,slice_cir_avg,slice_cir_std  
 
            gc.collect()
    
 
    if 1  :#and len(newstates)-1 == 4 and cir_no==0:
        fig, ax = plt.subplots()
        
        maker_s=['o','*','^','8','s','v','^']
        avg_data = []
        cl_data = []
        minlen=len(cir_avg_all[1]) 
        sminlen=len(slice_cir_avg_all[1]) 
        for i in range(1,circle_img_cnt+1 ):
            minlen=min(minlen,len(cir_avg_all[i]))
            sminlen=min(sminlen,len(slice_cir_avg_all[i]))
        print(circle_img_cnt,minlen)
       
        lx = 0
        data_len=int(3*minlen/4)
        for k in range (1, circle_img_cnt + 1 ):
          
            r = random.random()
            b = random.random()
            g = random.random()
            line_color = (r, g, b)
            ax.plot(range(0, minlen ,1  ),cir_avg_all[k][0:minlen] ,marker=maker_s[k], markersize=1,color=line_color, label=str(k))   
            
            for q in range(k+1,   circle_img_cnt + 1   ):
                SAE=np.round(sum(abs(x - y) for x, y in zip(sorted(cir_avg_all[k][0: minlen]), sorted(cir_avg_all[q][0: minlen]))),4)
                rmse = np.sqrt(mean_squared_error(cir_avg_all[k][0: minlen],cir_avg_all[q][0: minlen]))
                rmsestd=np.sqrt(mean_squared_error(cir_std_all[k][0: minlen],cir_std_all[q][0: minlen]))
                psnr = 20*math.log10(255/rmse)
                SSAE = np.round(sum(abs(x - y) for x, y in zip(sorted(slice_cir_avg_all[k][0: sminlen]), sorted(slice_cir_avg_all[q][0: sminlen]))),4)
                SRMSE =  np.sqrt(mean_squared_error(slice_cir_avg_all[k][0: sminlen],slice_cir_avg_all[q][0: sminlen]))
                nMSE,ssim=compute_newMSE(slice_cir_avg_all[k][0: sminlen],slice_cir_avg_all[q][0: sminlen],slice_cir_std_all[k][0: sminlen],slice_cir_std_all[q][0: sminlen],minlen,sminlen,k,q)
                txt_color = 'b'
                
                # h,per=scipy.stats.kruskal( cir_avg_all[k],cir_avg_all[q])
                # X=np.vstack([cir_avg_all[k][0: minlen],cir_avg_all[q][0: minlen]])  ## 1. pearsonr
                # d2=np.corrcoef(X)[0][1]
                # d3=scipy.stats.pearsonr(cir_avg_all[k][0: minlen],cir_avg_all[q][0: minlen])[0]  # 2.pearsonr
                # d4=scipy.stats.spearmanr(cir_avg_all[k][0: minlen],cir_avg_all[q][0: minlen])[0]
                # d5=scipy.stats.kendalltau(cir_avg_all[k][0: minlen],cir_avg_all[q][0: minlen])[0]           
                                           
                # if per< 0.05:
                #     txt_color = 'r'
                #show_text =  "SAE " +str(k)+"-"+str(q)+":"+str(SAE)+", RMSE:"+ str(np.round(rmse,3))+", rmsestd:" +str(np.round(rmsestd,4))+", SSAE="+str(np.round(SSAE,4))+", SRMSE="+str(np.round(SRMSE,4))+", nMSE="+str(np.round(nMSE,4))
                show_text =  "SAE " +str(k)+"-"+str(q)+":"+str(SAE)+", RMSE:"  +str(np.round(rmse,4))+", sSAE="+str(np.round(SSAE,4))+", sRMSE="+str(np.round(SRMSE,4))+", nmse="+str(np.round(nMSE,4))+", ssim="+str(np.round(ssim,4))
                ax.text(data_len, int(100+lx*6), show_text , color = txt_color, ha='center', va='bottom', fontsize=10)
                lx +=1
            
        r = random.random()
        b = random.random()
        g = random.random()
        line_color = (r, g, b)    
        ax.legend()
         
        title_plot= file_name #+":GaussianBlur3x3" #+':ATP 18000'
        ax.set_title( title_plot)
        ax.xaxis.set_label_text("circle no. (from inside to outside (x radius + "+str(thickness_num)+" ) )")
        ax.yaxis.set_label_text("avg gray value ")
        
        plt.show()  
        if save_files == 1:
            filename= save_folder + "circle_no_avg_gray.jpg"
        
            fig.savefig(filename )
    
    tkinter.messagebox.showinfo('Info','All done')

    #cv2.waitKey(0)
    cv2.destroyAllWindows()   
    newstates.clear()
    del img,img_gray,crop_img,resize_image,cir_avg_all,cir_std_all,slice_cir_avg_all,slice_cir_std_all
    gc.collect()
    return 1
def menu_fun():
 
    layout = [   [sg.Frame('Source : ALL / Single',key='-mainframe-',layout=[
                 [sg.Radio('',"All_IMAGE",key='-All_image-',default=False,size=(0, 1), enable_events=True,pad=(0,0)),\
                 sg.Text('Please enter image folder',size=(20,1),auto_size_text=False,justification='left') ,\
                 sg.InputText('d:/',key='-LOAD_SOU-', size=(50, 1),pad=(0,0)),sg.FolderBrowse()] ,
                [sg.Radio('',"All_IMAGE",default=True,key='-single_image-',size=(0, 1), enable_events=True,pad=(0,0)) ,\
                 sg.Text('Select your image file',size=(20,1),auto_size_text=False,justification='left') ,\
                 sg.Input(key='-imgfile-', size=(50, 1),pad=(0,0)  ),sg.FileBrowse( file_types=(("jpg Files", "*.*"),))] 
                 ])] ,  
                [sg.Frame('Parameter',key='-paraframe-',layout=[
                [sg.Radio('All Circle',"IMAGE_NO",key='-All_circle-',default=False,size=(10, 1), enable_events=True,pad=(0,0)),\
                 sg.Radio('Specify Circle',"IMAGE_NO",default=False,key='-Specify_circle-',size=(10, 1), enable_events=True,pad=(0,0)) ,\
                 sg.InputText('',size=(4, 1),text_color='black', background_color='white',key='-in_circle_no-' , pad=(0, 0),enable_events=False ),\
                     sg.T("  "),sg.Text('Slice num(1~360)',size=(12,1),auto_size_text=False,justification='left', pad=(0, 0)),\
                     sg.InputText('20',size=(5, 1),text_color='black', background_color='white',key='-in_slice_num-'  ,enable_events=False  ),\
                     sg.Text('Thickness num(1~100)',size=(18,1),auto_size_text=False,justification='left', pad=(0, 0)),\
                     sg.InputText('10',size=(5, 1),text_color='black', background_color='white',key='-in_thickness_num-' , pad=(0, 0),enable_events=False  )
                     ], \
                 [   sg.Checkbox('Save result',key='-chk_save-', default=False,size=(10, 1)  ), \
                     sg.Checkbox('Show result',key='-chk_show-', default=False,size=(10, 1)  ),\
                     sg.Checkbox('Show to csv',key='-csv_save-', default=False,size=(10, 1)  ),
                     sg.Checkbox('Equal increasing rate',key='-chk_xy_equal-', default=False,size=(20, 1)) ]  
                 ])] ,  
                [sg.Frame('Result',key='-resultframe-',layout=[
                [sg.Text('Please enter the save destination folder',size=(30,1),auto_size_text=False,justification='left',pad=(0,0)) ,\
                 sg.InputText('d:/', key='-SAVE_DST-',justification='left', size=(40, 1),pad=(0,0)),sg.FolderBrowse()] \
                ])] , 
                [sg.Button('Submit'), sg.Button('Cancel')]  
               
             ]
    # Create the Window
    window = sg.Window('noise analysis tool', layout, element_justification='l')
 
    while True:
        event, values   = window.read()
        if event == sg.WIN_CLOSED or event == 'Cancel':	# if user closes window or clicks Close
             break
        image_fname = values["-imgfile-"]
        global save_folder,show_results,save_files,slice_num,thickness_num,x_equal_y,file_name,foldername,image_folder,save_to_csv
        foldername = os.path.split(image_fname)[0] +"/"
        save_folder = values['-SAVE_DST-'] + "/"+ os.path.basename(image_fname).split('.')[0]+"_"
        image_folder =  values['-LOAD_SOU-']
        file_name = os.path.basename(image_fname).split('.')[0]
        show_results = 0
        save_files = 0
        slice_num = 1
        thickness_num = 1
        save_to_csv = 0
        x_equal_y = 0
        func=0
        if event == 'Submit':
             cir_no = -1
             err_flag=0
             if values['-All_image-']:
                 func=1
                 cir_no = 0
             
             if values['-Specify_circle-']:
                cir_no=values['-in_circle_no-']
                if not cir_no.isdigit  :                        
                     
                     err_flag = 1
                     tkinter.messagebox.showerror('Error','Specified circle does not exist')
             elif values['-All_circle-']:
                 cir_no = 0
             if values['-chk_save-'] == True:
                 save_files = 1
             if values['-chk_show-'] == True:
                 show_results = 1
             if values['-csv_save-'] == True:
                 save_to_csv = 1
             if values['-in_slice_num-'] :
                 
                 if not values['-in_slice_num-'].isdigit()   :
                     slice_num =1
                     tkinter.messagebox.showerror('Error','Slice number error')
                 else:
                     slice_num = int(values['-in_slice_num-'] )
                     if  (  slice_num  <=0 or   slice_num>=360 ):
                         tkinter.messagebox.showerror('Error','Slice number invalid')
                         slice_num = 1  
             if values['-in_thickness_num-'] :
                 
                 if not values['-in_thickness_num-'].isdigit()   :
                     thickness_num =1
                     tkinter.messagebox.showerror('Error','Thickness number error')
                 else:
                     thickness_num = int(values['-in_thickness_num-'] )
                     if  (  thickness_num  <=0 or   thickness_num>100 ):
                         tkinter.messagebox.showerror('Error','thickness number invalid')
                         thickness_num = 1   
             if values["-chk_xy_equal-"]:
                 x_equal_y = 1
             if not err_flag and int(cir_no)>=0 :
               
                if func == 1:
                    if os.path.isdir(image_folder):
                        image_analy_folder(image_fname,int(cir_no))
                        
                    else:  print(r"folder does not exist!")
                elif os.path.isfile(image_fname):
                   
                   #image_analy_files(image_fname,int(cir_no))
                   image_analy(image_fname,int(cir_no))                 
                else:
                   print(r"image is not exist!")

    window.close()
if __name__ == "__main__":   
    menu_fun() 
