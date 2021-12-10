import cv2
import numpy as np
import sys
from tqdm import tqdm
import os

SPEEDUP = 0.75

def cartoonize(video_in, video_out, start_sec=0, end_sec=10):
    print(video_in)
    cap = cv2.VideoCapture('./static/newdata/'+ video_in)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start_frame, end_frame = start_sec*fps, end_sec*fps

    #min_y,max_y = int(h/4), h
    #min_x,max_x = 0, int(w*3/4)
    
    min_y,max_y = 0, h
    min_x,max_x = 0, w
    out_h = max_y - min_y
    out_w = max_x - min_x

    writer = cv2.VideoWriter('./static/cartdata/'+video_out, cv2.VideoWriter_fourcc(*'WEBM'), SPEEDUP*fps, (out_w, out_h))
    for i in tqdm(range(length)):
        ret, img = cap.read()
    
        if start_frame <= i <= end_frame: 
            img = img[min_y:max_y, min_x:max_x]
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray = cv2.medianBlur(gray, 5)
            edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)
            color = cv2.bilateralFilter(img, 5, 5, 7)
            cartoon = cv2.bitwise_and(color, color, mask=edges)
            writer.write(cartoon)
    writer.release()
    cap.release()

def datalist():
    mylist = os.listdir(os.getcwd() + '/static/dataset')
    ls = []
    for x in mylist:
        ls.append(os.path.splitext(x)[0])
    return ls

"""
if __name__ == '__main__':
    ls = datalist()
    print(ls)
    for i in ls:
        video_in = i + '.mp4'
        video_out = 'cart'+ i + '.mp4'
        cartoonize(video_in, video_out, 0, 10)
    video_in = 'Teach.mp4'
    video_out = 'cartTeach.mp4'
    cartoonize(video_in, video_out, 0, 10)"""