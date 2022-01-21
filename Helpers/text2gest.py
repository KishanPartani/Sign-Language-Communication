# from nltk import text
import nltk
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('omw-1.4')
from nltk.util import pr
import pandas as pd
import numpy as np
# from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import os
import cv2
from werkzeug.utils import redirect

def datalist():
    mylist = os.listdir(os.getcwd() + '/static/dataset')
    ls = []
    for x in mylist:
        ls.append(os.path.splitext(x)[0])
    return ls

def cartoonize(img, ds_factor=4, sketch_mode=False):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.medianBlur(img_gray, 7)
    edges = cv2.Laplacian(img_gray, cv2.CV_8U, ksize=5)
    ret, mask = cv2.threshold(edges, 100, 255, cv2.THRESH_BINARY_INV)
    if sketch_mode:
        return cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    img_small = cv2.resize(img, None, fx=1.0/ds_factor, fy=1.0/ds_factor, interpolation=cv2.INTER_AREA)
    num_repetitions = 10
    sigma_color = 5
    sigma_space = 7
    size = 5

    for i in range(num_repetitions):
        img_small = cv2.bilateralFilter(img_small, size, sigma_color, sigma_space)

    img_output = cv2.resize(img_small, None, fx=ds_factor, fy=ds_factor, interpolation=cv2.INTER_LINEAR)
    dst = np.zeros(img_gray.shape)

    dst = cv2.bitwise_and(img_output, img_output, mask=mask)
    return dst

def gen_frames(camera):  
    while True:
        success, frame = camera.read()  # read the camera frame
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', cartoonize(frame, sketch_mode=False))
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result
    return redirect('index.html')


def convert(text):
    print('hello')
    text = text.title()
    #print(text)
    word_Lemmatized = WordNetLemmatizer()
    text = word_Lemmatized.lemmatize(text)
    print(text)
    ls = datalist()
    print(ls)
    if(text == 'Bye'):
        print('Good Bye !!')
        exit
    elif(text in ls):
        temp_ls = os.listdir()
        print(temp_ls)
        cap = cv2.VideoCapture('static/dataset/'+ text + '.mp4')
        if (cap.isOpened()== False): 
            print("Error opening video file")
            exit
        while(cap.isOpened()):
            ret, frame = cap.read()
            if ret == True:
                cv2.imshow('Corresponding Video', cartoonize(frame, sketch_mode=False))
                if cv2.waitKey(50) & 0xFF == ord('q'):
                    break
            else: 
                break
        cap.release()
        cv2.destroyAllWindows()
    else:
        print("Sorry, currently we don't have the word in our back end !")

"""if __name__ == '__main__':
    print('Enter text: ')
    text = input()
    convert(text)"""
