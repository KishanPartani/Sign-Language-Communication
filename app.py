from pymongo import MongoClient
from flask import Flask, render_template, url_for, request, flash, redirect, jsonify, send_file, session, Response
from flask_cors import CORS, cross_origin
from flask_bcrypt import Bcrypt
from Helpers import text2gest as tg
from Helpers import gest2text as gt
from Helpers import cartoonize as ct
import intervention as it
import os
import sys
from nltk.stem import WordNetLemmatizer
import cv2
import webbrowser
from tensorflow import keras
from csv import writer
import csv

global flag, text
flag = 0

app = Flask(__name__)
bcrypt = Bcrypt(app)
client = MongoClient('localhost', 27017)
db = client.cnt_db
majusers = db.majusers

cors = CORS(app, resources={r"/foo": {"origins": "*"}})
app.config['CORS_HEADERS'] = 'Content-Type'
app.config['SECRET_KEY'] = '14ec258c169f5c19f78385bcc83a51df7444624b2ff90449b4a9832e6fe706a1'

@app.route('/')
def index():
    global letters_img
    # print(letters_img)
    return render_template('index.html',letters_img=letters_img)

@app.route('/login-page')
def show_login_page():
    return render_template('login.html')

@app.route("/login", methods=['GET', 'POST'])
def login():
    print(request.form['user'])
    print(request.form['pwd'])
    user_cred = majusers.find_one({'user': request.form['user']}, {
        'email': 1, 'password': 1})
    print(user_cred)
    if(not bool(user_cred)):
        return render_template('login.html', message="User not found. Please Sign Up!", classm="alert alert-danger")
    if(bcrypt.check_password_hash(user_cred['password'], request.form['pwd'])):
        # create session variable with email
        session['email'] = user_cred['email']
        print(session['email'])
        return render_template('human-intervention.html', email=session['email'])
    else:     
        return render_template('login.html', message="Incorrect password!", classm="alert alert-danger")


@app.route("/signup", methods=['GET', 'POST'])
def signup():
    if(request.form['pwd'] != request.form['rpwd']):
        return render_template('login.html', message="Passwords don't match!", classm="alert alert-danger")

    user = request.form['user']
    password_hash = bcrypt.generate_password_hash(request.form['pwd'], 10)
    user_details = {"user": user,
                    "email": request.form['email'],
                    "password": password_hash}
    empr_id = db.majusers.insert_one(user_details).inserted_id
    print(empr_id)
    
    return render_template('login.html', message="Sign up successful", classm="alert alert-success")

@app.route("/text_gest", methods=['GET', 'POST'])
def text_gest():
    global text
    text = request.form['text']
    print(text)
    text = text.title()
    #print(text)
    word_Lemmatized = WordNetLemmatizer()
    text = word_Lemmatized.lemmatize(text)
    ls = tg.datalist()
    print(ls)
    word = it.similarWords(text)

    if(len(word) != 0):
        video = '../static/cartdata/cart'+word[0]+'.mp4'
        return render_template('index.html', video=video)
    else:
        with open('data.csv', 'a', newline='') as fp:
            writer_object = writer(fp)
            lt = []
            lt.append(text)
            writer_object.writerow(lt)
            fp.close()
        return render_template('index.html', alertt = 'Video not Found !')
    #return Response(tg.gen_frames(cap), mimetype='multipart/x-mixed-replace; boundary=frame')

"""@app.route("/gest_text", methods=['GET', 'POST'])
def gest_text():
    model = gt.model"""

@app.route("/inter", methods=['GET', 'POST'])
def inter():
    global flag
    if 'email' not in session:
        return render_template('login.html')
    rows = []
    with open('data.csv', 'r') as fp:
        csvreader = csv.reader(fp)
        for row in csvreader:
            rows.append(row)
    #print(rows)
    if flag == 1:
        flag = 0
        return render_template('human-intervention.html', rows = rows, alertt='Video Uploaded Successfully !')
    elif flag == 2:
        flag = 0
        return render_template('human-intervention.html', rows = rows, alertt='Concern Deleted Successfully !')
    else:
        return render_template('human-intervention.html', rows = rows)

@app.route("/uploadvideo/<file_name>", methods=['GET', 'POST'])
def download(file_name):
    global flag
    rows = []
    #print('debuggggg')
    #print(file_name)
    temp = []
    temp.append(file_name)
    with open('data.csv', 'r') as fp:
        csvreader = csv.reader(fp)
        for row in csvreader:
            rows.append(row)
    #print(rows)
    rows.remove(temp)
    with open('data.csv', 'w', newline='') as fw:
        writerobj = writer(fw)
        writerobj.writerows(rows)
        fw.close()
    if(request.method == 'POST'):
        file = request.files['f1']
        file.save("./static/newdata/"+ str(file_name)+ ".mp4")
    inf = str(file_name) + '.mp4'
    outf = 'cart'+ str(file_name) + '.mp4'
    #print("debugggggggggggg" ,outf)
    ct.cartoonize(inf, outf, 0, 10)
    flag = 1
    return redirect('/inter')

@app.route("/deleteconc/<file_name>", methods=['GET', 'POST'])
def delete_concern(file_name):
    global flag
    rows = []
    temp = []
    temp.append(file_name)
    with open('data.csv', 'r') as fp:
        csvreader = csv.reader(fp)
        for row in csvreader:
            rows.append(row)
    rows.remove(temp)
    with open('data.csv', 'w', newline='') as fw:
        writerobj = writer(fw)
        writerobj.writerows(rows)
        fw.close()
    flag = 2
    return redirect('/inter')

@app.route("/raiseconc", methods=['GET', 'POST'])
def human_intervention():
    text = request.form['text']
    #print(text)
    if(text == ''):
        return render_template('index.html', alertt = 'Please enter a word...')
    with open('data.csv', 'a', newline='') as fp:
        writer_object = writer(fp)
        lt = []
        lt.append(text)
        writer_object.writerow(lt)
        fp.close()
    return render_template('index.html', alertt = 'Concern raised successfully !')

@app.route("/gest_text_upload", methods=['GET', 'POST'])
def gest_text_upload():
    file = request.files['f2']
    file.save("./static/uploaded/video.mp4")
    gt.convert(1)
    os.remove("./static/uploaded/video.mp4")
    return render_template('index.html')

@app.route("/gest_text_cam", methods=['GET', 'POST'])
def gest_text_cam():
    gt.convert(0)
    return render_template('index.html')

if __name__ == "__main__":
    global letters_img
    letters_img = []
    for i in "abcdefghijklmnopqrstuvwxyz":
        url = "../static/assets/img/letters/" + i + ".jpg"
        letters_img.append(url)
    
    print('Server Started !!')
    app.run(debug=True, use_reloader=False)