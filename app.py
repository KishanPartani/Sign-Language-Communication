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

global flag, text, check
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
    if 'email' in session:
        check = 1
        return render_template('index.html', letters_img=letters_img, check=check, email=session['email'])
    else:
        check = 0
    # print(letters_img)
        return render_template('index.html', letters_img=letters_img, check=check)


@app.route('/login-page')
def show_login_page():
    return render_template('login.html')


@app.route("/login", methods=['GET', 'POST'])
def login():
    global check
    check = 0
    rows = []
    with open('data.csv', 'r') as fp:
        csvreader = csv.reader(fp)
        for row in csvreader:
            rows.append(row)
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
        return render_template('human-intervention.html', rows=rows, check=1, email=session['email'])
    else:
        check = 0
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
    global text, check
    if 'email' in session:
        check = 1
        email = session['email']
    else:
        check = 0
        email = ''
    text = request.form['text']
    print(text)
    text = text.title()
    # print(text)
    word_Lemmatized = WordNetLemmatizer()
    text = word_Lemmatized.lemmatize(text)
    ls = tg.datalist()
    print(ls)
    word = it.similarWords(text)
    print("word = ", word)
    if(len(word) != 0):
        video = '../static/cartdata/cart'+word[0]+'.mp4'
        return render_template('index.html', scroll='text2sign', check=check, email=email, letters_img=letters_img, video=video)
    else:
        with open('data.csv', 'a', newline='') as fp:
            writer_object = writer(fp)
            lt = []
            lt.append(text)
            writer_object.writerow(lt)
            fp.close()
        return render_template('index.html', check=check, email=email, letters_img=letters_img, alertt='Video not Found !')
    # return Response(tg.gen_frames(cap), mimetype='multipart/x-mixed-replace; boundary=frame')


"""@app.route("/gest_text", methods=['GET', 'POST'])
def gest_text():
    model = gt.model"""


@app.route("/inter", methods=['GET', 'POST'])
def inter():
    global flag, check
    if 'email' not in session:
        check = 0
        return render_template('login.html')
    print(session['email'])
    rows = []
    with open('data.csv', 'r') as fp:
        csvreader = csv.reader(fp)
        for row in csvreader:
            rows.append(row)
    # print(rows)
    if flag == 1:
        flag = 0
        return render_template('human-intervention.html', rows=rows, check=1, alertt='Video Uploaded Successfully !', email=session['email'])
    elif flag == 2:
        flag = 0
        return render_template('human-intervention.html', rows=rows, check=1, alertt='Concern Deleted Successfully !', email=session['email'])
    else:
        return render_template('human-intervention.html', rows=rows, check=1, email=session['email'])


@app.route("/uploadvideo/<file_name>", methods=['GET', 'POST'])
def download(file_name):
    global flag
    rows = []
    # print('debuggggg')
    # print(file_name)
    temp = []
    temp.append(file_name)
    with open('data.csv', 'r') as fp:
        csvreader = csv.reader(fp)
        for row in csvreader:
            rows.append(row)
    # print(rows)
    rows.remove(temp)
    with open('data.csv', 'w', newline='') as fw:
        writerobj = writer(fw)
        writerobj.writerows(rows)
        fw.close()
    if(request.method == 'POST'):
        file = request.files['f1']
        file.save("./static/dataset/" + str(file_name) + ".mp4")
    inf = str(file_name) + '.mp4'
    outf = 'cart' + str(file_name) + '.mp4'
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
    if 'email' in session:
        check = 1
        email = session['email']
    else:
        check = 0
        email = ''
    text = request.form['text']
    text = text.title()
    # print(text)
    if(text == ''):
        return render_template('index.html',  email=email, letters_img=letters_img, alertt='Please enter a word...')
    with open('data.csv', 'a', newline='') as fp:
        writer_object = writer(fp)
        lt = []
        lt.append(text)
        writer_object.writerow(lt)
        fp.close()
    return render_template('index.html',  email=email, check=check, letters_img=letters_img, alertt='Concern raised successfully !')


@app.route('/video_feed1')
def video_feed1():
    return Response(gt.convert(1), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route("/gest_text_upload", methods=['GET', 'POST'])
def gest_text_upload():
    if 'email' in session:
        check = 1
        email = session['email']
    else:
        check = 0
        email = ''
    file = request.files['f2']
    file.save("./static/uploaded/video.mp4")
    gt.convert(1)
    # os.remove("./static/uploaded/video.mp4")
    return render_template('index.html',scroll='sign2text', upload=1, email=email,  check=check, letters_img=letters_img)


@app.route('/video_feed')
def video_feed():
    return Response(gt.convert(0), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route("/gest_text_cam", methods=['GET', 'POST'])
def gest_text_cam():
    if 'email' in session:
        check = 1
        email = session['email']
    else:
        check = 0
        email = ''
    gt.convert(0)
    return render_template('index.html',scroll='sign2text', webcam=1,  email=email, check=check, letters_img=letters_img)


@app.route("/logout")
def logout():
    if 'email' not in session:
        return render_template('index.html', check=0, letters_img=letters_img)
    session.pop('email')
    return render_template("index.html", check=0, letters_img=letters_img)


if __name__ == "__main__":
    global letters_img
    letters_img = []
    for i in "abcdefghijklmnopqrstuvwxyz":
        url = "../static/assets/img/letters/" + i + ".jpg"
        letters_img.append(url)

    print('Server Started !!')
    app.run(debug=True, use_reloader=False,port=80, host='0.0.0.0')
