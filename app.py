from pymongo import MongoClient
from flask import Flask, render_template, url_for, request, flash, redirect, jsonify, send_file, session, Response
from flask_cors import CORS, cross_origin
from flask_bcrypt import Bcrypt
from Helpers import text2gest as tg
#from Helpers import gest2text as gt
import os
import sys
from nltk.stem import WordNetLemmatizer
import cv2
import webbrowser
from tensorflow import keras

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
    return render_template('index.html')

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
        return render_template('index.html', email=session['email'])
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
    text = request.form['text']
    print(text)
    text = text.title()
    #print(text)
    word_Lemmatized = WordNetLemmatizer()
    text = word_Lemmatized.lemmatize(text)
    ls = tg.datalist()
    print(ls)
    if(text == 'Bye'):
        print('Good Bye !!')
        exit
    elif(text in ls):
        print('debug')
        #cap = cv2.VideoCapture('static/dataset/'+ text + '.mp4')
        video = '../static/cartdata/cart'+text+'.mp4'
    #Human Intervention Part
    else:
        print("Sorry, currently we don't have the word in our back end !")
    #return Response(tg.gen_frames(cap), mimetype='multipart/x-mixed-replace; boundary=frame')
    return render_template('index.html', video=video, alert='working')

"""@app.route("/gest_text", methods=['GET', 'POST'])
def gest_text():
    model = gt.model"""

if __name__ == "__main__":
    print('Server Started !!')
    app.run(debug=True, use_reloader=False)