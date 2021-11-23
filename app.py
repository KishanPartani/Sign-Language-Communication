from pymongo import MongoClient
from flask import Flask, render_template, url_for, request, flash, redirect, jsonify, send_file, session, Response
from flask_cors import CORS, cross_origin
from flask_bcrypt import Bcrypt
from Helpers import text2gest as tg
import os
import sys
from nltk.stem import WordNetLemmatizer
import cv2

app = Flask(__name__)
bcrypt = Bcrypt(app)
client = MongoClient('localhost', 27017)
db = client.cnt_db
users = db.users

cors = CORS(app, resources={r"/foo": {"origins": "*"}})
app.config['CORS_HEADERS'] = 'Content-Type'
app.config['SECRET_KEY'] = '14ec258c169f5c19f78385bcc83a51df7444624b2ff90449b4a9832e6fe706a1'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/login-page')
def show_login_page():
    return render_template('login-signup.html')

@app.route("/login", methods=['GET', 'POST'])
def login():
    print(request.form['email'])
    print(request.form['password'])
    user_cred = users.find_one({'email': request.form['email']}, {
        'name': 1, 'email': 1, 'password': 1})
    if(not bool(user_cred)):
        return render_template('login-signup.html', message="User not found. Please Sign Up!", classm="alert alert-danger")
    if(bcrypt.check_password_hash(user_cred['password'], request.form['password'])):
        # create session variable with email
        session['email'] = user_cred['email']
        return render_template('dashboard.html', name=user_cred['name'])
    else:
        return render_template('login-signup.html', message="Incorrect password!", classm="alert alert-danger")


@app.route("/signup", methods=['GET', 'POST'])
def signup():
    if(request.form['pwd'] != request.form['rpwd']):
        return render_template('login-signup.html', message="Passwords don't match!", classm="alert alert-danger")

    name = request.form['fname'] + " " + request.form['lname']
    password_hash = bcrypt.generate_password_hash(request.form['pwd'], 10)
    user_details = {"name": name,
                    "email": request.form['email'],
                    "password": password_hash}
    empr_id = db.users.insert_one(user_details).inserted_id
    print(empr_id)
    
    return render_template('login-signup.html', message="Sign up successful", classm="alert alert-success")

@app.route("/text_gest", methods=['GET', 'POST'])
def text_gest():
    text = request.form['text']
    print(text)
    #tg.convert(text)
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
        cap = cv2.VideoCapture('static/dataset/'+ text + '.mp4')
    return Response(tg.gen_frames(cap), mimetype='multipart/x-mixed-replace; boundary=frame')
    #tg.convert(text)
    #return render_template('index.html')

if __name__ == "__main__":
    print('started')
    app.run(debug=True, use_reloader=False)