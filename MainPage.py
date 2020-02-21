# coding=utf-8
"""Face Detection, Recognition and Verification inside a gui includes  age, gender, object and race or ethnicity recognition """
# MIT License
#
# Copyright (c) 2019 Xolani Dube
#
# This is the work of Xolani Dube remodelled into a
# high level container. It's an attempt to simplify the use of such
# technology and provide an easy to use facial recognition package.
#
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from PyQt5.QtWidgets import QMessageBox


from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.image import Image
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.floatlayout import FloatLayout
from kivy.core.window import Window
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.uix.scrollview import ScrollView
import kivy.input.providers.mouse
import kivy
from kivy.lang import Builder
from kivy.properties import BooleanProperty, ObjectProperty
from kivy.garden.matplotlib.backend_kivyagg import FigureCanvasKivyAgg
from functools import partial
from datetime import datetime
from datetime import timedelta
import face as f
import time

import numpy as np
import os
from os import listdir
from os.path import isfile, join
import re
import cv2
import random
import smtplib, ssl
import tensorflow as tf
import facenet
import detect_face
from scipy import misc
import email
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from keras.models import load_model
from statistics import mode
from utils.datasets import get_labels
from utils.inference import detect_faces
from utils.inference import draw_text
from utils.inference import draw_bounding_box
from utils.inference import apply_offsets
from utils.inference import load_detection_model
from utils.preprocessor import preprocess_input
from kivy.config import Config
from email.policy import SMTP
gpu_memory_fraction = 0.3
import threading
import pyttsx3
from kivy.core.window import Window
from email import message
from kivy.uix.textinput import TextInput
from kivy.uix.popup import Popup
import matplotlib.pyplot as plt
import core.utils as utils
from core.yolov3 import YOLOv3, decode
import mysql.connector
from kivy.uix.gridlayout import GridLayout
from twilio.rest import Client
import base64
from email.mime.image import MIMEImage
from email import encoders
from face_network import create_face_network
from keras.optimizers import Adam, SGD
from kivy.uix import textinput
from time import sleep
import imageio
import math
import pickle
from sklearn.svm import SVC
from multiprocessing import Process, Queue
from keras.preprocessing import image
from keras.models import load_model
from kivy.uix.spinner import Spinner
from kivy.uix.recycleview import RecycleView
from distutils.command import install
import shutil
from sklearn.naive_bayes import GaussianNB



from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit

    
engine = pyttsx3.init()
engine.setProperty("rate", 150)
engine.setProperty("volume", 1)
Clock.max_iteration = 200
port = 465
paswrd = "9802185362084XOLANIDUBE@x"
emil = "projectclockin@gmail.com"
# context = ssl.create_default_context()

# s = smtplib.SMTP(host='smtp.gmail.com', port=587)

# s.starttls()
##s.connect(host='smtp.gmail.com', port=587)
# s.login("projectclockin@gmail.com", "9802185362084XOLANIDUBE@x")


server = "127.0.0.1"
username = "root"
password = ""
database = "employees"
conn = None

USA = "+12027514983"


def cdigit(string):
    return bool(re.search(r'-?\d+', string))

def dChar(string):
    return bool(re.search("[^a-zA-Z]", string))

def Connect(host_, user_, passwd_, database_):
    try:
        conn = mysql.connector.connect(
            host=host_, user=user_, passwd=passwd_, database=database_
        )

        return conn
    
    except Exception as e:
        #QMessageBox.about("Error", "Unable to connect to server stopping application....")
        print(e)
        
        return 
        


class auth(FloatLayout):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.conn = None
        try:
            self.conn = Connect(server, username, password, database)
        except:
            print("Error couldn’t connect to your server please start your server")
            return
        self.cursor = self.conn.cursor()
        self.lbl = Label(
            text="Project Clock In",
            pos_hint={"x": 0.02, "y": 0.8},
            size_hint=(0.12, 0.25),
            font_size="28sp",
        )
        self.compare = Image(
            source="C://Users//GabhaDi//eclipse-workspace2//Project Clock-in//data//Images//face_avatar.jpg"
        )
        self.rec = False
        self.face_recognition = f.Recognition()
        self.face_verification = f.Verification()
        self.now = datetime.now()
        self.padding = 20
        self.count = 0
        self.usedkey = {}
        self.ageProto = "..\\models\\age_deploy.prototxt"
        self.ageModel = "..\\models\\age_net.caffemodel"

        self.genderProto = "..\\models\\gender_deploy.prototxt"
        self.genderModel = "..\\models\\gender_net.caffemodel"

        self.emotionsModel = "..\\models\\emotion_model.hdf5"
        self.emotionsLabels = get_labels("fer2013")
        self.emotionsClassifier = load_model(self.emotionsModel)
        self.emotion_target_size = self.emotionsClassifier.input_shape[1:3]
        self.emotionsWindow = []

        self.classes = "..\\models\\yolov3.txt"
        self.weights = "..\\models\\yolov3.weights"
        self.config = "..\\models\\yolov3.cfg"
        self.threat = False

        # with open(self.classes, 'r') as v:
        #    self.classes = [line.strip() for line in v.readlines()]

        self.btnEmp = Button(
            text="Go to Employee Page",
            font_size=12,
            size_hint_y=None,
            size_hint_x=None,
            width=150,
            height=30,
            pos_hint={"x": 0.35, "y": 0.01},
        )
        self.btnEmp.bind(on_press=self.emp)

        self.btnTrainNTest = Button(
            text="Go to Train and Test Page",
            font_size=12,
            size_hint_y=None,
            size_hint_x=None,
            width=150,
            height=30,
            pos_hint={"x": 0.47, "y": 0.01},
        )
        self.btnTrainNTest.bind(on_press=self.tt)

        self.btnCollectData = Button(
            text="Go to Collect Data Page",
            font_size=12,
            size_hint_y=None,
            size_hint_x=None,
            width=150,
            height=30,
            pos_hint={"x": 0.59, "y": 0.01},
        )
        self.btnCollectData.bind(on_press=self.cd)

        self.btnReport = Button(
            text="Go to Report Page",
            font_size=12,
            size_hint_y=None,
            size_hint_x=None,
            width=150,
            height=30,
            pos_hint={"x": 0.71, "y": 0.01},
        )
        self.btnReport.bind(on_press=self.rp)

        self.MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
        self.ageList = [
            "(0-2)",
            "(3-7)",
            "(8-14)",
            "(15-20)",
            "(21-32)",
            "(33-43)",
            "(43-53)",
            "(60+)",
        ]
        self.genderList = ["Male", "Female"]

        self.ageNet = cv2.dnn.readNet(self.ageModel, self.ageProto)
        self.genderNet = cv2.dnn.readNet(self.genderModel, self.genderProto)

        self.CurrentTime = Label(
            text=self.now.strftime("Current Time : %H:%M:%S, %d %b %Y"),
            pos_hint={"x": 0.76, "y": 0.8},
            size_hint=(0.12, 0.25),
            font_size="28sp",
        )
        self.compare.allow_stretch = False
        self.compare.keep_ratio = False

        self.compare.size_hint_x = None  # 0.2
        self.compare.size_hint_y = None  # 0.3

        self.compare.width = 160
        self.compare.height = 160

        self.compare.pos_hint = {"x": 0.1, "y": 0.24}
        self.recognized = Image(
            source="C://Users//GabhaDi//eclipse-workspace2//Project Clock-in//data//Images//face_avatar.jpg"
        )

        self.recognized.allow_stretch = False
        self.recognized.keep_ratio = False

        self.recognized.size_hint_x = None  # 0.2
        self.recognized.size_hint_y = None  # 0.3

        self.recognized.width = 160
        self.recognized.height = 160

        self.recognized.pos_hint = {"x": 0.1, "y": 0.009}

        self.processed = Image()
        self.processed.allow_stretch = False
        self.processed.keep_ratio = False

        self.processed.size_hint_x = None  # 0.3
        self.processed.size_hint_y = None  # 1

        self.processed.width = 400
        self.processed.height = 400
        self.processed.pos_hint = {"x": 0.01, "y": 0.4}
        ###self.img1.size_hint = (.5, .25)
        self.dic = {}

        self.unprocessed = Image()

        self.unprocessed.allow_stretch = True
        self.unprocessed.keep_ratio = False

        self.unprocessed.size_hint_x = None  ##0.8
        self.unprocessed.size_hint_y = None  ##1
        self.unprocessed.width = 880
        self.unprocessed.height = 580

        self.unprocessed.pos_hint = {"x": 0.35, "y": 0.075}  ##x 0.28 y 0.001
        ###layout = FloatLayout(size=(1024, 760))
        self.means = np.load("means_ethnic.npy")

        self.model = create_face_network(
            nb_class=4, hidden_dim=512, shape=(224, 224, 3)
        )
        self.model.load_weights("weights_ethnic.hdf5")
        self.ETHNIC = {0: "Asian", 1: "Caucasion", 2: "African", 3: "Hispanic"}
        self.num_classes = 80
        self.input_size = 416

        # self.input_layer = tf.keras.layers.Input([self.input_size, self.input_size, 3])
        # self.feature_maps = YOLOv3(self.input_layer)

        # self.bbox_tensors =[]
        # for i, fm in enumerate(self.feature_maps):
        #    self.bbox_tensor = decode(fm, i)
        #    self.bbox_tensors.append(self.bbox_tensor)

        # self.kmodel = tf.keras.Model(self.input_layer, self.bbox_tensors)
        # utils.load_weights(self.kmodel, "..\\models\\yolov3.weights")
        # self.kmodel.summary()

        self.minsize = 20  # minimum size of face
        self.threshold = [0.6, 0.7, 0.7]  # three steps's threshold
        self.factor = 0.709  # scale factor

        with tf.Graph().as_default():
            gpu_options = tf.compat.v1.GPUOptions(
                per_process_gpu_memory_fraction=gpu_memory_fraction
            )
            sess = tf.compat.v1.Session(
                config=tf.compat.v1.ConfigProto(
                    gpu_options=gpu_options, log_device_placement=False
                )
            )
            with sess.as_default():
                self.pnet, self.rnet, self.onet = detect_face.create_mtcnn(sess, None)

        self.sess = tf.compat.v1.Session()
        # Load the model
        facenet.load_model("20180402-114759.pb")

        # Get input and output tensors
        self.images_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name(
            "input:0"
        )
        self.embeddings = tf.compat.v1.get_default_graph().get_tensor_by_name(
            "embeddings:0"
        )
        self.phase_train_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name(
            "phase_train:0"
        )
        widgets = [
            self.btnReport,
            self.btnCollectData,
            self.btnTrainNTest,
            self.btnEmp,
            self.processed,
            self.unprocessed,
            self.compare,
            self.lbl,
            self.recognized,
            self.CurrentTime,
        ]

        # self.threats_classifier = load_model("threats_cnn.h5")
        # self.class_labels = {0:"laptop", 1:"phone"}

        for wid in widgets:
            self.add_widget(wid)

        try:
            self.capture = cv2.VideoCapture(1)
        except:
            self.capture = cv2.VideoCapture(0)
        
        #finally:
        #    self.pop("Error : Camera Failer", "COULDN'T OPEN CAMERA PLEASE CONNECT A CAMERA")

        ###cv2.namedWindow("Testing System...")
        Clock.schedule_interval(self.update, 1.0 / 10000.0)
        Clock.schedule_interval(self.update_clock, 1)

        ###return layout

    def speak(self, txt):
        threading.Thread(target=self.run_pyttsx3, args=(txt,), daemon=True).start()

    def run_pyttsx3(self, text):
        engine.say(text)
        engine.runAndWait()

    def unsched(self, func):
        threading.Thread(target=Clock.unschedule, args=(func,), daemon=True).start()

    def emp(self, instance):

        app_.screen_manager.current = "employee"

        self.unsched(self.update)

    def tt(self, instance):

        app_.screen_manager.current = "trainNtest"
        self.unsched(self.update)

    def cd(self, instance):

        app_.screen_manager.current = "collect"
        self.unsched(self.update)

    def rp(self, instance):
        app_.screen_manager.current = "report"
        self.unsched(self.update)

    def threaded(self, normal):
        threading.Thread(
            target=self.threatDetection, args=(normal,), daemon=True
        ).start()

    def threaded2(self, frame):
        threading.Thread(
            target=self.youOnlyLookOnce,
            args=(frame, self.weights, self.config),
            daemon=True,
        ).start()

    def threaded3(self, dt):
        threading.Thread(target=self.update, args=(dt,), daemon=True).start()

    def update(self, dt):

        ret, frame = self.capture.read()
        frame = cv2.flip(frame, 1, 0)

        gry = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        normal = cv2.cvtColor(gry, cv2.COLOR_GRAY2BGR)

        if self.face_recognition.identify(normal) is None:
            faces = None

        if self.face_recognition.identify(normal) is not None:
            faces, points = self.face_recognition.identify(normal)

        ##threading.Thread(target=self.threatDetection, args = (normal,))
        ##thread3.start(); 
        ##thread3.join()

        if faces is not None:
            for face in faces:
                # Clock.schedule_once(partial(self.threaded, normal))
                # self.threat = self.threaded2(normal)
                face_bb = face.bounding_box.astype(int)
                yourface = normal[
                    max(0, face_bb[1] - self.padding) : min(
                        face_bb[3] + self.padding, normal.shape[0] - 1
                    ),
                    max(0, face_bb[0] - self.padding) : min(
                        face_bb[2] + self.padding, normal.shape[1] - 1
                    ),
                ]

                ###label = "{},{},{}".format(gender, age, emotion_text)

                ###cv2.rectangle(gray, (face_bb[0], face_bb[1]), (face_bb[2], face_bb[3]), (255,255,255), 1)

                if face.name is not None:

                    ###self.countdown(5)
                    if face.name == "Unrecognized":
                        if self.threat == False:
                            gray_face = gry[
                                max(0, face_bb[1]) : min(
                                    face_bb[3], normal.shape[0] - 1
                                ),
                                max(0, face_bb[0]) : min(
                                    face_bb[2], normal.shape[1] - 1
                                ),
                            ]

                            try:
                                gray_face = cv2.resize(
                                    gray_face, (self.emotion_target_size)
                                )
                            except:
                                continue

                            gray_face = preprocess_input(gray_face, True)
                            gray_face = np.expand_dims(gray_face, 0)
                            gray_face = np.expand_dims(gray_face, -1)
                            emotion_prediction = self.emotionsClassifier.predict(
                                gray_face
                            )
                            # emotion_probability = np.max(emotion_prediction)

                            emotion_label_arg = np.argmax(emotion_prediction)
                            emotion_text = self.emotionsLabels[emotion_label_arg]

                            blob = cv2.dnn.blobFromImage(
                                yourface,
                                1.0,
                                (227, 227),
                                self.MODEL_MEAN_VALUES,
                                swapRB=False,
                            )
                            self.genderNet.setInput(blob)
                            genderPreds = self.genderNet.forward()

                            gender = self.genderList[genderPreds[0].argmax()]

                            self.ageNet.setInput(blob)
                            agePreds = self.ageNet.forward()
                            age = self.ageList[agePreds[0].argmax()]

                            im = cv2.cvtColor(gry, cv2.COLOR_GRAY2RGB)
                            im = cv2.resize(im, (224, 224))
                            im = np.float64(im)
                            im /= 255.0
                            im = im - self.means

                            # self.threat= self.thr(frame)

                            race = self.ETHNIC[
                                np.argmax(self.model.predict(np.array([im])))
                            ]

                            lis = [face.name, gender, age, emotion_text, race]
                            conf = face.threshold * 100

                            cv2.rectangle(
                                gry,
                                (face_bb[0], face_bb[1]),
                                (face_bb[2], face_bb[3]),
                                (255, 77, 77),
                                1,
                            )

                            cv2.putText(
                                gry,
                                face.name,
                                (face_bb[0] + 100, face_bb[3] - 140),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.7,
                                (220, 20, 60),
                                1,
                                cv2.LINE_AA,
                            )
                            cv2.putText(
                                gry,
                                lis[1],
                                (face_bb[0] + 100, face_bb[3] - 100),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.7,
                                (220, 20, 60),
                                1,
                                cv2.LINE_AA,
                            )
                            cv2.putText(
                                gry,
                                lis[2],
                                (face_bb[0] + 100, face_bb[3] - 70),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.7,
                                (220, 20, 60),
                                1,
                                cv2.LINE_AA,
                            )
                            cv2.putText(
                                gry,
                                lis[3],
                                (face_bb[0] + 100, face_bb[3] - 30),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.7,
                                (220, 20, 60),
                                1,
                                cv2.LINE_AA,
                            )
                            cv2.putText(
                                gry,
                                lis[4],
                                (face_bb[0] + 100, face_bb[3]),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.7,
                                (220, 20, 60),
                                1,
                                cv2.LINE_AA,
                            )

                            cv2.rectangle(
                                frame,
                                (face_bb[0], face_bb[1]),
                                (face_bb[2], face_bb[3]),
                                (58, 7, 255),
                                3,
                            )

                            self.count += 1
                            for i in range(points.shape[1]):
                                pts = points[:, i].astype(np.int32)
                                for j in range(pts.size // 2):
                                    pt = (pts[j], pts[5 + j])
                                    cv2.circle(
                                        gry,
                                        center=pt,
                                        radius=1,
                                        color=(255, 0, 0),
                                        thickness=2,
                                    )

                            if self.count >= 20:
                                self.count = 0
                                unk = frame[
                                    max(0, face_bb[1]) : min(
                                        face_bb[3], normal.shape[0] - 1
                                    ),
                                    max(0, face_bb[0]) : min(
                                        face_bb[2], normal.shape[1] - 1
                                    ),
                                ]
                                resized = cv2.resize(
                                    unk, (160, 160), interpolation=cv2.INTER_AREA
                                )

                                name = face.name + "_" + str(random.randint(1000, 9999))
                                cv2.imwrite(
                                    "C://Users//GabhaDi//eclipse-workspace2//Project Clock-in//Unrecognized_blobs//"
                                    + str(name)
                                    + ".png",
                                    resized,
                                )

                                print("Unrecognized")

                                self.clear_image()
                                self.recognized.source = (
                                    "C://Users//GabhaDi//eclipse-workspace2//Project Clock-in//Unrecognized_blobs//"
                                    + name
                                    + ".png"
                                )

                                self.speak(
                                    "Sorry, I couldn't recognize you please check with security, you have been denied access.",
                                )

                        else:
                            print("Security breach detected ")
                            text = "Security breach detected please do not use a cell phone to clock in"
                            self.speak(text)

                    else:
                        self.count = 0
                        conf = face.threshold * 100
                        
                        cv2.rectangle(
                            gry,
                            (face_bb[0], face_bb[1]),
                            (face_bb[2], face_bb[3]),
                            (0, 128, 0),
                            1,
                        )
                        cv2.putText(
                            gry,
                            face.name + " " + str(conf),
                            (face_bb[0], face_bb[3] + 20),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, 
                            (0, 128, 0),
                            1,
                            cv2.LINE_AA,
                        )
                        cv2.rectangle(
                            frame,
                            (face_bb[0], face_bb[1]),
                            (face_bb[2], face_bb[3]),
                            (20, 255, 57),
                            3,
                        )

                        phase = frame[
                            max(0, face_bb[1]) : min(face_bb[3], normal.shape[0] - 1),
                            max(0, face_bb[0]) : min(face_bb[2], normal.shape[1] - 1),
                        ]
                        for i in range(points.shape[1]):
                            pts = points[:, i].astype(np.int32)
                            for j in range(pts.size // 2):
                                pt = (pts[j], pts[5 + j])
                                cv2.circle(
                                    gry,
                                    center=pt,
                                    radius=1,
                                    color=(0, 0, 255),
                                    thickness=2,
                                )

                        # self.recognized.source = face.name + ".png"

                        if self.checkKey(self.dic, face.name) == False:
                            

                            self.dic[face.name] = True
                            resized = cv2.resize(
                                phase, (160, 160), interpolation=cv2.INTER_AREA
                            )
                            ###cv2.imshow("this", resized)
                            path = (
                                "C://Users//GabhaDi//eclipse-workspace2//Project Clock-in//Clock_in//employee_data//train_aligned//"
                                + face.name
                                + "//"
                            )
                            rd = path + self.getRandomImage(path)
                            ###self.countdown(5)
                            img1 = face.name + ".png"
                            img2 = rd
                            cv2.imwrite(img1, resized)
                            self.emp_blobpath = img1
                            self.recognized.source = img1

                            self.compare.source = rd
                            
                            query = f"UPDATE employee_checks SET emp_blobpath = '{self.emp_blobpath}' WHERE emp_name = '{face.name}'"
                            self.cursor.execute(query)
                            self.conn.commit()

                            #t = threading.Thread(target=self.distance, args = (model, image_files, image_size, margin, gpu_memory_fraction), daemon=True)
                            #t.start()
                            #t.join()

                            self.conf = self.face_verification.verify(img1, img2)
                            if self.conf >= 60:
                                self.u = face.name

                                self.clck()
                                """ try:
                                    
                                    #threading.Thread(target=self.sendemail, args=(img1, face.name), daemon=True).start()
                                    
                                    self.sendemail(img1, self.u)
                                    layout = GridLayout(cols = 1, padding = 10)
                                    
                                    #self.speak("Please enter your Secret Code from your email : %s PRESS (Clock In) when you are done" % self.emp_email)
                                    popupLabel = Label(text="Please enter your Secret Code from your email : %s PRESS (Clock In) when you are done" % self.emp_email)
                                    closeButton = Button(text="close")
                                    ok = Button(text="Clock In")
                                    ###no = Button(text="no")
                                    self.txt = TextInput(text="")
                                    layout.add_widget(popupLabel)
                                    layout.add_widget(self.txt)
                                    layout.add_widget(ok)
                                    layout.add_widget(closeButton)
   
                                    self.popup = Popup(title="INFO : MESSAGE", content = layout, size_hint=(.6,.4))
                                    self.popup.open()
                                    ###no.bind(on_press= self.popup.dismiss)
                                    ok.bind(on_press=self.verify)
                                    closeButton.bind(on_press= self.popup.dismiss)
                                    self.popup.bind(on_dismiss=self.verify)
                                    ###del self.dic[face.name]
                                    
                                    #self.clck()
                                    
                                    print()
                                    
                                except:
                                    
                                    
                                    print("Please try again and clock in again")
                                    
                                    text = "I encountered an error. Please try again and clock in!"
                                    #self.speak(text)
                                    ###self.pop("ERROR : MESSAGE", "COULDNT SEND EMAIL, PLEASE TRY AGAIN AND CLOCK IN")
                                    del self.dic[face.name]"""

                            else:

                                print(
                                    "Comparisen confidence is low please try again. %2f percent"
                                    % self.conf
                                )

                                text = (
                                    "Comparisen confidence is low please try again. %f percent"
                                    % round(self.conf, 1)
                                )
                                
                                if self.conf >= 50 and  self.conf <=59:
                                    self.speak("I'm not sure if its you, please try again!")
                                
                                if self.conf <=49:
                                    self.speak("You are not the same person, please try again!")

                                ###self.pop("INFO : MESSAGE", "Comparisen confidence is too low please try again. %2f" % self.conf)
                                ###engine.say("Comparisen confidence was low please try again. %2f" % self.conf)
                                ###engine.runAndWait()
                                del self.dic[face.name]

        if faces is None:
            ###print("No Faces detected.")
            self.rec = False
            self.compare.source = (
                "C://Users//GabhaDi//eclipse-workspace2//Project Clock-in//data//Images//face_avatar.jpg"
            )
            self.recognized.source = (
                "C://Users//GabhaDi//eclipse-workspace2//Project Clock-in//data//Images//face_avatar.jpg"
            )
            self.count = 0

        font = cv2.FONT_HERSHEY_SIMPLEX

        cv2.putText(
            frame,
            "Unprocessed Video Stream",
            (10, 20),
            font,
            0.7,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            gry,
            "Processed Video Stream",
            (10, 20),
            font,
            0.7,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        buf1 = cv2.flip(frame, 0)

        buf = buf1.tostring()

        buf2 = cv2.flip(gry, 0)
        bf2 = buf2.tostring()

        texture1 = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt="bgr")
        texture1.blit_buffer(buf, colorfmt="bgr", bufferfmt="ubyte")

        texture2 = Texture.create(
            size=(gry.shape[1], gry.shape[0]), colorfmt="luminance"
        )
        texture2.blit_buffer(bf2, colorfmt="luminance", bufferfmt="ubyte")

        self.processed.texture = texture1
        self.unprocessed.texture = texture2

        # Clock.schedule_once(partial(self.set, frame))
        # Clock.schedule_once(partial(self.set_, gry))

    def set_(self, gry, dt):
        buf2 = cv2.flip(gry, 0)
        bf2 = buf2.tostring()

        texture2 = Texture.create(
            size=(gry.shape[1], gry.shape[0]), colorfmt="luminance"
        )
        texture2.blit_buffer(bf2, colorfmt="luminance", bufferfmt="ubyte")

        self.unprocessed.texture = texture2

    def set(self, frame, dt):
        buf1 = cv2.flip(frame, 0)

        buf = buf1.tostring()

        texture1 = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt="bgr")
        texture1.blit_buffer(buf, colorfmt="bgr", bufferfmt="ubyte")

        self.processed.texture = texture1

    def thr(self, frame):
        p_resize = cv2.resize(frame, (32, 32))
        x = image.img_to_array(p_resize)
        x = x * 1.0 / 255
        x = np.expand_dims(x, axis=0)

        pred_ = self.threats_classifier.predict_classes(x)

        print(pred_[0])
        prediction = self.class_labels[np.argmax(pred_)]

        if prediction == "phone":
            self.threat = True
            print(self.threat)

        return self.threat

    def sendemail(self, img1, name):
        msg = MIMEMultipart()
        secret_code = str(random.randint(100000, 999999))
        self.code = secret_code

        data_uri = base64.b64encode(open(img1, "rb").read()).decode()

        img_tag = '<img src="data:image/png;base64, {0}">'.format(data_uri)
        ##print(img_tag)
        message = "This is your secert code : %s " % secret_code
        message2 = f""" 
                                
                                <html>
                                    
                                        
                                    
                                        
                                    <body>
                                        <h1 align="center">Project Clock In</h1>
                                        
                                        <h3>Welcome this is for demonstration purposes you have been sent this mail to for to receive your secret code and you being a part in the project</h3><br><br>
                                        
                                        
                                        
                                        
                                        
                                        
                                        
                                     
                                        <tr><td><img src = "cid:image1"  height="213" width="213"></td><td>{name}</td></tr><br>
                                        <tr><td>Clock In Time :</td><td>{self.now.strftime('%H:%M:%S, %d %b %Y')}</td></tr><br>
                                        <tr><td>Your Secret Code :</td><td><b>{self.code}</b></td></tr>
                                            
                                        
                                    
                                    <body>
                                    
                                    
                                    <footer>
                                    
                                            <h6 align='center'>Copyright 2019 &copy;  Project Clock In&trade;</h6>
                                    
                                    </footer>
                                
                                
                                
                                
                                </html>

                                
                                """

        query = f"SELECT * FROM employee_details WHERE emp_name = '{name}'"

        self.cursor.execute(query)
        result = self.cursor.fetchall()
        result = list(sum(result, ()))

        self.emp_id = str(result[0])
        self.emp_name = str(result[1])
        self.emp_surname = str(result[2])
        self.emp_phone = str(result[3])
        self.emp_email = str(result[4])
        self.emp_blobpath = img1

        msg = MIMEMultipart("related")
        msg["From"] = emil
        msg["To"] = self.emp_email
        msg["Subject"] = f"Welcome {name} This is your secret code!!!"

        msgTxt = MIMEText(message2, "html", "utf-8")
        msg.attach(msgTxt)

        fp = open(img1, "rb")
        msgImage = MIMEImage(fp.read())
        fp.close()

        msgImage.add_header("Content-ID", "<image1>")
        msg.attach(msgImage)
        ##s.send_message(msg)
        self.speak("I have successfully sent you an email,. Please check your inbox")

    def clck(self):

        querychk = (
            f"SELECT emp_access_state FROM employee_checks WHERE emp_name = '{self.u}'"
        )

        self.cursor.execute(querychk)
        result = self.cursor.fetchall()
        result = list(sum(result, ()))
        self.access = -1

        if len(result) >= 1:
            self.access = int(result[0])

        if self.access == 1:
            # self.popup.on_open = self.popup.dismiss
            print("Access granted")
            # self.pop("INFO : MESSAGE ", "Good bye, Have a wonderful day,. " + self.u + ", you clocked out at " + str(self.now.strftime('%H:%M:%S, %d %b %Y')) + " and you have been logged out.")
            self.speak(
                "Good bye, Have a wonderful day,. "
                + self.u
                + ", you clocked out at "
                + str(self.now.strftime("%H:%M:%S, %d %b %Y"))
                + " ."
            )

            query = f"UPDATE employee_checks SET emp_clockouttime = '{self.now}', emp_access_state = 0, emp_blobpath = '{self.emp_blobpath}' WHERE emp_name = '{self.u}';"

            self.cursor.execute(query)
            self.conn.commit()

            ###self.clear_image()
            # self.usedkey[self.code] = self.u
            # del self.code
            ###self.clear_image()
            time.sleep(5)
            del self.dic[self.u]
        if self.access == 0:
            # self.popup.on_open = self.popup.dismiss
            print("Access granted")
            # self.pop("INFO : MESSAGE ", "Welcome back, " + self.u + ", you clocked in at " + str(self.now.strftime('%H:%M:%S, %d %b %Y')) + " and you have been logged in.")

            self.speak(
                "Welcome back, "
                + self.u
                + ", you clocked in at "
                + str(self.now.strftime("%H:%M:%S, %d %b %Y"))
                + " and you have been logged in."
            )

            query = f"UPDATE employee_checks SET emp_clockintime = '{self.now}', emp_access_state = 1, emp_blobpath = '{self.emp_blobpath}' WHERE emp_name = '{self.u}';"

            self.cursor.execute(query)
            self.conn.commit()

            ###self.clear_image()
            # self.usedkey[self.code] = self.u
            # del self.code
            ###self.clear_image()

            del self.dic[self.u]
        if self.access == -1:

            # self.popup.on_open = self.popup.dismiss
            print("Access granted")
            # self.pop("INFO : MESSAGE ", "Welcome, " + self.u + ", you clocked in at " + str(self.now.strftime('%H:%M:%S, %d %b %Y')) + " and you have been granted access.")

            self.speak(
                "Welcome "
                + self.u
                + ", you clocked in at "
                + str(self.now.strftime("%H:%M:%S, %d %b %Y"))
                + " ."
            )
            id = int(self.emp_id)
            query = f"INSERT INTO employee_checks(emp_id, emp_name, emp_surname, emp_email, emp_phoneNo, emp_clockInTime, emp_clockOutTime, emp_access_state, emp_blobpath) VALUES ({id}, '{self.emp_name}', '{self.emp_surname}','{self.emp_email}','{self.emp_phone}', '{self.now}', '', 1, '{self.emp_blobpath}');"

            self.cursor.execute(query)
            self.conn.commit()

            print("Successfully inserted the employee")

            ###self.clear_image()
            # self.usedkey[self.code] = self.u
            # del self.code
            ###self.clear_image()
            time.sleep(5)
            del self.dic[self.u]

        """else:
                    print("Access denied!")
                    ###self.pop("INFO : MESSAGE ", "Access Denied your secret code didn't match! please try again.")
                    self.popup.on_open = self.popup.dismiss
                      
                    self.speak("Access Denied, Your secret code didn't match! please try again.")
                        
                       
                    del self.dic[self.u]
                    del self.code
                    #self.clear_image()"""

        ##self.popup.on_open = self.popup.dismiss
        ###self.pop("ERROR : MESSAGE", "THIS SECRET CODE HAS BEEN USED YOU CANT USE IT MORE THAN ONCE!")

    def pop(self, tle, txt):
        layout = GridLayout(cols=1, padding=10)

        popupLabel = Label(text=txt)
        closeButton = Button(text="close")

        layout.add_widget(popupLabel)

        layout.add_widget(closeButton)

        popup = Popup(title=tle, content=layout, size_hint=(0.5, 0.3))
        popup.open()
        popup.auto_dismiss = True
        closeButton.bind(on_press=popup.dismiss)

    def clear_image(self):

        self.compare.source = "..\\data\\Images\\default.jpg"
        self.recognized.source = "..\\data\\Images\\default.jpg"

    def threatDetection(self, frame):

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        frame_size = frame.shape[:2]

        image_data = utils.image_preprocess(
            np.copy(frame), [self.input_size, self.input_size]
        )
        image_data = image_data[np.newaxis, ...].astype(np.float32)

        prev_time = time.time()
        pred_bbox = self.kmodel.predict(image_data)
        curr_time = time.time()
        exec_time = curr_time - prev_time

        pred_bbox = [tf.reshape(x, (-1, tf.shape(x)[-1])) for x in pred_bbox]
        pred_bbox = tf.concat(pred_bbox, axis=0)
        bboxes = utils.postprocess_boxes(pred_bbox, frame_size, self.input_size, 0.3)
        bboxes = utils.nms(bboxes, 0.45, method="nms")
        image, names = utils.draw_bbox(frame, bboxes)

        if names == "cell phone" or names == "laptop":
            self.threat = True

        # return self.threat

    def youOnlyLookOnce(self, frame, weights, config):
        Width = frame.shape[1]
        Height = frame.shape[0]
        scale = 0.00392

        yoloNet = cv2.dnn.readNet(weights, config)
        yoloblob = cv2.dnn.blobFromImage(
            frame, scale, (416, 416), (0, 0, 0), True, crop=False
        )
        yoloNet.setInput(yoloblob)
        yoloPreds = yoloNet.forward(self.get_output_layers(yoloNet))

        class_ids = []
        confidences = []
        boxes = []
        conf_threshold = 0.5
        nms_threshold = 0.4

        for out in yoloPreds:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence >= 0.9:
                    center_x = int(detection[0] * Width)
                    center_y = int(detection[1] * Height)
                    w = int(detection[2] * Width)
                    h = int(detection[3] * Height)
                    x = center_x - w / 2
                    y = center_y - h / 2
                    class_ids.append(class_id)
                    confidences.append(float(confidence))
                    boxes.append([x, y, w, h])
        indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

        for i in indices:
            i = i[0]
            if self.classes[class_ids[i]] == "cell phone":
                box = boxes[i]
                x = box[0]
                y = box[1]
                w = box[2]
                h = box[3]

                cv2.rectangle(
                    frame,
                    (int(x), int(y)),
                    (int(round(x + w)), int(round(y + h))),
                    (0, 0, 255),
                    1,
                )

                cv2.putText(
                    frame,
                    str(self.classes[class_ids[i]]) + " NOT ALLOWED!",
                    (int(x) - 10, int(y) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    1,
                    cv2.LINE_AA,
                )
                self.threat = True
                # engine.say("Security breach detected : Please dont use a " +  classes[class_ids[i]] + " to verify yourself. please check with security if theres a problem. This issue has been flagged")
                # engine.runAndWait()

                return self.threat

    def get_output_layers(self, net):

        layer_names = net.getLayerNames()

        output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

        return output_layers

    def update_clock(self, *args):

        self.now = self.now + timedelta(seconds=1)
        self.CurrentTime.text = self.now.strftime("Current Time : %H:%M:%S, %d %b %Y")

    def checkKey(self, dic, key):

        if key in dic.keys():

            return True
        else:

            return False

    def distance(self, model, image_files, image_size, margin, gpu_memory_fraction):

        images = self.load_and_align_data(
            image_files, image_size, margin, gpu_memory_fraction
        )

        # Run forward pass to calculate embeddings
        feed_dict = {
            self.images_placeholder: images,
            self.phase_train_placeholder: False,
        }
        emb = self.sess.run(self.embeddings, feed_dict=feed_dict)

        nrof_images = len(image_files)

        print("Images:")
        for i in range(nrof_images):
            print("%1d: %s" % (i, image_files[i]))
        print("")

        # Print distance matrix
        print("Distance matrix")
        print("    ", end="")
        for i in range(nrof_images):
            print("    %1d     " % i, end="")
        print("")
        for i in range(nrof_images):
            print("%1d  " % i, end="")
            for j in range(nrof_images):
                dist = np.sqrt(np.sum(np.square(np.subtract(emb[i, :], emb[j, :]))))
                print("  %1.4f  " % dist, end="")
            print("")
            if i == 0:
                self.conf = (2 - dist) * 50
                print("id : %1d Confidence : %1.4f" % (i, self.conf))

    def load_and_align_data(self, image_paths, image_size, margin, gpu_memory_fraction):

        tmp_image_paths = image_paths.copy()
        img_list = []
        for image in tmp_image_paths:
            img = misc.imread(os.path.expanduser(image), mode="RGB")
            img_size = np.asarray(img.shape)[0:2]
            bounding_boxes, _ = detect_face.detect_face(
                img,
                self.minsize,
                self.pnet,
                self.rnet,
                self.onet,
                self.threshold,
                self.factor,
            )
            if len(bounding_boxes) < 1:
                image_paths.remove(image)
                print("can't detect face, remove ", image)
                continue
            det = np.squeeze(bounding_boxes[0, 0:4])
            bb = np.zeros(4, dtype=np.int32)
            bb[0] = np.maximum(det[0] - margin / 2, 0)
            bb[1] = np.maximum(det[1] - margin / 2, 0)
            bb[2] = np.minimum(det[2] + margin / 2, img_size[1])
            bb[3] = np.minimum(det[3] + margin / 2, img_size[0])
            cropped = img[bb[1] : bb[3], bb[0] : bb[2], :]
            aligned = misc.imresize(
                cropped, (image_size, image_size), interp="bilinear"
            )
            prewhitened = facenet.prewhiten(aligned)
            img_list.append(prewhitened)
        images = np.stack(img_list)
        return images

    def countdown(self, t):
        while t:
            mins, secs = divmod(t, 60)
            timeformat = "{:02d}:{:02d}".format(mins, secs)
            print(timeformat, end="\r")
            time.sleep(1)
            t -= 1

    def getRandomImage(self, path):
        """function loads a random images from a random folder in our test path """
        random_filename = random.choice(
            [x for x in os.listdir(path) if os.path.isfile(os.path.join(path, x))]
        )

        return random_filename

        ###return image.load_img(final_path, target_size = (img_width, img_height)), final_path, path_class


kv = """
<Cell>:
    size_hint: (None, None)
    width: 200
    height: 160
    canvas.before:
        Color:
            rgba: [0.23, 0.23, 0.23, 1] if self.is_even else [0.2, 0.2, 0.2, 1]
        Rectangle:
            pos: self.pos
            size: self.size

<Table>:
    grid: grid
    bar_width: 15
    scroll_type: ['bars', 'content']
    bar_color: [0.4, 0.7, 0.9, 1]
    bar_inactive_color: [0.2, 0.7, 0.9, .5]
    do_scroll_x: False
    do_scroll_y: True
    GridLayout:
        id: grid
        cols: 5
        spacing: 5
        size_hint: (None, None)
        width: self.minimum_width
        height: self.minimum_height
<Cell2>:
    size_hint: (None, None)
    width: 100
    height: 30
    canvas.before:
        Color:
            rgba: [0.23, 0.23, 0.23, 1] if self.is_even else [0.2, 0.2, 0.2, 1]
        Rectangle:
            pos: self.pos
            size: self.size
<Table2>:
    grid: grid
    bar_width: 15
    scroll_type: ['bars', 'content']
    bar_color: [0.4, 0.7, 0.9, 1]
    bar_inactive_color: [0.2, 0.7, 0.9, .5]
    do_scroll_x: False
    do_scroll_y: True
    GridLayout:
        id: grid
        cols: 3
        spacing: 5
        size_hint: (None, None)
        width: self.minimum_width
        height: self.minimum_height

<Cell3>:
    size_hint: (None, None)
    width: 200
    height: 160
    canvas.before:
        Color:
            rgba: [0.23, 0.23, 0.23, 1] if self.is_even else [0.2, 0.2, 0.2, 1]
        Rectangle:
            pos: self.pos
            size: self.size
<Table3>:
    grid: grid
    bar_width: 15
    scroll_type: ['bars', 'content']
    bar_color: [0.4, 0.7, 0.9, 1]
    bar_inactive_color: [0.2, 0.7, 0.9, .5]
    do_scroll_x: True
    do_scroll_y: True
    GridLayout:
        id: grid
        cols: 8
        spacing: 5
        size_hint: (None, None)
        width: self.minimum_width
        height: self.minimum_height

"""

Builder.load_string(kv)


class Cell(Label):
    is_even = BooleanProperty(None)


class Cell2(Label):
    is_even = BooleanProperty(None)


class Cell3(Label):
    is_even = BooleanProperty(None)


class Table2(ScrollView):

    grid = ObjectProperty(None)

    def __init__(self, *args, **kwargs):
        super(Table2, self).__init__(*args, **kwargs)

        conn = mysql.connector.connect(
            host="127.0.0.1", user="root", password="", database="employees"
        )
        cursor = conn.cursor()

        query = "SELECT * FROM results"

        cursor.execute(query)

        rows = cursor.fetchall()
        self.grid.clear_widgets()
        # emp pic, emp name, emp clock in time, emp clock out time, emp access state
        self.grid.add_widget(Cell2(text="id"))
        self.grid.add_widget(Cell2(text="Class Name"))
        self.grid.add_widget(Cell2(text="Confidence"))

        for row in rows:
            for col in row:
                # text = "data row: {}, column: {}".format(i, j)

                self.grid.add_widget(Cell2(text=str(col)))


class Table(ScrollView):

    grid = ObjectProperty(None)

    def __init__(self, *args, **kwargs):
        super(Table, self).__init__(*args, **kwargs)

        conn = mysql.connector.connect(
            host="127.0.0.1", user="root", password="", database="employees"
        )
        cursor = conn.cursor()

        query = "SELECT emp_blobpath, emp_name, emp_clockintime, emp_clockouttime, emp_access_state FROM employee_checks"

        cursor.execute(query)

        rows = cursor.fetchall()

        self.grid.clear_widgets()
        # emp pic, emp name, emp clock in time, emp clock out time, emp access state
        self.grid.add_widget(Cell(text="DP"))
        self.grid.add_widget(Cell(text="Name"))
        self.grid.add_widget(Cell(text="Clock In Time"))
        self.grid.add_widget(Cell(text="Clock Out Time"))
        self.grid.add_widget(Cell(text="Access State"))

        for row in rows:
            for col in row:
                # text = "data row: {}, column: {}".format(i, j)

                if "." in str(col):

                    img = Image(source=str(col))
                    img.allow_stretch = False
                    img.keep_ratio = False
                    img.size_hint_x = None
                    img.size_hint_y = None
                    img.width = 200
                    img.height = 160

                    self.grid.add_widget(img)

                elif str(col) == "1":
                    img = Image(source="lgn3.gif")
                    img.allow_stretch = False
                    img.keep_ratio = False
                    img.size_hint_x = None
                    img.size_hint_y = None
                    img.width = 200
                    img.height = 160

                    self.grid.add_widget(img)

                elif str(col) == "0":

                    img = Image(source="lgno1.gif")
                    img.allow_stretch = False
                    img.keep_ratio = False
                    img.size_hint_x = None
                    img.size_hint_y = None
                    img.width = 200
                    img.height = 160

                    self.grid.add_widget(img)

                else:

                    self.grid.add_widget(Cell(text=str(col)))


class Table3(ScrollView):
    grid = ObjectProperty(None)

    def __init__(self, *args, **kwargs):
        super(Table3, self).__init__(*args, **kwargs)

        conn = mysql.connector.connect(
            host="127.0.0.1", user="root", password="", database="employees"
        )
        cursor = conn.cursor()

        query = "SELECT emp_blobpath ,emp_name, emp_surname, emp_phoneNo, emp_email, emp_gender, emp_dob, emp_age FROM employee_details"

        cursor.execute(query)

        rows = cursor.fetchall()

        self.grid.clear_widgets()
        # emp pic, emp name, emp clock in time, emp clock out time, emp access state
        self.grid.add_widget(Cell3(text="DP"))
        self.grid.add_widget(Cell3(text="Name"))
        self.grid.add_widget(Cell3(text="Surname"))
        self.grid.add_widget(Cell3(text="Phone Number"))
        self.grid.add_widget(Cell3(text="Email"))
        self.grid.add_widget(Cell3(text="Gender"))
        self.grid.add_widget(Cell3(text="Date of Birth"))
        self.grid.add_widget(Cell3(text="Age"))

        for row in rows:
            for col in row:

                if ".jpg" in str(col):

                    img = Image(source=str(col))
                    img.allow_stretch = False
                    img.keep_ratio = False
                    img.size_hint_x = None
                    img.size_hint_y = None
                    img.width = 200
                    img.height = 160

                    self.grid.add_widget(img)

                else:

                    self.grid.add_widget(Cell3(text=str(col)))


class Main_app(App):
    def build(self):
        

        # We are going to use screen manager, so we can add multiple screens
        # and switch between them
        self.screen_manager = ScreenManager()

        # Initial, connection screen (we use passed in name to activate screen)
        # First create a page, then a new screen, add page to screen and screen to screen manager
        self.clockin = auth()
        screen = Screen(name="clockin")
        screen.add_widget(self.clockin)
        self.screen_manager.add_widget(screen)

        # Employee page
        self.RegEmp = Employee()
        screen = Screen(name="employee")
        screen.add_widget(self.RegEmp)
        self.screen_manager.add_widget(screen)

        # Collect Data page
        self.collect = CollectData()
        screen = Screen(name="collect")
        screen.add_widget(self.collect)
        self.screen_manager.add_widget(screen)

        # Train And test Page
        self.trainAndtest = TrainAndTest()
        screen = Screen(name="trainNtest")
        screen.add_widget(self.trainAndtest)
        self.screen_manager.add_widget(screen)

        # Report Page
        self.rep = Report()
        screen = Screen(name="report")
        screen.add_widget(self.rep)
        self.screen_manager.add_widget(screen)

        return self.screen_manager


class TrainAndTest(FloatLayout):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        ###layout = FloatLayout()
        dataset = facenet.get_dataset("employee_data\\train_raw")
        paths, labels = facenet.get_image_paths_and_labels(dataset)

        class_names = [cls.name.replace("_", " ") for cls in dataset]
        image_paths = []

        for cls in dataset:
            for path in cls.image_paths:

                image_paths.append(path)

        self.class_len = len(class_names)
        self.len_paths = len(image_paths)
        self.now = f"Showing All Classes : {self.class_len} No of Images : {str(self.len_paths)}"
        images = []

        i = 0
        k = 0
        for no in range(self.len_paths):

            images.append(Image())
            images[k].allow_stretch = False
            images[k].keep_ratio = False
            images[k].size_hint_x = None
            images[k].size_hint_y = None
            images[k].width = 160
            images[k].height = 160
            k += 1

        random.shuffle(image_paths)
        for img in image_paths:

            images[i].source = str(img)
            i += 1

        self.Grid = GridLayout(cols=6, spacing=5, size_hint_y=None)
        # Grid.size_hint_x = None
        # Grid.size_hint_y = None
        # Grid.width = 350
        # Grid.height = 300
        # Grid.pos_hint = {"x":0.25, "y":0.53}
        self.Grid.bind(minimum_height=self.Grid.setter("height"))
        for wid in images:
            self.Grid.add_widget(wid)

        scrollpane = ScrollView(
            size_hint=(1, None), size=(1000, 630), pos_hint={"x": 0.25, "y": 0.05}
        )
        scrollpane.bar_width = 25
        scrollpane.scroll_type = ["bars", "content"]
        scrollpane.bar_color = [0.4, 0.7, 0.9, 1]
        scrollpane.bar_inactive_color = [0.2, 0.7, 0.9, 0.5]
        scrollpane.do_scroll_x = False
        scrollpane.do_scroll_y = True
        scrollpane.add_widget(self.Grid)

        self.margin = 44
        self.gpu_memory_fraction = 1.0
        self.random_order = True
        self.detect_multiple_faces = False
        self.mode = ["TRAIN", "CLASSIFY"]
        self.use_split_database = False
        self.seed = 666
        self.image_size = 160
        self.batch_size = 90
        self.min_nrof_images_per_class = 20
        self.nrof_train_images_per_class = 10
        self.res = {}

        self.btnMain = Button(
            text="Go to Main Page",
            font_size=12,
            size_hint_y=None,
            size_hint_x=None,
            width=150,
            height=30,
            pos_hint={"x": 0.25, "y": 0.004},
        )
        self.btnMain.bind(on_press=self.main)

        self.btnEmp = Button(
            text="Go to Employee Page",
            font_size=12,
            size_hint_y=None,
            size_hint_x=None,
            width=150,
            height=30,
            pos_hint={"x": 0.37, "y": 0.004},
        )
        self.btnEmp.bind(on_press=self.emp)

        self.btnCollectData = Button(
            text="Go to Collect Data Page",
            font_size=12,
            size_hint_y=None,
            size_hint_x=None,
            width=150,
            height=30,
            pos_hint={"x": 0.49, "y": 0.004},
        )
        self.btnCollectData.bind(on_press=self.cd)

        self.btnReport = Button(
            text="Go to Report Page",
            font_size=12,
            size_hint_y=None,
            size_hint_x=None,
            width=150,
            height=30,
            pos_hint={"x": 0.61, "y": 0.004},
        )
        self.btnReport.bind(on_press=self.rp)

        self.lblinput = Label(
            text="Input Dir",
            font_size=12,
            size_hint_y=None,
            size_hint_x=None,
            width=150,
            height=30,
            pos_hint={"x": 0.01, "y": 0.90},
        )
        self.lbloutput = Label(
            text="Output Dir",
            font_size=12,
            size_hint_y=None,
            size_hint_x=None,
            width=150,
            height=30,
            pos_hint={"x": 0.01, "y": 0.85},
        )
        self.lblimgsize = Label(
            text="Image Size",
            font_size=12,
            size_hint_y=None,
            size_hint_x=None,
            width=150,
            height=30,
            pos_hint={"x": 0.01, "y": 0.80},
        )
        self.lblData = Label(
            text="Data Dir",
            font_size=12,
            size_hint_y=None,
            size_hint_x=None,
            width=150,
            height=30,
            pos_hint={"x": 0.01, "y": 0.60},
        )
        self.lblmode = Label(
            text="Mode",
            font_size=12,
            size_hint_y=None,
            size_hint_x=None,
            width=150,
            height=30,
            pos_hint={"x": 0.01, "y": 0.55},
        )
        self.lblModel = Label(
            text="Model Name",
            font_size=12,
            size_hint_y=None,
            size_hint_x=None,
            width=150,
            height=30,
            pos_hint={"x": 0.01, "y": 0.50},
        )
        self.lblClassifier = Label(
            text="Classifier Name",
            font_size=12,
            size_hint_y=None,
            size_hint_x=None,
            width=150,
            height=30,
            pos_hint={"x": 0.01, "y": 0.45},
        )
        self.lblsearch = Label(
            text="Search",
            font_size=12,
            size_hint_y=None,
            size_hint_x=None,
            width=150,
            height=30,
            pos_hint={"x": 0.65, "y": 0.95},
        )
        self.lblAct = Label(
            text=self.now,
            font_size=12,
            size_hint_y=None,
            size_hint_x=None,
            width=150,
            height=30,
            pos_hint={"x": 0.28, "y": 0.95},
        )

        self.txtinput = Spinner(
            text="employee_data\\train_raw",
            values=("employee_data\\train_raw", "employee_data\\test_raw"),
            background_color=(0.784, 0.443, 0.216, 1),
            font_size=12,
            size_hint_y=None,
            size_hint_x=None,
            width=180,
            height=30,
            pos_hint={"x": 0.1, "y": 0.90},
        )
        self.txtoutput = Spinner(
            text="employee_data\\train_aligned",
            values=("employee_data\\train_aligned", "employee_data\\test_aligned"),
            background_color=(0.784, 0.443, 0.216, 1),
            font_size=12,
            size_hint_y=None,
            size_hint_x=None,
            width=180,
            height=30,
            pos_hint={"x": 0.1, "y": 0.85},
        )
        self.txtimgsize = TextInput(
            text="160",
            font_size=12,
            size_hint_y=None,
            size_hint_x=None,
            width=180,
            height=30,
            pos_hint={"x": 0.1, "y": 0.80},
        )
        self.txtData = Spinner(
            text="employee_data\\train_aligned",
            values=("employee_data\\train_aligned", "employee_data\\test_aligned"),
            background_color=(0.784, 0.443, 0.216, 1),
            font_size=12,
            size_hint_y=None,
            size_hint_x=None,
            width=180,
            height=30,
            pos_hint={"x": 0.1, "y": 0.60},
        )
        self.txtmode = Spinner(
            text="TRAIN",
            values=("TRAIN", "CLASSIFY"),
            background_color=(0.784, 0.443, 0.216, 1),
            font_size=12,
            size_hint_y=None,
            size_hint_x=None,
            width=180,
            height=30,
            pos_hint={"x": 0.1, "y": 0.55},
        )
        self.txtmodel = Spinner(
            text="20180402-114759.pb",
            values=("20180402-114759.pb",),
            font_size=12,
            size_hint_y=None,
            size_hint_x=None,
            width=180,
            height=30,
            pos_hint={"x": 0.1, "y": 0.50},
        )
        self.txtClassifier = TextInput(
            text="employees.pkl",
            font_size=12,
            size_hint_y=None,
            size_hint_x=None,
            width=180,
            height=30,
            pos_hint={"x": 0.1, "y": 0.45},
        )
        self.txtsearch = TextInput(
            text="",
            font_size=12,
            size_hint_y=None,
            size_hint_x=None,
            width=160,
            height=30,
            pos_hint={"x": 0.73, "y": 0.95},
        )

        self.btnAlign = Button(
            text="Align Data",
            font_size=12,
            size_hint_y=None,
            size_hint_x=None,
            width=150,
            height=30,
            pos_hint={"x": 0.1, "y": 0.75},
        )
        self.btnTrain = Button(
            text="Train",
            font_size=12,
            size_hint_y=None,
            size_hint_x=None,
            width=150,
            height=30,
            pos_hint={"x": 0.1, "y": 0.35},
        )
        self.btnTest = Button(
            text="Test",
            font_size=12,
            size_hint_y=None,
            size_hint_x=None,
            width=150,
            height=30,
            pos_hint={"x": 0.1, "y": 0.30},
        )
        self.btnResult = Button(
            text="Show Results",
            font_size=12,
            size_hint_y=None,
            size_hint_x=None,
            width=150,
            height=30,
            pos_hint={"x": 0.1, "y": 0.25},
        )
        self.btnsearch = Button(
            text="Search",
            font_size=12,
            size_hint_y=None,
            size_hint_x=None,
            width=150,
            height=30,
            pos_hint={"x": 0.86, "y": 0.95},
        )

        self.btnAlign.bind(on_press=self.callAlign)
        self.btnTrain.bind(on_press=self.callTrainNTest)
        self.btnTest.bind(on_press=self.callTrainNTest)
        self.btnsearch.bind(on_press=self.search)
        self.btnResult.bind(on_press=self.show_Res)

        widgets = [
            self.btnReport,
            self.btnMain,
            self.btnEmp,
            self.btnCollectData,
            self.lblAct,
            self.lblsearch,
            self.txtsearch,
            self.btnsearch,
            self.lblClassifier,
            self.lblData,
            self.lblimgsize,
            self.lblinput,
            self.lblmode,
            self.lblModel,
            self.lbloutput,
            self.txtClassifier,
            self.txtData,
            self.txtimgsize,
            self.txtinput,
            self.txtmode,
            self.txtmodel,
            self.txtoutput,
            self.btnAlign,
            self.btnTrain,
            self.btnTest,
            self.btnResult,
            scrollpane,
        ]

        for wid in widgets:
            self.add_widget(wid)

        Clock.schedule_interval(self.nw, 1.0 / 33.0)

    def sched(self, func):
        threading.Thread(
            target=Clock.schedule_interval, args=(func, 1.0 / 10000.0), daemon=True
        ).start()

    def setImageSource(self, imgpaths, dt):
        n = 0
        for img in imgpaths:
            
            
            self.images[n].source = str(img)
            
            n += 1

    def setImage(self, imgpaths, dt):
        m = 0
        for img in imgpaths:

            
            self.images[m].source = str(img)
            
            m += 1

    def main(self, instance):
        app_.screen_manager.current = "clockin"
        self.sched(app_.clockin.update)

    def emp(self, instance):
        app_.screen_manager.current = "employee"

    def cd(self, instance):
        app_.screen_manager.current = "collect"

    def rp(self, instance):
        app_.screen_manager.current = "report"

    def nw(self, dt):

        self.lblAct.text = self.now

    def callSearch(self, instance):
        threading.Thread(target=self.search, daemon=True).start()


    
    def search(self, instance):

        if(self.txtsearch.text.isnumeric() == True):
            self.pop("Error", f"Numbers and Characters({self.txtsearch.text}) are not allowed!")
            return 
        
        if(cdigit(self.txtsearch.text) == True):
            self.pop("Error", f"Numbers and Characters({self.txtsearch.text}) are not allowed!")
            return 
        
        if re.match("^[a-zA-Z]", self.txtsearch.text) == False:
            self.pop("Error", f"Numbers and Characters({self.txtsearch.text}) are not allowed!")
            return 
        
        if dChar(self.txtsearch.text) == True:
            self.pop("Error", f"Numbers and Characters({self.txtsearch.text}) are not allowed!")
            return 

        if (
            self.txtsearch.text == "all"
            or self.txtsearch.text == "All"
            or self.txtsearch.text == "ALL"
            or self.txtsearch.text == "default"
            or self.txtsearch.text == "Default"
        ):

            query = f"SELECT features FROM employees_traindata"

            app_.clockin.cursor.execute(query)

            result = app_.clockin.cursor.fetchall()
            result = list(sum(result, ()))

            self.Grid.clear_widgets()
            len_path = len(result)
            #print("len of paths : ", len_path, result)
            self.now = (
                f"Showing All Classes : {self.class_len} No of Images : {str(len_path)}"
            )
            self.images = []

            
            k = 0
            print("creating images")
            for no in range(len_path):
                
                self.images.append(Image())
                self.images[k].allow_stretch = False
                self.images[k].keep_ratio = False
                self.images[k].size_hint_x = None
                self.images[k].size_hint_y = None
                self.images[k].width = 160
                self.images[k].height = 160
                k += 1
            print("Done creating images")
            random.shuffle(result)

            
            v =0
            print("assigning images to their sources")
            for im in result:
                self.images[v].source =str(im)
                v+=1
            Clock.schedule_once(partial(self.setImageSource, result))
            print("done assigning images")
            #time.sleep(5)
            
            
            
            try:
                print("Adding widgets to grid")
                for wid in self.images:
                    self.Grid.add_widget(wid)
                print("done adding widgets")
            except:
                print("Error while trying a add widgets")

            return

        if self.txtsearch.text == "":
            self.pop("INFO : MESSAGE", "Search field cannot be empty!")
            return

        else:

            query = f"SELECT features FROM employees_traindata WHERE labels = '{self.txtsearch.text}'"

            app_.clockin.cursor.execute(query)

            result = app_.clockin.cursor.fetchall()
            result = list(sum(result, ()))

            if len(result) <= 0:
                self.pop(
                    "INFO : MESSAGE",
                    f"NO SUCH LABEL({self.txtsearch.text}) IN THE DATABASE",
                )

            else:

                self.Grid.clear_widgets()
                len_path = len(result)
                print("len of paths : ", len_path, result)
                self.now = f"Showing One Class : {self.txtsearch.text}, No of Images : {str(len_path)}"
                self.images = []
          
                k = 0
                for no in range(len_path):

                    self.images.append(Image())
                    self.images[k].allow_stretch = False
                    self.images[k].keep_ratio = False
                    self.images[k].size_hint_x = None
                    self.images[k].size_hint_y = None
                    self.images[k].width = 160
                    self.images[k].height = 160
                    k += 1

                random.shuffle(result)

                
                v =0
                for im in result:
                    self.images[v].source =str(im)
                    v+=1
                Clock.schedule_once(partial(self.setImageSource, result))
                #time.sleep(5)
                try:
                    for wid in self.images:
                        self.Grid.add_widget(wid)
                except:
                    print("Error while triyng to add widgets")

                return

    def pop(self, tle, txt):
        layout = GridLayout(cols=1, padding=10)

        popupLabel = Label(text=txt)
        closeButton = Button(text="close")

        layout.add_widget(popupLabel)

        layout.add_widget(closeButton)

        popup = Popup(title=tle, content=layout, size_hint=(0.3, 0.3))
        popup.open()
        closeButton.bind(on_press=popup.dismiss)

    def callAlign(self, instance):
        threading.Thread(target=self.align, daemon=True).start()

    def callTrainNTest(self, instance):
        threading.Thread(target=self.TrainNtest, daemon=True).start()

    def plot_learning_curve(self, estimator, title, X, y, ylim=None, cv=None, n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
        """
        Generate a simple plot of the test and training learning curve.
    
        Parameters
        ----------
        estimator : object type that implements the "fit" and "predict" methods
            An object of that type which is cloned for each validation.
    
        title : string
            Title for the chart.
    
        X : array-like, shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.
    
        y : array-like, shape (n_samples) or (n_samples, n_features), optional
            Target relative to X for classification or regression;
            None for unsupervised learning.
    
        ylim : tuple, shape (ymin, ymax), optional
            Defines minimum and maximum yvalues plotted.
    
        cv : int, cross-validation generator or an iterable, optional
            Determines the cross-validation splitting strategy.
            Possible inputs for cv are:
              - None, to use the default 3-fold cross-validation,
              - integer, to specify the number of folds.
              - :term:`CV splitter`,
              - An iterable yielding (train, test) splits as arrays of indices.
    
            For integer/None inputs, if ``y`` is binary or multiclass,
            :class:`StratifiedKFold` used. If the estimator is not a classifier
            or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.
    
            Refer :ref:`User Guide <cross_validation>` for the various
            cross-validators that can be used here.
    
        n_jobs : int or None, optional (default=None)
            Number of jobs to run in parallel.
            ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
            ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
            for more details.
    
        train_sizes : array-like, shape (n_ticks,), dtype float or int
            Relative or absolute numbers of training examples that will be used to
            generate the learning curve. If the dtype is float, it is regarded as a
            fraction of the maximum size of the training set (that is determined
            by the selected validation method), i.e. it has to be within (0, 1].
            Otherwise it is interpreted as absolute sizes of the training sets.
            Note that for classification the number of samples usually have to
            be big enough to contain at least one sample from each class.
            (default: np.linspace(0.1, 1.0, 5))
        """
        plt.figure()
        plt.title(title)
        if ylim is not None:
            plt.ylim(*ylim)
        plt.xlabel("Training examples")
        plt.ylabel("Score")
        train_sizes, train_scores, test_scores = learning_curve(
            estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)
        plt.grid()
    
        plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
        plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1, color="g")
        plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
        plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")
    
        plt.legend(loc="best")
        plt.show()
        return plt
    
    def align(self):

        if self.txtimgsize.text == "":
            self.pop("Error", f"Image Size cannot be empty")
            return 

        if(self.txtimgsize.text.isalpha() == True):
            self.pop("Error", f"Characters({self.txtimgsize.text}) are not allowed!")
            return 
        if re.match("^[0-9]", self.txtimgsize.text) == False:
            self.pop("Error", f"Characters({self.txtimgsize.text}) are not allowed!")
            return 
        
        #if dChar(self.txtimgsize.text) == True:
        #   self.pop("Error", f"Characters({self.txtimgsize.text}) are not allowed!")
        #   return 
        
        if(int(self.txtimgsize.text) <= 0):
            self.pop("Error", f"image size cannot be zero or less than zero")
            return 
        
        
        
        

        self.now = "Aligning Images"
        app_.clockin.speak(self.now)
        time.sleep(8)
        sleep(random.random())

        output_dir = os.path.expanduser(self.txtoutput.text)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        dataset = facenet.get_dataset(self.txtinput.text)

        print("Creating networks and loading parameters")
        self.now = "Creating networks and loading parameters"
        app_.clockin.speak(self.now)
        time.sleep(8)
        with tf.Graph().as_default():
            gpu_options = tf.compat.v1.GPUOptions(
                per_process_gpu_memory_fraction=self.gpu_memory_fraction
            )
            sess = tf.compat.v1.Session(
                config=tf.compat.v1.ConfigProto(
                    gpu_options=gpu_options, log_device_placement=False
                )
            )
            with sess.as_default():
                pnet, rnet, onet = detect_face.create_mtcnn(sess, None)

        minsize = 20  # minimum size of face
        threshold = [0.6, 0.7, 0.8]  # three steps's threshold
        factor = 0.709  # scale factor

        random_key = np.random.randint(0, high=99999)
        bounding_boxes_filename = os.path.join(
            output_dir, "bounding_boxes_%05d.txt" % random_key
        )

        with open(bounding_boxes_filename, "w") as text_file:
            nrof_images_total = 0
            nrof_successfully_aligned = 0
            if self.random_order:
                random.shuffle(dataset)
            for cls in dataset:
                output_class_dir = os.path.join(output_dir, cls.name)
                if not os.path.exists(output_class_dir):
                    os.makedirs(output_class_dir)
                    if self.random_order:
                        random.shuffle(cls.image_paths)
                for image_path in cls.image_paths:
                    nrof_images_total += 1
                    filename = os.path.splitext(os.path.split(image_path)[1])[0]
                    output_filename = os.path.join(output_class_dir, filename + ".png")

                    if not os.path.exists(output_filename):
                        try:
                            img = imageio.imread(image_path)
                        except (IOError, ValueError, IndexError) as e:
                            errorMessage = "{}: {}".format(image_path, e)
                            print(errorMessage)
                        else:
                            if img.ndim < 2:
                                print('Unable to align "%s"' % image_path)
                                os.remove(image_path)
                                
                                self.now = 'Unable to align "%s"' % image_path
                                app_.clockin.speak(self.now)
                                time.sleep(5)
                                text_file.write("%s\n" % (output_filename))
                                continue
                            if img.ndim == 2:
                                img = facenet.to_rgb(img)
                            img = img[:, :, 0:3]

                            bounding_boxes, _ = detect_face.detect_face(
                                img, minsize, pnet, rnet, onet, threshold, factor
                            )
                            nrof_faces = bounding_boxes.shape[0]
                            if nrof_faces > 0:
                                det = bounding_boxes[:, 0:4]
                                det_arr = []
                                img_size = np.asarray(img.shape)[0:2]
                                if nrof_faces > 1:
                                    if self.detect_multiple_faces:
                                        for i in range(nrof_faces):
                                            det_arr.append(np.squeeze(det[i]))
                                    else:
                                        bounding_box_size = (det[:, 2] - det[:, 0]) * (
                                            det[:, 3] - det[:, 1]
                                        )
                                        img_center = img_size / 2
                                        offsets = np.vstack(
                                            [
                                                (det[:, 0] + det[:, 2]) / 2
                                                - img_center[1],
                                                (det[:, 1] + det[:, 3]) / 2
                                                - img_center[0],
                                            ]
                                        )
                                        offset_dist_squared = np.sum(
                                            np.power(offsets, 2.0), 0
                                        )
                                        index = np.argmax(
                                            bounding_box_size
                                            - offset_dist_squared * 2.0
                                        )  # some extra weight on the centering
                                        det_arr.append(det[index, :])
                                else:
                                    det_arr.append(np.squeeze(det))

                                for i, det in enumerate(det_arr):
                                    det = np.squeeze(det)
                                    bb = np.zeros(4, dtype=np.int32)
                                    bb[0] = np.maximum(det[0] - self.margin / 2, 0)
                                    bb[1] = np.maximum(det[1] - self.margin / 2, 0)
                                    bb[2] = np.minimum(
                                        det[2] + self.margin / 2, img_size[1]
                                    )
                                    bb[3] = np.minimum(
                                        det[3] + self.margin / 2, img_size[0]
                                    )
                                    cropped = img[bb[1] : bb[3], bb[0] : bb[2], :]
                                    scaled = misc.imresize(
                                        cropped,
                                        (
                                            int(self.txtimgsize.text),
                                            int(self.txtimgsize.text),
                                        ),
                                        interp="bilinear",
                                    )
                                    nrof_successfully_aligned += 1
                                    self.now = f"Aligned : {nrof_successfully_aligned} / {nrof_images_total}"
                                    filename_base, file_extension = os.path.splitext(
                                        output_filename
                                    )
                                    if self.detect_multiple_faces:
                                        output_filename_n = "{}_{}{}".format(
                                            filename_base, i, file_extension
                                        )
                                    else:
                                        output_filename_n = "{}{}".format(
                                            filename_base, file_extension
                                        )
                                    misc.imsave(output_filename_n, scaled)
                                    text_file.write(
                                        "%s %d %d %d %d\n"
                                        % (
                                            output_filename_n,
                                            bb[0],
                                            bb[1],
                                            bb[2],
                                            bb[3],
                                        )
                                    )
                            else:
                                print('Unable to align "%s"' % image_path)
                                os.remove(image_path)
                                self.now = 'Unable to align "%s"' % image_path
                                app_.clockin.speak(self.now)
                                time.sleep(4)
                                
                                text_file.write("%s\n" % (output_filename))

        print("Total number of images: %d" % nrof_images_total)
        app_.clockin.speak("Total number of images: %d" % nrof_images_total)
        time.sleep(8)
        print("Number of successfully aligned images: %d" % nrof_successfully_aligned)
        app_.clockin.speak("Number of successfully aligned images: %d" % nrof_successfully_aligned)
        time.sleep(8)
        self.now = "Alignment Done."
        app_.clockin.speak(self.now)
        time.sleep(8)
       
        self.now = f"Showing All Classes : {self.class_len} No of Images : {str(self.len_paths)}"
        app_.clockin.speak(self.now)
        time.sleep(8)
    def TrainNtest(self):
        if(self.txtClassifier.text.isnumeric() == True):
            self.pop("Error", f"Numbers ({self.txtClassifier.text}) are not allowed!")
            return 
        
        if(cdigit(self.txtClassifier.text) == True):
            self.pop("Error", f"Numbers ({self.txtClassifier.text}) are not allowed!")
            return 
        
        if re.match("^[a-zA-Z.]", self.txtClassifier.text) == False:
            self.pop("Error", f"Numbers ({self.txtClassifier.text}) are not allowed!")
            return 
        
        #if dChar(self.txtsearch.text) == True:
        #    self.pop("Error", f"Numbers and Characters({self.txtClassifier.text}) are not allowed!")
        #    return
        
        
        with tf.Graph().as_default():

            with tf.compat.v1.Session() as sess:

                np.random.seed(seed=self.seed)

                if self.use_split_database:
                    dataset_tmp = facenet.get_dataset(self.txtData.text)
                    train_set, test_set = self.split_dataset(
                        dataset_tmp,
                        self.min_nrof_images_per_class,
                        self.nrof_train_images_per_class,
                    )
                    if self.txtmode.text == "TRAIN":
                        dataset = train_set
                    elif self.txtmode.text == "CLASSIFY":
                        dataset = test_set
                else:
                    dataset = facenet.get_dataset(self.txtData.text)

                # Check that there are at least one training image per class
                for cls in dataset:
                    assert (
                        len(cls.image_paths) > 0,
                        "There must be at least one image for each class in the dataset",
                    )

                paths, labels = facenet.get_image_paths_and_labels(dataset)

                print("Number of classes: %d" % len(dataset))
                print("Number of images: %d" % len(paths))
                self.now = (
                    "Number of classes: %d" % len(dataset)
                    + ".,"
                    + "Number of images: %d" % len(paths)
                )
                app_.clockin.speak("The Number of classes: %d" % len(dataset)
                    + " and the "
                    + "Number of images: %d" % len(paths))
                time.sleep(8)
                # Load the model
                print("Loading feature extraction model")
                self.now = "Loading feature extraction model"
                
                app_.clockin.speak(self.now)
                time.sleep(8)
                facenet.load_model(self.txtmodel.text)

                # Get input and output tensors
                images_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name(
                    "input:0"
                )
                embeddings = tf.compat.v1.get_default_graph().get_tensor_by_name(
                    "embeddings:0"
                )
                phase_train_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name(
                    "phase_train:0"
                )
                embedding_size = embeddings.get_shape()[1]

                # Run forward pass to calculate embeddings
                print("Calculating features for images")
                self.now = "Calculating features for images. Please wait"
                app_.clockin.speak(self.now)
                time.sleep(8)
                
                start = time.time()
                nrof_images = len(paths)
                nrof_batches_per_epoch = int(
                    math.ceil(1.0 * nrof_images / self.batch_size)
                )
                emb_array = np.zeros((nrof_images, embedding_size))
                for i in range(nrof_batches_per_epoch):
                    start_index = i * self.batch_size
                    end_index = min((i + 1) * self.batch_size, nrof_images)
                    paths_batch = paths[start_index:end_index]
                    images = facenet.load_data(
                        paths_batch, False, False, int(self.txtimgsize.text)
                    )
                    feed_dict = {
                        images_placeholder: images,
                        phase_train_placeholder: False,
                    }
                    emb_array[start_index:end_index, :] = sess.run(
                        embeddings, feed_dict=feed_dict
                    )
                end = time.time() - start
                self.now = "Calculating features for images done. Process took %.2f seconds." % end
                app_.clockin.speak(self.now)
                time.sleep(8)
                
                classifier_filename_exp = os.path.expanduser(self.txtClassifier.text)

                if self.txtmode.text == "TRAIN" and self.btnTrain.text == "Train":
                    # Train classifier
                    print("Training classifier")
                    self.now = "Training classifier"
                    app_.clockin.speak(self.now)
                    time.sleep(8)
                    model = SVC(kernel="linear", probability=True)
                    try:
                        title = "Learning Curves (SVC linear)"
                        cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
                        self.plot_learning_curve(model, title, emb_array, labels, ylim=(0.7, 1.01), cv=cv, n_jobs=4).show
                        #plt.show()
                    except:
                        print("Couldn't display graph")
                    
                    #model = GaussianNB()
                    model.fit(emb_array, labels)
                    
                
                

                    # Create a list of class names
                    class_names = [cls.name.replace("_", " ") for cls in dataset]

                    # Saving classifier model
                    with open(classifier_filename_exp, "wb") as outfile:
                        pickle.dump((model, class_names), outfile)
                    print(
                        'Saved classifier model to file "%s"' % classifier_filename_exp
                    )
                    self.now = (
                        'Saved classifier model to file "%s"' % classifier_filename_exp
                    )
                    app_.clockin.speak(self.now)
                    time.sleep(8)
                    app_.clockin.speak("I'm done training your classifier %s, and I have saved it." % classifier_filename_exp)
                    self.now = f"Showing All Classes : {self.class_len} No of Images : {str(self.len_paths)}"
                    time.sleep(8)
                    app_.clockin.speak(f"Showing All {self.class_len} Classes and the number of total Images : {str(self.len_paths)}")
                elif self.txtmode.text == "CLASSIFY" and self.btnTest.text == "Test":
                    # Classify images
                    print("Testing classifier")
                    self.now = "Testing classifier"
                    app_.clockin.speak(self.now)
                    time.sleep(8)
                    with open(classifier_filename_exp, "rb") as infile:
                        (model, class_names) = pickle.load(infile)

                    print(
                        'Loaded classifier model from file "%s"'
                        % classifier_filename_exp
                    )
                    self.now = (
                        'Loaded classifier model from file "%s"'
                        % classifier_filename_exp
                    )
                    app_.clockin.speak('Loaded classifier model from file "%s"' % classifier_filename_exp)
                    time.sleep(8)
                    predictions = model.predict_proba(emb_array)
                    best_class_indices = np.argmax(predictions, axis=1)
                    best_class_probabilities = predictions[
                        np.arange(len(best_class_indices)), best_class_indices
                    ]
                    
                    app_.clockin.speak("These are the results for each Class by its Confidence Probability")
                    time.sleep(8)
                    ind = 1
                    for i in range(len(best_class_indices)):
                        print(
                            "%4d  %s: %.3f"
                            % (
                                i,
                                class_names[best_class_indices[i]],
                                best_class_probabilities[i],
                              )
                        )
                        app_.clockin.speak("Number: "+str(ind)+ " " + class_names[best_class_indices[i]]+" "+str(round(best_class_probabilities[i] * 100,3)) + " % ")

                        self.res[class_names[best_class_indices[i]]] = (
                            best_class_probabilities[i] * 100
                        )
                        
                        fl = round(best_class_probabilities[i], 3)
                        query = f"Update results SET  confidence_probability = {fl} WHERE class_name = '{class_names[best_class_indices[i]]}';"
                        app_.clockin.cursor.execute(query)

                        app_.clockin.conn.commit()
                        time.sleep(8)
                        ind+=1
                    bcp = best_class_probabilities
                    bcplen = len(best_class_probabilities)
                    self.accuracy = np.divide(np.sum(bcp), bcplen)
                    #self.accuracy = np.mean(np.equal(best_class_indices, labels))
                    print("Accuracy: %.3f" % self.accuracy)
                    app_.clockin.speak("Accuracy: %.3f percent" % (self.accuracy *100))
                    time.sleep(8)
                    app_.clockin.speak("I'm done testing your classifier %s, and I have saved it." % classifier_filename_exp)

                    time.sleep(8)
                    self.now = f"Showing All Classes : {self.class_len} No of Images : {str(self.len_paths)}"
                    app_.clockin.speak(f"Showing All {self.class_len} Classes and the number of total Images : {str(self.len_paths)}")
    def show_Res(self, instance):
        
        
        layout = GridLayout(cols=1, padding=5)

        length = len(self.res)

        if length >= 1:
            app_.clockin.speak(f"Showing Results., There is a total number of {self.class_len} Classes")
            time.sleep(4)
            self.name = Label(text="Class    :Confidence Probability")
            layout.add_widget(self.name)

            lbls = []
            for l in range(length):
                lbls.append(Label(text=""))

            classes = []
            for i in self.res:
                classes.append(i)

            k = 0
            j = 0
            for lbl in lbls:

                for i in self.res:

                    if classes[k] == i:
                        lbl.text = i + " :" + str(self.res[i])

                k += 1

            for wid in lbls:
                layout.add_widget(wid)

            self.lblAcc = Label(text="Accuracy : " + str(self.accuracy * 100))

            layout.add_widget(self.lblAcc)

            closeButton = Button(text="close")

            layout.add_widget(closeButton)

            popup = Popup(title="Results", content=layout, size_hint=(0.5, 0.8))
            popup.open()
            closeButton.bind(on_press=popup.dismiss)
        else:
            app_.clockin.speak(f"No results available!")
            time.sleep(4)
            self.lblNoRes = Label(text="No results available!")
            layout.add_widget(self.lblNoRes)

            closeButton = Button(text="close")

            layout.add_widget(closeButton)

            popup = Popup(title="No Results", content=layout, size_hint=(0.6, 0.3))
            popup.open()
            closeButton.bind(on_press=popup.dismiss)

    def split_dataset(
        self, dataset, min_nrof_images_per_class, nrof_train_images_per_class
    ):
        train_set = []
        test_set = []
        for cls in dataset:
            paths = cls.image_paths
            # Remove classes with less than min_nrof_images_per_class
            if len(paths) >= min_nrof_images_per_class:
                np.random.shuffle(paths)
                train_set.append(
                    facenet.ImageClass(cls.name, paths[:nrof_train_images_per_class])
                )
                test_set.append(
                    facenet.ImageClass(cls.name, paths[nrof_train_images_per_class:])
                )
        return train_set, test_set


class CollectData(FloatLayout):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        ###layout = FloatLayout()
        self.nw = ""
        ###self.face_recognition = f.Recognition()
        self.user = Image(
            source="C://Users//GabhaDi//eclipse-workspace2//Project Clock-in//data//Images//face_avatar.jpg"
        )
        self.trainingSample = Image(
            source="C://Users//GabhaDi//eclipse-workspace2//Project Clock-in//data//Images//face_avatar.jpg"
        )
        self.testSample = Image(
            source="C://Users//GabhaDi//eclipse-workspace2//Project Clock-in//data//Images//face_avatar.jpg"
        )

        self.lblTrain = Label(
            text="Training Sample",
            font_size=12,
            size_hint_y=None,
            size_hint_x=None,
            width=150,
            height=30,
            pos_hint={"x": 0.6, "y": 0.65},
        )
        self.lblTest = Label(
            text="Test Sample",
            font_size=12,
            size_hint_y=None,
            size_hint_x=None,
            width=150,
            height=30,
            pos_hint={"x": 0.6, "y": 0.37},
        )
        self.collect = Button(
            text="Collect Data",
            font_size=12,
            size_hint_y=None,
            size_hint_x=None,
            width=150,
            height=95,
            pos_hint={"x": 0.29, "y": 0.16},
        )
        self.collect.bind(on_press=self.cdata)
        ##self.capture = cv2.VideoCapture(0)

        self.btnMain = Button(
            text="Go to Main Page",
            font_size=12,
            size_hint_y=None,
            size_hint_x=None,
            width=150,
            height=30,
            pos_hint={"x": 0.23, "y": 0.004},
        )
        self.btnMain.bind(on_press=self.main)

        self.btnEmp = Button(
            text="Go to Employee Page",
            font_size=12,
            size_hint_y=None,
            size_hint_x=None,
            width=150,
            height=30,
            pos_hint={"x": 0.35, "y": 0.004},
        )
        self.btnEmp.bind(on_press=self.emp)

        self.btnTrainNTest = Button(
            text="Go to Train and Test Page",
            font_size=12,
            size_hint_y=None,
            size_hint_x=None,
            width=150,
            height=30,
            pos_hint={"x": 0.47, "y": 0.004},
        )
        self.btnTrainNTest.bind(on_press=self.tt)

        self.btnReport = Button(
            text="Go to Report Page",
            font_size=12,
            size_hint_y=None,
            size_hint_x=None,
            width=150,
            height=30,
            pos_hint={"x": 0.6, "y": 0.004},
        )
        self.btnReport.bind(on_press=self.rp)

        self.lblclass_name = Label(
            text="Employee name : ",
            font_size=12,
            size_hint_y=None,
            size_hint_x=None,
            width=150,
            height=30,
            pos_hint={"x": 0.05, "y": 0.16},
        )
        self.txtclass_name = TextInput(
            text="",
            multiline=False,
            font_size=12,
            size_hint_y=None,
            size_hint_x=None,
            width=150,
            height=30,
            pos_hint={"x": 0.15, "y": 0.16},
        )

        self.lblNoOfSample = Label(
            text="Number of Samples : ",
            font_size=12,
            size_hint_y=None,
            size_hint_x=None,
            width=150,
            height=30,
            pos_hint={"x": 0.05, "y": 0.20},
        )
        self.txtSample = TextInput(
            text="",
            multiline=False,
            font_size=12,
            size_hint_y=None,
            size_hint_x=None,
            width=150,
            height=30,
            pos_hint={"x": 0.15, "y": 0.20},
        )

        self.lblVariationStep = Label(
            text="Variation Step : ",
            font_size=12,
            size_hint_y=None,
            size_hint_x=None,
            width=150,
            height=30,
            pos_hint={"x": 0.05, "y": 0.24},
        )
        self.txtVar = TextInput(
            text="",
            multiline=False,
            font_size=12,
            size_hint_y=None,
            size_hint_x=None,
            width=150,
            height=30,
            pos_hint={"x": 0.15, "y": 0.24},
        )

        self.fn_dir = "employee_data\\train_raw"
        self.fn_test_dir = "employee_data\\test_raw"

        self.lblActivity = Label(
            text="",
            font_size=12,
            size_hint_y=None,
            size_hint_x=None,
            width=150,
            height=30,
            pos_hint={"x": 0.45, "y": 0.3},
        )
        self.user.allow_stretch = False
        self.user.keep_ratio = False

        self.user.size_hint_x = None
        self.user.size_hint_y = None
        self.user.width = 600
        self.user.height = 450

        self.user.pos_hint = {"x": 0.05, "y": 0.3}

        self.trainingSample.allow_stretch = False
        self.trainingSample.keep_ratio = False

        self.trainingSample.size_hint_x = None
        self.trainingSample.size_hint_y = None

        self.trainingSample.width = 160
        self.trainingSample.height = 160

        self.trainingSample.pos_hint = {"x": 0.6, "y": 0.68}

        self.testSample.allow_stretch = False
        self.testSample.keep_ratio = False

        self.testSample.size_hint_x = None
        self.testSample.size_hint_y = None

        self.testSample.width = 160
        self.testSample.height = 160

        self.testSample.pos_hint = {"x": 0.6, "y": 0.4}

        self.comp = [
            self.btnMain,
            self.btnTrainNTest,
            self.btnEmp,
            self.btnReport,
            self.lblVariationStep,
            self.txtSample,
            self.txtVar,
            self.lblNoOfSample,
            self.collect,
            self.user,
            self.lblclass_name,
            self.txtclass_name,
            self.lblActivity,
            self.trainingSample,
            self.testSample,
            self.lblTest,
            self.lblTrain,
        ]

        for c in self.comp:
            self.add_widget(c)

        Clock.schedule_interval(self.update_lbl, 1.0 / 33.0)
        # Clock.schedule_interval(self.updte, 1.0/66.0)

    def sched(self, func):
        threading.Thread(
            target=Clock.schedule_interval, args=(func, 1.0 / 10000.0), daemon=True
        ).start()

    def main(self, instance):
        app_.screen_manager.current = "clockin"
        self.sched(app_.clockin.update)

    def emp(self, instance):
        app_.screen_manager.current = "employee"

    def tt(self, instance):
        app_.screen_manager.current = "trainNtest"

    def rp(self, instance):
        app_.screen_manager.current = "report"

    def update_lbl(self, *args):

        self.lblActivity.text = self.nw

    def cdata(self, instance):
        threading.Thread(target=self.coll, daemon=True).start()

    def coll(self):

        if(self.txtSample.text.isalpha() == True):
            self.pop("Error", f"Characters({self.txtSample.text}) are not allowed!")
            return 
        if re.match("^[0-9]", self.txtSample.text) == False:
            self.pop("Error", f"Characters({self.txtSample.text}) are not allowed!")
            return 

        if(self.txtVar.text.isalpha() == True):
            self.pop("Error", f"Characters({self.txtVar.text}) are not allowed!")
            return 
        if re.match("^[0-9]", self.txtVar.text) == False:
            self.pop("Error", f"Characters({self.txtVar.text}) are not allowed!")
            return 
        
        if(self.txtclass_name.text.isnumeric() == True):
            self.pop("Error", f"Numbers and Characters({self.txtclass_name.text}) are not allowed!")
            return 
        
        if(cdigit(self.txtclass_name.text) == True):
            self.pop("Error", f"Numbers and Characters({self.txtclass_name.text}) are not allowed!")
            return 
        
        if re.match("^[a-zA-Z]", self.txtclass_name.text) == False:
            self.pop("Error", f"Numbers and Characters({self.txtclass_name.text}) are not allowed!")
            return 
        
        if dChar(self.txtclass_name.text) == True:
            self.pop("Error", f"Numbers and Characters({self.txtclass_name.text}) are not allowed!")
            return 

        if (
            self.txtclass_name.text == ""
            or self.txtVar.text == ""
            or self.txtSample.text == ""
        ):
            print("Please fill in all missing fields")
            self.pop("ERROR : MESSAGE", "Please fill in all missing fields")
            return
        else:

            queryChk = f"SELECT emp_name FROM employee_details WHERE emp_name = '{self.txtclass_name.text}'"

            app_.clockin.cursor.execute(queryChk)
            result = app_.clockin.cursor.fetchall()
            result = list(sum(result, ()))

            if len(result) <= 0:
                self.name = ""

            if len(result) >= 1:
                self.name = result[0]

            if self.name == "":

                print(
                    "Employee Not found in the Database please register first and then Collect Data"
                )
                self.pop(
                    "ERROR : MESSAGE",
                    "Employee Not found in the Database please register first and then Collect Data",
                )

                app_.screen_manager.current = "employee"

                return
                ###then return to Employee Registration Page

            else:
                self.path = os.path.join(self.fn_dir, self.txtclass_name.text)
                if not os.path.isdir(self.fn_test_dir):
                    os.mkdir(self.fn_test_dir)

                if not os.path.isdir(self.path):
                    os.mkdir(self.path)

                (self.im_width, self.im_height) = (160, 160)

                self.pin = (
                    sorted(
                        [
                            int(n[: n.find(".")])
                            for n in os.listdir(self.path)
                            if n[0] != "."
                        ]
                        + [0]
                    )[-1]
                    + 1
                )

                self.count = 0
                self.pause = 0
                self.count_max = int(self.txtSample.text)

                ###Clock.Schedule(self.collect, 1.0/66.0)

                while self.count < self.count_max:

                    ret = False

                    while not ret:
                        (ret, frame) = app_.clockin.capture.read()
                        if not ret:
                            self.popup = Popup(
                                title="Error Message",
                                content=Label(
                                    text="Failed to open webcam, trying again..."
                                ),
                                size_hint=(None, None),
                                size=(100, 100),
                            )

                        height, width, channels = frame.shape

                        frame = cv2.flip(frame, 1, 0)
                        normal = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        normal = cv2.cvtColor(normal, cv2.COLOR_RGB2BGR)

                        if app_.clockin.face_recognition.detection(frame) is None:
                            faces = None

                        if app_.clockin.face_recognition.detection(frame) is not None:
                            faces, points = app_.clockin.face_recognition.detection(
                                frame
                            )

                        if faces is not None:
                            for face in faces:
                                face_bb = face.bounding_box.astype(int)
                                yourface = normal[
                                    max(0, face_bb[1]) : min(
                                        face_bb[3], normal.shape[0] - 1
                                    ),
                                    max(0, face_bb[0]) : min(
                                        face_bb[2], normal.shape[1] - 1
                                    ),
                                ]

                                for i in range(points.shape[1]):
                                    pts = points[:, i].astype(np.int32)
                                    for j in range(pts.size // 2):
                                        pt = (pts[j], pts[5 + j])
                                        cv2.circle(
                                            frame,
                                            center=pt,
                                            radius=1,
                                            color=(255, 0, 0),
                                            thickness=2,
                                        )

                                face_resize = cv2.resize(
                                    yourface, (self.im_width, self.im_height)
                                )
                                cv2.rectangle(
                                    frame,
                                    (face_bb[0], face_bb[1]),
                                    (face_bb[2], face_bb[3]),
                                    (220, 20, 60),
                                    1,
                                )
                                cv2.putText(
                                    frame,
                                    self.txtclass_name.text,
                                    (face_bb[0], face_bb[3] + 20),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.7,
                                    (220, 20, 60),
                                    1,
                                    cv2.LINE_AA,
                                )

                                if self.pause == 0:

                                    print(
                                        "Saving training sample "
                                        + str(self.count + 1)
                                        + "/"
                                        + str(self.count_max)
                                    )
                                    self.nw = (
                                        "Saving training sample "
                                        + str(self.count + 1)
                                        + "/"
                                        + str(self.count_max)
                                    )

                                    path = os.path.abspath(
                                        "%s/%s.png" % (self.path, self.pin)
                                    )
                                    path_ = path.replace("\\", "\\\\")
                                    query = f"INSERT INTO employees_traindata(labels, features) VALUES ('{self.name}', '{path_}')"

                                    app_.clockin.cursor.execute(query)

                                    app_.clockin.conn.commit()

                                    # self.nw = self.name " " path_, "Successfully Added")
                                    cv2.imwrite(
                                        "%s/%s.png" % (self.path, self.pin), face_resize
                                    )
                                    ###self.user.source = '%s/%s.png' % (self.path, self.pin)
                                    self.source = "%s/%s.png" % (self.path, self.pin)
                                    Clock.schedule_once(
                                        partial(self.setImageSource, self.source)
                                    )

                                    self.pin += 1
                                    self.count += 1
                                    self.pause += 1

                        if faces is None:
                            self.nw = "No face detected"
                        if self.pause > 0:
                            self.pause = (self.pause + 1) % int(self.txtVar.text)

                        Clock.schedule_once(partial(self.setText, frame))

                self.nw = "Collecting samples done."
                time.sleep(3)
                self.nw = "No Activity"
        path = (
            "C://Users//GabhaDi//eclipse-workspace2//Project Clock-in//Clock_in//employee_data//train_raw//"
            + self.txtclass_name.text
            + "//"
        )
        _ = "C://Users//GabhaDi//eclipse-workspace2//Project Clock-in//Clock_in//employee_data//"
        pic = self.getRandomImage(path)
        rndpic = path + pic

        img = cv2.imread(rndpic)
        # self.testSample.source = rndpic
        Clock.schedule_once(partial(self.setImage, rndpic))

        path_ = os.path.join(self.fn_test_dir, self.txtclass_name.text)
        if not os.path.isdir(path_):
            os.mkdir(path_)

        time.sleep(3)
        Clock.schedule_once(
            partial(
                self.default,
                "C://Users//GabhaDi//eclipse-workspace2//Project Clock-in//data//Images//face_avatar.jpg",
            )
        )
        self.txtVar.text = ""
        self.txtclass_name.text = ""
        self.txtSample.text = ""
        # cv2.imshow('img',img)

        cv2.imwrite("%s/%s" % (path_, pic), img)
        os.remove(rndpic)

        ###cv2.destroyAllWindows()

    def setText(self, frame, dt):
        self.buf1 = cv2.flip(frame, 0)

        self.buf = self.buf1.tostring()

        self.texture1 = Texture.create(
            size=(frame.shape[1], frame.shape[0]), colorfmt="bgr"
        )
        self.texture1.blit_buffer(self.buf, colorfmt="bgr", bufferfmt="ubyte")

        self.user.texture = self.texture1

    def default(self, src, dt):
        self.user.source = src
        self.trainingSample.source = src
        self.testSample.source = src
        self.user.source = src

    def setImageSource(self, src, dt):
        self.trainingSample.source = src

    def setImage(self, src, dt):
        self.testSample.source = src

    def getRandomImage(self, path):
        """function loads a random file from a folder in a given path """
        random_filename = random.choice(
            [x for x in os.listdir(path) if os.path.isfile(os.path.join(path, x))]
        )

        return random_filename

    def updte(self, dt):

        # self.lblActivity.text = self.now

        ret, frame = app_.clockin.capture.read()

        if app_.clockin.face_recognition.detection(frame) is None:
            faces = None

        if app_.clockin.face_recognition.detection(frame) is not None:
            faces, points = app_.clockin.face_recognition.detection(frame)

        if faces is not None:
            for face in faces:
                face_bb = face.bounding_box.astype(int)
                for i in range(points.shape[1]):
                    pts = points[:, i].astype(np.int32)
                    for j in range(pts.size // 2):
                        pt = (pts[j], pts[5 + j])
                        cv2.circle(
                            frame, center=pt, radius=1, color=(255, 0, 0), thickness=2
                        )

                cv2.rectangle(
                    frame,
                    (face_bb[0], face_bb[1]),
                    (face_bb[2], face_bb[3]),
                    (220, 20, 60),
                    1,
                )
                cv2.putText(
                    frame,
                    self.txtclass_name.text,
                    (face_bb[0], face_bb[3] + 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (220, 20, 60),
                    1,
                    cv2.LINE_AA,
                )

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(
            frame,
            "Unprocessed Video Stream",
            (10, 80),
            font,
            0.7,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        Clock.schedule_once(partial(self.setText, frame))

    def pop(self, tle, txt):
        layout = GridLayout(cols=1, padding=10)

        popupLabel = Label(text=txt)
        closeButton = Button(text="close")

        layout.add_widget(popupLabel)

        layout.add_widget(closeButton)

        popup = Popup(title=tle, content=layout, size_hint=(0.3, 0.3))
        popup.open()
        closeButton.bind(on_press=popup.dismiss)


class Report(FloatLayout):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.btnMain = Button(
            text="Go to Main Page",
            font_size=12,
            size_hint_y=None,
            size_hint_x=None,
            width=150,
            height=30,
            pos_hint={"x": 0.63, "y": 0.35},
        )
        self.btnMain.bind(on_press=self.main)

        self.btnEmp = Button(
            text="Go to Employee Page",
            font_size=12,
            size_hint_y=None,
            size_hint_x=None,
            width=150,
            height=30,
            pos_hint={"x": 0.63, "y": 0.3},
        )
        self.btnEmp.bind(on_press=self.emp)

        self.btnTrainNTest = Button(
            text="Go to Train and Test Page",
            font_size=12,
            size_hint_y=None,
            size_hint_x=None,
            width=150,
            height=30,
            pos_hint={"x": 0.63, "y": 0.25},
        )
        self.btnTrainNTest.bind(on_press=self.tt)

        self.btnCollect = Button(
            text="Go to Collect Page",
            font_size=12,
            size_hint_y=None,
            size_hint_x=None,
            width=150,
            height=30,
            pos_hint={"x": 0.63, "y": 0.2},
        )
        self.btnCollect.bind(on_press=self.rp)

        self.tbl = Table()
        self.tbl.size_hint_x = None
        self.tbl.size_hint_y = None
        self.tbl.width = 1000
        self.tbl.height = 400
        self.tbl.pos_hint = {"x": 0, "y": 0.43}

        self.tbl2 = Table2()
        self.tbl2.size_hint_x = None
        self.tbl2.size_hint_y = None
        self.tbl2.width = 320
        self.tbl2.height = 300
        self.tbl2.pos_hint = {"x": 0.4, "y": 0}

        self.fig = FigureCanvasKivyAgg(self.pl())
        self.fig.size_hint_x = 0.4
        self.fig.size_hint_y = 0.427
        # fig.size_hint_x = None
        # fig.size_hint_y =None
        # fig.width = 400
        # fig.height= 300
        self.fig.pos_hint = {"x": 0, "y": 0}

        widgets = [
            self.btnMain,
            self.btnTrainNTest,
            self.btnEmp,
            self.btnCollect,
            self.tbl,
            self.tbl2,
            self.fig,
        ]

        for wid in widgets:
            self.add_widget(wid)

        Clock.schedule_interval(self.updateTbl, 30)

    def sched(self, func):
        threading.Thread(
            target=Clock.schedule_interval, args=(func, 1.0 / 10000.0), daemon=True
        ).start()

    def main(self, instance):
        app_.screen_manager.current = "clockin"
        self.sched(app_.clockin.update)

    def emp(self, instance):
        app_.screen_manager.current = "employee"

    def tt(self, instance):
        app_.screen_manager.current = "trainNtest"

    def rp(self, instance):
        app_.screen_manager.current = "collect"

    def updateTbl(self, dt):
        self.remove_widget(self.tbl)
        self.remove_widget(self.tbl2)
        self.remove_widget(self.fig)
        self.tbl = Table()
        self.tbl.size_hint_x = None
        self.tbl.size_hint_y = None
        self.tbl.width = 1000
        self.tbl.height = 400
        self.tbl.pos_hint = {"x": 0, "y": 0.43}
        self.tbl.canvas.ask_update()
        self.tbl2 = Table2()
        self.tbl2.size_hint_x = None
        self.tbl2.size_hint_y = None
        self.tbl2.width = 320
        self.tbl2.height = 300
        self.tbl2.pos_hint = {"x": 0.4, "y": 0}
        self.tbl2.canvas.ask_update()
        self.fig = FigureCanvasKivyAgg(self.pl())
        self.fig.size_hint_x = 0.4
        self.fig.size_hint_y = 0.427
        # fig.size_hint_x = None
        # fig.size_hint_y =None
        # fig.width = 400
        # fig.height= 300
        self.fig.pos_hint = {"x": 0, "y": 0}
        self.fig.canvas.ask_update()

        self.add_widget(self.tbl)
        self.add_widget(self.tbl2)
        self.add_widget(self.fig)

    def pl(self):
        query = "SELECT * FROM results"

        app_.clockin.cursor.execute(query)

        rows = app_.clockin.cursor.fetchall()

        self.figure, self.ax = plt.subplots()
        N = len(rows)

        confidences = "SELECT Confidence_Probability FROM results"

        app_.clockin.cursor.execute(confidences)

        confs = app_.clockin.cursor.fetchall()
        confs = list(sum(confs, ()))
        conf = ()
        confstd = ()

        for c in confs:
            c = round(c * 100, 1)

            conf = conf + (float(c),)

            strconf = str(c).split(".")[1]

            confstd = confstd + (int(strconf),)

        ind = np.arange(N)  # the x locations for the groups
        width = 0.60  # the width of the bars

        # fig1 = plt.gcf()
        rects1 = self.ax.bar(ind, conf, width, color="black", yerr=confstd)

        classNames = "SELECT Class_Name FROM results"

        app_.clockin.cursor.execute(classNames)

        names = app_.clockin.cursor.fetchall()
        names = list(sum(names, ()))
        newn = tuple(names)
        self.ax.set_ylabel("Confidence Probabilities")
        self.ax.set_title("Confidence Probability by class")
        self.ax.set_xticks(ind)
        self.ax.set_xticklabels(newn)
        self.ax.set_facecolor("xkcd:gray")
        # ax.legend(rects1[0], 'Confidence')

        self.autolabel(rects1, self.ax)
        # autolabel(rects2)

        # plt.figure(figsize=(8,6))

        return self.figure

    def autolabel(self, rects, ax):
        # attach some text labels
        for rect in rects:
            height = rect.get_height()
            ax.text(
                rect.get_x() + rect.get_width() / 2.0,
                0.92 * height,
                height,
                ha="center",
                va="bottom",
                color="w",
            )


class Employee(FloatLayout):
    def __init__(self, **kwargs):
        super().__init__()
        try:
            self.conn = Connect(server, username, password, database)
        except:
            print("Error please start your server")
        self.cursor = self.conn.cursor()
        self.user = Image(
            source="C://Users//GabhaDi//eclipse-workspace2//Project Clock-in//employee_blobs//default.png"
        )

        self.lbl = Label(
            text="Project Clock In",
            font_size="28sp",
            size_hint_y=None,
            size_hint_x=None,
            width=150,
            height=30,
            pos_hint={"x": 0.07, "y": 0.9},
        )

        self.btnMain = Button(
            text="Go to Main Page",
            font_size=12,
            size_hint_y=None,
            size_hint_x=None,
            width=150,
            height=100,
            pos_hint={"x": 0.06, "y": 0.04},
        )
        self.btnMain.bind(on_press=self.main)

        self.btnEmp = Button(
            text="Go to Report Page",
            font_size=12,
            size_hint_y=None,
            size_hint_x=None,
            width=150,
            height=100,
            pos_hint={"x": 0.2, "y": 0.04},
        )
        self.btnEmp.bind(on_press=self.emp)

        self.btnTrainNTest = Button(
            text="Go to Train and Test Page",
            font_size=12,
            size_hint_y=None,
            size_hint_x=None,
            width=150,
            height=100,
            pos_hint={"x": 0.34, "y": 0.04},
        )
        self.btnTrainNTest.bind(on_press=self.tt)

        self.btnCollect = Button(
            text="Go to Collect Page",
            font_size=12,
            size_hint_y=None,
            size_hint_x=None,
            width=150,
            height=100,
            pos_hint={"x": 0.48, "y": 0.04},
        )
        self.btnCollect.bind(on_press=self.rp)

        self.lblemp_id = Label(
            text="Emp ID : ",
            font_size=12,
            size_hint_y=None,
            size_hint_x=None,
            width=150,
            height=30,
            pos_hint={"x": 0.29, "y": 0.8},
        )
        self.lblemp_name = Label(
            text="Emp Name : ",
            font_size=12,
            size_hint_y=None,
            size_hint_x=None,
            width=150,
            height=30,
            pos_hint={"x": 0.29, "y": 0.75},
        )
        self.lblemp_surname = Label(
            text="Emp Surname : ",
            font_size=12,
            size_hint_y=None,
            size_hint_x=None,
            width=150,
            height=30,
            pos_hint={"x": 0.29, "y": 0.70},
        )
        self.lblemp_phoneNo = Label(
            text="Emp Phone No : ",
            font_size=12,
            size_hint_y=None,
            size_hint_x=None,
            width=150,
            height=30,
            pos_hint={"x": 0.29, "y": 0.65},
        )
        self.lblemp_email = Label(
            text="Emp Email : ",
            font_size=12,
            size_hint_y=None,
            size_hint_x=None,
            width=150,
            height=30,
            pos_hint={"x": 0.29, "y": 0.60},
        )
        self.lblemp_gender = Label(
            text="Emp Gender : ",
            font_size=12,
            size_hint_y=None,
            size_hint_x=None,
            width=150,
            height=30,
            pos_hint={"x": 0.29, "y": 0.55},
        )
        self.lblemp_age = Label(
            text="Emp Age : ",
            font_size=12,
            size_hint_y=None,
            size_hint_x=None,
            width=150,
            height=30,
            pos_hint={"x": 0.29, "y": 0.50},
        )
        self.lblemp_dob = Label(
            text="Emp DOB : ",
            font_size=12,
            size_hint_y=None,
            size_hint_x=None,
            width=150,
            height=30,
            pos_hint={"x": 0.29, "y": 0.45},
        )
        self.lblemp_blobpath = Label(
            text="Emp DP : ",
            font_size=12,
            size_hint_y=None,
            size_hint_x=None,
            width=150,
            height=30,
            pos_hint={"x": 0.29, "y": 0.40},
        )

        self.txtemp_id = TextInput(
            text="",
            multiline=False,
            font_size=12,
            size_hint_y=None,
            size_hint_x=None,
            width=150,
            height=30,
            pos_hint={"x": 0.4, "y": 0.8},
        )
        self.txtemp_name = TextInput(
            text="",
            multiline=False,
            font_size=12,
            size_hint_y=None,
            size_hint_x=None,
            width=150,
            height=30,
            pos_hint={"x": 0.4, "y": 0.75},
        )
        self.txtemp_surname = TextInput(
            text="",
            multiline=False,
            font_size=12,
            size_hint_y=None,
            size_hint_x=None,
            width=150,
            height=30,
            pos_hint={"x": 0.4, "y": 0.70},
        )
        self.txtemp_phoneNo = TextInput(
            text="",
            multiline=False,
            font_size=12,
            size_hint_y=None,
            size_hint_x=None,
            width=150,
            height=30,
            pos_hint={"x": 0.4, "y": 0.65},
        )
        self.txtemp_email = TextInput(
            text="",
            multiline=False,
            font_size=12,
            size_hint_y=None,
            size_hint_x=None,
            width=150,
            height=30,
            pos_hint={"x": 0.4, "y": 0.60},
        )
        self.txtemp_gender = TextInput(
            text="",
            multiline=False,
            font_size=12,
            size_hint_y=None,
            size_hint_x=None,
            width=150,
            height=30,
            pos_hint={"x": 0.4, "y": 0.55},
        )
        self.txtemp_age = TextInput(
            text="",
            multiline=False,
            font_size=12,
            size_hint_y=None,
            size_hint_x=None,
            width=150,
            height=30,
            pos_hint={"x": 0.4, "y": 0.50},
        )
        self.txtemp_dob = TextInput(
            text="",
            multiline=False,
            font_size=12,
            size_hint_y=None,
            size_hint_x=None,
            width=150,
            height=30,
            pos_hint={"x": 0.4, "y": 0.45},
        )
        self.txtemp_blobpath = TextInput(
            text="",
            multiline=False,
            font_size=12,
            size_hint_y=None,
            size_hint_x=None,
            width=300,
            height=30,
            pos_hint={"x": 0.4, "y": 0.40},
        )

        self.txtemp_id.disabled = True
        self.txtemp_blobpath.disabled = True

        self.lblsearch = Label(
            text="Search",
            font_size=12,
            size_hint_y=None,
            size_hint_x=None,
            width=150,
            height=30,
            pos_hint={"x": 0.3, "y": 0.9},
        )
        self.txtsearch = TextInput(
            text="",
            multiline=False,
            font_size=12,
            size_hint_y=None,
            size_hint_x=None,
            width=150,
            height=30,
            pos_hint={"x": 0.4, "y": 0.9},
        )
        self.btnSearch = Button(
            text="Search",
            font_size=12,
            size_hint_y=None,
            size_hint_x=None,
            width=150,
            height=30,
            pos_hint={"x": 0.52, "y": 0.9},
        )
        self.btnSearch.bind(on_press=self.Search)

        self.btnAdd = Button(
            text="Add",
            font_size=12,
            size_hint_y=None,
            size_hint_x=None,
            width=150,
            height=30,
            pos_hint={"x": 0.06, "y": 0.2},
        )
        self.btnUpdate = Button(
            text="Update",
            font_size=12,
            size_hint_y=None,
            size_hint_x=None,
            width=150,
            height=30,
            pos_hint={"x": 0.2, "y": 0.2},
        )
        self.btnDelete = Button(
            text="Delete",
            font_size=12,
            size_hint_y=None,
            size_hint_x=None,
            width=150,
            height=30,
            pos_hint={"x": 0.34, "y": 0.2},
        )
        self.btnClear = Button(
            text="Clear All",
            font_size=12,
            size_hint_y=None,
            size_hint_x=None,
            width=150,
            height=30,
            pos_hint={"x": 0.48, "y": 0.2},
        )

        self.btnAdd.bind(on_press=self.Add)
        self.btnUpdate.bind(on_press=self.Update)
        self.btnDelete.bind(on_press=self.Delete)
        self.btnClear.bind(on_press=self.clear)

        self.user.allow_stretch = False
        self.user.keep_ratio = False

        self.user.size_hint_x = None
        self.user.size_hint_y = None
        self.user.width = 400
        self.user.height = 400

        self.user.pos_hint = {"x": 0.01, "y": 0.3}

        self.tbl = Table3()
        self.tbl.size_hint_x = None
        self.tbl.size_hint_y = None
        self.tbl.width = 600
        self.tbl.height = 650
        self.tbl.pos_hint = {"x": 0.63, "y": 0.03}

        Clock.schedule_interval(self.updatetbl, 30)

        comp = [
            self.tbl,
            self.btnMain,
            self.btnTrainNTest,
            self.btnEmp,
            self.btnCollect,
            self.lbl,
            self.btnClear,
            self.btnUpdate,
            self.btnSearch,
            self.btnDelete,
            self.btnAdd,
            self.lblemp_id,
            self.lblemp_name,
            self.lblemp_surname,
            self.lblemp_phoneNo,
            self.lblemp_gender,
            self.lblemp_dob,
            self.lblemp_age,
            self.lblemp_email,
            self.lblsearch,
            self.lblemp_blobpath,
            self.txtemp_age,
            self.txtemp_blobpath,
            self.txtemp_dob,
            self.txtemp_email,
            self.txtemp_gender,
            self.txtemp_id,
            self.txtemp_name,
            self.txtemp_phoneNo,
            self.txtemp_surname,
            self.txtsearch,
            self.user,
        ]
        for obj in comp:
            self.add_widget(obj)

    def sched(self, func):
        threading.Thread(
            target=Clock.schedule_interval, args=(func, 1.0 / 10000.0), daemon=True
        ).start()

    def updatetbl(self, dt):

        self.remove_widget(self.tbl)
        self.tbl = Table3()
        self.tbl.size_hint_x = None
        self.tbl.size_hint_y = None
        self.tbl.width = 600
        self.tbl.height = 650
        self.tbl.pos_hint = {"x": 0.63, "y": 0.03}

        self.tbl.canvas.ask_update()

        self.add_widget(self.tbl)

    def main(self, instance):
        app_.screen_manager.current = "clockin"
        self.sched(app_.clockin.update)

    def emp(self, instance):
        app_.screen_manager.current = "report"

    def tt(self, instance):
        app_.screen_manager.current = "trainNtest"

    def rp(self, instance):
        app_.screen_manager.current = "collect"

    def Add(self, instance):


       

        if(self.txtemp_age.text.isalpha() == True):
            self.pop("Error", f"Characters({self.txtemp_age.text}) are not allowed!")
            return 
        if re.match("^[a-zA-Z]", self.txtemp_name.text) == False:
            self.pop("Error", f"Numbers({self.txtemp_name.text}) are not allowed!")
            return 
        if(self.txtemp_name.text.isnumeric() == True):
            self.pop("Error", f"Numbers({self.txtemp_name.text}) are not allowed!")
            return 
        if self.txtemp_age.text.isalpha() == True:
            self.pop("Error", f"Characters({self.txtemp_age.text}) are not allowed!")
            return 
        if cdigit(self.txtemp_gender.text) == True:
            self.pop("Error", f"Numbers({self.txtemp_gender.text}) are not allowed!")
            return 
        if cdigit(self.txtemp_name.text) == True:
            self.pop("Error", f"Numbers({self.txtemp_name.text}) are not allowed!")
            return 
        if cdigit(self.txtemp_surname.text) == True:
            self.pop("Error", f"Numbers({self.txtemp_surname.text}) are not allowed!")
            return 
        #if cdigit(self.txtemp_email.text) == True:
        #    self.pop("Error", f"Characters({self.txtimgsize.text}) are not allowed!")
        #return 
        if re.match("""(?:[a-z0-9!#$%&'*+/=?^_`{|}~-]+(?:\.[a-z0-9!#$%&'*+/=?^_`{|}~-]+)*|"(?:[\x01-\x08\x0b\x0c\x0e-\x1f\x21\x23-\x5b\x5d-\x7f]|\\[\x01-\x09\x0b\x0c\x0e-\x7f])*")@(?:(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.)+[a-z0-9](?:[a-z0-9-]*[a-z0-9])?|\[(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?|[a-z0-9-]*[a-z0-9]:(?:[\x01-\x08\x0b\x0c\x0e-\x1f\x21-\x5a\x53-\x7f]|\\[\x01-\x09\x0b\x0c\x0e-\x7f])+)\])""", self.txtemp_email.text) == False:
            self.pop("Error", f"Invalid email address{self.txtemp_email}, please enter a valid email")
            return
        
         
        
        
        if (
            self.txtemp_age.text == ""
            or self.txtemp_dob.text == ""
            or self.txtemp_email.text == ""
            or self.txtemp_gender.text == ""
            or self.txtemp_name.text == ""
            or self.txtemp_phoneNo == ""
            or self.txtemp_surname.text == ""
        ):
            print("Plese fill in all missing values")
            self.pop("INFO : MESSAGE", "Plese fill in all missing values")

        else:

            check = f"SELECT emp_name, emp_surname FROM employee_details WHERE emp_name ='{self.txtemp_name.text}' AND emp_surname = '{self.txtemp_surname.text}'"

            self.cursor.execute(check)

            result = self.cursor.fetchall()

            result = list(sum(result, ()))

            if len(result) <= 0:
                print("Adding Employee")
                age = int(self.txtemp_age.text)
                key = cv2.waitKey(1)
                ###webcam = cv2.VideoCapture(0)
                while True:

                    check, frame = app_.clockin.capture.read()
                    ##print(check) #prints true as long as the webcam is running
                    ##print(frame) #prints matrix values of each framecd

                    if app_.clockin.face_recognition.identify(frame) is None:
                        faces = None
                    else:
                        faces, points = app_.clockin.face_recognition.identify(frame)
                    cv2.imshow("Capturing", frame)
                    key = cv2.waitKey(1)
                    file = (
                        "C://Users//GabhaDi//eclipse-workspace2//Project Clock-in//employee_blobs//%s.jpg"
                        % self.txtemp_name.text
                    )
                    if key == ord("s") or faces is not None:
                        cv2.imwrite(filename=file, img=frame)
                        # app_.clockin.capture.release()
                        img_i = cv2.imread(file, cv2.IMREAD_ANYCOLOR)
                        ###img_new = cv2.imshow("Captured Image", img_i)
                        # cv2.waitKey(1650)
                        # cv2.destroyAllWindows()
                        ##print("Processing image...")
                        img_ = cv2.imread(file, cv2.IMREAD_ANYCOLOR)
                        ##print("Converting RGB image to grayscale...")
                        ###gray = cv2.cvtColor(img_, cv2.COLOR_BGR2GRAY)
                        ##print("Converted RGB image to grayscale...")
                        ##print("Resizing image to 28x28 scale...")
                        img_ = cv2.resize(img_, (960, 1280))
                        ##print("Resized...")
                        img_resized = cv2.imwrite(filename=file, img=img_)
                        ###print("Image saved!")
                        self.txtemp_blobpath.text = file
                        break
                    elif key == ord("q"):
                        print("Turning off camera.")
                        self.pop("INFO : MESSAGE", "Turning off camera.")

                        app_.clockin.capture.release()
                        print("Camera off.")
                        print("Program ended.")
                        cv2.destroyAllWindows()
                        break

                query = f"INSERT INTO employee_details(emp_id, emp_name, emp_surname, emp_phoneno, emp_email, emp_gender, emp_dob, emp_age, emp_blobpath) VALUES(NULL, '{self.txtemp_name.text}', '{self.txtemp_surname.text}', '{self.txtemp_phoneNo.text}', '{self.txtemp_email.text}', '{self.txtemp_gender.text}', '{self.txtemp_dob.text}', {age}, '{self.txtemp_blobpath.text}');"

                self.cursor.execute(query)

                self.conn.commit()

                query_ = f"INSERT INTO employee_checks(emp_id, emp_name, emp_surname, emp_email, emp_phoneNo, emp_access_state, emp_blobpath) VALUES(NULL,'{self.txtemp_name.text}', '{self.txtemp_surname.text}', '{self.txtemp_email.text}', '{self.txtemp_phoneNo.text}', 0, '{self.txtemp_name.text}.png');"
                self.cursor.execute(query_)

                self.conn.commit()

                print("Successfully inserted the employee")

                print(self.cursor.rowcount, "record(s) affected")
                self.pop(
                    "INFO : MESSAGE",
                    "Successfully inserted the employee"
                    + str(self.cursor.rowcount)
                    + "record(s) affected",
                )
                app_.collect.txtclass_name.text = self.txtemp_name.text
                app_.screen_manager.current = "collect"

            else:
                self.pop(
                    "INFO : MESSAGE",
                    f"Employee {self.txtemp_name.text} already exists in the database",
                )

    def Update(self, instance):

        if self.txtemp_id.text == "":
            print("Please search for employee to update")

            self.pop("ERROR : MESSAGE", "Please search for employee to update")

        else:
            
            
            layout = GridLayout(cols=1, padding=10)

            popupLabel = Label(
                text="Are you sure you want to update user id %s at %s "
                % (self.txtemp_id.text, self.txtemp_name.text)
            )
            closeButton = Button(text="close")
            yes = Button(text="yes")
            no = Button(text="no")
            txt = TextInput(text="")
            layout.add_widget(popupLabel)
            layout.add_widget(yes)
            layout.add_widget(no)
            layout.add_widget(closeButton)

            self.popup = Popup(
                title="INFO : MESSAGE", content=layout, size_hint=(0.4, 0.4)
            )
            self.popup.open()
            no.bind(on_press=self.popup.dismiss)
            yes.bind(on_press=self.upd)
            closeButton.bind(on_press=self.popup.dismiss)
            
            
    def upd(self, instance):
        print("Updating Employee")
        id = int(self.txtemp_id.text)
        age = int(self.txtemp_age.text)
        query = f"UPDATE employee_details SET emp_name = '{self.txtemp_name.text}', emp_surname = '{self.txtemp_surname.text}', emp_phoneNo = '{self.txtemp_phoneNo.text}', emp_email = '{self.txtemp_email.text}', emp_gender = '{self.txtemp_gender.text}', emp_dob = '{self.txtemp_dob.text}', emp_age = {age}, emp_blobpath = '{self.txtemp_blobpath.text}' WHERE emp_id = {id}"

        self.cursor.execute(query)

        self.conn.commit()
        print("Successfully updated the employee")

        print(self.cursor.rowcount, "record(s) affected")
        self.pop(
            "INFO : MESSAGE",
            "Successfully updated the employee"
            + str(self.cursor.rowcount)
            + "record(s) affected",
        )         

    def Delete(self, instance):

        if self.txtemp_id.text == "":
            print("Please search for employee to delete")
            self.pop("ERROR : MESSAGE", "Please search for employee to delete")
        else:
            layout = GridLayout(cols=1, padding=10)

            popupLabel = Label(
                text="Are you sure you want to delete user id %s at %s "
                % (self.txtemp_id.text, self.txtemp_name.text)
            )
            closeButton = Button(text="close")
            yes = Button(text="yes")
            no = Button(text="no")
            txt = TextInput(text="")
            layout.add_widget(popupLabel)
            layout.add_widget(yes)
            layout.add_widget(no)
            layout.add_widget(closeButton)

            self.popup = Popup(
                title="INFO : MESSAGE", content=layout, size_hint=(0.4, 0.4)
            )
            self.popup.open()
            no.bind(on_press=self.popup.dismiss)
            yes.bind(on_press=self.dell)
            closeButton.bind(on_press=self.popup.dismiss)

    def dell(self, instance):
        print("Deleting Employee")
        self.popup.dismiss
        id = int(self.txtemp_id.text)
        query = f"DELETE FROM employee_details WHERE emp_id = {id}"
        query_ = (
            f"DELETE FROM employees_traindata WHERE labels = '{self.txtemp_name.text}'"
        )
        query__ = (
            f"DELETE FROM employee_checks WHERE emp_name = '{self.txtemp_name.text}'"
        )
        self.cursor.execute(query)
        self.cursor.execute(query_)
        self.cursor.execute(query__)

        try:
            os.remove(self.txtemp_blobpath.text)
            shutil.rmtree(
                f"C://Users//GabhaDi//eclipse-workspace2//Project Clock-in//Clock_in//employee_data//train_raw//{self.txtemp_name.text}"
            )
            shutil.rmtree(
                f"C://Users//GabhaDi//eclipse-workspace2//Project Clock-in//Clock_in//employee_data//test_raw//{self.txtemp_name.text}"
            )
            shutil.rmtree(
                f"C://Users//GabhaDi//eclipse-workspace2//Project Clock-in//Clock_in//employee_data//train_aligned//{self.txtemp_name.text}"
            )
            shutil.rmtree(
                f"C://Users//GabhaDi//eclipse-workspace2//Project Clock-in//Clock_in//employee_data//test_aligned//{self.txtemp_name.text}"
            )
        except:
            print("Error couldnt delete info")

        self.conn.commit()
        self.clear_all()
        print("Successfully deleted the employee")
        print(self.cursor.rowcount, "record(s) affected")
        self.pop(
            "INFO : MESSAGE",
            "Successfully deleted the employee"
            + str(self.cursor.rowcount)
            + "record(s) affected",
        )

    def Search(self, instance):
        if(self.txtsearch.text.isnumeric() == True):
            self.pop("Error", f"Numbers and Characters({self.txtsearch.text}) are not allowed!")
            return 
        
        if(cdigit(self.txtsearch.text) == True):
            self.pop("Error", f"Numbers and Characters({self.txtsearch.text}) are not allowed!")
            return 
        
        if re.match("^[a-zA-Z]", self.txtsearch.text) == False:
            self.pop("Error", f"Numbers and Characters({self.txtsearch.text}) are not allowed!")
            return 
        
        if dChar(self.txtsearch.text) == True:
            self.pop("Error", f"Numbers and Characters({self.txtsearch.text}) are not allowed!")
            return
        
        if self.txtsearch.text == "":
            print("Please enter employee name to search for!!")
            self.pop("ERROR : MESSAGE", "Please enter employee name to search for!!")
        else:

            print("Searching Employee")

            query = f"SELECT * FROM employee_details WHERE emp_name = '{self.txtsearch.text}'"

            self.cursor.execute(query)

            result = self.cursor.fetchall()
            result = list(sum(result, ()))

            if len(result) <= 0:
                print("Employee : %s not found in the database" % self.txtsearch.text)
                self.pop(
                    "INFO : EMPLOYEE NOT FOUND",
                    "Employee : %s not found in the database" % self.txtsearch.text,
                )
                return
            else:
                self.txtemp_id.text = str(result[0])
                self.txtemp_name.text = str(result[1])
                self.txtemp_surname.text = str(result[2])
                self.txtemp_phoneNo.text = str(result[3])
                self.txtemp_email.text = str(result[4])
                self.txtemp_gender.text = str(result[5])
                self.txtemp_age.text = str(result[6])
                self.txtemp_dob.text = str(result[7])
                self.txtemp_blobpath.text = str(result[8])
                self.user.source = str(result[8])
                for res in result:

                    print(res)

    def clear_all(self):
        self.txtemp_age.text = ""
        self.txtemp_blobpath.text = ""
        self.txtemp_dob.text = ""
        self.txtemp_email.text = ""
        self.txtemp_gender.text = ""
        self.txtemp_id.text = ""
        self.txtemp_name.text = ""
        self.txtemp_phoneNo.text = ""
        self.txtemp_surname.text = ""
        self.txtsearch.text = ""
        self.user.source = (
            "C://Users//GabhaDi//eclipse-workspace2//Project Clock-in//employee_blobs//default.png"
        )
        self.pop("INFORMATION : CLEARED", "everything was successfully clearly")

    def clear(self, instance):

        self.txtemp_age.text = ""
        self.txtemp_blobpath.text = ""
        self.txtemp_dob.text = ""
        self.txtemp_email.text = ""
        self.txtemp_gender.text = ""
        self.txtemp_id.text = ""
        self.txtemp_name.text = ""
        self.txtemp_phoneNo.text = ""
        self.txtemp_surname.text = ""
        self.txtsearch.text = ""
        self.user.source = (
            "C://Users//GabhaDi//eclipse-workspace2//Project Clock-in//employee_blobs//default.png"
        )
        self.pop("INFORMATION : CLEARED", "everything was successfully clearly")

    def pop(self, tle, txt):
        layout = GridLayout(cols=1, padding=10)

        popupLabel = Label(text=txt)
        closeButton = Button(text="close")

        layout.add_widget(popupLabel)

        layout.add_widget(closeButton)

        popup = Popup(title=tle, content=layout, size_hint=(0.3, 0.3))
        popup.open()
        closeButton.bind(on_press=popup.dismiss)


if __name__ == "__main__":
    Window.size = (1366, 768)
    ##Window.fullscreen = True
    Window.borderless = False

    app_ = Main_app()
    app_.run()
    cv2.destroyAllWindows()
