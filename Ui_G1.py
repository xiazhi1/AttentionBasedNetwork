# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'd:\大三\大三上\软件课设\gui\G1.ui'
#
# Created by: PyQt5 UI code generator 5.15.9
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.

import cv2
from PyQt5 import QtCore, QtGui, QtWidgets  
from PyQt5.QtCore import QFileInfo
from PyQt5.QtWidgets import QFileDialog
 
import sys
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
import cv2
import numpy as np
import math
import run      #连接图像识别程序


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):

    # 定义功能函数
    #打开文件   
    def openfile(self):
        global fname
        #定义文件读取函数，解决中文路径读取错误的问题
        def cv_imread(file_path):
            cv_img = cv2.imdecode(np.fromfile(file_path,dtype=np.uint8),-1)
            return cv_img
        fname, imgType = QFileDialog.getOpenFileName(None, "打开图片", "", "*;;*.png;;All Files(*)") 
        img = cv_imread(fname)  # opencv读取图片
        res = cv2.resize(img, (441, 341), interpolation=cv2.INTER_CUBIC) #用cv2.resize设置图片大小
        self.img_o = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)  # opencv读取的bgr格式图片转换成rgb格式
        _image = QtGui.QImage(self.img_o[:], self.img_o.shape[1], self.img_o.shape[0], self.img_o.shape[1] * 3,
                              QtGui.QImage.Format_RGB888)  # pyqt5转换成自己能放的图片格式
        jpg_out = QtGui.QPixmap(_image)  # 转换成QPixmap
        self.Input.setPixmap(jpg_out)  # 设置图片显示
        
    
    def run(self):
        #在此连接图像识别程序,未完成，需要图像识别软件的接口
        a=[0]
        run.start(a,fname)
        if a[0]==1:
            textOut="识别到乳腺癌！"
        elif a[0]==0:
            textOut="未检查出乳腺癌"
        _translate = QtCore.QCoreApplication.translate
        self.Output.setText(_translate("MainWindow", textOut))

    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.setEnabled(True)
        MainWindow.setMinimumSize(QtCore.QSize(1000, 750))
        MainWindow.setMaximumSize(QtCore.QSize(1900, 1000))
        MainWindow.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(300, 0, 60, 20))
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        font.setBold(False)
        font.setWeight(50)
        self.label.setFont(font)
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(850, 260, 60, 20))
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.Input = QtWidgets.QLabel(self.centralwidget)
        self.Input.setGeometry(QtCore.QRect(20, 30, 731, 611))
        self.Input.setText("")
        self.Input.setAlignment(QtCore.Qt.AlignCenter)
        self.Input.setObjectName("Input")
        self.Output = QtWidgets.QLabel(self.centralwidget)
        self.Output.setGeometry(QtCore.QRect(770, 300, 221, 20))
        self.Output.setAlignment(QtCore.Qt.AlignCenter)
        self.Output.setObjectName("Output")
        self.Open = QtWidgets.QPushButton(self.centralwidget)
        self.Open.setGeometry(QtCore.QRect(790, 560, 128, 25))
        self.Open.setMinimumSize(QtCore.QSize(50, 25))
        self.Open.setMaximumSize(QtCore.QSize(128, 25))
        self.Open.setCursor(QtGui.QCursor(QtCore.Qt.ArrowCursor))
        self.Open.setStyleSheet("font: 10pt \"黑体\";")
        self.Open.setObjectName("Open")
        self.Run = QtWidgets.QPushButton(self.centralwidget)
        self.Run.setGeometry(QtCore.QRect(790, 620, 128, 25))
        self.Run.setMaximumSize(QtCore.QSize(128, 25))
        self.Run.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.Run.setStyleSheet("font: 10pt \"黑体\";")
        self.Run.setObjectName("Run")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1000, 26))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

        self.Open.clicked.connect(self.openfile)
        self.Run.clicked.connect(self.run)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "乳腺癌CT图像检测"))
        self.label.setText(_translate("MainWindow", "输入图像"))
        self.label_2.setText(_translate("MainWindow", "处理结果"))
        self.Output.setText(_translate("MainWindow", "待运行"))
        self.Open.setText(_translate("MainWindow", "选取图像"))
        self.Run.setText(_translate("MainWindow", "进行识别"))
