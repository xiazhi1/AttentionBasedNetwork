# coding:utf-8
import sys
import cv2
import numpy as np
import math
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt, QUrl,QFileInfo
from PyQt5.QtGui import QIcon, QDesktopServices
from PyQt5.QtWidgets import QApplication, QFrame, QHBoxLayout,QWidget,QGridLayout,QFileDialog
from qfluentwidgets import (NavigationItemPosition, MessageBox, setTheme, Theme, MSFluentWindow,
                            NavigationAvatarWidget, qrouter, SubtitleLabel,ImageLabel, setFont,PrimaryPushButton,PushButton)
from qfluentwidgets import FluentIcon as FIF
from network import network


#生成Home主页
class Home(QFrame):

    #配置输入输出方法
    def openfile(self):
        global fname
        #定义文件读取函数，解决中文路径读取错误的问题
        def cv_imread(file_path):
            cv_img = cv2.imdecode(np.fromfile(file_path,dtype=np.uint8),-1)
            return cv_img
        fname, imgType = QFileDialog.getOpenFileName(None, "打开图片", "", "*;;*.png;;All Files(*)") 
        img = cv_imread(fname)  # opencv读取图片
        res = cv2.resize(img, (440, 330), interpolation=cv2.INTER_CUBIC) #用cv2.resize设置图片大小
        self.img_o = img #将读取的图片保存到全局变量中
        present_img = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)  # opencv读取的bgr格式图片转换成rgb格式
        _image = QtGui.QImage(present_img[:], present_img.shape[1], present_img.shape[0], present_img.shape[1] * 3,
                              QtGui.QImage.Format_RGB888)  # pyqt5转换成自己能放的图片格式
        jpg_out = QtGui.QPixmap(_image)  # 转换成QPixmap
        self.Input.setPixmap(jpg_out)  # 设置图片显示
        
    
    def run(self):
        def cv_imread(file_path):
            cv_img = cv2.imdecode(np.fromfile(file_path,dtype=np.uint8),-1)
            return cv_img
        #在此连接图像识别程序
        input_img = self.img_o
        imgo = network(input_img) #调用模型并返回RGB图片
        reso = cv2.resize(imgo, (440, 330), interpolation=cv2.INTER_CUBIC) #用cv2.resize设置图片大小
        result_image = cv2.cvtColor(reso, cv2.COLOR_BGR2RGB) #转换图片通道
        self.img_oo = result_image
        _imageo = QtGui.QImage(self.img_oo[:], self.img_oo.shape[1], self.img_oo.shape[0], self.img_oo.shape[1] * 3,
                              QtGui.QImage.Format_RGB888)  # pyqt5转换成自己能放的图片格式
        jpg_outo = QtGui.QPixmap(_imageo)  # 转换成QPixmap
        self.Output.setPixmap(jpg_outo)  # 设置图片显示
    
    #初始化，设置界面大小范围
    def __init__(self, text: str, parent=None):
        super().__init__()
        
        self.setObjectName(text.replace(' ', '-'))
        self.setMinimumSize(QtCore.QSize(1000, 750))
        self.setMaximumSize(QtCore.QSize(1900, 1000))
        self.set()
        
    #配置label和button控件
    def set(self):
        self.Input = ImageLabel('', self)
        
        self.Output = ImageLabel('', self)
        
        self.Open = PushButton(FIF.FOLDER, '读取图片', self)
        self.Run = PrimaryPushButton(FIF.UPDATE, '识别', self)
        self.gridLayout = QGridLayout(self)
        self.gridLayout.addWidget(self.Open, 1, 0,alignment=Qt.AlignCenter)
        self.gridLayout.addWidget(self.Run, 1, 1,alignment=Qt.AlignCenter)
        self.gridLayout.addWidget(self.Input, 0, 0,alignment=Qt.AlignCenter)
        self.gridLayout.addWidget(self.Output, 0, 1,alignment=Qt.AlignCenter)
        self.Open.clicked.connect(self.openfile)
        self.Run.clicked.connect(self.run)

        self.setLayout(self.gridLayout)
#生成库页面,目前只预留了库的位置，但详细内容暂未实现
class library(QFrame):

    def __init__(self, text: str, parent=None):
        super().__init__(parent=parent)
        self.setMinimumSize(QtCore.QSize(1000, 750))
        self.setMaximumSize(QtCore.QSize(1900, 1000))
        self.label = SubtitleLabel(text, self)
        self.hBoxLayout = QHBoxLayout(self)

        setFont(self.label, 24)
        self.label.setAlignment(Qt.AlignCenter)
        self.hBoxLayout.addWidget(self.label, 1, Qt.AlignCenter)
        self.setObjectName(text.replace(' ', '-'))


#生成窗口
class Window(MSFluentWindow):

    def __init__(self):
        super().__init__()

        # 生成主页和库
        self.homeInterface = Home('Home', self)
        self.libraryInterface = library('敬请期待', self)

        self.initNavigation()
        self.initWindow()
    #初始化侧边导航
    def initNavigation(self):
        self.addSubInterface(self.homeInterface, FIF.HOME, '主页', FIF.HOME_FILL)
        self.addSubInterface(self.libraryInterface, FIF.BOOK_SHELF, '库', FIF.LIBRARY_FILL, NavigationItemPosition.BOTTOM)
        self.navigationInterface.addItem(
            routeKey='Help',
            icon=FIF.HELP,
            text='帮助',
            onClick=self.showMessageBox,
            selectable=False,
            position=NavigationItemPosition.BOTTOM,
        )

        self.navigationInterface.setCurrentItem(self.homeInterface.objectName())
    #初始化窗口，设置名称和图标等
    def initWindow(self):
        self.resize(900, 700)
        self.setWindowIcon(QIcon(r"D:\学习资料\大学功课\软件课设\AttentionBasedNetwork\images\TOM.png"))#设置图标
        self.setWindowTitle('乳腺癌CT图像识别')

        desktop = QApplication.desktop().availableGeometry()
        w, h = desktop.width(), desktop.height()
        self.move(w//2 - self.width()//2, h//2 - self.height()//2)
    #帮助页面
    def showMessageBox(self):
        w = MessageBox(
            '帮助',
            '选择图片后点击识别，输出注意力图。\n我们的github仓库：https://github.com/xiazhi1/AttentionBasedNetwork',
            self
        )
        w.yesButton.setText('访问主页')
        w.cancelButton.setText('OK')

        if w.exec():
            QDesktopServices.openUrl(QUrl("https://github.com/xiazhi1/AttentionBasedNetwork"))


if __name__ == '__main__':
    QApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps)

    

    app = QApplication(sys.argv)
    w = Window()
    w.show()
    app.exec_()
