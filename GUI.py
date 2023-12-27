import sys
from PyQt5.QtWidgets import QMainWindow,QApplication,QWidget
from Ui_G1 import Ui_MainWindow  #导入界面类
 
 
class MyMainWindow(QMainWindow,Ui_MainWindow): #继承类并初始化以便打开界面
    def __init__(self,parent =None):
        super(MyMainWindow,self).__init__(parent)
        self.setupUi(self)
 
if __name__ == "__main__":
    app = QApplication(sys.argv)
    myWin = MyMainWindow()
    myWin.show()
    sys.exit(app.exec_())
