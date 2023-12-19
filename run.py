import cv2
import numpy as np
def cv_imread(file_path):
            cv_img = cv2.imdecode(np.fromfile(file_path,dtype=np.uint8),-1)
            return cv_img
def start(fname):
    imgo = fname  # 读取图片地址
    return imgo