import cv2
from PyQt5.uic import loadUi
from PyQt5.QtWidgets import*
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtGui import QPixmap
import PyQt5.QtGui as QtGui
import numpy as np
import time

class loadUi_example(QMainWindow):


    def __init__(self):
        super().__init__()
        loadUi("untitled.ui",self)
        self.setWindowTitle("IMAGIFY")
        self.temp = 0
        self.msg = QMessageBox()
        self.filterimages()
        self.load.clicked.connect(self.load_)
        self.contup.clicked.connect(self.contup_)
        self.cropl.clicked.connect(self.cropl_)
        self.cropr.clicked.connect(self.cropr_)
        self.cropt.clicked.connect(self.cropt_)
        self.cropb.clicked.connect(self.cropb_)
        self.contdown.clicked.connect(self.contdown_)
        self.rotate.clicked.connect(self.rotate_)
        self.save.clicked.connect(self.save_)
        self.mirrorh.clicked.connect(self.mirrorh_)
        self.blur.clicked.connect(self.blur_)
        self.brightup.clicked.connect(self.brightness_up)
        self.brightdown.clicked.connect(self.brightness_down)
        self.enhancement.clicked.connect(self.enhancement_)
        self.mirrorv.clicked.connect(self.mirrorv_)
        self.filtre2.clicked.connect(self.filtre2_)
        self.filtre1.clicked.connect(self.filtre1_)
        self.filtre3.clicked.connect(self.filtre3_)
        self.filtre4.clicked.connect(self.filtre4_)
        self.filtre5.clicked.connect(self.filtre5_)
        self.filtre6.clicked.connect(self.filtre6_)
        self.filtre7.clicked.connect(self.filtre7_)
        self.filtre8.clicked.connect(self.filtre8_)
        self.filtre9.clicked.connect(self.filtre9_)
        self.filtre10.clicked.connect(self.filtre10_)
        self.filtre11.clicked.connect(self.filtre11_)
        self.filtre12.clicked.connect(self.filtre12_)
        self.filtre13.clicked.connect(self.filtre13_)
        self.filtre14.clicked.connect(self.filtre14_)
        self.filtre15.clicked.connect(self.filtre15_)
        self.filtre16.clicked.connect(self.filtre16_)
        self.filtre17.clicked.connect(self.filtre17_)
        self.filtre18.clicked.connect(self.filtre18_)
        self.filtre19.clicked.connect(self.filtre19_)
        self.filtre20.clicked.connect(self.filtre20_)

    def error_(self):

        self.msg.setText("Please, make sure you select a picture and no checkboxes are selected.")
        x =self.msg.exec()


    def filterimages(self):
        f_labels=[self.f1,self.f2,self.f3,self.f4,self.f5,self.f6,self.f7,self.f8,self.f9,self.f10,self.f11,self.f12,self.f13,self.f14, self.f15, self.f16, self.f17, self.f18, self.f19, self.f20]
        imglist=["1.jpg","2.jpg","3.jpg","4.jpg","5.jpg","6.jpg","7.jpg","8.jpg","9.jpg","10.jpg","11.jpg","12.jpg","13.jpg","14.jpg","15.jpg","16.jpg","17.jpg","18.jpg","19.jpg","20.jpg"]
        for each in range(len(imglist)):
            temp = QPixmap("filters/"+imglist[each]).scaled(f_labels[each].size())
            f_labels[each].setPixmap(temp)


    def load_(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        self.fileName, _ = QFileDialog.getOpenFileName(self,"QFileDialog.getOpenFileName()", "","All Files (*);;Python Files (*.py)", options=options)
        if self.fileName:

            self.label.resize(500,500)
            h = QPixmap(self.fileName).size().height()
            w=QPixmap(self.fileName).size().width()
            h_l=self.label.size().height()
            w_l=self.label.size().width()
            #Which edge of the image is larger, it is reduced in proportion with edge difference to fit the label .
            if (h>h_l) or (w>w_l):
                kh=h/h_l
                kw=w/w_l
                k=max(kh,kw)

                self.label.resize(QPixmap(self.fileName).size()/k)
                self.im = QPixmap(self.fileName).scaled(self.label.size())
                self.label.setPixmap(self.im)

            else:

                self.label.resize(QPixmap(self.fileName).size())
                self.im = QPixmap(self.fileName).scaled(self.label.size())
                self.label.setPixmap(self.im)

    def rotate_(self):
        try:
            src = self.im.toImage()
            src.save('filters/temp.png', 'png')
            # read the image
            src = cv2.imread('filters/temp.png')
            #rotate 90 degrees
            self.rot = cv2.rotate(src, cv2.cv2.ROTATE_90_CLOCKWISE)
            #cv2 to qimage
            self.image = QtGui.QImage(self.rot, self.rot.shape[1],self.rot.shape[0], self.rot.shape[1] * 3, QtGui.QImage.Format_BGR888)
            self.pix = QtGui.QPixmap(self.image)
            #label resize as rotated images size
            self.label.resize(self.label.size().height(), self.label.size().width())
            #set new image in label with qpixmap
            self.im = QPixmap(self.pix).scaled(self.label.size())
            self.label.setPixmap(self.im)
        #if there is any error, call error function
        except:
            self.error_()



    def contup_(self):
        try:
            src = self.im.toImage()
            src.save('filters/temp.png', 'png')
            # read the image
            src = cv2.imread('filters/temp.png')
            #contrast up
            self.contrast_img = cv2.addWeighted(src, 1.1, np.zeros(src.shape, src.dtype), 0, 0)
            # cv2 to qimage
            self.image = QtGui.QImage(self.contrast_img, self.contrast_img.shape[1], self.contrast_img.shape[0], self.contrast_img.shape[1] * 3, QtGui.QImage.Format_BGR888)
            # set new image in label with qpixmap
            self.pix = QtGui.QPixmap(self.image)
            self.im = QPixmap(self.pix).scaled(self.label.size())
            self.label.setPixmap(self.im)
        # if there is any error, call error function
        except:
            self.error_()

    def contdown_(self):
        try:
            src = self.im.toImage()
            src.save('filters/temp.png', 'png')
            # read the image
            src = cv2.imread('filters/temp.png')
            # contrast down
            self.contrast_img = cv2.addWeighted(src, 0.9, np.zeros(src.shape, src.dtype), 0, 0)
            # cv2 to qimage
            self.image = QtGui.QImage(self.contrast_img,self.contrast_img.shape[1], self.contrast_img.shape[0], self.contrast_img.shape[1] * 3, QtGui.QImage.Format_BGR888)
            # set new image in label with qpixmap
            self.pix = QtGui.QPixmap(self.image)
            self.im = QPixmap(self.pix).scaled(self.label.size())
            self.label.setPixmap(self.im)
        # if there is any error, call error function
        except:
            self.error_()

    def cropl_(self):
        try:
            src = self.im.toImage()
            src.save('filters/temp.png', 'png')
            # read the image
            src = cv2.imread('filters/temp.png')
            height, width = src.shape[0:2]
            #crop the image left with starting int(width * 0.10)
            self.croppedImage = src[0:int(height), int(width * 0.05):int(width)]
            self.croppedImage= np.array(self.croppedImage, dtype=np.uint8)
            # cv2 to qimage
            self.image = QtGui.QImage(self.croppedImage.data, self.croppedImage.shape[1], self.croppedImage.shape[0], self.croppedImage.shape[1] * 3, QtGui.QImage.Format_BGR888)
            # set new image in label with qpixmap
            self.pix = QtGui.QPixmap(self.image)
            # label resize as cropped images size
            self.label.resize(self.label.size().width()*0.95, self.label.size().height())
            # set new image in label with qpixmap
            self.im = QPixmap(self.pix).scaled(self.label.size())
            self.label.setPixmap(self.im)
        # if there is any error, call error function
        except:
            self.error_()


    def cropr_(self):
        try:
            src = self.im.toImage()
            src.save('filters/temp.png', 'png')
            # read the image
            src = cv2.imread('filters/temp.png')
            height, width = src.shape[0:2]
            # crop the image right with finishing int(width*0.9)
            self.croppedImage = src[0:int(height), 0:int(width*0.95)]
            self.croppedImage= np.array(self.croppedImage, dtype=np.uint8)
            # cv2 to qimage
            self.image = QtGui.QImage(self.croppedImage.data, self.croppedImage.shape[1], self.croppedImage.shape[0], self.croppedImage.shape[1] * 3, QtGui.QImage.Format_BGR888)
            # set new image in label with qpixmap
            self.pix = QtGui.QPixmap(self.image)
            # label resize as cropped images size
            self.label.resize(self.label.size().width()*0.95, self.label.size().height())
            self.im = QPixmap(self.pix).scaled(self.label.size())
            self.label.setPixmap(self.im)
        # if there is any error, call error function
        except:
            self.error_()

    def cropt_(self):
        try:
            src = self.im.toImage()
            src.save('filters/temp.png', 'png')
            # read the image
            src = cv2.imread('filters/temp.png')
            height, width = src.shape[0:2]
            # crop the image top with starting int(height * 0.10)
            self.croppedImage = src[int(height * 0.05):int(height), 0:int(width)]
            self.croppedImage= np.array(self.croppedImage, dtype=np.uint8)
            # cv2 to qimage
            self.image = QtGui.QImage(self.croppedImage.data, self.croppedImage.shape[1], self.croppedImage.shape[0], self.croppedImage.shape[1] * 3, QtGui.QImage.Format_BGR888)
            # set new image in label with qpixmap
            self.pix = QtGui.QPixmap(self.image)
            # label resize as cropped images size
            self.label.resize(self.label.size().width(), self.label.size().height()*0.95)
            self.im = QPixmap(self.pix).scaled(self.label.size())
            self.label.setPixmap(self.im)
        # if there is any error, call error function
        except:
            self.error_()

    def cropb_(self):
        try:
            src = self.im.toImage()
            src.save('filters/temp.png', 'png')
            # read the image
            src = cv2.imread('filters/temp.png')
            height, width = src.shape[0:2]
            # crop the image top with starting int(height * 0.10)
            self.croppedImage = src[0:int(height*0.95), 0:int(width)]
            self.croppedImage= np.array(self.croppedImage, dtype=np.uint8)
            # cv2 to qimage
            self.image = QtGui.QImage(self.croppedImage.data, self.croppedImage.shape[1], self.croppedImage.shape[0], self.croppedImage.shape[1] * 3, QtGui.QImage.Format_BGR888)
            # set new image in label with qpixmap
            self.pix = QtGui.QPixmap(self.image)
            # label resize as cropped images size
            self.label.resize(self.label.size().width(), self.label.size().height()*0.95)
            self.im = QPixmap(self.pix).scaled(self.label.size())
            self.label.setPixmap(self.im)
        # if there is any error, call error function
        except:
            self.error_()
    def blur_(self):

        try:
            if self.temp==0:
                src = self.im.toImage()
                src.save('filters/temp.png', 'png')
                # read the image
                img = cv2.imread('filters/temp.png')
                #turn back to first image
                self.original = img.copy()

            if self.blur.isChecked() == True:
                self.temp=1
                #gaussian blur
                self.img= cv2.GaussianBlur(img, (5, 5), cv2.BORDER_DEFAULT)
                # cv2 to qimage
                self.image = QtGui.QImage(self.img.data, self.img.shape[1], self.img.shape[0], self.img.shape[1] * 3, QtGui.QImage.Format_BGR888)
                # set new image in label with qpixmap
                self.pix = QtGui.QPixmap(self.image)
                self.im = QPixmap(self.pix).scaled(self.label.size())
                self.label.setPixmap(self.im)
            #self.blur.isChecked() == False
            else:
                self.temp = 0
                #turn back
                self.image = QtGui.QImage(self.original.data, self.original.shape[1],
                                     self.original.shape[0], self.original.shape[1] * 3, QtGui.QImage.Format_BGR888)
                self.pix = QtGui.QPixmap(self.image)
                self.im = QPixmap(self.pix).scaled(self.label.size())
                self.label.setPixmap(self.im)
        # if there is any error, call error function
        except:
            self.error_()

    def mirrorh_(self):
        try:
            src = self.im.toImage()
            src.save('filters/temp.png', 'png')
            # read the image
            src = cv2.imread('filters/temp.png')
            #flip the image horizontal
            self.mir = cv2.flip(src, 1)
            # cv2 to qimage
            self.image = QtGui.QImage(self.mir.data, self.mir.shape[1], self.mir.shape[0], self.mir.shape[1] * 3, QtGui.QImage.Format_BGR888)
            # set new image in label with qpixmap
            self.pix = QtGui.QPixmap(self.image)
            self.im = QPixmap(self.pix).scaled(self.label.size())
            self.label.setPixmap(self.im)
        # if there is any error, call error function
        except:
            self.error_()
    def mirrorv_(self):
        try:

            src = self.im.toImage()
            src.save('filters/temp.png', 'png')
            # read the image
            src = cv2.imread('filters/temp.png')
            #flip the image vertical
            self.mir = cv2.flip(src, 0)
            # cv2 to qimage
            self.image = QtGui.QImage(self.mir.data, self.mir.shape[1], self.mir.shape[0], self.mir.shape[1] * 3, QtGui.QImage.Format_BGR888)
            # set new image in label with qpixmap
            self.pix = QtGui.QPixmap(self.image)
            self.im = QPixmap(self.pix).scaled(self.label.size())
            self.label.setPixmap(self.im)
        # if there is any error, call error function
        except:
            self.error_()
    def save_(self):
        #save the image with time as name
        try:
            timee = time.asctime().replace(" ", "_").replace(":", "_")

            src = self.im.toImage()
            src.save(timee + ".png", 'png')
        # if there is any error, call error function
        except:
            self.error_()

    def brightness_up(self):
        try:
            src = self.im.toImage()
            src.save('filters/temp.png', 'png')
            # read the image
            src = cv2.imread('filters/temp.png')
            #bright up of the image
            self.image = cv2.addWeighted(src,1,np.zeros(src.shape,src.dtype),0,10)
            # cv2 to qimage
            self.image = QtGui.QImage(self.image.data, self.image.shape[1], self.image.shape[0], self.image.shape[1] * 3, QtGui.QImage.Format_BGR888)
            # set new image in label with qpixmap
            self.pix = QtGui.QPixmap(self.image)
            self.im = QPixmap(self.pix).scaled(self.label.size())
            self.label.setPixmap(self.im)
        # if there is any error, call error function
        except:
            self.error_()
    def brightness_down(self):

        try:
            src = self.im.toImage()
            src.save('filters/temp.png', 'png')
            # read the image
            src = cv2.imread('filters/temp.png')
            #bright down of the image
            self.image = cv2.addWeighted(src,1,np.zeros(src.shape,src.dtype),0,-10)
            # cv2 to qimage
            self.image = QtGui.QImage(self.image.data, self.image.shape[1], self.image.shape[0], self.image.shape[1] * 3, QtGui.QImage.Format_BGR888)
            # set new image in label with qpixmap
            self.pix = QtGui.QPixmap(self.image)
            self.im = QPixmap(self.pix).scaled(self.label.size())
            self.label.setPixmap(self.im)
        # if there is any error, call error function
        except:
            self.error_()
    def filtre1_(self):
        try:
            if self.temp == 0:
                src = self.im.toImage()
                src.save('filters/temp.png', 'png')
                # read the image
                img = cv2.imread('filters/temp.png')
                #turn back to first image
                self.original = img.copy()

            if self.filtre1.isChecked() == True:
                self.temp = 1
                # convert to float
                img1 = np.array(img, dtype=np.float64)
                # multipy image with matrix
                img1 = cv2.transform(img1, np.matrix([[0.272, 0.534, 0.131],
                                                      [0.349, 0.686, 0.168],
                                                      [0.393, 0.769, 0.189]]))
                #normalize
                img1[np.where(img1 > 255)] = 255
                self.img = np.array(img1, dtype=np.uint8)
                # cv2 to qimage
                self.image = QtGui.QImage(self.img.data, self.img.shape[1], self.img.shape[0], self.img.shape[1] * 3, QtGui.QImage.Format_BGR888)
                # set new image in label with qpixmap
                self.pix = QtGui.QPixmap(self.image)
                self.im = QPixmap(self.pix).scaled(self.label.size())
                self.label.setPixmap(self.im)
            #self.filtre1.isChecked() == False:
            #turn back
            else:
                self.temp = 0
                self.image = QtGui.QImage(self.original.data, self.original.shape[1], self.original.shape[0], self.original.shape[1] * 3, QtGui.QImage.Format_BGR888)

                self.pix = QtGui.QPixmap(self.image)
                self.im = QPixmap(self.pix).scaled(self.label.size())
                self.label.setPixmap(self.im)
        # if there is any error, call error function
        except:
            self.error_()
    def filtre2_(self):

        try:
            if self.temp == 0:
                src = self.im.toImage()
                src.save('filters/temp.png', 'png')
                # read the image
                img = cv2.imread('filters/temp.png')
                self.original = img.copy()

            if self.filtre2.isChecked() == True:
                self.temp = 1
                # convert to float
                img2 = np.array(img, dtype=np.float64)
                # multipy image with matrix
                img2 = cv2.transform(img2, np.matrix([[0.672, 0.334, 0.931],
                                                      [0.149, 0.386, 0.268],
                                                      [0.793, 0.469, 0.289]]))
                #normalize
                img2[np.where(img2 > 255)] = 255
                self.img = np.array(img2, dtype=np.uint8)
                # cv2 to qimage
                self.image = QtGui.QImage(self.img.data, self.img.shape[1],self.img.shape[0], self.img.shape[1] * 3, QtGui.QImage.Format_BGR888)
                # set new image in label with qpixmap
                self.pix = QtGui.QPixmap(self.image)
                self.im = QPixmap(self.pix).scaled(self.label.size())
                self.label.setPixmap(self.im)
            #self.filtre2.isChecked() == False, turn back
            else:
                self.temp = 0
                self.image = QtGui.QImage(self.original.data, self.original.shape[1], self.original.shape[0], self.original.shape[1] * 3, QtGui.QImage.Format_BGR888)
                self.pix = QtGui.QPixmap(self.image)
                self.im = QPixmap(self.pix).scaled(self.label.size())
                self.label.setPixmap(self.im)
        # if there is any error, call error function
        except:
            self.error_()
    def filtre3_(self):

        try:
            if self.temp == 0:
                src = self.im.toImage()
                src.save('filters/temp.png', 'png')
                # read the image
                img = cv2.imread('filters/temp.png')
                self.original = img.copy()

            if self.filtre3.isChecked() == True:
                self.temp = 1
                img3 = np.array(img, dtype=np.float64)
                img3 = cv2.transform(img3, np.matrix([[0.273, 0.535, 0.232],
                                                      [0.348, 0.685, 0.167],
                                                      [0.392, 0.768, 0.287]]))
                img3[np.where(img3 > 255)] = 255
                self.img = np.array(img3, dtype=np.uint8)
                # cv2 to qimage
                self.image = QtGui.QImage(self.img.data, self.img.shape[1], self.img.shape[0], self.img.shape[1] * 3, QtGui.QImage.Format_BGR888)
                # set new image in label with qpixmap
                self.pix = QtGui.QPixmap(self.image)
                self.im = QPixmap(self.pix).scaled(self.label.size())
                self.label.setPixmap(self.im)
            #self.filtre3.isChecked() == False, turn back
            else:
                self.temp = 0
                self.image = QtGui.QImage(self.original.data, self.original.shape[1],
                                          self.original.shape[0], self.original.shape[1] * 3, QtGui.QImage.Format_BGR888)

                self.pix = QtGui.QPixmap(self.image)
                self.im = QPixmap(self.pix).scaled(self.label.size())
                self.label.setPixmap(self.im)
        # if there is any error, call error function
        except:
            self.error_()
    def filtre4_(self):

        try:
            if self.temp == 0:
                src = self.im.toImage()
                src.save('filters/temp.png', 'png')
                # read the image
                img = cv2.imread('filters/temp.png')
                self.original = img.copy()

            if self.filtre4.isChecked() == True:
                self.temp = 1
                # converting to float
                img4 = np.array(img, dtype=np.float64)
                W = [0.2, 0.5, 0.3]
                # calculate the tensor dot product of two given tensors
                W_mean = np.tensordot(img4, W, axes=((-1, -1)))[..., None]
                img4[:] = W_mean.astype(img4.dtype)
                self.img = np.array(img4, dtype=np.uint8)
                # cv2 to qimage
                self.image = QtGui.QImage(self.img.data, self.img.shape[1], self.img.shape[0], self.img.shape[1] * 3, QtGui.QImage.Format_BGR888)
                # set new image in label with qpixmap
                self.pix = QtGui.QPixmap(self.image)
                self.im = QPixmap(self.pix).scaled(self.label.size())
                self.label.setPixmap(self.im)
            #self.filtre4.isChecked() == False , turn back
            else:
                self.temp = 0
                self.image = QtGui.QImage(self.original.data, self.original.shape[1], self.original.shape[0], self.original.shape[1] * 3, QtGui.QImage.Format_BGR888)
                self.pix = QtGui.QPixmap(self.image)
                self.im = QPixmap(self.pix).scaled(self.label.size())
                self.label.setPixmap(self.im)
        # if there is any error, call error function
        except:
            self.error_()
    def filtre5_(self):

        try:
            if self.temp == 0:
                src = self.im.toImage()
                src.save('filters/temp.png', 'png')
                # read the image
                img = cv2.imread('filters/temp.png')
                self.original = img.copy()

            if self.filtre5.isChecked() == True:
                self.temp = 1
                # converting to float
                img5 = np.array(img, dtype=np.float64)
                x, y, z = img5.shape
                for i in range(x):
                    for j in range(y):
                        # if img5[i, j, 0] < 200 and img5[i, j, 1] < 150) use mean func.
                        if (img5[i, j, 0] < 200 and img5[i, j, 1] < 150):
                            img5[i, j] = np.mean(img5[i, j])

                self.img = np.array(img5, dtype=np.uint8)
                # cv2 to qimage
                self.image = QtGui.QImage(self.img.data, self.img.shape[1], self.img.shape[0], self.img.shape[1] * 3, QtGui.QImage.Format_BGR888)
                # set new image in label with qpixmap
                self.pix = QtGui.QPixmap(self.image)
                self.im = QPixmap(self.pix).scaled(self.label.size())
                self.label.setPixmap(self.im)
            #self.filtre5.isChecked() == False , turn back
            else:
                self.temp = 0
                self.image = QtGui.QImage(self.original.data, self.original.shape[1], self.original.shape[0], self.original.shape[1] * 3, QtGui.QImage.Format_BGR888)
                self.pix = QtGui.QPixmap(self.image)
                self.im = QPixmap(self.pix).scaled(self.label.size())
                self.label.setPixmap(self.im)
        # if there is any error, call error function
        except:
            self.error_()
    def filtre6_(self):

        try:
            if self.temp == 0:
                src = self.im.toImage()
                src.save('filters/temp.png', 'png')
                img = cv2.imread('filters/temp.png')
                self.original = img.copy()

            if self.filtre6.isChecked() == True:
                self.temp = 1
                img6 = np.array(img, dtype=np.float64)
                img6[:] = np.max(img6, axis=-1, keepdims=1) / 3 + np.min(img6, axis=-1, keepdims=1) / 3
                self.img = np.array(img6, dtype=np.uint8)
                # cv2 to qimage
                self.image = QtGui.QImage(self.img.data, self.img.shape[1],
                                          self.img.shape[0], self.img.shape[1] * 3, QtGui.QImage.Format_BGR888)
                # set new image in label with qpixmap
                self.pix = QtGui.QPixmap(self.image)
                self.im = QPixmap(self.pix).scaled(self.label.size())
                self.label.setPixmap(self.im)

            else:
                self.temp = 0
                self.image = QtGui.QImage(self.original.data, self.original.shape[1],
                                          self.original.shape[0], self.original.shape[1] * 3, QtGui.QImage.Format_BGR888)

                self.pix = QtGui.QPixmap(self.image)
                self.im = QPixmap(self.pix).scaled(self.label.size())
                self.label.setPixmap(self.im)
        # if there is any error, call error function
        except:
            self.error_()

    def filtre7_(self):
        try:
            if self.temp == 0:
                src = self.im.toImage()
                src.save('filters/temp.png', 'png')
                # read the image
                img = cv2.imread('filters/temp.png')
                self.original = img.copy()

            if self.filtre7.isChecked() == True:
                self.temp = 1
                img7 = np.array(img, dtype=np.float64)
                # apply morphology
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (6, 6))
                morph = cv2.morphologyEx(img7, cv2.MORPH_OPEN, kernel)
                # bright darkness
                img7 = cv2.normalize(morph, None, 20, 255, cv2.NORM_MINMAX)
                self.img = np.uint8(img7)
                # cv2 to qimage
                self.image = QtGui.QImage(self.img.data, self.img.shape[1],
                                          self.img.shape[0], self.img.shape[1] * 3, QtGui.QImage.Format_BGR888)
                # set new image in label with qpixmap
                self.pix = QtGui.QPixmap(self.image)
                self.im = QPixmap(self.pix).scaled(self.label.size())
                self.label.setPixmap(self.im)

            else:
                self.temp = 0
                self.image = QtGui.QImage(self.original.data, self.original.shape[1],
                                          self.original.shape[0], self.original.shape[1] * 3, QtGui.QImage.Format_BGR888)

                self.pix = QtGui.QPixmap(self.image)
                self.im = QPixmap(self.pix).scaled(self.label.size())
                self.label.setPixmap(self.im)
        # if there is any error, call error function
        except:
            self.error_()


    def filtre8_(self):
        try:

            if self.temp == 0:
                src = self.im.toImage()
                src.save('filters/temp.png', 'png')
                # read the image
                img = cv2.imread('filters/temp.png')
                self.original = img.copy()

            if self.filtre8.isChecked() == True:
                self.temp = 1
                img8 = np.array(img, dtype=np.float64)
                img8[:] = np.max(img8, axis=-1, keepdims=1) / 2 + np.min(img8, axis=-1, keepdims=1) / 2
                height, width = img8.shape[:2]
                thresh = 0.9
                for i in range(height):
                    for j in range(width):
                        if np.random.rand() <= thresh:
                            if np.random.randint(2) == 0:
                                img8[i, j] = min(img8[i, j] + np.random.randint(0, 64))
                            else:
                                img8[i, j] = max(img8[i, j] - np.random.randint(0, 64))
                self.img = np.array(img8, dtype=np.uint8)
                # cv2 to qimage
                self.image = QtGui.QImage(self.img.data, self.img.shape[1],
                                          self.img.shape[0], self.img.shape[1] * 3, QtGui.QImage.Format_BGR888)
                # set new image in label with qpixmap
                self.pix = QtGui.QPixmap(self.image)
                self.im = QPixmap(self.pix).scaled(self.label.size())
                self.label.setPixmap(self.im)
            #self.filtre8.isChecked() == False, turn back
            else:
                self.temp = 0
                self.image = QtGui.QImage(self.original.data, self.original.shape[1],
                                          self.original.shape[0], self.original.shape[1] * 3, QtGui.QImage.Format_BGR888)

                self.pix = QtGui.QPixmap(self.image)
                self.im = QPixmap(self.pix).scaled(self.label.size())
                self.label.setPixmap(self.im)

        # if there is any error, call error function
        except:
            self.error_()


    def filtre9_(self):
        try:
            if self.temp == 0:
                src = self.im.toImage()
                src.save('filters/temp.png', 'png')
                # read the image
                img = cv2.imread('filters/temp.png')
                self.original = img.copy()

            if self.filtre9.isChecked() == True:
                self.temp = 1
                img9 = np.array(img, dtype=np.float64)
                x, y, z = img9.shape
                for i in range(x):
                    for j in range(y):
                        if (img9[i, j, 0] < 200 and img9[i, j, 1] < 150):
                            img9[i, j] = np.mean(img9[i, j]/2)
                self.img = np.array(img9, dtype=np.uint8)
                # cv2 to qimage
                self.image = QtGui.QImage(self.img.data, self.img.shape[1],
                                          self.img.shape[0], self.img.shape[1] * 3, QtGui.QImage.Format_BGR888)
                # set new image in label with qpixmap
                self.pix = QtGui.QPixmap(self.image)
                self.im = QPixmap(self.pix).scaled(self.label.size())
                self.label.setPixmap(self.im)
            else:
                self.temp = 0
                self.image = QtGui.QImage(self.original.data, self.original.shape[1],
                                          self.original.shape[0], self.original.shape[1] * 3, QtGui.QImage.Format_BGR888)

                self.pix = QtGui.QPixmap(self.image)
                self.im = QPixmap(self.pix).scaled(self.label.size())
                self.label.setPixmap(self.im)
        # if there is any error, call error function
        except:
            self.error_()


    def enhancement_(self):
        try:
            if self.temp == 0:
                src = self.im.toImage()
                src.save('filters/temp.png', 'png')
                # read the image
                img = cv2.imread('filters/temp.png')
                self.original = img.copy()

            if self.enhancement.isChecked() == True:
                self.temp = 1
                #applying histogram equalization
                flat = img.flatten()
                bins=256
                histogram = np.zeros(bins)
                for each in flat:
                    histogram[each] += 1
                a = iter(histogram)
                b = [next(a)]
                for each in a:
                    b.append(b[-1] + each)
                h=np.array(b)
                nj = (h - h.min()) * 255
                N = h.max() - h.min()
                h = nj / N
                img10= h[flat]
                img10 = np.reshape(img10, img.shape)
                self.img = np.array(img10, dtype=np.uint8)
                # cv2 to qimage
                self.image = QtGui.QImage(self.img.data, self.img.shape[1],
                                          self.img.shape[0], self.img.shape[1] * 3, QtGui.QImage.Format_BGR888)
                # set new image in label with qpixmap
                self.pix = QtGui.QPixmap(self.image)
                self.im = QPixmap(self.pix).scaled(self.label.size())
                self.label.setPixmap(self.im)

            else:
                self.temp = 0
                self.image = QtGui.QImage(self.original.data, self.original.shape[1],
                                          self.original.shape[0], self.original.shape[1] * 3, QtGui.QImage.Format_BGR888)

                self.pix = QtGui.QPixmap(self.image)
                self.im = QPixmap(self.pix).scaled(self.label.size())
                self.label.setPixmap(self.im)
        # if there is any error, call error function
        except:
            self.error_()

    def filtre10_(self):
        try:

            if self.temp == 0:
                src = self.im.toImage()
                src.save('filters/temp.png', 'png')
                # read the image
                img = cv2.imread('filters/temp.png')
                self.original = img.copy()

            if self.filtre10.isChecked() == True:
                self.temp = 1
                height, width = img.shape[:2]
                y = np.ones((height, width), np.uint8) * 128
                output = np.zeros((height, width), np.uint8)
                # create  kernels
                kernel1 = np.array([[0, -1, -1],
                                    [1, 0, -1],
                                    [1, 1, 0]])
                kernel2 = np.array([[-1, -1, 0],
                                    [-1, 0, 1],
                                    [0, 1, 1]])
                #convert grayscale
                img10 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                # bottom left
                output1 = cv2.add(cv2.filter2D(img10, -1, kernel1), y)
                # bottom right
                output2 = cv2.add(cv2.filter2D(img10, -1, kernel2), y)
                for i in range(height):
                    for j in range(width):
                        output[i, j] = max(output1[i, j], output2[i, j])

                self.img = output
                # cv2 to qimage
                self.image = QtGui.QImage(self.img.data, self.img.shape[1],
                                          self.img.shape[0],  QtGui.QImage.Format_Grayscale8)
                # set new image in label with qpixmap
                self.pix = QtGui.QPixmap(self.image)
                self.im = QPixmap(self.pix).scaled(self.label.size())
                self.label.setPixmap(self.im)

            else:
                self.temp = 0
                self.image = QtGui.QImage(self.original.data, self.original.shape[1],
                                          self.original.shape[0], self.original.shape[1] * 3, QtGui.QImage.Format_BGR888)

                self.pix = QtGui.QPixmap(self.image)
                self.im = QPixmap(self.pix).scaled(self.label.size())
                self.label.setPixmap(self.im)
        # if there is any error, call error function
        except:
            self.error_()


    def filtre11_(self):
        try:

            if self.temp == 0:
                src = self.im.toImage()
                src.save('filters/temp.png', 'png')
                # read the image
                img = cv2.imread('filters/temp.png')
                self.original = img.copy()

            if self.filtre11.isChecked() == True:
                self.temp = 1
                height, width = img.shape[:2]
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                thresh = 0.8  # noise will be added to 80% pixels
                for i in range(height):
                    for j in range(width):
                        if np.random.rand() <= thresh:
                            if np.random.randint(2) == 0:
                                # adding random value between 0 to 64
                                gray[i, j] = min(gray[i, j] + np.random.randint(0, 64),255)

                            else:
                                # subtracting random values between 0 to 64.
                                gray[i, j] = max(gray[i, j] - np.random.randint(0, 64),0)

                self.img = gray
                # cv2 to qimage
                self.image = QtGui.QImage(self.img.data, self.img.shape[1],
                                          self.img.shape[0],  QtGui.QImage.Format_Grayscale8)
                # set new image in label with qpixmap
                self.pix = QtGui.QPixmap(self.image)
                self.im = QPixmap(self.pix).scaled(self.label.size())
                self.label.setPixmap(self.im)

            else:
                self.temp = 0
                self.image = QtGui.QImage(self.original.data, self.original.shape[1],
                                          self.original.shape[0], self.original.shape[1] * 3, QtGui.QImage.Format_BGR888)

                self.pix = QtGui.QPixmap(self.image)
                self.im = QPixmap(self.pix).scaled(self.label.size())
                self.label.setPixmap(self.im)
        # if there is any error, call error function
        except:
            self.error_()

    def filtre12_(self):
        try:
            if self.temp == 0:
                src = self.im.toImage()
                src.save('filters/temp.png', 'png')
                # read the image
                img = cv2.imread('filters/temp.png')
                self.original = img.copy()

            if self.filtre12.isChecked() == True:
                self.temp = 1

                def gamma_function(channel, gamma):
                    invGamma = 1 / gamma
                    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
                    channel = cv2.LUT(channel, table)
                    return channel

                img[:, :, 0] = gamma_function(img[:, :, 0], 0.75)  # b channel down
                img[:, :, 2] = gamma_function(img[:, :, 2], 1.25)  # r channel up
                hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                hsv[:, :, 1] = gamma_function(hsv[:, :, 1], 1.2)  # saturation up
                self.img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

                # cv2 to qimage
                self.image = QtGui.QImage(self.img.data, self.img.shape[1],
                                          self.img.shape[0], self.img.shape[1] * 3, QtGui.QImage.Format_BGR888)
                # set new image in label with qpixmap
                self.pix = QtGui.QPixmap(self.image)
                self.im = QPixmap(self.pix).scaled(self.label.size())
                self.label.setPixmap(self.im)

            else:
                self.temp = 0
                self.image = QtGui.QImage(self.original.data, self.original.shape[1],
                                          self.original.shape[0], self.original.shape[1] * 3, QtGui.QImage.Format_BGR888)

                self.pix = QtGui.QPixmap(self.image)
                self.im = QPixmap(self.pix).scaled(self.label.size())
                self.label.setPixmap(self.im)
        # if there is any error, call error function
        except:
            self.error_()



    def filtre13_(self):
        try:
            if self.temp == 0:
                src = self.im.toImage()
                src.save('filters/temp.png', 'png')
                # read the image
                img = cv2.imread('filters/temp.png')
                self.original = img.copy()

            if self.filtre13.isChecked() == True:
                self.temp = 1

                output = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
                L, A, B = cv2.split(output)
                # Interpolation values
                originalValues = np.array([0, 15, 30, 50, 70, 90, 120, 160, 180, 210, 255])
                values = np.array([0, 0, 5, 15, 60, 110, 150, 190, 210, 230, 255])
                allValues = np.arange(0, 256)
                lookuptable = np.interp(allValues, originalValues, values)
                L = cv2.LUT(L, lookuptable)
                L = np.uint8(L)
                # merge back the channels
                self.img = cv2.merge([L, A, B])
                # convert to BGR
                self.img = cv2.cvtColor(self.img, cv2.COLOR_LAB2BGR)
                # convert to HSV color space
                self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)
                # split H, S, V channels
                H, S, V = cv2.split(self.img)
                # Multiply S channel by saturation scale value
                S = S * 0.01
                S = np.uint8(S)
                S = np.clip(S, 0, 255)
                # merge
                self.img = cv2.merge([H, S, V])
                # convert back to BGR
                self.img = cv2.cvtColor(self.img, cv2.COLOR_HSV2BGR)

                # cv2 to qimage
                self.image = QtGui.QImage(self.img.data, self.img.shape[1],
                                          self.img.shape[0], self.img.shape[1] * 3, QtGui.QImage.Format_BGR888)
                # set new image in label with qpixmap
                self.pix = QtGui.QPixmap(self.image)
                self.im = QPixmap(self.pix).scaled(self.label.size())
                self.label.setPixmap(self.im)

            else:
                self.temp = 0
                self.image = QtGui.QImage(self.original.data, self.original.shape[1],
                                          self.original.shape[0], self.original.shape[1] * 3, QtGui.QImage.Format_BGR888)
                # set new image in label with qpixmap
                self.pix = QtGui.QPixmap(self.image)
                self.im = QPixmap(self.pix).scaled(self.label.size())
                self.label.setPixmap(self.im)
        # if there is any error, call error function
        except:
            self.error_()


    def filtre14_(self):
        try:
            if self.temp == 0:
                src = self.im.toImage()
                src.save('filters/temp.png', 'png')
                # read the image
                img = cv2.imread('filters/temp.png')
                self.original = img.copy()

            if self.filtre14.isChecked() == True:
                self.temp = 1
                # Convert to HLS
                image_HLS = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
                image_HLS = np.array(image_HLS, dtype=np.float64)
                daylight = 1.15
                # channel 1(Lightness)
                image_HLS[:, :, 1] = image_HLS[:, :, 1] * daylight
                image_HLS[:, :, 1][image_HLS[:, :, 1] > 255] = 255
                image_HLS = np.array(image_HLS, dtype=np.uint8)
                # Conversion to RGB
                self.img = cv2.cvtColor(image_HLS, cv2.COLOR_HLS2BGR)

                # cv2 to qimage
                self.image = QtGui.QImage(self.img.data, self.img.shape[1],
                                          self.img.shape[0], self.img.shape[1] * 3, QtGui.QImage.Format_BGR888)
                # set new image in label with qpixmap
                self.pix = QtGui.QPixmap(self.image)
                self.im = QPixmap(self.pix).scaled(self.label.size())
                self.label.setPixmap(self.im)

            else:
                self.temp = 0
                self.image = QtGui.QImage(self.original.data, self.original.shape[1],
                                          self.original.shape[0], self.original.shape[1] * 3, QtGui.QImage.Format_BGR888)
                # set new image in label with qpixmap
                self.pix = QtGui.QPixmap(self.image)
                self.im = QPixmap(self.pix).scaled(self.label.size())
                self.label.setPixmap(self.im)

        # if there is any error, call error function
        except:
            self.error_()

    def filtre15_(self):
        try:
            if self.temp == 0:
                src = self.im.toImage()
                src.save('filters/temp.png', 'png')
                # read the image
                img = cv2.imread('filters/temp.png')
                self.original = img.copy()

            if self.filtre15.isChecked() == True:
                self.temp = 1

                img15 = np.array(img, dtype=np.float64)
                img15 = cv2.transform(img15, np.matrix([[0.072, 0.534, 0.131],
                                                      [0.149, 0.386, 0.068],
                                                      [0.193, 0.269, 0.089]]))
                img15[np.where(img15 > 255)] = 255
                self.img = np.array(img15, dtype=np.uint8)
                # cv2 to qimage
                self.image = QtGui.QImage(self.img.data, self.img.shape[1],
                                          self.img.shape[0], self.img.shape[1] * 3, QtGui.QImage.Format_BGR888)
                # set new image in label with qpixmap
                self.pix = QtGui.QPixmap(self.image)
                self.im = QPixmap(self.pix).scaled(self.label.size())
                self.label.setPixmap(self.im)

            else:
                self.temp = 0
                self.image = QtGui.QImage(self.original.data, self.original.shape[1],
                                          self.original.shape[0], self.original.shape[1] * 3, QtGui.QImage.Format_BGR888)
                # set new image in label with qpixmap
                self.pix = QtGui.QPixmap(self.image)
                self.im = QPixmap(self.pix).scaled(self.label.size())
                self.label.setPixmap(self.im)
        # if there is any error, call error function
        except:
            self.error_()



    def filtre16_(self):
        try:

            if self.temp == 0:
                src = self.im.toImage()
                src.save('filters/temp.png', 'png')
                # read the image
                img = cv2.imread('filters/temp.png')
                self.original = img.copy()

            if self.filtre16.isChecked() == True:
                self.temp = 1
                #convert grayscale
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                #apply sobel filter
                self.img = gray
                sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

                self.y = cv2.filter2D(self.img, -1, sobel_y)

                sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
                self.x = cv2.filter2D(self.img, -1, sobel_x)
                self.img = cv2.bitwise_or(self.x, self.y)
                # cv2 to qimage
                self.image = QtGui.QImage(self.img.data, self.img.shape[1],
                                          self.img.shape[0],  QtGui.QImage.Format_Grayscale8)
                # set new image in label with qpixmap
                self.pix = QtGui.QPixmap(self.image)
                self.im = QPixmap(self.pix).scaled(self.label.size())
                self.label.setPixmap(self.im)
            #if self.filtre16.isChecked() == False, turn back
            else:
                self.temp = 0
                self.image = QtGui.QImage(self.original.data, self.original.shape[1],
                                          self.original.shape[0], self.original.shape[1] * 3, QtGui.QImage.Format_BGR888)
                # set new image in label with qpixmap
                self.pix = QtGui.QPixmap(self.image)
                self.im = QPixmap(self.pix).scaled(self.label.size())
                self.label.setPixmap(self.im)
        # if there is any error, call error function
        except:
            self.error_()




    def filtre17_(self):
        try:
            if self.temp == 0:
                src = self.im.toImage()
                src.save('filters/temp.png', 'png')
                # read the image
                img = cv2.imread('filters/temp.png')
                self.original = img.copy()

            if self.filtre17.isChecked() == True:
                self.temp = 1
                # converting to float
                img1 = np.array(img, dtype=np.float64)
                img1 = cv2.transform(img1, np.matrix([[0.672, 0.734, 0.531],
                                                      [0.549, 0.386, 0.658],
                                                      [0.593, 0.369, 0.759]]))
                img1[np.where(img1 > 255)] = 255
                self.img = np.array(img1, dtype=np.uint8)
                # cv2 to qimage
                self.image = QtGui.QImage(self.img.data, self.img.shape[1],
                                          self.img.shape[0], self.img.shape[1] * 3, QtGui.QImage.Format_BGR888)
                # set new image in label with qpixmap
                self.pix = QtGui.QPixmap(self.image)
                self.im = QPixmap(self.pix).scaled(self.label.size())
                self.label.setPixmap(self.im)

            else:
                self.temp = 0
                self.image = QtGui.QImage(self.original.data, self.original.shape[1],
                                          self.original.shape[0], self.original.shape[1] * 3, QtGui.QImage.Format_BGR888)
                # set new image in label with qpixmap
                self.pix = QtGui.QPixmap(self.image)
                self.im = QPixmap(self.pix).scaled(self.label.size())
                self.label.setPixmap(self.im)
        # if there is any error, call error function
        except:
            self.error_()


    def filtre18_(self):
        try:
            if self.temp == 0:
                src = self.im.toImage()
                src.save('filters/temp.png', 'png')
                # read the image
                img = cv2.imread('filters/temp.png')
                self.original = img.copy()

            if self.filtre18.isChecked() == True:
                self.temp = 1
                #split the image BGR channels
                (B, G, R) = cv2.split(img)
                max = np.maximum(np.maximum(R, G), B)
                R[R < max] = 0
                G[G < max] = 0
                B[B < max] = 0
                filtered=cv2.merge([B, G, R])

                self.img = np.array(filtered, dtype=np.uint8)
                # cv2 to qimage
                self.image = QtGui.QImage(self.img.data, self.img.shape[1],
                                          self.img.shape[0], self.img.shape[1] * 3, QtGui.QImage.Format_BGR888)
                # set new image in label with qpixmap
                self.pix = QtGui.QPixmap(self.image)
                self.im = QPixmap(self.pix).scaled(self.label.size())
                self.label.setPixmap(self.im)

            else:
                self.temp = 0
                self.image = QtGui.QImage(self.original.data, self.original.shape[1],
                                          self.original.shape[0], self.original.shape[1] * 3, QtGui.QImage.Format_BGR888)

                self.pix = QtGui.QPixmap(self.image)
                self.im = QPixmap(self.pix).scaled(self.label.size())
                self.label.setPixmap(self.im)
        # if there is any error, call error function
        except:
            self.error_()

    def filtre19_(self):
        try:
            if self.temp == 0:
                src = self.im.toImage()
                src.save('filters/temp.png', 'png')
                # read the image
                img = cv2.imread('filters/temp.png')
                self.original = img.copy()

            if self.filtre19.isChecked() == True:
                self.temp = 1
                kernel = np.array([[0.0, -1.0, 0.0],
                                   [-1.0, 5.0, -1.0],
                                   [0.0, -1.0, 0.0]])

                kernel = kernel / (np.sum(kernel) if np.sum(kernel) != 0 else 1)
                fltr = cv2.filter2D(img, -1, kernel)

                self.img = np.array(fltr, dtype=np.uint8)
                # cv2 to qimage
                self.image = QtGui.QImage(self.img.data, self.img.shape[1], self.img.shape[0], self.img.shape[1] * 3, QtGui.QImage.Format_BGR888)
                # set new image in label with qpixmap
                self.pix = QtGui.QPixmap(self.image)
                self.im = QPixmap(self.pix).scaled(self.label.size())
                self.label.setPixmap(self.im)

            else:
                self.temp = 0
                self.image = QtGui.QImage(self.original.data, self.original.shape[1],
                                          self.original.shape[0], self.original.shape[1] * 3,
                                          QtGui.QImage.Format_BGR888)

                self.pix = QtGui.QPixmap(self.image)
                self.im = QPixmap(self.pix).scaled(self.label.size())
                self.label.setPixmap(self.im)
        # if there is any error, call error function
        except:
            self.error_()
    def filtre20_(self):
        try:
            if self.temp == 0:
                src = self.im.toImage()
                src.save('filters/temp.png', 'png')
                # read the image
                img = cv2.imread('filters/temp.png')
                self.original = img.copy()

            if self.filtre20.isChecked() == True:
                self.temp = 1
                #create mask
                edge = cv2.bitwise_not(cv2.Canny(img, 100, 200))
                dst = cv2.edgePreservingFilter(img, flags=2, sigma_s=64, sigma_r=0.25)
                img20 = cv2.bitwise_and(dst, dst, mask=edge)
                self.img = np.array(img20, dtype=np.uint8)
                # cv2 to qimage
                self.image = QtGui.QImage(self.img.data, self.img.shape[1], self.img.shape[0], self.img.shape[1] * 3, QtGui.QImage.Format_BGR888)
                # set new image in label with qpixmap
                self.pix = QtGui.QPixmap(self.image)
                self.im = QPixmap(self.pix).scaled(self.label.size())
                self.label.setPixmap(self.im)
            #if self.filtre20.isChecked() == False, turn back
            else:
                self.temp = 0
                self.image = QtGui.QImage(self.original.data, self.original.shape[1],
                                          self.original.shape[0], self.original.shape[1] * 3,
                                          QtGui.QImage.Format_BGR888)

                self.pix = QtGui.QPixmap(self.image)
                self.im = QPixmap(self.pix).scaled(self.label.size())
                self.label.setPixmap(self.im)
        # if there is any error, call error function
        except:
            self.error_()


app=QApplication([])
window=loadUi_example()
window.show()
app.exec_()

