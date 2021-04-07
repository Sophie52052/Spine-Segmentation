# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'C:\Users\Sophie\Desktop\test.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets
import sys
from PyQt5.QtWidgets import QFileDialog
import numpy as np
import glob
import cv2
import tensorflow as tf
from tensorflow import keras
import random
import matplotlib.pyplot as plt
from pylab import *
from scipy.ndimage import label, generate_binary_structure


class Ui_MainWindow(object):
    
    def __init__(self):
        self.unet_model = self.unet5()
        self.test_img = None
        self.test_mask = None
        self.weight = None
        
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1105, 651)
        MainWindow.setStyleSheet("background-color: rgb(255, 255, 255);\n"
"border-color: rgb(0, 0, 127);")
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(80, 30, 151, 41))
        self.pushButton.setStyleSheet("background-color: rgb(0, 0, 127);\n"
"color: rgb(255, 255, 255);")
        self.pushButton.setObjectName("pushButton")
        self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_2.setGeometry(QtCore.QRect(350, 30, 151, 41))
        self.pushButton_2.setStyleSheet("background-color: rgb(0, 0, 127);\n"
"color: rgb(255, 255, 255);")
        self.pushButton_2.setObjectName("pushButton_2")
        self.pushButton_3 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_3.setGeometry(QtCore.QRect(620, 30, 151, 41))
        self.pushButton_3.setStyleSheet("border-color: rgb(0, 0, 127);\n"
"background-color: rgb(0, 0, 127);\n"
"color: rgb(255, 255, 255);")
        self.pushButton_3.setObjectName("pushButton_3")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(40, 80, 231, 481))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Ignored)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_2.sizePolicy().hasHeightForWidth())
        self.label_2.setSizePolicy(sizePolicy)
        self.label_2.setText("")
        self.label_2.setPixmap(QtGui.QPixmap(""))
        self.label_2.setScaledContents(True)
        self.label_2.setObjectName("label_2")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(580, 80, 231, 481))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Ignored)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_3.sizePolicy().hasHeightForWidth())
        self.label_3.setSizePolicy(sizePolicy)
        self.label_3.setText("")
        self.label_3.setPixmap(QtGui.QPixmap(""))#result
        self.label_3.setScaledContents(True)
        self.label_3.setObjectName("label_3")
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setGeometry(QtCore.QRect(310, 80, 231, 481))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Ignored)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_4.sizePolicy().hasHeightForWidth())
        self.label_4.setSizePolicy(sizePolicy)
        self.label_4.setText("")
        self.label_4.setPixmap(QtGui.QPixmap(""))
        self.label_4.setScaledContents(True)
        self.label_4.setObjectName("label_4")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(870, 70, 71, 41))
        self.label.setObjectName("label")
        self.label_5 = QtWidgets.QLabel(self.centralwidget)
        self.label_5.setGeometry(QtCore.QRect(870, 540, 111, 18))
        self.label_5.setObjectName("label_5")
        self.label_6 = QtWidgets.QLabel(self.centralwidget)
        self.label_6.setGeometry(QtCore.QRect(140, 570, 70, 18))
        self.label_6.setObjectName("label_6")
        self.label_7 = QtWidgets.QLabel(self.centralwidget)
        self.label_7.setGeometry(QtCore.QRect(390, 570, 101, 21))
        self.label_7.setObjectName("label_7")
        self.label_8 = QtWidgets.QLabel(self.centralwidget)
        self.label_8.setGeometry(QtCore.QRect(660, 570, 70, 18))
        self.label_8.setObjectName("label_8")
        self.label_9 = QtWidgets.QLabel(self.centralwidget)
        self.label_9.setGeometry(QtCore.QRect(870, 110, 101, 421))
        self.label_9.setObjectName("label_9")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1105, 29))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.pushButton.setText(_translate("MainWindow", "Select Image"))
        self.pushButton.clicked.connect(self.function1)
        self.pushButton_2.setText(_translate("MainWindow", "Select Model"))
        self.pushButton_2.clicked.connect(self.function2)
        self.pushButton_3.setText(_translate("MainWindow", "Run"))
        self.pushButton_3.clicked.connect(self.function3)
        self.label.setText(_translate("MainWindow", "DC"))
        self.label_5.setText(_translate("MainWindow", ""))
        self.label_6.setText(_translate("MainWindow", "Sourse"))
        self.label_7.setText(_translate("MainWindow", "Ground truth"))
        self.label_8.setText(_translate("MainWindow", "Result"))
        self.label_9.setText(_translate("MainWindow", ""))
        
        
    def unet3(self):
      IMG_HEIGHT = 1200
      IMG_WIDTH = 512
      IMG_CHANNELS = 1
      # Build U-Net model
      inputs = tf.keras.layers.Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
      #s = tf.keras.layers.Lambda(lambda x: x / 255)(inputs)
      
      c1 = tf.keras.layers.Conv2D(16, (3, 3), activation=tf.keras.activations.relu,  
                                  padding='same')(inputs)
      #c1 = tf.keras.layers.Dropout(0.1)(c1)
      c1 = tf.keras.layers.Conv2D(16, (3, 3), activation=tf.keras.activations.relu,  
                                  padding='same')(c1)
      p1 = tf.keras.layers.MaxPooling2D((2, 2))(c1)
      
      c2 = tf.keras.layers.Conv2D(32, (3, 3), activation=tf.keras.activations.relu,  
                                  padding='same')(p1)
      #c2 = tf.keras.layers.Dropout(0.1)(c2)
      c2 = tf.keras.layers.Conv2D(32, (3, 3), activation=tf.keras.activations.relu,  
                                  padding='same')(c2)
      p2 = tf.keras.layers.MaxPooling2D((2, 2))(c2)
      
      c3 = tf.keras.layers.Conv2D(64, (3, 3), activation=tf.keras.activations.relu,  
                                  padding='same')(p2)
      #c3 = tf.keras.layers.Dropout(0.2)(c3)
      c3 = tf.keras.layers.Conv2D(64, (3, 3), activation=tf.keras.activations.relu,  
                                  padding='same')(c3)
      p3 = tf.keras.layers.MaxPooling2D((2, 2))(c3)
      
      c4 = tf.keras.layers.Conv2D(128, (3, 3), activation=tf.keras.activations.relu,  
                                  padding='same')(p3)
      #c4 = tf.keras.layers.Dropout(0.2)(c4)
      c4 = tf.keras.layers.Conv2D(128, (3, 3), activation=tf.keras.activations.relu,  
                                  padding='same')(c4)
      p4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(c4)
      
      c5 = tf.keras.layers.Conv2D(256, (3, 3), activation=tf.keras.activations.relu,  
                                  padding='same')(p4)
      #c5 = tf.keras.layers.Dropout(0.3)(c5)
      c5 = tf.keras.layers.Conv2D(256, (3, 3), activation=tf.keras.activations.relu,  
                                  padding='same')(c5)
      
      u6 = tf.keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
      u6 = tf.keras.layers.concatenate([u6, c4])
      c6 = tf.keras.layers.Conv2D(128, (3, 3), activation=tf.keras.activations.relu,  
                                  padding='same')(u6)
      #c6 = tf.keras.layers.Dropout(0.2)(c6)
      c6 = tf.keras.layers.Conv2D(128, (3, 3), activation=tf.keras.activations.relu,  
                                  padding='same')(c6)
      
      u7 = tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
      u7 = tf.keras.layers.concatenate([u7, c3])
      c7 = tf.keras.layers.Conv2D(64, (3, 3), activation=tf.keras.activations.relu,  
                                  padding='same')(u7)
      #c7 = tf.keras.layers.Dropout(0.2)(c7)
      c7 = tf.keras.layers.Conv2D(64, (3, 3), activation=tf.keras.activations.relu,  
                                  padding='same')(c7)
      
      u8 = tf.keras.layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
      u8 = tf.keras.layers.concatenate([u8, c2])
      c8 = tf.keras.layers.Conv2D(32, (3, 3), activation=tf.keras.activations.relu,  
                                  padding='same')(u8)
      #c8 = tf.keras.layers.Dropout(0.1)(c8)
      c8 = tf.keras.layers.Conv2D(32, (3, 3), activation=tf.keras.activations.relu,  
                                  padding='same')(c8)
      
      u9 = tf.keras.layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
      u9 = tf.keras.layers.concatenate([u9, c1], axis=3)
      c9 = tf.keras.layers.Conv2D(16, (3, 3), activation=tf.keras.activations.relu,padding='same')(u9)
      #c9 = tf.keras.layers.Dropout(0.1)(c9)
      c9 = tf.keras.layers.Conv2D(16, (3, 3), activation=tf.keras.activations.relu,padding='same')(c9)
      
      outputs = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(c9)
      
      model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
      #model.compile(optimizer='adam', loss=weighted_cross_entropy, metrics=[dice_coef1])
      #model.summary()
      #results = model.fit(x = train_image, y = train_mask, validation_split=0.1, batch_size=4, epochs=300,)
      return model
  
    def unet5(self):
        IMG_HEIGHT = 1200
        IMG_WIDTH = 512
        IMG_CHANNELS = 1
        # Build U-Net model
        inputs = tf.keras.layers.Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
        #inputs = Input(input_size)
        conv1 = tf.keras.layers.Conv2D(32, 3, activation = 'relu', padding = 'same'  )(inputs)
        conv1 = tf.keras.layers.Conv2D(32, 3, activation = 'relu', padding = 'same'   )(conv1)
        pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)
        conv2 = tf.keras.layers.Conv2D(64, 3, activation = 'relu', padding = 'same'  ) (pool1)
        conv2 = tf.keras.layers.Conv2D(64, 3, activation = 'relu', padding = 'same'  ) (conv2)
        pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)
        conv3 = tf.keras.layers.Conv2D(128, 3, activation = 'relu', padding = 'same' )  (pool2)
        conv3 = tf.keras.layers.Conv2D(128, 3, activation = 'relu', padding = 'same' ) (conv3)
        pool3 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv3)
        conv4 = tf.keras.layers.Conv2D(256, 3, activation = 'relu', padding = 'same' )  (pool3)
        conv4 = tf.keras.layers.Conv2D(256, 3, activation = 'relu', padding = 'same' )  (conv4)
        #drop4 = tf.keras.layers.Dropout(0.5)(conv4)
        #drop4 = conv4
        pool4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv4)
    
        conv5 = tf.keras.layers.Conv2D(512, 3, activation = 'relu', padding = 'same'  ) (pool4)
        conv5 = tf.keras.layers.Conv2D(512, 3, activation = 'relu', padding = 'same' )  (conv5)
        #drop5=conv5
        #drop5 = tf.keras.layers.Dropout(0.5)(conv5)
    
        up6 = tf.keras.layers.UpSampling2D(size=(2,2) )(conv5)
        up6 = tf.keras.layers.Conv2D(256, 2, activation = 'relu', padding = 'same' )  (up6)
        merge6 = tf.keras.layers.concatenate([conv4,up6], axis = 3)
        conv6 = tf.keras.layers.Conv2D(256, 3, activation = 'relu', padding = 'same' ) (merge6)
        conv6 = tf.keras.layers.Conv2D(256, 3, activation = 'relu', padding = 'same' ) (conv6)
    
        up7 = tf.keras.layers.UpSampling2D(size=(2,2) )(conv6)
        up7 = tf.keras.layers.Conv2D(128, 2, activation = 'relu', padding = 'same'  ) (up7)
        merge7 = tf.keras.layers.concatenate([conv3,up7], axis = 3)
        conv7 = tf.keras.layers.Conv2D(128, 3, activation = 'relu', padding = 'same'  ) (merge7)
        conv7 = tf.keras.layers.Conv2D(128, 3, activation = 'relu', padding = 'same'  ) (conv7)
    
        up8 = tf.keras.layers.UpSampling2D(size=(2,2) )(conv7)
        up8 = tf.keras.layers.Conv2D(64, 2, activation = 'relu', padding = 'same'  ) (up8)
        merge8 = tf.keras.layers.concatenate([conv2,up8], axis = 3)
        conv8 = tf.keras.layers.Conv2D(64, 3, activation = 'relu', padding = 'same'  ) (merge8)
        conv8 = tf.keras.layers.Conv2D(64, 3, activation = 'relu', padding = 'same'  ) (conv8)
    
        up9 = tf.keras.layers.UpSampling2D(size=(2,2) )(conv8)
        up9 = tf.keras.layers.Conv2D(32, 2, activation = 'relu', padding = 'same' )  (up9)
        merge9 = tf.keras.layers.concatenate([conv1,up9], axis = 3)
        conv9 = tf.keras.layers.Conv2D(32, 3, activation = 'relu', padding = 'same'  ) (merge9)
        conv9 = tf.keras.layers.Conv2D(32, 3, activation = 'relu', padding = 'same'  ) (conv9)
        conv9 = tf.keras.layers.Conv2D(16, 3, activation = 'relu', padding = 'same')   (conv9)
        conv10 = tf.keras.layers.Conv2D(1, 1, activation = 'sigmoid')(conv9)
    
        model = tf.keras.Model(inputs = inputs, outputs = conv10)
        #model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
        #adam=tf.keras.optimizers.Adam(learning_rate=1e-4, beta_1=0.9, beta_2=0.999, amsgrad=False)
        #model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4), loss = weighted_cross_entropy, metrics = [dice_coef])
        
        #model.summary()
        return model
    
    

  
    
    def function1(self):
        fname, _ = QFileDialog.getOpenFileName(None, 'Open file', 
         'c:\\',"Image files (*.jpg *.gif *.png)")
        self.label_2.setPixmap(QtGui.QPixmap(fname))
        
        test_img = []
        for img in glob.glob(fname):
            n= cv2.imread(img,0)
            #外層補0讓圖片加大
            n = cv2.copyMakeBorder(n,0,0,6,6, cv2.BORDER_CONSTANT,value=[0,0,0])
            test_img.append(n)
            
        fname1, _ = QFileDialog.getOpenFileName(None, 'Open file', 
         'c:\\',"Image files (*.jpg *.gif *.png)")
        self.label_4.setPixmap(QtGui.QPixmap(fname1))
        #print(fname1)
        
        test_mask = []
        for img in glob.glob(fname1):
            n= cv2.imread(img,0)
            #外層補0讓圖片加大
            n = cv2.copyMakeBorder(n,0,0,6,6, cv2.BORDER_CONSTANT,value=[0,0,0])
            test_mask.append(n)
            
        self.test_img = test_img
        self.test_mask = test_mask
        print(test_img)
        
    def function2(self):
        weight, _ = QFileDialog.getOpenFileName(None, 'Open file', 
         'c:\\',"files (*.h5 *.gif *.png)")
        self.unet_model.load_weights(weight)
        #predict = self.unet_model.predict(self.test_img[0].reshape(1,1200,512,1))
        #plt.imshow(predict.reshape(1200,512),cmap='gray')
        #plt.show()
        self.weight = weight
        
    def function3(self):
        def dice_2img(img, img2):
            if img.shape != img2.shape:
                raise ValueError("Shape mismatch: img and img2 must have to be of the same shape.")
            else:
                intersection = np.logical_and(img, img2)
                value = (2. * intersection.sum())  / (img.sum() + img2.sum())
            return value
        def dice_cof(label_gt,predict):
          label_gt_abs = len(np.where(label_gt >0 )[0])
          predict_abs = len(np.where(predict > 0)[0])
          intersection = np.array(cv2.bitwise_and(label_gt,predict),dtype=np.uint8)
          intersection_abs = len(np.where(intersection > 0)[0])
          #print(label_gt_abs,predict_abs,intersection_abs)
          dice = (2*intersection_abs) / (label_gt_abs + predict_abs)
          return dice
      
        def per_aspine(test_mask):
          mask_list = []
          #取出塊
          s = generate_binary_structure(2,2)
          labeled_array, num_features = label(test_mask, structure=s)
          #看有幾塊
          for area_number in range(0,num_features):
            one = np.where(labeled_array == area_number+1,1,0)
            mask_list.append(one)
          return mask_list
      
        def per_dice(test_mask,predict):
          
          dice_list = []
          img_pre = np.array(predict.reshape(1200,512)*255, dtype = np.uint8)
          test_mask = np.array(test_mask.reshape(1200,512)*255, dtype = np.uint8)
          per_mask_total = per_aspine(test_mask)
          kernel = np.ones((3,3),np.uint8)
          for per_mask in per_mask_total:
            one_bone_gt = np.array(per_mask.reshape(1200,512)*255, dtype = np.uint8)
            dilate_mask = cv2.dilate(one_bone_gt,kernel,iterations = 4)
            one_bone_pre = cv2.bitwise_and(img_pre,dilate_mask)
         
            dice_list.append(round(dice_cof(one_bone_gt,one_bone_pre),3))
            
          return dice_list
                
        def overlap(image,predict):
              img_ct = np.array(image.reshape(1200,512)*255, dtype = np.uint8)
              backtorgb = cv2.cvtColor(img_ct,cv2.COLOR_GRAY2RGB)
            
              img_perdit = np.array(predict.reshape(1200,512)*255, dtype = np.uint8)
              ret, binary = cv2.threshold(img_perdit.astype(np.uint8),0,255,cv2.THRESH_BINARY)
              contours, hierarchy = cv2.findContours(binary,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
              overlap = np.zeros((1200,512,3), dtype = np.uint8)
              cv2.drawContours(overlap,contours,-1,(255,0,0),3)
              
              overlapping = cv2.addWeighted(backtorgb, 0.8, overlap, 0.2, 0)
              plt.imshow(overlapping)
              plt.imsave('test1.png',overlapping)
              return overlapping
        
        
        #unet_model = self.unet3()
        self.unet_model.load_weights(self.weight)
        predict = self.unet_model.predict(self.test_img[0].reshape(1,1200,512,1)/255)
        plt.imshow(predict.reshape(1200,512),cmap='gray')
        #plt.show()
        overlap(self.test_img[0].reshape(1200,512)/255,predict.reshape(1200,512))
        #plt.imsave('test.png',predict.reshape(1200,512),cmap='gray')
        self.label_3.setPixmap(QtGui.QPixmap('test1.png'))
        
        threshold=0.7
        predict[predict<=threshold]=0
        predict[predict>threshold]=1
        dice=dice_2img(predict.reshape(1200,512),self.test_mask[0].reshape(1200,512)/255)
        #self.label_5.setText(_translate("MainWindow", dice))
        print(dice)
        dice=round(dice,3)
        avg_dice=str(dice)
        avg_dice_string='Avg : '+avg_dice
        
        self.label_5.setText(avg_dice_string)
        A=per_dice(self.test_mask[0],predict)
        #print(A)
        
        DC_string = ''
        for i in range(0,len(A)):
            DC_string += 'V%d : %0.3f\n'%(i, A[i])
        self.label_9.setText(DC_string)
        
if __name__ == '__main__':  
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow) 
    MainWindow.show()
    sys.exit(app.exec_())

