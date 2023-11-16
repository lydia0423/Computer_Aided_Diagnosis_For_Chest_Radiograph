from PyQt6 import uic
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import QApplication, QWidget, QLabel, QGraphicsScene, QGraphicsPixmapItem, QGraphicsView, QPushButton
import sys
from PIL import Image, ImageQt
import numpy as np
from tensorflow import keras
import time
import pyautogui
import cv2
import scipy
from skimage.feature.peak import peak_local_max
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt

class UI(QWidget):
    def __init__(self):
        super().__init__()
        uic.loadUi('Desktop/MainWindow.ui', self)
        

        # define widgets
        self.load_image = self.findChild(QPushButton, 'load_image')
        self.predict_image = self.findChild(QPushButton, 'predict_image')
        self.oriCXR = self.findChild(QGraphicsView, 'oriCXR')
        self.result = self.findChild(QLabel, 'result')
        
        self.load_image.clicked.connect(self.load_CXR)
        self.predict_image.clicked.connect(self.predict_CXR)

        self.model = keras.models.load_model('Desktop/mobile_net4_model.h5')
        self.labels = ['Normal', 'Pneumothorax']
        # show the app
        self.show()


    def load_CXR(self):
        self.result.setText("")
        time.sleep(5)  # Wait for 2 seconds before taking the screenshot
        im = pyautogui.screenshot()  # Take a screenshot of the entire screen
        ImageQt.ImageQt(im).save('Desktop/CXR_screenshot/CXR.png')

    # display image in graphics view
        pixmap = QPixmap('Desktop/CXR_screenshot/CXR.png')
        pixmap_item = QGraphicsPixmapItem(pixmap)
        scene = QGraphicsScene()
        scene.addItem(pixmap_item)

        self.oriCXR = QGraphicsView(scene, self)
        self.oriCXR.setGeometry(20, 100, 711, 671)
        self.oriCXR.show()

    def predict_CXR(self):
        file_path = 'Desktop/CXR_screenshot/CXR.png'
        image = Image.open(file_path).resize((224, 224))
        image_array = np.array(image)
        image_array = keras.applications.mobilenet.preprocess_input(image_array)

        prediction = self.model.predict(np.expand_dims(image_array, axis=0))
        predicted_class = np.argmax(prediction)
        print(predicted_class)
        predicted_label = self.labels[predicted_class]
        
        self.result.setText(f"Predicted class: {predicted_label}")
        print(f"Predicted class: {predicted_label}")

        if predicted_label != 'Normal':
            img = cv2.imread(file_path)
            img = Image.fromarray(img)
            img = img.resize((224,224))
            img  = np.array(img)
            img = img / 255.0
            self.plot_heatmap(img, predicted_class)


    def plot_heatmap(self, img, predicted_class):
            last_layer_weights = self.model.layers[-1].get_weights()[0]
            last_layer_weights_for_pred = last_layer_weights[:, predicted_class]
            last_conv_model = keras.models.Model(self.model.input, self.model.get_layer("out_relu").output)
            last_conv_output = last_conv_model.predict(img[np.newaxis,:,:,:])
            last_conv_output = np.squeeze(last_conv_output)
            
            h = int(img.shape[0]/last_conv_output.shape[0])
            w = int(img.shape[1]/last_conv_output.shape[1])
            upsampled_last_conv_output = scipy.ndimage.zoom(last_conv_output, (h, w, 1), order=1)
            
            heat_map = np.dot(upsampled_last_conv_output.reshape((img.shape[0]*img.shape[1], 1280)), 
                        last_layer_weights_for_pred).reshape(img.shape[0],img.shape[1])
            
            heat_map[img[:,:,0] == 0] = 0 
            
            peak_coords = peak_local_max(heat_map, num_peaks=5, threshold_rel=0.5, min_distance=10) 

            plt.figure(figsize=(12, 12))
            plt.imshow(img.astype('float32').reshape(img.shape[0],img.shape[1],3))
            plt.imshow(heat_map, cmap='jet', alpha=0.20)
            for i in range(0,peak_coords.shape[0]):
                print(i)
                y = peak_coords[i,0]
                x = peak_coords[i,1]
                plt.gca().add_patch(Rectangle((x-25, y-25), 50,50,linewidth=1,edgecolor='r',facecolor='none'))
                plt.savefig('Desktop/CXR_screenshot/CXR_heatmap.png') 

            pixmap = QPixmap('Desktop/CXR_screenshot/CXR_heatmap.png')
            pixmap_item = QGraphicsPixmapItem(pixmap)
            scene = QGraphicsScene()
            scene.addItem(pixmap_item)

            self.oriCXR = QGraphicsView(scene, self)
            self.oriCXR.setGeometry(20, 100, 711, 671)
            self.oriCXR.show() 
        
app = QApplication(sys.argv)
window = UI()
sys.exit(app.exec())