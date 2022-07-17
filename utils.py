import os
import tensorflow as tf
import pathlib
import pandas as pd
import numpy as np
from PIL import Image

from keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
import matplotlib.pyplot as plt

def get_models():
    return os.listdir('models')

def get_bird_types():
    return os.listdir('data\\test')

def get_bird_pics(bird_type):
    return os.listdir(f'data\\test\\{bird_type}')

class BirdWatcher():

    def __init__(self, model_name):
        self.model = self.read_model(model_name)
        self.ref = self.get_output_classes()

    def read_model(self, model_name):
        return tf.keras.models.load_model('D:\\GitHub\\Bird-Classifier\\models\\' + model_name, compile = False)

    def get_output_classes(self):
        with open('output_classes.txt', 'r') as inputFile:
            dataset = inputFile.readlines()

        for i in range(len(dataset)):
            dataset[i] = dataset[i].strip('\n')

        return dataset

    def classify(self, image_location):

        #load the image
        my_image = load_img(image_location, target_size=(224, 224))

        #preprocess the image
        my_image = img_to_array(my_image)
        my_image = my_image.reshape((1, my_image.shape[0], my_image.shape[1], my_image.shape[2]))
        my_image = preprocess_input(my_image)

        #make the prediction
        prediction = self.ref[np.argmax(self.model.predict(my_image))]

        return prediction

    def my_summary(self):
        return self.model.summary()

if __name__ == '__main__':
    bw = BirdWatcher('inception')

    print(get_bird_pics('CANARY'))

    print(bw.classify('.\\data\\test\\CANARY\\1.jpg'))