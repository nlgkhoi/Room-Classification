from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import sys
import urllib.request
import tensorflow as tf 
import os
import time

# Configure session
config = tf.compat.v1.ConfigProto()

config.gpu_options.allow_growth=True

sess = tf.compat.v1.Session(config=config)


image_shape = 240

model_path = "Trained_model"
CATEGORIES = ['bathroom', 'bedroom', 'dining room', 'exterior', 'interior', 'kitchen', 'living room']
# load the model we saved
model = load_model(model_path)

def download_image_ipg(url, file_path, file_name):
    fullpath=file_path+file_name+".png"
    urllib.request.urlretrieve(url,fullpath)

def download_images(urls, file_path):
    for i, url in enumerate(urls):
        try:
            download_image_ipg(url, file_path, str(i))
        except:
            pass
    
def batch_predict(urls):
    # print('Loading images...')
    download_images(urls, 'resources/')
    # print('Predicting...')
    images = []
    for file_name in os.listdir('resources/'):
        img = image.load_img('resources/' + file_name, target_size=(image_shape, image_shape))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0) / 255
        images.append(x)
    # print(images)
    predictions = model.predict(np.vstack(images))
    labels = [CATEGORIES[np.argmax(x)] for x in predictions]
    
    # Delete temporary files
    # print('Deleting temporary files')
    file_names = os.listdir('resources/')
    for file_name in file_names:
        os.remove("resources/"+file_name)
    
    # print('Predict successfully')
    return labels