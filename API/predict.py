from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import sys
from urllib.request import urlretrieve
import PIL
import cv2

# dimensions of our images
image_shape = 240
# CATEGORIES = ['bathroom', 'bedroom', 'dining room', 'exterior', 'interior', 'kitchen', 'living room']

# load the model we saved
model = load_model("Trained_model")

# path_to_image = sys.argv[1]
print('Predicting...')
CATEGORIES = ['bathroom', 'bedroom', 'dining room', 'exterior', 'interior', 'kitchen', 'living room']
url = "https://i.pinimg.com/originals/22/63/3c/22633cd4dc4b98fe248d224475d54b88.jpg"
urlretrieve(url, "resources/test.jpg")
path_to_image = "resources/test.jpg"
# predicting images
img = image.load_img(path_to_image, target_size=(image_shape, image_shape))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)/255

images = np.vstack([x])
classes = model.predict(images)
pred_name = CATEGORIES[np.argmax(classes)]

print(pred_name)

# while True:
#     path_to_image = input('image_path: ')

#     try:
#         # predicting images
#         img = image.load_img(path_to_image, target_size=(image_shape, image_shape))
#         x = image.img_to_array(img)
#         x = np.expand_dims(x, axis=0)/255

#         images = np.vstack([x])
#         classes = model.predict(images)
#         pred_name = CATEGORIES[np.argmax(classes)]

#         print(pred_name)
#     except:
#         pass