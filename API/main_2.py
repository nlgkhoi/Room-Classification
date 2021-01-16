from flask import Flask
from flask_restful import Api, Resource, reqparse, abort
# from predict import init_model, label
import sys
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
from urllib.request import urlretrieve
import PIL
import cv2
app = Flask(__name__)
api = Api(app)

# dimensions of our images
image_shape = 240
# CATEGORIES = ['bathroom', 'bedroom', 'dining room', 'exterior', 'interior', 'kitchen', 'living room']

# load the model we saved
model = load_model("Trained_model")

print("Model initialized successfully!")

img_put_args = reqparse.RequestParser()
img_put_args.add_argument("url", type=str, help="URL of the image is required", required=True)


class Predict(Resource):
    def get(self):
        print('Predicting...')
        CATEGORIES = ['bathroom', 'bedroom', 'dining room', 'exterior', 'interior', 'kitchen', 'living room']
        args = img_put_args.parse_args()
        url = args["url"]
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
        return {"label": pred_name}
    
api.add_resource(Predict, "/predict")

if __name__ == "__main__":
    app.run(debug=True)
    
