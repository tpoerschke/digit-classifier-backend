import os
os.environ['KERAS_BACKEND'] = 'theano'

import numpy as np
from PIL import Image
from keras.models import load_model

from flask import Flask
from flask import request
app = Flask(__name__)

@app.route('/classify', methods=["POST"])
def classify():
    image = request.files['image']

    img_rows, img_cols = 28, 28

    image = np.array(Image.open(image).convert('L'))
    image = image.reshape(img_rows * img_cols)

    model = load_model(os.path.abspath(os.path.dirname(__file__)) + '/simple-mnist-model')

    prediction = model.predict_classes(np.array([image]))[0]

    return {
        "prediction": int(prediction)
    }