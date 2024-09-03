from __future__ import division, print_function
import sys
import os
import numpy as np
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import pickle

app = Flask(__name__)

MODEL_PATH = 'models/modelAdam.h5'
model = load_model(MODEL_PATH, compile=False)

# model = load_model(MODEL_PATH, compile=False)

print('Model loaded. Check http://127.0.0.1:5000/')


def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224), color_mode="grayscale")
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    
    # Make predictions using the model
    predictions = model.predict(x)
    return predictions[0][0]

@app.route('/', methods=['GET'])
def index():
    return render_template('index1.html')
@app.route('/predict', methods=['POST'])
def upload():
    if request.method == 'POST':
        # Get the file from the request
        f = request.files['file']

        # Save the file to the uploads folder
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make the prediction using the model
        preds = model_predict(file_path, model)

        # Interpret the predictions
        if preds > 0.5:
            result = "The image is classified as lumpy skin."
        else:
            result = "The image is classified as normal skin."

        # Return the result in a JSON response
        return jsonify({"result": result})

    return result

if __name__ == '__main__':
    app.run(debug=True)