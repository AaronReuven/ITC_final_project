import flask

from config import *
import numpy as np
from flask import Flask, request
import tensorflow as tf
from tensorflow.keras.models import load_model
import os
import cv2
import json

app = Flask(__name__)


@app.route('/predict_one', methods=['POST'])
def predict_bulk():
    """
    runs the predictions on the model of dataframe passed as json
    :return: json of predictions
    """
    r = flask.request.get_data()
    # convert string of image data to uint8
    nparr = np.fromstring(r, np.uint8)
    # decode image
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (28, 28))
    results = model.predict(np.reshape(img, (1, *img.shape, 1)))
    letter_index = results.argmax()
    result = {RESULT_JSON_TAG: LETTERS[letter_index]}
    return flask.jsonify(result)


# @app.route('/predict_churn')
# def predict():
#     """
#     predicts a single sample from the parameters of the calling route
#     :return: string of prediction
#     """
#     return str(
#         model.predict(np.array([request.args.get(col) for col in columns]).astype(dtype=float).reshape(1, -1))[0])


def read_model(file):
    """
    reads the model from file
    :param file: file to read the model from
    :return: trained model from file
    """
    if os.path.exists(file):
        return load_model(file)
    else:
        raise FileNotFoundError("Can't find model.")


def main():
    model = read_model(MODEL_FILE)
    return model


if __name__ == '__main__':
    physcial_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physcial_devices[0], True)
    model = main()
    app.run()
    # app.run(host='0.0.0.0')
