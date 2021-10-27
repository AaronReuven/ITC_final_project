import os

import cv2
import flask
import numpy as np
import tensorflow as tf
from flask import Flask
from tensorflow.keras.models import load_model

from config import *

app = Flask(__name__)


@app.route('/predict_one', methods=['POST'])
def predict_bulk():
    """
    runs the prediction of the model on encoded picture
    :return: json of predictions
    """
    if 'model' not in globals():
        model = read_model(MODEL_FILE)
    r = flask.request.get_data()
    # convert string of image data to uint8
    # print(r)
    nparr = np.fromstring(r, np.uint8)
    # print(nparr)
    # decode image
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (28, 28))
    results = model.predict(np.reshape(img, (1, *img.shape, 1)))
    letter_index = results.argmax()
    print(results)
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
    # physcial_devices = tf.config.list_physical_devices('GPU')
    # tf.config.experimental.set_memory_growth(physcial_devices[0], True)
    # model = main()
    # app.run()
    model = main()
    port = os.environ.get('PORT')
    if port:
        # 'PORT' variable exists - running on Heroku, listen on external IP and on given by Heroku port
        app.run(host='0.0.0.0', port=int(port))
    else:
        # 'PORT' variable doesn't exist, running not on Heroku, presumabely running locally, run with default
        #   values for Flask (listening only on localhost on default Flask port)
        app.run()
    # app.run(host='0.0.0.0')
