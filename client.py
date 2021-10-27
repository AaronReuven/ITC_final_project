import json

import cv2
import matplotlib.pyplot as plt
import pandas as pd
import requests
from PIL import Image

from config import *


def demo(test_pics):
    keep_going = True
    content_type = 'image/jpeg'
    headers = {'content-type': content_type}
    print('Our Demo')
    while keep_going:
        random_pic = test_pics.sample(1)
        label = random_pic[TARGET_COLUMN_NAME]
        data = random_pic.drop(columns=TARGET_COLUMN_NAME)
        im = Image.fromarray(data.to_numpy().reshape(28, 28)).convert("L")
        im.save('tmp.jpeg')
        img = cv2.imread('tmp.jpeg')
        _, img_encoded = cv2.imencode('.jpeg', img)
        response = requests.post(URL, data=img_encoded.tostring(), headers=headers)
        results = json.loads(response.text)
        plt.imshow(data.to_numpy().reshape(28, 28), cmap='gray')
        plt.title(f'True label:{LETTERS[label.values[0]]}, predicted label:{results["result"]}')
        plt.show()
        keep_going = False if input(
            'Would you like to keep testing our model?\nPress Enter otherwise enter any character: ').strip() != '' else True
        print(f'True label:{LETTERS[label.values[0]]}, predicted label:{results["result"]}')
        plt.close()


def load_pics(FILE_NAME):
    return pd.read_csv(FILE_NAME, index_col=0)


def main():
    test_pics = load_pics(TEST_FILE_NAME)
    demo(test_pics)


if __name__ == '__main__':
    main()
