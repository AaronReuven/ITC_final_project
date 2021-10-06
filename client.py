import requests
import json
import cv2
from config import *

# prepare headers for http request
content_type = 'image/jpeg'
headers = {'content-type': content_type}

img = cv2.imread('test.jpg')
# encode image as jpeg
_, img_encoded = cv2.imencode('.jpg', img)
# send http request with image and receive response
response = requests.post(URL, data=img_encoded.tostring(), headers=headers)
# decode response
print(json.loads(response.text))