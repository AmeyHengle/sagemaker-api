import requests 
import json
import os

ocrApiUrl = "http://127.0.0.1:5000/invocations"

files1 = {
    'image': ('PDF test.jpg', open('PDF test.jpg', 'rb')),
}

r = requests.post(ocrApiUrl, files = files1)
print(r.content)