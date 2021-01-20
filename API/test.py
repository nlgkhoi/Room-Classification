import requests
import string
import json
BASE = "http://127.0.0.1:5000/"

input_urls = input("URLs (seperated by comma): ")
urls = input_urls.split(",")

response = requests.get(BASE + "predict", json=urls)
# print(response.json())
print(json.dumps(response.json(), indent=4))