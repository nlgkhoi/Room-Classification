import requests

BASE = "http://127.0.0.1:5000/"

# response = requests.put(BASE + "video/1", {"likes": 10, "name": "tim", "views": 12})
# print(response.json()["name"])
# input()
# response = requests.get(BASE + "video/1")
# print(response.json())
# input()
# response = requests.get(BASE + "video/2")
# print(response.json())


input_urls = input("URLs (seperated by comma): ")
URLs = input_urls.split(',')
for URL in URLs:
    response = requests.get(BASE + "predict/", {"url": URL})
    print(response.json()["label"])

URL = "https://i.pinimg.com/originals/22/63/3c/22633cd4dc4b98fe248d224475d54b88.jpg"
response = requests.get(BASE + "predict", {"url": URL})
print(response.json()["label"])