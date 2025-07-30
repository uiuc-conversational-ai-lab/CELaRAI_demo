# import json

# import requests

# url = "https://celarai-demo.onrender.com/process"

# # Example data
# files = {'files': open('backend/scratch.txt', 'rb')}  # Replace with your file path
# config = {
#     'questionTypes': 'multiple_choice',
#     'difficulty': 'medium',
#     'customInstructions': 'Generate 5 questions'
# }
# data = {'config': json.dumps(config)}

# response = requests.post(url, files=files, data=data)

# print("Status Code:", response.status_code)
# print("Response:", response.text)

import os

print(os.path.dirname(os.path.abspath(__file__)))