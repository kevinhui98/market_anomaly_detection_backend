import requests
import json

# URL of your deployed model
url = ""

import json

with open('data.json') as f:
    df = json.load(f)

response = requests.post(url,json=df)

# Check if the request was successful
if response.status_code ==200:
    # Parse the JSON response
    result = response.json()
    print(result)
else:
    print("Error: " , response.status_code,response.text)