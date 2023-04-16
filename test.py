import requests
import json

data = {"max_tokens": 2048, "top_p": 0.2, "temperature": 0.8, "mix":True, "messages": [{"role":"user", "content":"请问你知道 Inhouse 吗"}]}
headers = {
  "Content-Type": "application/json",
}
url = "https://xsre.workflow.sz.shopee.io/chat/completions"
response = requests.post(url, headers=headers, json=data, stream=True)
# Check if the request was successful
if response.status_code == 200:
  # Extract the generated text from the response
  for line in response.iter_lines():
        print(line)
        data = json.loads(line.decode("utf-8"))
        response_text = data["response"]
        print(response_text)
else:
    print(f"Request failed with status code {response.status_code}")