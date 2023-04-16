import requests
import json

data = {"max_tokens": 2048, "top_p": 0.2, "temperature": 0.8, "mix":True, "messages": [{"role":"user", "content":"请问你知道 Inhouse 吗"}]}
headers = {
  "Content-Type": "application/json",
  "Accept": "text/event-stream"
}
url = "https://xsre.workflow.sz.shopee.io/chat/completions"
response = requests.post(url, headers=headers, json=data, stream=True)
# Check if the request was successful
if response.status_code == 200:
  import re
  # Extract the generated text from the response
  for line in response.iter_lines():
    text = json.load(line.decode("utf-8"))
    print(text["response"])
    """ text = line.decode("utf-8")
    json_strings = re.findall(r"\{[^}]*\}", text)
    for json_string in json_strings:
      data = json.loads(json_string)
      response_text = data["response"]
      print(response_text) """
else:
    print(f"Request failed with status code {response.status_code}")