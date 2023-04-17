import requests
import json

data = {"max_tokens": 2048, "top_p": 0.2, "temperature": 0.8, "library":"fess", "messages": [{"role":"user", "content":"如何使用监控平台"}]}
headers = {
  "Content-Type": "application/json",
  "Accept": "text/event-stream"
}
url = "/chat/completions"
response = requests.post(url, headers=headers, json=data, stream=True)
# Check if the request was successful
if response.status_code == 200:
  # Extract the generated text from the response
  response_text = ""
  previous_response = ''
  for line in response.iter_lines():
    if line:
      data = json.loads(line.decode("utf-8"))
      current_response = data['response']
      if current_response == "[DONE]":
        break
      new_characters = current_response[len(previous_response):]
      print(new_characters, end='', flush=True)
      previous_response = current_response
else:
    print(f"Request failed with status code {response.status_code}")