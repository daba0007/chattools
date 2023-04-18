""" import requests
import json

data = {"max_tokens": 2048, "top_p": 0.2, "temperature": 0.8, "library": "fess",
        "messages": [{"role": "user", "content": "你好"}]}
headers = {
    "Content-Type": "application/json",
    "Accept": "text/event-stream"
}
url = "http://127.0.0.1:17860/chat/completions"
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
 """
 
from __future__ import print_function

import os.path

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# If modifying these scopes, delete the file token.json.
SCOPES = ['https://www.googleapis.com/auth/drive.metadata.readonly']


def main():
    """Shows basic usage of the Drive v3 API.
    Prints the names and ids of the first 10 files the user has access to.
    """
    creds = None
    # The file token.json stores the user's access and refresh tokens, and is
    # created automatically when the authorization flow completes for the first
    # time.
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                'credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        # Save the credentials for the next run
        with open('token.json', 'w') as token:
            token.write(creds.to_json())

    try:
        service = build('drive', 'v3', credentials=creds)

        # Call the Drive v3 API
        results = service.files().list(
            pageSize=10, fields="nextPageToken, files(id, name)").execute()
        items = results.get('files', [])

        if not items:
            print('No files found.')
            return
        print('Files:')
        for item in items:
            print(u'{0} ({1})'.format(item['name'], item['id']))
    except HttpError as error:
        # TODO(developer) - Handle errors from drive API.
        print(f'An error occurred: {error}')


if __name__ == '__main__':
    main()