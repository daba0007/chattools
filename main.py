# -*- coding: UTF-8 -*-
from utils.main import app

from controller.chat import chat_ops
import os

@app.route('/ping')
def pong():  # put application's code here
    #app.logger.info('pong')
    return 'pong'
app.register_blueprint(chat_ops)

if __name__ == '__main__':
    port = os.environ.get("PORT", 17860)
    app.run(host="0.0.0.0", port=int(port))
