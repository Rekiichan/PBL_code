from werkzeug.utils import secure_filename
from flask import Flask, request, redirect, url_for
from flask_s3 import FlaskS3
import boto3, botocore
from flask_cors import CORS, cross_origin
from model import predict
import cv2 as cv
import numpy as np
# Khởi tạo Flask Server Backend
app = Flask(__name__)
app.config['FLASKS3_BUCKET_NAME'] = 'pioneerstorage'
s3 = FlaskS3(app)
# Apply Flask CORS
CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'
app.config['UPLOAD_FOLDER'] = ''
app.config['S3_BUCKET'] = "pioneerstorage"
app.config['S3_KEY'] = "AKIA54U57CQ6AKHP2AX6"
app.config['S3_SECRET'] = "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"
app.config['S3_LOCATION'] = 'http://{}.s3.amazonaws.com/'.format("pioneerstorage")

s3 = boto3.client(
   "s3",
   aws_access_key_id=app.config['S3_KEY'],
   aws_secret_access_key=app.config['S3_SECRET']
)

@app.route('/', methods=['POST', 'GET'])
@cross_origin(origin='*')
def home():
    if request.method == 'POST':
        image = request.files['file'].read()
        file_bytes = np.fromstring(image, np.uint8)
        img = cv.imdecode(file_bytes, cv.IMREAD_UNCHANGED)  
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        res = predict(img)
        return res
    if request.method == 'GET':
        return 'get'
    return ''

# @app.route("/upload", methods=["POST"])
# @cross_origin(origin='*')
# def upload_file():
#     # if "url" not in request.files:
#     #     return "No user_file key in request.files"

#     file = request.files["url"]

#     if file.filename == "":
#         return "Please select a file"

#     if file:
#         file.filename = secure_filename(file.filename)
#         output = send_to_s3(file, app.config['S3_BUCKET'])
#         return str(output)

#     else:
#         return 'home'

# Start Backend
if __name__ == '__main__':
    # app.run()
    app.run(host='0.0.0.0',port=4000)

