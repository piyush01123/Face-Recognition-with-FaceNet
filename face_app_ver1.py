from keras import backend as K
import time
from multiprocessing.dummy import Pool
K.set_image_data_format('channels_first')
import cv2
import os
import glob
import numpy as np
from numpy import genfromtxt
import tensorflow as tf
from fr_utils import *
from inception_blocks_v2 import *
# import win32com.client as wincl
import pickle as pkl

PADDING = 50
ready_to_detect_identity = True
FRmodel = faceRecoModel(input_shape=(3, 96, 96))

def who_is_it(image, database, model):

    encoding = img_path_to_encoding(image, model)
    
    min_dist = 100
    identity = None
    
    # Loop over the database dictionary's names and encodings.
    for (name, db_enc) in database.items():
        
        # Compute L2 distance between the target "encoding" and the current "emb" from the database.
        dist = np.linalg.norm(db_enc - encoding)

#         print('distance for %s is %s' %(name, dist))

        # If this distance is less than the min_dist, then set min_dist to dist, and identity to name
        if dist < min_dist:
            min_dist = dist
            identity = name
    
    if min_dist > 0.52:
        return 'Please register yourself with the system'
    else:
        return str(identity)

with open('faces.pkl', 'rb') as f:
    faces = pkl.load(f)



from flask import Flask, url_for, send_from_directory, request, jsonify
import logging, os
from werkzeug import secure_filename
import random
# import json


app = Flask(__name__)
file_handler = logging.FileHandler('server.log')
app.logger.addHandler(file_handler)
app.logger.setLevel(logging.INFO)

PROJECT_HOME = os.path.dirname(os.path.realpath(__file__))
UPLOAD_FOLDER = '{}/uploads/'.format(PROJECT_HOME)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def create_new_folder(local_dir):
    newpath = local_dir
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    return newpath

@app.route('/', methods = ['POST'])
def api_root():
    app.logger.info(PROJECT_HOME)
    if request.method == 'POST' and request.files['image']:
        app.logger.info(app.config['UPLOAD_FOLDER'])
        img = request.files['image']
        print(img)
        # img_name = secure_filename(img.filename) #original image name
        img_name = str(random.randint(0,10**10))+'.jpg'   #our image name
        create_new_folder(app.config['UPLOAD_FOLDER'])
        saved_path = os.path.join(app.config['UPLOAD_FOLDER'], img_name)
        app.logger.info("saving {}".format(saved_path))
        img.save(saved_path)

        # return send_from_directory(app.config['UPLOAD_FOLDER'],img_name, as_attachment=True)
        # return json.dumps({'identity': 'Amitabh Bachchan'})
        identified = who_is_it(saved_path, faces, FRmodel)
        return jsonify(identity=identified)
    else:
        return "Where is the image?"

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=False)


