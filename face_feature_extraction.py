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
import requests
import urllib.request as ulib
import ssl
import pickle as pkl

FRmodel = faceRecoModel(input_shape=(3, 96, 96))

def triplet_loss(y_true, y_pred, alpha = 0.3):
    anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]
    
    # Step 1: Compute the (encoding) distance between the anchor and the positive, you will need to sum over axis=-1
    pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), axis=-1)
    # Step 2: Compute the (encoding) distance between the anchor and the negative, you will need to sum over axis=-1
    neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), axis=-1)
    # Step 3: subtract the two previous distances and add alpha.
    basic_loss = tf.add(tf.subtract(pos_dist, neg_dist), alpha)
    # Step 4: Take the maximum of basic_loss and 0.0. Sum over the training examples.
    loss = tf.reduce_sum(tf.maximum(basic_loss, 0.0))
    
    return loss

FRmodel.compile(optimizer = 'adam', loss = triplet_loss, metrics = ['accuracy'])
load_weights_from_FaceNet(FRmodel)

r=requests.get('http://billspill.com/camerademodata.php')
myjson=  r.json()

for person in myjson:
    img_url = person['image']
    img = ulib.urlopen(img_url, context = ssl.SSLContext(ssl.PROTOCOL_TLSv1)).read()
    f = open('%s/%s.%s' %('images_new', person['name'], 'jpg'), 'wb')
    f.write(img)
    f.close()


database = {}
# load all the images of individuals to recognize into the database
for file in glob.glob("images_new/*"):
    identity = os.path.splitext(os.path.basename(file))[0]
    database[identity] = img_path_to_encoding(file, FRmodel)

with open('face_features.pkl', 'wb') as f:
    pkl.dump(database, f, protocol=pkl.HIGHEST_PROTOCOL)
