#Part 1 FR Model
from keras import backend as K
K.set_image_data_format('channels_first')
from fr_utils import *
from inception_blocks_v2 import *
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
print('Model loaded')


#Part 2 Loading face features
import pickle as pkl
with open('face_features.pkl', 'rb') as f:
    faces = pkl.load(f)
print('Features loaded')


#Part 3 Face recognition
def find_identity(frame, database, x1, y1, x2, y2, max_allowed_dist):
    """
    Determine whether the face contained within the bounding box exists in our database
    """
    height, width, channels = frame.shape
    # The padding is necessary since the OpenCV face detector creates the bounding box around the face and not the head
    part_image = frame[max(0, y1):min(height, y2), max(0, x1):min(width, x2)]
    
    return who_is_it(part_image, database, FRmodel, max_allowed_dist)

def who_is_it(image, database, model, max_allowed_dist):
    encoding = img_to_encoding(image, model)
    
    min_dist = 100
    identity = None
    
    # Loop over the database dictionary's names and encodings.
    for (name, db_enc) in database.items():
        
        # Compute L2 distance between the target "encoding" and the current "emb" from the database.
        dist = np.linalg.norm(db_enc - encoding)

        print('distance for %s is %s' %(name, dist))

        # If this distance is less than the min_dist, then set min_dist to dist, and identity to name
        if dist < min_dist:
            min_dist = dist
            identity = name
    
    if min_dist > max_allowed_dist:
        return 'Please register yourself with the system'
    else:
        print('Detected identity: ', identity)
        return str(identity)


#Part 4 Summarise all Facial Recognition
PADDING=50
def find_who_is_it(img_path, database, saved_model, max_allowed_dist):
    frame = cv2.imread(img_path)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Loop through all the faces detected and determine whether or not they are in the database
    identities = []
    for (x, y, w, h) in faces:
        x1 = x-PADDING
        y1 = y-PADDING
        x2 = x+w+PADDING
        y2 = y+h+PADDING

        identity = find_identity(frame, database, x1, y1, x2, y2, max_allowed_dist)
        identities.append(identity)
    return identities




#Part 5 API

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
        identified = find_who_is_it(saved_path, faces, FRmodel, 1.0)
        print('returned val', identified)
        return jsonify(identity=identified)
    else:
        return "Where is the image?"

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=False)


