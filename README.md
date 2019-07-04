## Face Recognition with FaceNet
FaceNet is a [pre-trained network](https://github.com/davidsandberg/facenet) trained on multiple faces datasets (LFW, VGGFace2) with minimizing the triplet loss as the training objective.


This repository contains a demonstration of face recognition using this network. For a database consisting FaceNet embeddings (1-D vectors) of N images of N distinct people, if we pass a new image of one of those N people through the FaceNet and compare its embedding with those in the database, the closet one (min L2 distance) will be the person predicted.


Our implementation uses OpenCV to detect the part of the image containing face (OpenCV provides an API called `cv2.CascadeClassifier` for pre-implemented Haar Cascade classifier) and then we pass that part of the image to the FaceNet to calculate the embedding.


This repository was tested on a Raspberry Pi 3B.
