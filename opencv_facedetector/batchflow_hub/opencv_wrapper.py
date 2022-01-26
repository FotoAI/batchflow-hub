import cv2
import os
import pandas as pd
import numpy as np
from PIL import Image
import math


def build_model():

	detector ={}

	detector["face_detector"] = build_cascade('haarcascade')
	detector["eye_detector"] = build_cascade('haarcascade_eye')

	return detector

def build_cascade(model_name = 'haarcascade'):
	opencv_path = get_opencv_path()

	if model_name == 'haarcascade':

		face_detector_path = opencv_path+"haarcascade_frontalface_default.xml"

		if os.path.isfile(face_detector_path) != True:
			raise ValueError("Confirm that opencv is installed on your environment! Expected path ",face_detector_path," violated.")


		face_detector = cv2.CascadeClassifier(face_detector_path)
		return face_detector

	elif model_name == 'haarcascade_eye':
		eye_detector_path = opencv_path+"haarcascade_eye.xml"

		if os.path.isfile(eye_detector_path) != True:
			raise ValueError("Confirm that opencv is installed on your environment! Expected path ",eye_detector_path," violated.")

		eye_detector = cv2.CascadeClassifier(eye_detector_path)
		return eye_detector

def detect_face(detector, img, align = True):

	resp = []

	detected_face = None
	img_region = [0, 0, img.shape[0], img.shape[1]]

	faces = []
	try:
		#faces = detector["face_detector"].detectMultiScale(img, 1.3, 5)
		faces = detector["face_detector"].detectMultiScale(img, 1.1, 10)
	except:
		pass

	if len(faces) > 0:

		for x,y,w,h in faces:
			detected_face = img[int(y):int(y+h), int(x):int(x+w)]

			if align:
				detected_face = align_face(detector["eye_detector"], detected_face)

			img_region = [x, y, w, h]

			resp.append((detected_face, img_region))

	return resp

def align_face(eye_detector, img):

	detected_face_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #eye detector expects gray scale image

	#eyes = eye_detector.detectMultiScale(detected_face_gray, 1.3, 5)
	eyes = eye_detector.detectMultiScale(detected_face_gray, 1.1, 10)

	if len(eyes) >= 2:

		#find the largest 2 eye
		"""
		base_eyes = eyes[:, 2]

		items = []
		for i in range(0, len(base_eyes)):
			item = (base_eyes[i], i)
			items.append(item)

		df = pd.DataFrame(items, columns = ["length", "idx"]).sort_values(by=['length'], ascending=False)

		eyes = eyes[df.idx.values[0:2]] #eyes variable stores the largest 2 eye
		"""
		#eyes = eyes[0:2]

		#-----------------------
		#decide left and right eye

		eye_1 = eyes[0]; eye_2 = eyes[1]

		if eye_1[0] < eye_2[0]:
			left_eye = eye_1; right_eye = eye_2
		else:
			left_eye = eye_2; right_eye = eye_1

		#-----------------------
		#find center of eyes
		left_eye = (int(left_eye[0] + (left_eye[2] / 2)), int(left_eye[1] + (left_eye[3] / 2)))
		right_eye = (int(right_eye[0] + (right_eye[2]/2)), int(right_eye[1] + (right_eye[3]/2)))
		img = alignment_procedure(img, left_eye, right_eye)
	return img  #return img anyway

def get_opencv_path():
	opencv_home = cv2.__file__
	folders = opencv_home.split(os.path.sep)[0:-1]

	path = folders[0]
	for folder in folders[1:]:
		path = path + "/" + folder

	return path+"/data/"

def findEuclideanDistance(source_representation, test_representation):
    euclidean_distance = source_representation - test_representation
    euclidean_distance = np.sum(np.multiply(euclidean_distance, euclidean_distance))
    euclidean_distance = np.sqrt(euclidean_distance)
    return euclidean_distance

def alignment_procedure(img, left_eye, right_eye):

	#this function aligns given face in img based on left and right eye coordinates

	left_eye_x, left_eye_y = left_eye
	right_eye_x, right_eye_y = right_eye

	#-----------------------
	#find rotation direction

	if left_eye_y > right_eye_y:
		point_3rd = (right_eye_x, left_eye_y)
		direction = -1 #rotate same direction to clock
	else:
		point_3rd = (left_eye_x, right_eye_y)
		direction = 1 #rotate inverse direction of clock

	#-----------------------
	#find length of triangle edges

	a = findEuclideanDistance(np.array(left_eye), np.array(point_3rd))
	b = findEuclideanDistance(np.array(right_eye), np.array(point_3rd))
	c = findEuclideanDistance(np.array(right_eye), np.array(left_eye))

	#-----------------------

	#apply cosine rule

	if b != 0 and c != 0: #this multiplication causes division by zero in cos_a calculation

		cos_a = (b*b + c*c - a*a)/(2*b*c)
		angle = np.arccos(cos_a) #angle in radian
		angle = (angle * 180) / math.pi #radian to degree

		#-----------------------
		#rotate base image

		if direction == -1:
			angle = 90 - angle

		img = Image.fromarray(img)
		img = np.array(img.rotate(direction * angle))

	#-----------------------

	return img #return img anyway
