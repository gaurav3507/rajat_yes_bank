from flask import Flask, jsonify, request, render_template  
from flask_cors import CORS
from althappy.faceDetection.get_expressions_test1 import ml_run
from PIL import Image
import base64
import os	
import subprocess
from althappy.faceDetection.extractFaces import extract


def getData(uploaded_img,profile_img):
	print("function called")

	# this will decode and save the image string to storage
	#name = name.partition(",")[2]
	#name = base64.b64decode(name)
	#with open("name.jpeg", 'wb') as f:
	#	f.write(name)
	#f.close()
	flag = 0
	# flag 1 will indicates that person is found in group picture
	flag = extract(uploaded_img,profile_img)
	if(flag==1):
		print("Recognised successfully")
		flag = 0
		# unknown.jpeg is cropped image of that person
		value = ml_run(uploaded_img.split('.')[0]+'_unknown.jpeg')
		print("ml returned")
		print("value")
		# os.remove("final.jpeg")
		return value
	elif flag==0:
		return "person not found"
	elif flag==2:
		return "face_locations not found"		
