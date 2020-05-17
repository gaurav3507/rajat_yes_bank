#!/usr/bin/env python
from flask import Flask, jsonify, request, render_template  
from flask_cors import CORS
from get_expressions_test1 import ml_run
from PIL import Image
import base64
import os	
import subprocess
from extractFaces import extract

app = Flask(__name__)
CORS(app)

languages = [{'name':'javascript'},{'name':'python'}]

@app.route('/', methods=['GET'])
def test():
    return jsonify({'message': 'It works'})


@app.route('/lang', methods=['POST'])
def test1():
	request.get_json(force=True)
	global languages
	language = {'name':request.json['name']}
	res = getData(request.json['name'])
	print(res)
	language = {'name':res}
	languages=language
	response = jsonify({'languages': languages})
	response.headers.add('Access-Control-Allow-Origin', '*')
	return response
    
    
@app.route('/aaa/<string:condition>', methods=['GET'])
def ow2(condition):
	aa = condition
	
	return jsonify({'message': "heeee"})
    
# function that will return string of expressions
def getData(name):
	print("function called")

	# this will decode and save the image string to storage
	name = name.partition(",")[2]
	name = base64.b64decode(name)
	with open("name.jpeg", 'wb') as f:
		f.write(name)
	f.close()
	flag = 0
	# flag 1 will indicates that person is found in group picture
	flag = extract("name.jpeg")
	if(flag==1):
		print("Recognised successfully")
		flag = 0
		# unknown.jpeg is cropped image of that person
		value = ml_run('unknown.jpeg')
		print("ml returned")
		print("value")
		# os.remove("final.jpeg")
		return value
	else:
		return "person not found"

	

	# return the string



# for testing 
# if __name__ == '__main__':
#    app.run(debug=True)

# for deployment
if __name__ == "__main__":
	app.run(host="0.0.0.0", port="80")





