import cv2, glob, random, math, numpy as np, dlib, itertools

from sklearn.svm import SVC
import os
import pickle
from sklearn.externals import joblib

# Set paths for files



clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")  # Or set this to whatever you named the downloaded file

clf = joblib.load('yes_bank_train.pkl')

#print("analyse imported")
def start_analysis(img_path):
    moods = {}
    max_mood = []
    image = cv2.imread(img_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # convert to grayscale
    clahe_image = clahe.apply(gray)
    landmarks_vectorised = get_landmarks(clahe_image,1)
    if landmarks_vectorised == "error":
        return
    print("return: ", landmarks_vectorised)
    cdata=[]
    max_emo=[]
    cdata.append(np.array(landmarks_vectorised))
    print("cdata: ", cdata)
    a=clf.predict_proba(cdata)
    moods["Anger proability"] = str(round((a[0][0] *100),2))+"%"
    max_mood.append(round(a[0][0],2))
    moods["Disgust proability"]= str(round((a[0][1] *100),2))+"%"
    max_mood.append(round(a[0][1],2))
    #moods["Fear proability"]= str(round((a[0][2] *100),2))+"%"
    moods["Happy proability"]= str(round((a[0][2] *100),2))+"%"
    max_mood.append(round(a[0][2],2))

    moods["Sad proability"]= str(round((a[0][3] *100),2))+"%"
    max_mood.append(round(a[0][3],2))
    moods["Surprised proability"]= str(round((a[0][4] *100),2))+"%"
    max_mood.append(round(a[0][4],2))

    return [max(max_mood),moods]

# Define facial landmarks
'''def get_landmarks(image,ii):
    detections = detector(image, 1)
    for k, d in enumerate(detections):  # For all detected face instances individually
        shape = predictor(image, d)  # Draw Facial Landmarks with the predictor class
        xlist = []
        ylist = []
        for i in range(1, 68):  # Store X and Y coordinates in two lists
            xlist.append(float(shape.part(i).x))
            ylist.append(float(shape.part(i).y))
        xmean = np.mean(xlist)
        ymean = np.mean(ylist)
        xcentral = [(x - xmean) for x in xlist]
        ycentral = [(y - ymean) for y in ylist]
        landmarks_vectorised = []
        for x, y, w, z in zip(xcentral, ycentral, xlist, ylist):
                landmarks_vectorised.append(w)
                landmarks_vectorised.append(z)
                meannp = np.asarray((ymean, xmean))
                coornp = np.asarray((z, w))
                dist = np.linalg.norm(coornp - meannp)
                landmarks_vectorised.append(dist)
                landmarks_vectorised.append((math.atan2(y, x) * 360) / (2 * math.pi))
        # data['landmarks_vectorised'] = landmarks_vectorised
    if len(detections) < 1:
            landmarks_vectorised = "error"

    return landmarks_vectorised '''

def get_landmarks(image,ii):
    detections = detector(image, 1)
    # For all detected face instances individually
    for k, d in enumerate(detections):
        # get facial landmarks with prediction model
        shape = predictor(image, d)
        xpoint = []
        ypoint = []
        for i in range(17, 68):
            xpoint.append(float(shape.part(i).x))
            ypoint.append(float(shape.part(i).y))

        # center points of both axis
        xcenter = np.mean(xpoint)
        ycenter = np.mean(ypoint)
        # Calculate distance between particular points and center point
        xdistcent = [(x - xcenter) for x in xpoint]
        ydistcent = [(y - ycenter) for y in ypoint]

        # prevent divided by 0 value
        if xpoint[11] == xpoint[14]:
            angle_nose = 0
        else:
            # point 14 is the tip of the nose, point 11 is the top of the nose brigde
            angle_nose = int(math.atan((ypoint[11] - ypoint[14]) / (xpoint[11] - xpoint[14])) * 180 / math.pi)

        # Get offset by finding how the nose brigde should be rotated to become perpendicular to the horizontal plane
        if angle_nose < 0:
            angle_nose += 90
        else:
            angle_nose -= 90

        landmarks = []
        for cx, cy, x, y in zip(xdistcent, ydistcent, xpoint, ypoint):
            # Add the coordinates relative to the centre of gravity
            landmarks.append(cx)
            landmarks.append(cy)

            # Get the euclidean distance between each point and the centre point (the vector length)
            meanar = np.asarray((ycenter, xcenter))
            centpar = np.asarray((y, x))
            dist = np.linalg.norm(centpar - meanar)

            # Get the angle the vector describes relative to the image, corrected for the offset that the nosebrigde has when the face is not perfectly horizontal
            if x == xcenter:
                angle_relative = 0
            else:
                angle_relative = (math.atan(float(y - ycenter) / (x - xcenter)) * 180 / math.pi) - angle_nose
                # print(anglerelative)
            landmarks.append(dist)
            landmarks.append(angle_relative)

    if len(detections) < 1:
        # In case no case selected, print "error" values
        landmarks = "error"
    return landmarks
