import cv2, glob, random, math, numpy as np, dlib, itertools
from PIL import Image, ImageDraw, ImageFont
import matplotlib.image as mpimg

from sklearn.svm import SVC
import os
import pickle
from sklearn.externals import joblib
import sys
def ml_run(image_path_codec):
    # Set paths for files
    print("inside get Expression")
    image_process = image_path_codec
    print(image_process)
    # test_path = "D:\\Defence Project\\emotion\\test\\"

    # cap = cv2.VideoCapture(1)

    # Initialise predictor
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("althappy/faceDetection/shape_predictor_68_face_landmarks.dat")
    print("predicted properly")
    # Or set this to whatever you named the downloaded file


    # Function to check emotion in a file
    def test_files(files):
        # Define function to get file list, randomly shuffle it and split 80/20

        # files = sorted(glob.glob(os.path.join(test_path, "*")), key=os.path.getmtime)

        # print(files)

        # for i in range(0, len(files)):
        #     print ("testing file ", files[i])
        print("inside test_files")
        tt(cv2.imread(files), files)
        print("testfiles completed")

    # Fetch files from folder


    def getexpressions(a, image):
        index_max = np.argmax(a[0])
        font_normal = ImageFont.truetype("Verdana.ttf", size=20)
        font_max = ImageFont.truetype("Verdana.ttf", size=30)
        if index_max == 0:
            c = "Anger probability = " + str(a[0][0] * 100) + "%\n"
            image.text((50, 50), c, fill=(255, 0, 0), font=font_max)
            c = "Disgust probability = " + str(a[0][1] * 100) + "%\n" + "Happy probability = " + str(
                a[0][2] * 100) + "%\n" + "Sad probability = " + str(
                a[0][3] * 100) + "%\n" + "Surprised probability = " + str(a[0][4] * 100) + "%\n"
            image.text((50, 83), c, fill=(255, 0, 0), font=font_normal)

        elif index_max == 1:
            c = "Disgust probability = " + str(a[0][1] * 100) + "%\n"
            image.text((50, 50), c, fill=(255, 0, 0), font=font_max)
            c = "Anger probability = " + str(a[0][0] * 100) + "%\n" + "Happy probability = " + str(
                a[0][2] * 100) + "%\n" + "Sad probability = " + str(
                a[0][3] * 100) + "%\n" + "Surprised probability = " + str(a[0][4] * 100) + "%\n"
            image.text((50, 83), c, fill=(255, 0, 0), font=font_normal)

        elif index_max == 2:
            c = "Happy probability = " + str(a[0][2] * 100) + "%\n"
            image.text((50, 50), c, fill=(255, 0, 0), font=font_max)
            c = "Anger probability = " + str(a[0][0] * 100) + "%\n" + "Disgust probability = " + str(
                a[0][1] * 100) + "%\n" + "Sad probability = " + str(
                a[0][3] * 100) + "%\n" + "Surprised probability = " + str(a[0][4] * 100) + "%\n"
            image.text((50, 83), c, fill=(255, 0, 0), font=font_normal)

        elif index_max == 3:
            c = "Sad probability = " + str(a[0][3] * 100) + "%\n"
            image.text((50, 50), c, fill=(255, 0, 0), font=font_max)
            c = "Anger probability = " + str(a[0][0] * 100) + "%\n" + "Disgust probability = " + str(
                a[0][1] * 100) + "%\n" + "Happy probability = " + str(
                a[0][2] * 100) + "%\n" + "Surprised probability = " + str(a[0][4] * 100) + "%\n"
            image.text((50, 83), c, fill=(255, 0, 0), font=font_normal)

        elif index_max == 4:
            c = "Surprised probability = " + str(a[0][4] * 100) + "%\n"
            image.text((50, 50), c, fill=(255, 0, 0), font=font_max)
            c = "Anger probability = " + str(a[0][0] * 100) + "%\n" + "Disgust probability = " + str(
                a[0][1] * 100) + "%\n" + "Happy probability = " + str(
                a[0][2] * 100) + "%\n" + "Sad probability = " + str(a[0][3] * 100) + "%\n"
            image.text((50, 83), c, fill=(255, 0, 0), font=font_normal)
        print("getExpressions completed")
        return image
    # c = ""

    def tt(image, name):
        print("inside tt")
        global c
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # convert to grayscale
        clahe_image = clahe.apply(gray)
        landmarks_vectorised = get_landmarks(clahe_image)
        print("landmarks returned")
        # print(landmarks_vectorised)
        cdata = []
        max_emo = []
        cdata.append(np.array(landmarks_vectorised))
        a = []
        try:
            a = clf.predict_proba(cdata)
        except :
            b = []
            b.append(0)
            b.append(0)
            b.append(0)
            b.append(0)
            b.append(0)
            a.append(b)
        a[0][0] = round(a[0][0], 2)
        a[0][1] = round(a[0][1], 2)
        a[0][2] = round(a[0][2], 2)
        a[0][3] = round(a[0][3], 2)
        a[0][4] = round(a[0][4], 2)
        # c = "Anger probability = " + str(a[0][0] * 100) + "%\n" + "Disgust probability = " + str(
        #     a[0][1] * 100) + "%\n" + "Happy probability = " + str(
        #     a[0][2] * 100) + "%\n" + "Sad probability = " + str(
        #     a[0][3] * 100) + "%\n" + "Surprised probability = " + str(a[0][4] * 100) + "%\n"


        c = [a[0][2]
            ,{"Anger proability" : str(a[0][0] * 100),
            "Disgust proability" : str(a[0][1] * 100),
            "Happy proability" : str(a[0][2] * 100),
            "Sad proability" : str(a[0][3] * 100),
            "Surprised proability" : str(a[0][4] * 100)}]


        # print "Fear probability", a[0][2]*100, "%"
        # print(c)

        # Inserting text in the Images(Frames)...
        # img = Image.open(name)
        # d = ImageDraw.Draw(img)
        # d = getexpressions(a, d)
        # d.text((50, 50), c, fill=(255, 255, 0), font=font)

        # Image is saved after insertion of text...
        # img.save(name)

        # Images are read again from the disk to show it like a video...
        # img = cv2.imread(name)
        # cv2.imshow("video", img)
        print("tt completed")
        # print("tt completed")
        return None
        

    # Define facial landmarks
    def get_landmarks(image):
        print("inside getlandmarks")
        detections = detector(image, 1)
        # For all detected face instances individually
        landmarks = []
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

                # Get the angle the vector describes relative to the image, corrected for the
                # offset that the nosebrigde has when the face is not perfectly horizontal
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
            print("landmarks returned")
        return landmarks

    try:
        clf = joblib.load('althappy/faceDetection/big_dataset3_linear.pkl')
    except:
        print("Oops!",sys.exc_info()[0],"occured.")


    
    currentframe = 0
    
    frame = mpimg.imread(image_process)
    
    ret = frame

    name = image_path_codec
    # print('Creating...' + name)

    # writing the extracted images
    cv2.waitKey(10)
    cv2.imwrite(name, frame)
    # cv2.imshow("Video", frame)
    cv2.waitKey(10)
    test_files(name)
  
    currentframe += 1
    # cap.release()
    cv2.destroyAllWindows()
    # print(c)
    # os.remove(image_path_codec)
    print("removed")
    print(c)
    return c




