import face_recognition
import cv2
def extract(imname,profile_img):
	## New Code for resize 
    img = cv2.imread(imname)
    res= cv2.resize(img,(300,300),interpolation = cv2.INTER_NEAREST)
    cv2.imwrite(imname,res)
    ##

    image = face_recognition.load_image_file(imname)
    face_locations = face_recognition.face_locations(image)
    # print(face_locations)
    i=0
    # x,y,w,h
    found =0
    if len(face_locations) > 0:
        for(top, right, bottom, left) in face_locations:
            # print(top,right,bottom,left)
            # i=i+1
            # name = "final/"
            # name+=str(i)
            # name+=".jpeg"
            crop_img = image[top:bottom, left:right]
            #cv2.imwrite("unknown.jpeg", crop_img)
            cv2.imwrite(imname.split('.')[0]+"_unknown.jpeg", crop_img)

            #known_image = face_recognition.load_image_file("known.jpeg")
            #unknown_image = face_recognition.load_image_file("unknown.jpeg")
            known_image = face_recognition.load_image_file(profile_img)
            unknown_image = face_recognition.load_image_file(imname.split('.')[0]+"_unknown.jpeg")

            biden_encoding = face_recognition.face_encodings(known_image)[0]
            unknown_encoding = face_recognition.face_encodings(unknown_image)[0]

            results = face_recognition.compare_faces([biden_encoding], unknown_encoding)

            if(str(results[0])=='True'):
                    print("found")
                    found = 1
                    break
        if(found==1):
            return 1
        else:
            return 0
    else:
        return 2

# extract("group2.jpeg")