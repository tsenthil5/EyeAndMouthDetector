# import the necessary packages
from imutils import face_utils
import numpy as np
import argparse
import imutils
import dlib
import cv2
import sys
import vlc
import time

from scipy.spatial import distance as dist
count = 0
ear=0.3
eye_thr=0
x=0
count1=0
count2=0
list = []
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
	help="path to facial landmark predictor")
#ap.add_argument("-i", "--image", required=True,
	#help="path to input image")
args = vars(ap.parse_args())
# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])
# load the input image, resize it, and convert it to grayscale
cap = cv2.VideoCapture(0)
while 1:
        ret,image = cap.read()
#image = cv2.imread(args["image"])
        image = imutils.resize(image, width=500)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
 
# detect faces in the grayscale image
        rects = detector(gray, 1)
        def eye_aspect_ratio(eye):
                
                
                
                A=dist.euclidean(eye[1], eye[5])
                B=dist.euclidean(eye[2], eye[4])
                C=dist.euclidean(eye[0], eye[4])
                ear = (A+B)/(2.0*C)
                #print(ear)
                return ear        
                              
                        
        #def count_ear(eye):
                
                
                #for i in range(10):
                        #D=dist.euclidean(eye[1], eye[5])
                        #E=dist.euclidean(eye[2], eye[4])
                        #F=dist.euclidean(eye[0], eye[4])
                        #ear1 = (D+E)/(2.0*F)
                        #print(ear1)
                        #list.append(ear1)
                #for i in range(11):
                        #print(list[i])
                #for i in range(11):
                        #if list[i]<0.3:
                                #x=x+1
                #return x
                
                        
                        
                        
                
                
# loop over the face detections
        for (i, rect) in enumerate(rects):
	# determine the facial landmarks for the face region, then
	# convert the facial landmark (x, y)-coordinates to a NumPy
	# array
                shape = predictor(gray, rect)
                shape = face_utils.shape_to_np(shape)
                (x, y, w, h) = cv2.boundingRect(np.array([shape[36:42]]))
                roi = image[y:y+h,x:x+w]
                roi = imutils.resize(roi, width=250, inter=cv2.INTER_CUBIC)
                cv2.imshow("eye", roi)
                roi=cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                ret,threshold = cv2.threshold(roi, 55, 200, cv2.THRESH_BINARY)
                blur = cv2.GaussianBlur(roi,(5,5),0)
                cv2.imshow("eye1", blur)
                th = cv2.adaptiveThreshold(roi, 200, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 1)
                cv2.imshow("threshold",th)
                #contours, heirarchy = cv2.findContours(th, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                #cv2.drawContours(roi,contours,-1,(0,0,255),3)
                (X,Y)=face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
                righteye = shape[X:Y]
                rightEAR = eye_aspect_ratio(righteye)
                if rightEAR<0.3:
                        count+=1
                else:
                        count=0
                if count>10:
                        p = vlc.MediaPlayer("C:\Python36\Loud_Alarm_Clock_Buzzer-Muk1984-493547174.mp3")
                        p.play()
                        time.sleep(5)
                        p.stop()
                        count=0
                        
                                
                                
                        
                #print(count)       
                #cv2.imshow("eye2", roi)
                        #mouth_aspect_ratio
                J=dist.euclidean(shape[51],shape[59])
                K=dist.euclidean(shape[53],shape[57])
                L=dist.euclidean(shape[61],shape[65])
                O=((J+K)/2.0*L)
                print(O)
                if (O>1200):
                        count1=count1+1
                if (count1>3):
                        count2=count2+1
                        count1=0
                        if count2>3:
                                
                                s = vlc.MediaPlayer("C:\Python36\Fatigue.mp3")
                                s.play()
                                time.sleep(7)
                                s.stop()
                                count2=0

                

        
 
	# convert dlib's rectangle to a OpenCV-style bounding box
	# [i.e., (x, y, w, h)], then draw the face bounding box
                (x, y, w, h) = face_utils.rect_to_bb(rect)
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
 
	# show the face number
                cv2.putText(image, "Face #{}".format(i + 1), (x - 10, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
 
	# loop over the (x, y)-coordinates for the facial landmarks
	# and draw them on the image
                for (x, y) in shape:
                        cv2.circle(image, (x, y), 1, (0, 0, 255), -1)
 
# show the output image with the face detections + facial landmarks
        #if right_EAR < ear:
                #count=count+1
                #if count>=ear_thr:
                        
                        
        cv2.imshow("Output", image)
        #x = len(contours)
        #print("COUNT:",x)
        #if eye_thr>20:
                
                #if rightEAR<0.3:
                        
                        #p = vlc.MediaPlayer("C:\Python36\Loud_Alarm_Clock_Buzzer-Muk1984-493547174.mp3")
                        #p.play()
                        #time.sleep(5)
                        #p.stop()
                        #eye_thr=0
                
                
        #E=dist.euclidean(shape[63], shape[67])
        #F=dist.euclidean(shape[49], shape[55])
        
        #print(E/F)
        
        k = cv2.waitKey(30) & 0xff
        if k == ord("q"):
                break
                
                
cap.release()
cv2.destroyALLWindows()

