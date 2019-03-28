# USAGE
# python object_tracker.py --prototxt deploy.prototxt --model res10_300x300_ssd_iter_140000.caffemodel

# import the necessary packages
from pyimagesearch.centroidtracker import CentroidTracker
from pyimagesearch.udp import EnviaDatos
from pyimagesearch.emotionDetect import Emotions
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2

bGhost = False
sOrigen = "Notebook Hernan"
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=True,
	help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True,
	help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
#args = vars(ap.parse_args())

# initialize our centroid tracker and frame dimensions
ct = CentroidTracker()
envia = EnviaDatos()
envia.senddata("INICIO,,," + sOrigen, bGhost)
(H, W) = (None, None)

# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe("C:/Desarrollo/Ejemplos/Python ejemplos/201902/simple-object-tracking/deploy.prototxt", "C:/Desarrollo/Ejemplos/Python ejemplos/201902/simple-object-tracking/res10_300x300_ssd_iter_140000.caffemodel")

# initialize the video stream and allow the camera sensor to warmup
print("[INFO] starting video stream...")
vs = cv2.VideoCapture(0)
#vs = VideoStream(src=0,resolution=(640,480),framerate=5).start()
time.sleep(2.0)

# loop over the frames from the video stream
count = 0
rects = []
width = vs.get(3)  # float
height = vs.get(4) # float
label = "N/A"
emociones=[]
while True:
	# read the next frame from the video stream and resize it
	frame = vs.read()[1]	
	frame = imutils.resize(frame, width=800)
	frameidentificado, box, rects, emociones = ct.getFaceBox(net,frame,0.5)
# Identifico la emocion
	em = Emotions()
	label = em.crop_minAreaRect(frame, rects)
#----------------------
	objects = ct.update(rects, sOrigen, emociones)	
	
	# loop over the tracked objects
	# Esta parte agrega el ID del objeto detectado
	for (objectID, centroid) in objects.items():
		# draw both the ID of the object and the centroid of the
		# object on the output frame
		try:			
			text = "ID {}".format(objectID) + " - " + emociones[objectID]
			cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
				cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
			cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)
		except:
			text = "ID {}".format(objectID)
			cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
				cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
			cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)
			pass
	# show the output frame
	cv2.imshow("Banco Galicia Torre Peron 430 Piso 6to.", frame)
	key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

# do a bit of cleanup
envia.senddata("FIN"+ " - " + sOrigen, bGhost)
cv2.destroyAllWindows()
try:
	vs.stop()
except:
	pass