# import the necessary packages
import numpy as np
import argparse
import imutils
import time
import cv2
import os
from tqdm import tqdm

def add_padding(img, pad_l, pad_t, pad_r, pad_b):
	height, width, colors = img.shape
	# Adding padding to the left side.
	pad_left = np.zeros([height, pad_l, 3])
	img = np.concatenate((pad_left, img), axis=1)

	# Adding padding to the top.
	pad_up = np.zeros([pad_t, pad_l + width, 3])
	img = np.concatenate((pad_up, img), axis=0)

	# Adding padding to the right.
	pad_right = np.zeros([height + pad_t, pad_r, 3])
	img = np.concatenate((img, pad_right), axis=1)

	# Adding padding to the bottom
	pad_bottom = np.zeros([pad_b, pad_l + width + pad_r, 3])
	img = np.concatenate((img, pad_bottom), axis=0)

	return img

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True,	help="path to input video")
ap.add_argument("-o", "--output", required=True, help="path to output video")
ap.add_argument("-y", "--yolo", required=True, help="base path to YOLO directory")
ap.add_argument("-c", "--confidence", type=float, default=0.5, help="minimum probability to filter weak detections")
ap.add_argument("-t", "--threshold", type=float, default=0.3, help="threshold when applyong non-maxima suppression")
args = vars(ap.parse_args())

# load the COCO class labels our YOLO model was trained on
labelsPath = os.path.sep.join(["classes.txt"])
LABELS = open(labelsPath).read().strip().split("\n")

# initialize a list of colors to represent each possible class label
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")

# choose weights and config files
# DEFAULT (YOLO) -- 
weightsPath = os.path.sep.join([args["yolo"], "yolov3.weights"])
# MODEL 1 (25000)-- weightsPath = os.path.sep.join([args["yolo"], "player4-yolov3_25000.weights"])
# MODEL 2 (14000)-- weightsPath = os.path.sep.join([args["yolo"], "player4-yolov3_14000.weights"])
# MODEL 3 (24000)-- weightsPath = os.path.sep.join([args["yolo"], "player5-yolov3_24000.weights"])
configPath = os.path.sep.join([args["yolo"], "yolo.cfg"])

# load our YOLO object detector trained on COCO dataset (80 classes)
# and determine only the *output* layer names that we need from YOLO
print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# initialize the video stream, pointer to output video file, and
# frame dimensions
vs = cv2.VideoCapture(args["input"])
writer = None
(W, H) = (None, None)

# try to determine the total number of frames in the video file
try:
	prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() else cv2.CAP_PROP_FRAME_COUNT
	total = int(vs.get(prop))
	print("[INFO] {} total frames in video".format(total))

# an error occurred while trying to determine the total
# number of frames in the video file
except:
	print("[INFO] could not determine # of frames in video")
	print("[INFO] no approx. completion time can be provided")
	total = -1

# initialize progress bar
pbar = tqdm(total=total)

# loop over frames from the video file stream
while True:
	# read the next frame from the file
	(grabbed, frame) = vs.read()
	
	# frame width and height variables
	fw = int(vs.get(cv2.CAP_PROP_FRAME_WIDTH))
	fh = int(vs.get(cv2.CAP_PROP_FRAME_HEIGHT))

	# if the frame was not grabbed, then we have reached the end
	# of the stream
	if not grabbed:
		pbar.close()
		break

	# if the frame dimensions are empty, grab them
	if W is None or H is None:
		(H, W) = frame.shape[:2]

	# construct a blob from the input frame and then perform a forward
	# pass of the YOLO object detector, giving us our bounding boxes
	# and associated probabilities
	blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
	net.setInput(blob)
	start = time.time()
	layerOutputs = net.forward(ln)
	end = time.time()

	# initialize our lists of detected bounding boxes, confidences,
	# and class IDs, respectively
	boxes = []
	confidences = []
	classIDs = []
	avgx = 0
	avgy = 0
	players = 0
	# loop over each of the layer outputs
	for output in layerOutputs:
		# loop over each of the detections
		for detection in output:
			# extract the class ID and confidence (i.e., probability)
			# of the current object detection
			scores = detection[5:]
			classID = np.argmax(scores)
			confidence = scores[classID]

			# filter out weak predictions by ensuring the detected
			# probability is greater than the minimum probability
			if confidence > args["confidence"] and classID == 0:
				# scale the bounding box coordinates back relative to
				# the size of the image, keeping in mind that YOLO
				# actually returns the center (x, y)-coordinates of
				# the bounding box followed by the boxes' width and
				# height
				box = detection[0:4] * np.array([W, H, W, H])
				(centerX, centerY, width, height) = box.astype("int")

				# use the center (x, y)-coordinates to derive the top
				# and and left corner of the bounding box
				players += 1
				avgx = (avgx * (players-1) + centerX)/players
				avgy = (avgy * (players-1) + centerY)/players

			# update our list of bounding box coordinates,
			# confidences, and class IDs

	classIDs.append(1)
	confidences.append(float(1))
	boxes.append([int(avgx), int(avgy), 3, 3])
	classIDs.append(2)
	confidences.append(float(1))
	boxes.append([int(fw/2), int(fh/2), 3, 3])
	# apply non-maxima suppression to suppress weak, overlapping
	# bounding boxes
	idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"], args["threshold"])

	# ensure at least one detection exists
	if len(idxs) > 0:
		# loop over the indexes we are keeping
		for i in idxs.flatten():
			# extract the bounding box coordinates
			(x, y) = (boxes[i][0], boxes[i][1])
			(w, h) = (boxes[i][2], boxes[i][3])

			# draw a bounding box rectangle and label on the frame
			color = [int(c) for c in COLORS[classIDs[i]]]
			cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
			if i == 0:
				text = "Center of Players"
			else:
				text = "Center of Frame"
			cv2.line(frame, (x, y), (boxes[i-1][0], boxes[i-1][1]), color, thickness=1, lineType=8, shift=0)
			cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
		# check if the video writer is None
	if writer is None:
		# initialize our video writer
		fourcc = cv2.VideoWriter_fourcc(*"XVID")
		
		sz = (int(vs.get(cv2.CAP_PROP_FRAME_WIDTH)), int(vs.get(cv2.CAP_PROP_FRAME_HEIGHT)))
		fps = vs.get(cv2.CAP_PROP_FPS)
		
		# open and set props
		writer = cv2.VideoWriter(args["output"], fourcc, fps, sz, True)
	
	# write the output frame to disk
	################################################################################
	img = np.asarray(frame)
	
	pad_l = int(max(-1 * (boxes[0][0] - fw/2), 0))
	pad_u = int(max(-1 * (boxes[0][1] - fh/2), 0))
	pad_r = int(max(boxes[0][0] - fw/2, 0))
	pad_d = int(max(boxes[0][1] - fh/2, 0))
	
	crop_l = int(max(boxes[0][0] - fw/2, 0))
	crop_u = int(max(boxes[0][1] - fh/2, 0))
	crop_r = int(min(fw, boxes[0][0] + fw/2))
	crop_d = int(min(fh, boxes[0][1] + fh/2))
	
	cropped_image = img[crop_u:crop_d, crop_l:crop_r]
	im2 = add_padding(cropped_image, pad_l, pad_u, pad_r, pad_d)
	#################################################################################
	writer.write(np.uint8(im2))
	pbar.update(1)

# release the file pointers
print("[INFO] cleaning up...")
writer.release()
vs.release()
