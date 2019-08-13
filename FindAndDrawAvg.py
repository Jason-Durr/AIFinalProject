# Import Packages
import cv2
import argparse
import numpy as np


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


# Handle Command Line arguments
ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True,
                help='path to input image')
ap.add_argument('-c', '--config', required=True,
                help='path to yolo config file')
ap.add_argument('-w', '--weights', required=True,
                help='path to yolo pre-trained weights')
ap.add_argument('-cl', '--classes', required=True,
                help='path to text file containing class names')
args = ap.parse_args()

# read input image
image = cv2.imread(args.image)

Width = image.shape[1]
Height = image.shape[0]
scale = 0.00392

# read class names from text file
classes = None
with open(args.classes, 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# generate different colors for different classes
COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

# read pre-trained model and config file
net = cv2.dnn.readNet(args.weights, args.config)  # pylint: disable=no-member
# create input blob
blob = cv2.dnn.blobFromImage(
    image, scale, (416, 416), (0, 0, 0), True, crop=False)  # pylint: disable=no-member

# set input blob for the network
net.setInput(blob)

# function to get the output layer names
# in the architecture


def get_output_layers(net):

    layer_names = net.getLayerNames()

    output_layers = [layer_names[i[0] - 1]
                     for i in net.getUnconnectedOutLayers()]

    return output_layers

# function to draw bounding box on the detected object with class name


def draw_bounding_box(img, class_id, confidence, x, y, x_plus_w, y_plus_h, old_x, old_y):

    if(old_x == 0):
        old_x = x
    if(old_y == 0):
        old_y = y
    label = class_id
    color = COLORS[3]
    cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)
    cv2.putText(img, label, (x-10, y-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    cv2.line(img, (x, y), (old_x, old_y), color,
             thickness=1, lineType=8, shift=0)


# run inference through the network
# and gather predictions from output layers
outs = net.forward(get_output_layers(net))

# initialization
class_ids = []
confidences = []
boxes = []
conf_threshold = 0.5
nms_threshold = 0.4
avgx = 0
avgy = 0
players = 0

# for each detetion from each output layer
# get the confidence, class id, bounding box params
# and ignore weak detections (confidence < 0.5)
for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5:
            center_x = int(detection[0] * Width)
            center_y = int(detection[1] * Height)
            players += 1
            avgx = (avgx * (players-1) + center_x)/players
            avgy = (avgy * (players-1) + center_y)/players
class_ids.append(1)
confidences.append(float(1))
boxes.append([avgx, avgy, 3, 3])
class_ids.append(1)
confidences.append(float(1))
boxes.append([1920/2, 1080/2, 3, 3])
# apply non-max suppression
indices = cv2.dnn.NMSBoxes(
    boxes, confidences, conf_threshold, nms_threshold)  # pylint: disable=no-member

# go through the detections remaining
# after nms and draw bounding box
old_x = 0
old_y = 0
for i in indices:
    i = i[0]
    box = boxes[i]
    x = box[0]
    y = box[1]
    w = box[2]
    h = box[3]
    if i == 0:
        draw_bounding_box(image, "Center of Players", confidences[i], round(
            x), round(y), round(x+w), round(y+h), old_x, old_y)
        old_x = round(x)
        old_y = round(y)
    else:
        draw_bounding_box(image, "Center of Frame", confidences[i], round(
            x), round(y), round(x+w), round(y+h), old_x, old_y)
        old_x = round(x)
        old_y = round(y)
img = np.asarray(image)
pad_l = int(max(-1*(boxes[0][0]-1920/2), 0))
crop_l = int(max(boxes[0][0]-1920/2, 0))
print(boxes[0][0], boxes[0][0]-1920/2, 0)
pad_u = int(max(-1*(boxes[0][1]-1080/2), 0))
crop_u = int(max(boxes[0][1]-1080/2, 0))
pad_r = int(max(boxes[0][0]-1920/2, 0))
crop_r = int(min(1920, boxes[0][0]+1920/2))
pad_d = int(max(boxes[0][1]-1080, 0))
crop_d = int(min(1080, boxes[0][1] + 1080/2))
cropped_image = img[crop_u:crop_d, crop_l:crop_r]
print(cropped_image.shape)
im2 = add_padding(cropped_image, pad_l, pad_u, pad_r, pad_d)

# display output image
# cv2.imshow("object detection", image)

# wait until any key is pressed
cv2.waitKey()

# save output image to disk
cv2.imwrite("object-detection.jpg", im2)

# release resources
cv2.destroyAllWindows()
