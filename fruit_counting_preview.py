import argparse
import datetime
import time

time.sleep(0.0001)

import cv2

import counting.tracker as trackingModule
import nn.nn as nn

ap = argparse.ArgumentParser()

ap.add_argument("-i", "--input", required=True,
                help="Path to input video file")
ap.add_argument("-o", "--output", required=False,
                default="output.avi", help="Path to output video file")
ap.add_argument("-w", "--weights", required=False,
                default="yolov3.weights", help="Path to yolo weights file")
ap.add_argument("-c", "--config", required=False,
                default="yolov3.cfg", help="Path to yolo config file")
ap.add_argument("-n", "--names", required=False,
                default="yolov3.names", help="Path to class names file")
ap.add_argument("-r", "--rotate", required=False,
                default="0", help="Rotation")
args = vars(ap.parse_args())

_DETECTION_X_AREA = 1200
_ENABLE_NN = False

if _ENABLE_NN:
    net = nn.NeuralNet(args["weights"], args["config"], args["names"], conf=0.5)
# Create the video capture object
video = cv2.VideoCapture(args["input"])
# Read video
video.set(3, 1280)
video.set(4, 720)
_COUNTED = 0


# Detection line parameters:


def draw_vertical_detection_line(f, x=_DETECTION_X_AREA):
    h, w, c = f.shape
    cv2.line(f, pt1=(x, 0), pt2=(x, h), color=(0, 0, 255), thickness=4)


"""
_, frame = video.read()
for i in range(0, 1):
    if not _:
        print("Check camera")
        exit(1)
    bbox = cv2.selectROI(frame, False)
    tracker = t.TrackedObject(frame, bbox)
"""
while True:
    # Read a new frame
    ok, frame = video.read()
    if not ok:
        break
    if _ENABLE_NN:
        nn_detections = net.inference(frame)
        for i in nn_detections:
            if i[1] > _DETECTION_X_AREA:
                continue
            # cv2.rectangle(frame, (i[1], i[2]), (i[1] + i[3], i[2] + i[4]), (0, 255, 255), 2, 1)
            if trackingModule.is_object_tracked(i[1], i[2], i[3], i[4]):
                # print("Object already tracked")
                continue
            new_object_box = (i[1], i[2], i[3], i[4])
            trackingModule.NewTrackedObject(frame, new_object_box)
    for i in trackingModule.trackedObjects:
        # Update tracker
        status = i.update(frame)
        if (datetime.datetime.now() - i.last_detected) > datetime.timedelta(0, 30):
            trackingModule.trackedObjects.remove(i)
        if status:
            cv2.circle(frame, i.mid, 10, (255, 0, 0))
            # cv2.rectangle(frame, i.xy_start, i.xy_stop, (255, 0, 0), 2, 1)
            cv2.rectangle(frame, (i.x, i.y), (i.x + i.width, i.y + i.height), (255, 0, 0), 2, 1)
            if i.mid[0] > _DETECTION_X_AREA:
                # Object is behind counting line
                _COUNTED += 1
                trackingModule.trackedObjects.remove(i)
        else:
            trackingModule.trackedObjects.remove(i)

    cv2.putText(frame, "Tracking: " + str(len(trackingModule.trackedObjects)), (100, 20), cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (50, 170, 50), 2)
    cv2.putText(frame, "Counted: " + str(_COUNTED), (100, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)

    # Display result
    draw_vertical_detection_line(frame)
    cv2.imshow("Tracking", frame)
    # Exit if ESC pressed
    k = cv2.waitKey(25) & 0xff
    if k == 27:
        break
    if k == 9:
        print("roi")
        bbox = cv2.selectROI(frame, False)
        if not trackingModule.is_object_tracked(bbox[0], bbox[1], bbox[2], bbox[3]):
            tracker = trackingModule.NewTrackedObject(frame, bbox)
        else:
            print("Already tracking this object")
