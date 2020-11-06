import cv2
import numpy as np


class NeuralNet:
    _weights = None
    _cfg = None
    _names = None
    _classes = []
    _net = None
    _output_layers = None
    _res = 416
    _conf = 0.5

    def __init__(self, weights, cfg, names, res=416, conf=0.5,use_cuda = False):
        """

        :param weights:
        :param cfg:
        :param names:
        :param res:
        :param conf:
        """
        self._weights = weights
        self._cfg = cfg
        self._net = cv2.dnn.readNetFromDarknet(cfg, weights)
        if use_cuda:
            print("NN will use CUDA Backend")
            self._net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self._net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

        layer_names = self._net.getLayerNames()
        self._output_layers = [layer_names[i[0] - 1] for i in self._net.getUnconnectedOutLayers()]
        if names:
            self._names = names
            with open(names, "r") as f:
                self._classes = [line.strip() for line in f.readlines()]
        self._res = res
        self._conf = conf

    def inference(self, img):
        """

        :param img:
        :return:
        """
        blob = cv2.dnn.blobFromImage(
            img, 0.00392, (self._res, self._res), (0, 0, 0), swapRB=True, crop=False)
        self._net.setInput(blob)
        outs = self._net.forward(self._output_layers)
        height, width, channels = img.shape
        class_ids = []
        confidences = []
        boxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > self._conf:
                    # get object detected position and size
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    boxes.append([x, y, w, h])  # put all rectangle areas
                    confidences.append(
                        float(confidence))  # how confidence was that object detected and show that percentage
                    # name of the object tha was detected
                    class_ids.append(class_id)
        # apply non maximum suppression
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.4, 0.6)
        good_detections = []
        for i in range(len(boxes)):
            if i in indexes:
                x_start, y_start, width, height = boxes[i]
                # x_end = x_start + width
                # y_end = y_start + height
                class_id = int(class_ids[i])
                label = str(self._classes[class_id])
                good_detection = [label, x_start, y_start, width, height]
                good_detections.append(good_detection)
        return good_detections
