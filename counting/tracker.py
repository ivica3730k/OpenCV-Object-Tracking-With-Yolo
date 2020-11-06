import datetime

import cv2

trackedObjects = []


def _intersection(a, b):
    """
    Returns intersect point between two provided rectangles.

    Use (x,y,w,h) tuple format.
    :param a: First rectangle
    :param b: Second rectangle
    :return: Intersection point of two rectangles
    """
    x = max(a[0], b[0])
    y = max(a[1], b[1])
    w = min(a[0] + a[2], b[0] + b[2]) - x
    h = min(a[1] + a[3], b[1] + b[3]) - y
    if w < 0 or h < 0:
        return ()  # or (0,0,0,0) ?
    return x, y, w, h


def is_object_tracked(x, y, w, h):
    """
    Function that check if the object is currently on the tracking list.

    Returns True if the provided object intersects with any of the objects
    in the trackedObjects list.
    :param x: X start point of rectangle
    :param y: Y start point of rectangle
    :param w: Width of rectangle
    :param h: Height of rectangle
    :return: Boolean value, true if provided polygon is intersection with any polygon on tracking list
    """
    a = (x, y, w, h)
    for i in trackedObjects:
        b = (i.x, i.y, i.width, i.height)
        intersection = _intersection(a, b)
        if intersection:
            return True
    return False


class NewTrackedObject:
    x = 0
    y = 0
    width = 0
    height = 0
    xy_start = None
    xy_stop = None
    mid = None
    last_detected = None
    _tracker = None

    def midpoint(self):
        """

        :return:
        """
        return int((self.xy_start[0] + self.xy_stop[0]) / 2), int((self.xy_start[1] + self.xy_stop[1]) / 2)

    def __init__(self, frame, bbox):
        """

        :param frame:
        :param bbox:
        """
        self._tracker = cv2.TrackerKCF_create()
        self._tracker.init(frame, bbox)
        self.xy_start = (bbox[0], bbox[1])
        self.xy_stop = (bbox[2], bbox[3])
        self.mid = self.midpoint()
        self.last_detected = datetime.datetime.now()
        global trackedObjects
        trackedObjects.append(self)
        pass

    def update(self, frame):
        """

        :param frame:
        :return:
        """
        ok, bbox = self._tracker.update(frame)
        # Draw bounding box
        if ok:
            # Tracking success
            self.xy_start = (int(bbox[0]), int(bbox[1]))
            self.xy_stop = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            self.x = self.xy_start[0]
            self.y = self.xy_start[1]
            self.width = int(bbox[2])
            self.height = int(bbox[3])
            self.mid = self.midpoint()
            self.last_detected = datetime.datetime.now()
            return True
        return False
