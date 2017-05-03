"""
@author lmiguelmh
@since 20170503
"""

import cv2
import dlib
import numpy as np

predictor_model = "/home/deeplearning/Desktop/projects/tesis/data/dlib/shape_predictor_68_face_landmarks.dat"
file_name = "/home/deeplearning/Desktop/projects/tesis/pedestrian-faces-detection/datasets/images/newtest/cfb.gif"
face_detector = dlib.get_frontal_face_detector()
face_pose_predictor = dlib.shape_predictor(predictor_model)


def draw_landmarks(image):
    detected_faces = face_detector(image, 1)
    for i, face_rect in enumerate(detected_faces):
        pose_landmarks = face_pose_predictor(image, face_rect)
        pose_landmarks_iterable = np.matrix([[p.x, p.y] for p in pose_landmarks.parts()])
        for l, point in enumerate(pose_landmarks_iterable):
            pos = (point[0, 0], point[0, 1])
            cv2.circle(image, pos, 1, color=(255, 255, 255))


# http://www.pyimagesearch.com/2017/04/03/facial-landmarks-dlib-opencv-python/
def shape_to_np(shape, dtype="int"):
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((68, 2), dtype=dtype)

    # loop over the 68 facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)

    # return the list of (x, y)-coordinates
    return coords


# http://www.pyimagesearch.com/2017/04/03/facial-landmarks-dlib-opencv-python/
def rect_to_bb(rect):
    # take a bounding predicted by dlib and convert it
    # to the format (x, y, w, h) as we would normally do
    # with OpenCV
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y

    # return a tuple of (x, y, w, h)
    return (x, y, w, h)


import collections

# define a dictionary that maps the indexes of the facial
# landmarks to specific face regions
FACIAL_LANDMARKS_IDXS = collections.OrderedDict([
    ("mouth", (48, 68)),
    ("right_eyebrow", (17, 22)),
    ("left_eyebrow", (22, 27)),
    ("right_eye", (36, 42)),
    ("left_eye", (42, 48)),
    ("nose", (27, 35)),
    ("jaw", (0, 17))
])
colors = [(19, 199, 109), (79, 76, 240), (230, 159, 23),
          (168, 100, 168), (158, 163, 32),
          (163, 38, 32), (180, 42, 220)]


def draw_landmarks_2(image):
    detected_faces = face_detector(image, 1)
    for i, face_rect in enumerate(detected_faces):
        face_landmarks = face_pose_predictor(image, face_rect)
        face_landmarks = shape_to_np(face_landmarks)
        # for (x, y) in face_landmarks:
        #     cv2.circle(image, (x, y), 1, (0, 0, 255), -1)

        # loop over the facial landmark regions individually
        for (i, name) in enumerate(FACIAL_LANDMARKS_IDXS.keys()):
            # grab the (x, y)-coordinates associated with the
            # face landmark
            (j, k) = FACIAL_LANDMARKS_IDXS[name]
            pts = face_landmarks[j:k]
            for p in pts:
                cv2.circle(image, tuple(p), 1, colors[i], -1)


def align_face(image, landmarks, required_landmarks):
    # landmarks must be in numpy format (shape_to_np)
    (r_start, r_end) = FACIAL_LANDMARKS_IDXS["right_eye"]
    (l_start, l_end) = FACIAL_LANDMARKS_IDXS["left_eye"]
    right_eye = landmarks[r_start, r_end]
    left_eye = landmarks[l_start, l_end]

