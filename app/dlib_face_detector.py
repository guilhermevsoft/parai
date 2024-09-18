import dlib

import numpy as np


class DlibFaceDetector:
    def __init__(self):
        self.dlib_face_detection_model = dlib.get_frontal_face_detector()
        shape_predictor_path = 'face-alignment/shape_predictor_5_face_landmarks.dat'
        self.shape_predictor = dlib.shape_predictor(shape_predictor_path)


    def __get_face_chip(self, original_img:np.ndarray, detected_face_bbox:tuple, size:int):
        img_shape = self.shape_predictor(original_img, detected_face_bbox)
        face_chip = dlib.get_face_chip(original_img, img_shape, size=size)
        return face_chip


    def detect_faces(self, original_img: np.array):
        """
        Detects faces in the given image.

        Args:
            original_img (np.array): The input image in the form of a NumPy array in BGR format.

        Returns:
            list: A list of dictionaries containing information about each detected face.
                    Each dictionary contains the following keys:
                    - 'bbox': The bounding box coordinates of the face in the format (x1,y1,x2,y2).
                    - 'face_img': The cropped face image in BGR format.
                    - 'face_chip': The aligned face image (face chip) in BGR format.
                    - 'score': The confidence score of the detection.
                    - 'landmarks': The facial landmarks of the detected face.
        """

        detections = self.dlib_face_detection_model(original_img, 1)        

        detected_faces = []

        for d in detections:
            left = d.left(); right = d.right()
            top = d.top(); bottom = d.bottom()

            detected_face = original_img[max(0, top): min(bottom, original_img.shape[0]), max(0, left): min(right, original_img.shape[1])]

            bbox = (left, top, right, bottom)
            
            face_chip = self.__get_face_chip(original_img, d, size=detected_face.shape[0])

            detected_faces.append({
                'bbox': bbox,
                'score': None,
                'landmarks': None,
                'face_img': detected_face,
                'face_chip': face_chip
            })

        return detected_faces
