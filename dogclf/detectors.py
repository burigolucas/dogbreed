import numpy as np
import cv2

from keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.keras.applications.resnet50 import ResNet50

from .utils import convert_path_to_tensor
from . import session

# Class for human face detection
class HumanFaceDetector():
    def __init__(self):

        # pre-trained face detector
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_alt.xml"
            )

    # Implementation of face detection
    def detect_faces(self,img_path):
        '''
        INPUT:
        img_path    path of image

        OUTPUT:
        boolean     True if face detected

        Description:
        Returns ndarray with faces detected in image
        '''

        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray)

        return faces

# Class for dog detection
class DogDetector():

    def __init__(self):
        # Pre-trained model on imagenet dataset used to detect dogs
        with session.graph.as_default() and session.session.as_default():
            self.__imagenet_model = ResNet50(weights='imagenet')

    # Function to predict the labels from ImageNet
    def predict_imagenet_labels(self,img_path):
        '''
        INPUT:
        img_path    path of image

        OUTPUT:
        predicted label

        Description:
        Returns label from prediction vector for image located at img_path
        '''

        img = preprocess_input(convert_path_to_tensor(img_path))
        prediction = self.__imagenet_model.predict(img)

        return np.argmax(prediction)


    # Dog detector according to labels from ImageNet
    def detect_dog(self,img_path):
        '''
        INPUT:
        img_path    path of image

        OUTPUT:
        boolean     True if dog detected

        Description:
        Returns "True" if a dog is detected in the image stored at img_path
        '''
        prediction = self.predict_imagenet_labels(img_path)
        # Labels 151-268 (inclusive) at ImageNet correspond to dog
        return ((prediction <= 268) & (prediction >= 151))
