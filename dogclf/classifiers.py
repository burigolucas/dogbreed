import numpy as np
import json
import cv2

from keras.applications.resnet50 import preprocess_input
from tensorflow.keras.applications.resnet50 import ResNet50
from keras.layers import GlobalAveragePooling2D
from keras.layers import Dense
from keras.models import Sequential

from .utils import convert_path_to_tensor

from . import session

from .detectors import HumanFaceDetector, DogDetector

# classifier of dog brees
class DogBreedClassifier():

    def __init__(self):

        self._faceDetector = HumanFaceDetector()
        self._dogDetector = DogDetector()

        with open('data/dog_names.json','r') as f:
            self._dog_names = json.load(f)
        f.close()

        # ResNet-50 model for dog breed classification
        with session.graph.as_default() and session.session.as_default():
            self._dogBreedCNN = Sequential()
            self._dogBreedCNN.add(GlobalAveragePooling2D(input_shape=(7,7,2048)))
            self._dogBreedCNN.add(Dense(133, activation='softmax'))
            self._dogBreedCNN.load_weights('data/model_weights_best_Resnet50.hdf5')


        # ResNet-50 model for feature extration
        with session.graph.as_default() and session.session.as_default():
            self._featureExtractor = ResNet50(weights='imagenet', include_top=False)

    # Helper function to generate bottleneck features from CNN
    def _extract_bottleneck_features(self,tensor):
        '''
        INPUT:
        Image tensor

        OUTPUT:
        bottleneck features

        Description:
        Extract bottleneck features
        '''

        return self._featureExtractor.predict(preprocess_input(tensor))

    # Classification of dog breed
    def predict_breed(self,img_path):
        '''
        INPUT:
        img_path    path of image

        OUTPUT:
        dog breed

        Description:
        Takes a path to an image as input and returns the dog breed that is predicted by the model.
        '''

        features = self._extract_bottleneck_features(convert_path_to_tensor(img_path))
        # obtain predicted vector
        predicted_vector = self._dogBreedCNN.predict(features)
        # return dog breed that is predicted by the model
        return self._dog_names[np.argmax(predicted_vector)]

    # Algorithm to obtain dog breed from dog images or resembling dog breed from human face
    def classify_dog_breed(self,img_path):
        '''
        INPUT:
        img_path    path of image 

        OUTPUT:
        dog breed

        Description:
        Takes a path to an image and check if a dog or a human face is in the image.
        If True, then, predicts the dog breed (resembling dog breed for human face).
        If neither a dog or human face in the image, return (-1,-1)
        '''

        if self._dogDetector.detect_dog(img_path):
            # image contains a dog
            breed_id = self.predict_breed(img_path)
            isDog = 1

        else:
            faces = self._faceDetector.detect_faces(img_path)
            if len(faces) and faces.shape[0] == 1:
                # image contains at least a human face
                breed_id =  self.predict_breed(img_path)
                isDog = 0
            else:
                # Image does not contain a dog nor a human face
                breed_id = -1
                isDog = -1

        return (isDog,breed_id)