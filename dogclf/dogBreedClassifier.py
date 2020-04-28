import numpy as np                      
import json

import cv2

from keras.preprocessing import image                  
from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.applications.resnet50 import ResNet50
from keras.layers import GlobalAveragePooling2D
from keras.layers import Dense
from keras.models import Sequential
import tensorflow as tf
from tqdm import tqdm

from tensorflow.python.keras.backend import set_session
from tensorflow.python.keras.models import load_model

sess = tf.Session()
# to ensure the graph is the same across all threads
graph = tf.compat.v1.get_default_graph()

with open('data/dog_names.json','r') as f:
  dog_names = json.load(f)
f.close()

# extract pre-trained face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt.xml")

# pre-trained model on imagenet dataset
with graph.as_default():
    set_session(sess)
    ResNet50_model = ResNet50(weights='imagenet')
                            
# ResNet-50 model to detect dog breed
with graph.as_default():
    set_session(sess)
    Resnet50_model_breed = Sequential()
    Resnet50_model_breed.add(GlobalAveragePooling2D(input_shape=(1,1,2048)))
    Resnet50_model_breed.add(Dense(133, activation='softmax'))
    Resnet50_model_breed.load_weights('data/model_weights_best_Resnet50.hdf5')

# Implementation of face detector
def face_detector(img_path):
    '''
    INPUT:
    img_path    path of image 

    OUTPUT:
    boolean     True if face detected
    
    Description:
    Returns "True" if face is detected in image stored at img_path
    '''    
    
    global graph
    with graph.as_default():
        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray)
        return len(faces) > 0

# Implementation of transformatin of image to tensor
def path_to_tensor(img_path):
    '''
    INPUT:
    img_path    path of image 

    OUTPUT:
    4d tensor
    
    Description:
    Convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    '''    
    global graph
    with graph.as_default():
        # loads RGB image as PIL.Image.Image type
        img = image.load_img(img_path, target_size=(224, 224))
        # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
        x = image.img_to_array(img)
        
        return np.expand_dims(x, axis=0)

# Helper funcation to load set of images and transfor to tensors
def paths_to_tensor(img_paths):
    '''
    INPUT:
    img_path    (list) paths of images to convert to tensor

    OUTPUT:
    vstack of 4D tensors
    '''    

    list_of_tensors = [path_to_tensor(img_path) for img_path in tqdm(img_paths)]
    return np.vstack(list_of_tensors)

# Function to predict the labels from ImageNet
def ResNet50_predict_labels(img_path):
    '''
    INPUT:
    img_path    path of image 

    OUTPUT:
    predicted label
    
    Description:
    Returns label from prediction vector for image located at img_path
    '''    
    global graph
    with graph.as_default():
        set_session(sess)
        img = preprocess_input(path_to_tensor(img_path))
        prediction = ResNet50_model.predict(img)

        return np.argmax(prediction)

# Dog detector according to labels from ImageNet
def dog_detector(img_path):
    '''
    INPUT:
    img_path    path of image 

    OUTPUT:
    boolean     True if dog detected
    
    Description:
    Returns "True" if a dog is detected in the image stored at img_path
    '''    
    prediction = ResNet50_predict_labels(img_path)
    return ((prediction <= 268) & (prediction >= 151)) 

# Helper function to generate bottleneck features from ResNet-50 model
def extract_Resnet50(tensor):
    '''
    INPUT:
    Image tensor

    OUTPUT:
    ResNet50 features
    
    Description:
    Extract ResNet50 features
    '''    
    global graph
    with graph.as_default():
        set_session(sess)

        return ResNet50(weights='imagenet', include_top=False).predict(preprocess_input(tensor))

# Classification of dog breed
def Resnet50_predict_breed(img_path):
    '''
    INPUT:
    img_path    path of image 

    OUTPUT:
    dog breed
    
    Description:
    Takes a path to an image as input and returns the dog breed that is predicted by the model.
    '''
    global graph
    with graph.as_default():
        set_session(sess)

        # extract bottleneck features
        bottleneck_feature = extract_Resnet50(path_to_tensor(img_path))
        # obtain predicted vector
        predicted_vector = Resnet50_model_breed.predict(bottleneck_feature)
        # return dog breed that is predicted by the model
        return dog_names[np.argmax(predicted_vector)]

# Algorithm to obtain dog breed from dog images or resembling dog breed from human face
def dog_breed_detector(img_path):
    '''
    INPUT:
    img_path    path of image 

    OUTPUT:
    dog breed
    
    Description:
    Takes a path to an image and check if a dog or a human face is in the image.
    If True, then, predicts the dog breed (resembling dog breed for human face).
    '''    
    global graph
    with graph.as_default():
        breed_id = None
        
        if dog_detector(img_path):
            # image contains a dog
            breed_id =  Resnet50_predict_breed(img_path)
            print('Dog breed: {:}'.format(breed_id))

        elif face_detector(img_path):
            # image contains a human face
            breed_id =  Resnet50_predict_breed(img_path)
            print('Resembling dog breed: {:}'.format(breed_id))

        else:
            raise("Image does not contain a dog nor a human face")

        return breed_id
