import numpy as np                      
from keras.preprocessing import image                  

# Implementation of transformatin of image to tensor
def convert_path_to_tensor(img_path):
    '''
    INPUT:
    img_path    path of image 

    OUTPUT:
    4d tensor
    
    Description:
    Convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    '''    
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    
    return np.expand_dims(x, axis=0)

# Helper funcation to load set of images and transfor to tensors
def convert_paths_to_tensor(img_paths):
    '''
    INPUT:
    img_path    (list) paths of images to convert to tensor

    OUTPUT:
    vstack of 4D tensors
    '''    

    list_of_tensors = [convert_path_to_tensor(img_path) for img_path in tqdm(img_paths)]
    return np.vstack(list_of_tensors)
