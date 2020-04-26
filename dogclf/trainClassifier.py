import matplotlib.pyplot as plt
import numpy as np
import json
from glob import glob

from sklearn import metrics
from sklearn.datasets import load_files
from keras.utils import np_utils
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Dropout, Flatten, Dense
from keras.models import Sequential
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from tqdm import tqdm

def load_dataset(dirpath):
    '''
    INPUT:
    dirpath  - (str)  path where images are located with sub-directories per category

    OUTPUT:
    None

    Description:
    Load datasets from dirpath
    '''
    data = load_files(dirpath)
    files = np.array(data['filenames'])
    targets = np_utils.to_categorical(np.array(data['target']), 133)
    return files, targets

def plot_fit_history(history,filepath):
    '''
    INPUT:
    history   - (dict) history dictionary from the History object ouput from the model fit
    filepath  - (str)  path where to save image with fit history

    OUTPUT:
    None

    Description:
    Plots history of CNN model fiting on training and validation sets.
    '''
    plt.clf() #clears matplotlib data and axes
    fig = plt.figure(figsize=[12,4])
    ax1 = plt.subplot(1, 2, 1)
    ax2 = plt.subplot(1, 2, 2)
        
    loss = history['loss']
    val_loss = history['val_loss']
    acc = history['acc']
    val_acc = history['val_acc']
    ep = range(len(loss))

    ax1.plot(ep,loss,
                label='train',
                color='C0')
    ax1.plot(ep,val_loss,
                label='val',
                color='C1')
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('loss')
    ax1.legend()

    ax2.plot(ep,acc,
                label='train',
                color='C0')
    ax2.plot(ep,val_acc,
                label='val',
                color='C1')
    ax2.set_xlabel('epoch')
    ax2.set_ylabel('accuracy')
    ax2.legend()

    fig.savefig(filepath)

def train_model(train_features,train_targets,
            valid_features,valid_targets,
            dropout=0.0,
            epochs=10,
            filepath='weights.best.model.hdf5'):
    '''
    INPUT:
    history   - (dict) history dictionary from the History object ouput from the model fit
    filepath  - (str)  path where to save image with fit history
    train_features - (tensor) train bottleneck features
    train_targets - (ndarray) array with train labels
    valid_features - (tensor) validation bottleneck features
    valid_targets - (ndarray) array with valid labels
    dropout     - (float) drop-out factor
    filepath    - (str) path where to save best model

    OUTPUT:
    model       - model with the best validation loss

    Description:
    Applies transfer learning using bottleneck features from a Keras CNN model to train the model
    and evaluating on test set. Results of the fit are stored togehter with best model on the filepath.
    A dropout parameter can be set to help with overfitting.
    '''
    # Model Architecture
    model = Sequential()
    model.add(GlobalAveragePooling2D(input_shape=train_features.shape[1:]))
    model.add(Dropout(dropout))
    model.add(Dense(133, activation='softmax'))
    model.summary()

    # Compile the Model
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    # Train the model.
    checkpointer = ModelCheckpoint(filepath=filepath, 
                                verbose=1, save_best_only=True)
    early_stopper = EarlyStopping(monitor = 'val_loss', patience = 5)

    fit_history = model.fit(train_features, train_targets, 
            validation_data=(valid_features, valid_targets),
            epochs=epochs, batch_size=32, callbacks=[checkpointer,early_stopper], verbose=1)

    with open(filepath.split('.hdf5')[0]+'_fit_history.json', 'w') as fp:
        json.dump(fit_history.history, fp)
        
    plot_fit_history(fit_history.history,filepath.split('.hdf5')[0]+'_fit_history.png')

    # Load the Model with the Best Validation Loss
    model.load_weights(filepath)

    return model

def evaluate_model(model,test_features,test_targets):
    '''
    INPUT:
    test_features - (tensor) test bottleneck features
    test_targets - (ndarray) array with test labels
    model       - model with the best validation loss

    OUTPUT:
    None

    Description:
    Evaluate model scores on test dataset.
    '''
    # Test the Model
    # Calculate classification accuracy on the test dataset.
    # Get index of class for each image in test set
    y_pred = np.array([np.argmax(model.predict(np.expand_dims(feature, axis=0))) for feature in test_features])
    y_true = np.argmax(test_targets, axis=1)

    # Report test accuracy
    test_accuracy = 100*np.sum(np.equal(y_pred,y_true))/len(y_pred)
    test_balanced_accuracy_score = metrics.balanced_accuracy_score(y_true, y_pred)
    test_confusion_matrix = metrics.confusion_matrix(y_true, y_pred)

    report = dict()
    report['accuracy']                  = test_accuracy
    report['balanced_accuracy_score']   = test_balanced_accuracy_score
    report['confusion_matrix']          = test_confusion_matrix

    print(f"Test accuracy: {test_accuracy:.4f}%%")
    print(f"Balanced accuracy score: {test_balanced_accuracy_score:.4f}")

    return report


# Import Dog Dataset
# load train, test, and validation datasets
train_files, train_targets = load_dataset('data/dogImages/train')
valid_files, valid_targets = load_dataset('data/dogImages/valid')
test_files, test_targets = load_dataset('data/dogImages/test')

# Load list of dog names
dog_names = [item[25:-1] for item in sorted(glob("data/dogImages/train/*/"))]
with open('models/dog_names.json', 'w') as fp:
    json.dump(dog_names, fp)

# Print statistics about the dataset
print('There are %d total dog categories.' % len(dog_names))
print('There are %s total dog images.\n' % len(np.hstack([train_files, valid_files, test_files])))
print('There are %d training dog images.' % len(train_files))
print('There are %d validation dog images.' % len(valid_files))
print('There are %d test dog images.'% len(test_files))

# Obtain Bottleneck Features
bottleneck_features = np.load('data/bottleneck_features/DogResnet50Data.npz')
train_Resnet50 = bottleneck_features['train']
valid_Resnet50 = bottleneck_features['valid']
test_Resnet50 = bottleneck_features['test']

# Train network
model = train_model(train_Resnet50,train_targets,
                    valid_Resnet50,valid_targets,
                    dropout=0.50,
                    epochs=50,
                    filepath='models/weights_best_Resnet50.hdf5')

# Evaluate network
report = evaluate_model(model,test_Resnet50,test_targets)
plt.clf() #clears matplotlib data and axes
plt.imshow(report['confusion_matrix'])
plt.savefig('models/weights_best_Resnet50_confusion_matrix.png')
report['confusion_matrix'] = report['confusion_matrix'].tolist()
with open('models/weights_best_Resnet50_eval_report.json', 'w') as fp:
    json.dump(report, fp)