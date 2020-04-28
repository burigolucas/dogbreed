import matplotlib.pyplot as plt
import numpy as np
import json

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

# Helper function to ouput history of training
def plot_fit_history(history,filepath):
    '''
    INPUT:
    history   - (dict) history dictionary from the History object ouput from the model fit
    filepath  - (str)  path where to save image with fit history

    OUTPUT:
    None

    Description:
    Plot history of CNN model fiting on training and validation sets to a file
    '''
    plt.clf() #clears matplotlib data and axes
    fig, (ax1, ax2) = plt.subplots(2, sharex=True)
        
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

    fig.savefig(filepath,
                bbox_inches='tight',
                size=(3,4),
                dpi=250)

# Function to train the CNN model
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

    print(fit_history.history)
    with open(filepath.split('.hdf5')[0]+'_fit_history.json', 'w') as fp:
        json.dump(str(fit_history.history), fp)
        
    plot_fit_history(fit_history.history,filepath.split('.hdf5')[0]+'_fit_history.png')

    # Load the Model with the Best Validation Loss
    model.load_weights(filepath)

    return model

# Function to evaluate the CNN model on test data
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

    y_true_arr = test_targets

    index_array = [np.arange(y_true_arr.shape[0]), y_pred]
    y_pred_arr = np.zeros(y_true_arr.shape)
    flat_index_array = np.ravel_multi_index(
        index_array,
        y_pred_arr.shape)
    np.ravel(y_pred_arr)[flat_index_array] = 1

    # Report test accuracy
    test_accuracy = metrics.accuracy_score(y_true,y_pred)
    test_balanced_accuracy_score = metrics.balanced_accuracy_score(y_true, y_pred)
    test_f1_micro = metrics.f1_score(y_true=y_true_arr,y_pred=y_pred_arr,average="micro")
    test_f1_macro = metrics.f1_score(y_true=y_true_arr,y_pred=y_pred_arr,average="macro")
    test_f1_weighted = metrics.f1_score(y_true=y_true_arr,y_pred=y_pred_arr,average="weighted")
    test_confusion_matrix = metrics.confusion_matrix(y_true, y_pred)
    # Evaluate model on each category
    test_category_reports = []
    for colIx in range(y_true_arr.shape[1]):
        print(f"Computing classification report for col: {colIx}")
        print(metrics.classification_report(y_true_arr[:,colIx], y_pred_arr[:,colIx],output_dict=False))
        test_category_reports.append(metrics.classification_report(y_true_arr[:,colIx], y_pred_arr[:,colIx],output_dict=True))

    report = dict()
    report['accuracy']                  = test_accuracy
    report['balanced_accuracy_score']   = test_balanced_accuracy_score
    report['f1_micro']                  = test_f1_micro
    report['f1_macro']                  = test_f1_macro
    report['f1_weighted']               = test_f1_weighted
    report['confusion_matrix']          = test_confusion_matrix
    report['category_reports']          = test_category_reports

    print("Model evaluation scores: ")
    print(f"\tAccuracy: {test_accuracy:.4f}")
    print(f"\tBalanced accuracy score: {test_balanced_accuracy_score:.4f}")
    print(f"\tF1 (micro): {test_f1_micro:.4f}")
    print(f"\tF1 (macro): {test_f1_macro:.4f}")
    print(f"\tF1 (weighted): {test_f1_weighted:.4f}")

    return report

# Load bottleneck features and targets
with np.load('data/bottleneck_resnet50.npz') as data:
    train_features = data['train_features']
    train_targets = data['train_targets']
    valid_features = data['valid_features']
    valid_targets = data['valid_targets']
    test_features = data['test_features']
    test_targets = data['test_targets']

# Label used to save the best model
model_label = 'Resnet50'

# Train network
model = train_model(train_features,train_targets,
                    valid_features,valid_targets,
                    dropout=0.50,
                    epochs=50,
                    filepath=f'data/model_weights_best_{model_label}.hdf5')

# Evaluate network
report = evaluate_model(model,test_features,test_targets)
confMatrix = report['confusion_matrix']

# Store the confusion matrix in a JSON file for analysis
report['confusion_matrix'] = report['confusion_matrix'].tolist()
with open(f'data/model_weights_best_{model_label}_eval_report.json', 'w') as fp:
    json.dump(report, fp)

# Plot the confusion matrix and save to file
plt.clf()
fig = plt.figure(figsize=[6,4],dpi=150)
ax = plt.subplot(1, 1, 1)
pos = ax.imshow(confMatrix,cmap=plt.cm.Blues)
fig.colorbar(pos, ax=ax)
ax.set_xlabel('Predicted category')
ax.set_ylabel('True category')
ax.set_title('Confusion matrix')
fig.savefig(f'data/model_weights_best_{model_label}_confusion_matrix.png',
            bbox_inches='tight',
            size=(3,2),
            dpi=250)

# Plot normalized confusion matrix and save to file
plt.clf()
fig = plt.figure(figsize=[6,4],dpi=150)
ax = plt.subplot(1, 1, 1)
# normalize confusion matrix
for ix in range(confMatrix.shape[0]):
    if confMatrix[ix,:].sum() > 0:
        confMatrix[ix,:] = confMatrix[ix,:]/confMatrix[ix,:].sum()
    else:
        print(f"No test data for category {ix}")
pos = ax.imshow(confMatrix,cmap=plt.cm.Blues)
fig.colorbar(pos, ax=ax)
ax.set_xlabel('Predicted category')
ax.set_ylabel('True category')
ax.set_title('Normalized confusion matrix')
fig.savefig(f'data/model_weights_best_{model_label}_confusion_matrix_normalized.png',
            bbox_inches='tight',
            size=(3,2),
            dpi=250)
