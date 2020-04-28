import numpy as np
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from tqdm import tqdm
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Load list of dog names
dog_names = [item[25:-1] for item in sorted(glob("data/dogImages/train/*/"))]
with open('data/dog_names.json', 'w') as fp:
    json.dump(dog_names, fp)

feature_extractor = ResNet50(weights='imagenet', include_top=False)

datagen = ImageDataGenerator(
            preprocessing_function=preprocess_input,
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.1,
            zoom_range=0.2,
            horizontal_flip=True)   

BATCH_SIZE = 32
IMAGE_RESIZE = 224
DATA_AUGMENTATION_FACTOR = 1

generator_train = datagen.flow_from_directory(
        'data/dogImages/train',
        target_size=(IMAGE_RESIZE, IMAGE_RESIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=True)

generator_valid = datagen.flow_from_directory(
        'data/dogImages/valid',
        target_size=(IMAGE_RESIZE, IMAGE_RESIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=True)

generator_test = datagen.flow_from_directory(
        'data/dogImages/test',
        target_size=(IMAGE_RESIZE, IMAGE_RESIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=True)

print("Generate bottleneck features for training data")
generator_train.reset()
list_tensors = []
list_targets = []
for ix in tqdm(range(DATA_AUGMENTATION_FACTOR*len(generator_train))):
    batch = next(generator_train)
    x = batch[0]
    y = batch[1]
    list_tensors.append(feature_extractor.predict(x))
    list_targets.append(y)  
train_features = np.vstack(list_tensors)
train_targets = np.vstack(list_targets)

print("Genearte bottleneck features for validation data")
generator_valid.reset()
list_tensors = []
list_targets = []
for ix in tqdm(range(DATA_AUGMENTATION_FACTOR*len(generator_valid))):
    batch = next(generator_valid)
    x = batch[0]
    y = batch[1]
    list_tensors.append(feature_extractor.predict(x))
    list_targets.append(y)  
valid_features = np.vstack(list_tensors)
valid_targets = np.vstack(list_targets)

print("Genearte bottleneck features for test data")
generator_test.reset()
list_tensors = []
list_targets = []
for ix in tqdm(range(DATA_AUGMENTATION_FACTOR*len(generator_test))):
    batch = next(generator_test)
    x = batch[0]
    y = batch[1]
    list_tensors.append(feature_extractor.predict(x))
    list_targets.append(y)  
test_features = np.vstack(list_tensors)
test_targets = np.vstack(list_targets)

print("Saving bottleneck data")
np.savez_compressed(open('data/bottleneck_resnet50.npz', 'wb'),
        train_features = train_features,
        train_targets = train_targets,
        valid_features = valid_features,
        valid_targets = valid_targets,
        test_features = test_features,
        test_targets = test_targets)
