import pickle

import numpy as np 
import tensorflow as tf
#import matplotlib.pyplot as plt
import tensorflow.keras as keras
import pickle
IMG_SHAPE = (128,128)
NUM_CLASSES = 2
BATCH_SIZE = 16
FOLDS = 10
SEED = 42
import os
import pathlib
total_this_count = 0
import os
import platform




from tensorflow.keras import layers

from keras.utils.np_utils import to_categorical
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split



from tensorflow.keras import layers

from keras.utils.np_utils import to_categorical
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from classification_models.keras import Classifiers

from keras.preprocessing.image import ImageDataGenerator

from sklearn.preprocessing import LabelEncoder
"ResNet18_Flat","ResNet18_Flat_pluslayer","ResNet18_Pool","ResNet18_Pool_pluslayer"


def rate_scheduler(lr=.001, decay=0.95):
    """Schedule the learning rate based on the epoch.

    Args:
        lr (float): initial learning rate
        decay (float): rate of decay of the learning rate

    Returns:
        function: A function that takes in the epoch
        and returns a learning rate.
    """
    def output_fn(epoch):
        epoch = np.int_(epoch)
        new_lr = lr * (decay ** epoch)
        return new_lr
    return output_fn

METRICS = [
            keras.metrics.BinaryAccuracy(name='accuracy'),
            keras.metrics.BinaryCrossentropy(name='crossentropy'),
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall'),
            keras.metrics.AUC(name='auc'),
            keras.metrics.AUC(name='prc', curve='PR'), # precision-recall curve
           # keras.metrics.F1Score(name='f1')
        ]

def get_model(which):
    if which == "ResNet18_Flat":
        ResNet18, preprocess_input = Classifiers.get('resnet18')
        base_model = ResNet18(input_shape=(128, 128, 3), weights="imagenet",include_top=False)
        base_model.trainable = True
        model = Sequential([
            base_model,
            Flatten(),
            #Dense(384, activation='relu'),
           # Dense(16, activation='relu') , # binary classification, so 1 neuron with sigmoid activation
            Dense(1,activation="sigmoid")
        ])
        lr_sched = rate_scheduler(lr=1e-5, decay=0.99)
        
        model.compile(
            optimizer=keras.optimizers.Adam(lr=1e-5, clipnorm=0.001),
            loss='binary_crossentropy',
            metrics=METRICS
        )
        return model, preprocess_input
    if which == "ResNet18_Flat_pluslayer":
        ResNet18, preprocess_input = Classifiers.get('resnet18')
        base_model = ResNet18(input_shape=(128, 128, 3), weights="imagenet",include_top=False)
        base_model.trainable = True
        model = Sequential([
            base_model,
            Flatten(),
            #Dense(384, activation='relu'),
            Dense(16, activation='relu') , # binary classification, so 1 neuron with sigmoid activation
            Dense(1,activation="sigmoid")
        ])
        lr_sched = rate_scheduler(lr=1e-5, decay=0.99)
        
        model.compile(
            optimizer=keras.optimizers.Adam(lr=1e-5, clipnorm=0.001),
            loss='binary_crossentropy',
            metrics=METRICS
        )
        return model, preprocess_input
    if which == "ResNet18_Pool":
        ResNet18, preprocess_input = Classifiers.get('resnet18')
        base_model = ResNet18(input_shape=(128, 128, 3), weights="imagenet",include_top=False)
        base_model.trainable = True
        x = keras.layers.GlobalAveragePooling2D()(base_model.output)
        output = keras.layers.Dense(1, activation='sigmoid')(x)
        model = keras.models.Model(inputs=[base_model.input], outputs=[output])
        lr_sched = rate_scheduler(lr=1e-5, decay=0.99)
        
        model.compile(
            optimizer=keras.optimizers.Adam(lr=1e-5, clipnorm=0.001),
            loss='binary_crossentropy',
            metrics=METRICS
        )
        return model, preprocess_input
    if which == "ResNet18_Pool_pluslayer":
        ResNet18, preprocess_input = Classifiers.get('resnet18')
        base_model = ResNet18(input_shape=(128, 128, 3), weights="imagenet",include_top=False)
        base_model.trainable = True
        x = keras.layers.GlobalAveragePooling2D()(base_model.output)
        output = keras.layers.Dense(16, activation='relu')(x)
        output = keras.layers.Dense(1, activation='sigmoid')(x)
        model = keras.models.Model(inputs=[base_model.input], outputs=[output])
        model.compile(
            optimizer=keras.optimizers.Adam(lr=1e-5, clipnorm=0.001),
            loss='binary_crossentropy',
            metrics=METRICS
        )
        return model, preprocess_input

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
import cv2
kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
def unsharp_mask(image, kernel_size=(5, 5), sigma=.5, amount=1, threshold=0):
    """Return a sharpened version of the image, using an unsharp mask."""
    blurred = cv2.GaussianBlur(image, kernel_size, sigma)
    sharpened = float(amount + 1) * image - float(amount) * blurred
    sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
    sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
    sharpened = sharpened.round().astype(np.uint8)
    if threshold > 0:
        low_contrast_mask = np.absolute(image - blurred) < threshold
        np.copyto(sharpened, image, where=low_contrast_mask)
    return sharpened

image_clahe = cv2.createCLAHE(clipLimit=.01,tileGridSize=(8,8))
def create_composite(images, colors):
    """
    Creates a composite image from multiple images using additive blending with clipping.

    :param images: List of 2D NumPy arrays of type float32, representing grayscale images.
    :param colors: List of RGB color tuples, each representing the color to tint the corresponding image.
    :return: A 3D NumPy array representing the colored and blended composite image.
    """
    # Ensure the input lists are of the same length
    if len(images) != len(colors):
        raise ValueError("The number of images and colors must be the same")

    height, width = images[0].shape
    composite = np.zeros((height, width, 3), dtype=np.float32)

    for image, color in zip(images, colors):
        if image.ndim != 2:
            raise ValueError("Each image must be a 2D array")

        for i in range(3):  # Loop over the RGB channels
            composite[:, :, i] += image * color[i]

    composite = np.clip(composite, 0.0, 255.0)

    return composite

def preprocess(this3):

    this3[:,:,1] = image_clahe.apply(np.expand_dims(((scaler.fit_transform(this3[:,:,1].reshape(-1, 1))*255)).reshape((128,128)).astype(np.uint8),-1))
    this3[:,:,2] = this3[:,:,2]*1.25
    this3[:,:,1] = this3[:,:,1]/2

    return create_composite(this3.transpose(2,0,1),colors=[(50,50,50),(0,0,1),(0,1,0)]).astype(int)

thisnewestX = []
thisnewestY = []
for dir in os.listdir("./dataset_tp53_sep"):
    for file in os.listdir("./dataset_tp53_sep/"+dir):
        rd = pickle.load(open("./dataset_tp53_sep/"+dir+"/"+file,"rb"))
        thisnewestX.append(preprocess(rd))
        thisnewestY.append(dir)


X = np.array(thisnewestX)
Y = np.array(thisnewestY)
encoder = LabelEncoder()
Y = encoder.fit_transform(Y)

Y = to_categorical(Y)

from sklearn.utils import shuffle
X, Y = shuffle(X, Y, random_state=2)
print(X.shape)
print(Y.shape)

train_inds, val_inds = train_test_split([*range(len(X))], test_size=0.15,random_state=SEED)
        
X_trn_ = X[train_inds]
Y_trn_ = Y[train_inds]
X_test = X[val_inds]
Y_test = Y[val_inds]

train_inds, val_inds = train_test_split([*range(len(X_trn_))], test_size=0.25,random_state=SEED)
X_trn = X_trn_[train_inds].copy()
Y_trn = Y_trn_[train_inds].copy()
X_val = X_trn_[val_inds].copy()
Y_val = Y_trn_[val_inds].copy()
print('number of images: %3d' % (len(X_trn) + len(X_val)))
print('- training:       %3d' % len(X_trn))
print('- positive training:       %3d' % np.sum(Y_trn))
print('- validation:     %3d' % len(X_val))
print('- positive val:       %3d' % np.sum(Y_val))
print('- testing:     %3d' % len(X_test))
print('- positive test:       %3d' % np.sum(Y_test))


#X_trn = np.array(newX)
#Y_trn = np.array(newY)

print('number of images: %3d' % (len(X_trn) + len(X_val)))
print('- training:       %3d' % len(X_trn))
print('- positive training:       %3d' % np.sum(Y_trn))
print('- validation:     %3d' % len(X_val))
print('- positive val:       %3d' % np.sum(Y_val))
print('- testing:     %3d' % len(X_test))
print('- positive test:       %3d' % np.sum(Y_test))
#######         Train           ##############
model, preprocess2 = get_model("ResNet18_Pool")


train_gen = ImageDataGenerator(horizontal_flip=True,vertical_flip=True,rotation_range=360,zoom_range=(.8,1.2),preprocessing_function=preprocess2)

val_gen  = ImageDataGenerator(preprocessing_function=preprocess2)
print(X_trn.shape)
print(Y_trn.shape)
print(X_val.shape)
print(Y_val.shape)
train_generator  = train_gen.flow(X_trn,Y_trn,batch_size=BATCH_SIZE)
val_generator = val_gen.flow(X_val,Y_val,batch_size=BATCH_SIZE)



checkpoint_filepath = '/media/redmondlab3/Onions/ryan/celltyping/checkpoints/round7_tp53_2.h5'
model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_crossentropy',
    mode='min',
    save_best_only=True)


lr_sched = rate_scheduler(lr=1e-5, decay=0.99)
lr_sched_callback = keras.callbacks.LearningRateScheduler(lr_sched)

from tensorflow.keras.models import clone_model


model.load_weights("/media/redmondlab3/Onions/ryan/celltyping/checkpoints/round7_ptrdec21_2.h5")
model_clone = clone_model(model)
model_clone.set_weights(model.get_weights())
model_truncated = keras.models.Model(
    inputs=model_clone.input, 
    outputs=model_clone.layers[-3].output
)



model_truncated.trainable = True
num_classes = 3  



x = keras.layers.GlobalAveragePooling2D()(model_truncated.output)
output = keras.layers.Dense(3, activation='softmax')(x)
new_model = keras.models.Model(inputs=[model_truncated.input], outputs=[output])
for i in range(len(model_clone.layers[:-3])):
    new_model.layers[i].set_weights(model_clone.layers[i].get_weights())
new_model.trainable = True




lr_sched = rate_scheduler(lr=1e-5, decay=0.99)
new_model.compile(
    optimizer=keras.optimizers.Adam(lr=1e-6, clipnorm=0.001), #CHANGING TO. 5e-3
    loss='categorical_crossentropy',   
    metrics=METRICS
)



history = new_model.fit(train_generator, validation_data=val_generator, epochs=25,callbacks=[model_checkpoint_callback,lr_sched_callback])
new_model.save_weights('/media/redmondlab3/Onions/ryan/celltyping/checkpoints/round7_tp53_end_2.h5')


import pickle

#pickle.dump(history.history,open("./pickdir/round7_poolpluslayer_new.pkl","wb"))
del model
model, preprocessing = get_model("ResNet18_Pool")

model.load_weights("/media/redmondlab3/Onions/ryan/celltyping/checkpoints/round7_ptrdec21_2.h5")
model_clone = clone_model(model)
model_clone.set_weights(model.get_weights())
model_truncated = keras.models.Model(
    inputs=model_clone.input, 
    outputs=model_clone.layers[-3].output
)



model_truncated.trainable = True
num_classes = 3  



x = keras.layers.GlobalAveragePooling2D()(model_truncated.output)
output = keras.layers.Dense(3, activation='softmax')(x)
new_model = keras.models.Model(inputs=[model_truncated.input], outputs=[output])
for i in range(len(model_clone.layers[:-3])):
    new_model.layers[i].set_weights(model_clone.layers[i].get_weights())
new_model.trainable = True




lr_sched = rate_scheduler(lr=1e-5, decay=0.99)
new_model.compile(
    optimizer=keras.optimizers.Adam(lr=1e-5, clipnorm=0.001), #CHANGING TO. 5e-3
    loss='categorical_crossentropy',   
    metrics=METRICS
)


new_model.load_weights("/media/redmondlab3/Onions/ryan/celltyping/checkpoints/round7_tp53_2.h5")
for i in range(len(X_test)):
    if np.sum(X_test[i,:,:,0]) > 500000:
        print(np.sum(X_test[i,:,:,0]),i)
val_gen  = ImageDataGenerator(preprocessing_function=preprocess2)
val_generator = val_gen.flow(X_test,Y_test,batch_size=BATCH_SIZE,shuffle=False)
output  = new_model.evaluate(val_generator)
output  = new_model.predict(val_generator)
pickle.dump(output,open("./thisoutput_feb20.pkl","wb"))
pickle.dump(X_test,open("./Xtest_feb20.pkl","wb"))
pickle.dump(Y_test,open("./Ytest_feb20.pkl","wb"))

