import os
import pathlib
import platform
import pickle
import numpy as np 
import tensorflow as tf
import tensorflow.keras as keras

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


IMG_SHAPE = (128,128)
NUM_CLASSES = 2
BATCH_SIZE = 16
FOLDS = 10
SEED = 42

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


membrane_data = np.load("data/membrane_annotations.npz")
X, Y = membrane_data["X"], membrane_data["Y"]
X, Y = shuffle(X, Y, random_state=SEED)
train_inds, val_inds = train_test_split([*range(len(X))], test_size=0.15,random_state=SEED)
        
X_trn_ = X[train_inds].copy()
Y_trn_ = Y[train_inds].copy()
X_test = X[val_inds].copy()
Y_test = Y[val_inds].copy()

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



model, preprocess = get_model("ResNet18_Pool")

train_gen = ImageDataGenerator(horizontal_flip=True,vertical_flip=True,rotation_range=360,zoom_range=(.8,1.2),brightness_range=(.95,1.05),preprocessing_function=preprocess)
val_gen  = ImageDataGenerator(preprocessing_function=preprocess)
train_generator  = train_gen.flow(X_trn,Y_trn,batch_size=BATCH_SIZE)
val_generator = val_gen.flow(X_val,Y_val,batch_size=BATCH_SIZE)



checkpoint_filepath = os.path.join("models","MembraneModel.h5")
model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_crossentropy',
    mode='min',
    save_best_only=True)

lr_sched = rate_scheduler(lr=1e-5, decay=0.99)
model.compile(
    optimizer=keras.optimizers.Adam(lr=1e-5, clipnorm=0.001), #CHANGING TO. 5e-3
    loss='binary_crossentropy',   
    metrics=METRICS
)
lr_sched_callback = keras.callbacks.LearningRateScheduler(lr_sched)
lr_sched = rate_scheduler(lr=1e-5, decay=0.99)
        

history = model.fit(train_generator, validation_data=val_generator, epochs=25,callbacks=[model_checkpoint_callback,lr_sched_callback])

pickle.dump(history.history,open(os.path.join("eval","history.pkl"),"wb"))


val_gen  = ImageDataGenerator(preprocessing_function=preprocessing)
val_generator = val_gen.flow(X_test,Y_test,batch_size=BATCH_SIZE,shuffle=False)
output  = model.evaluate(val_generator)
output  = model.predict(val_generator)
pickle.dump(output,open(os.path.join("eval","model_output.pkl"),"wb"))
pickle.dump(X_test,open(os.path.join("eval","X_test.pkl"),"wb"))
pickle.dump(Y_test,open(os.path.join("eval","Y_test.pkl"),"wb"))


