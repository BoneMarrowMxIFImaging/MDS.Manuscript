import os
import errno
import numpy as np 
import deepcell
import cv2
import pickle
os.environ.update({"LD_LIBRARY_PATH": "/media/redmondlab3/Onions/ryan/resnetenv/lib/python3.11/site-packages/nvidia/cudnn/lib"})

experiment_folder = "Wholecellpretrain"
MODEL_DIR = os.path.join("models", experiment_folder)
NPZ_DIR = "data/mytissuenet"
LOG_DIR = os.path.join("models",experiment_folder,"logs")

if not os.path.isdir(MODEL_DIR):
    os.makedirs(MODEL_DIR)

if not os.path.isdir(LOG_DIR):
    os.makedirs(LOG_DIR)

from deepcell.utils.data_utils import get_data
from skimage.segmentation import relabel_sequential



#Load augmented tissuenet training data into generators
train_data = np.load(os.path.join(NPZ_DIR,"train.npz"))
X_train, y_train = train_data["X"], train_data["y"]
val_data = np.load(os.path.join(NPZ_DIR,"val.npz"))
X_val, y_val = val_data['X'], val_data["y"]

from deepcell import image_generators
from deepcell.utils import train_utils


batch_size = 8
min_objects = 0  
seed=0

# training augmentation
datagen = image_generators.CroppingDataGenerator(
    rotation_range=180,
    shear_range=0,
    zoom_range=(0.8, 1/0.8),
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode="reflect",
    crop_size=(256, 256))

datagen_val = image_generators.SemanticDataGenerator(
    rotation_range=0,
    shear_range=0,
    zoom_range=0,
    horizontal_flip=0,
    vertical_flip=0)
    
train_data = datagen.flow(
    {'X': X_train, 'y': np.expand_dims(y_train,axis=-1)},
    seed=seed,
    transforms=['inner-distance', 'outer-distance', 'fgbg', 'pixelwise'],
    transforms_kwargs={'pixelwise':{'dilation_radius': 1}, 
                      'inner-distance': {'erosion_width': 1, 'alpha': 'auto'},
                      'outer-distance': {'erosion_width': 1}},
    min_objects=0,
    batch_size=batch_size)

val_data = datagen_val.flow(
    {'X': X_val, 'y': np.expand_dims(y_val,axis=-1)},
    seed=seed,
    transforms=['inner-distance', 'outer-distance', 'fgbg', 'pixelwise'],
    transforms_kwargs={'pixelwise':{'dilation_radius': 1}, 
                      'inner-distance': {'erosion_width': 1, 'alpha': 'auto'},
                      'outer-distance': {'erosion_width': 1}},
    min_objects=1,
    batch_size=batch_size)


#instantiate model

from deepcell.model_zoo.panopticnet import PanopticNet

new_model = PanopticNet(
    backbone='resnet50',
    input_shape=(256, 256, 4),
    norm_method=None,
    num_semantic_heads=4,
    num_semantic_classes=[1, 1, 2, 3], 
    location=True, 
    include_top=True)


from tensorflow.keras.optimizers import SGD, Adam
from deepcell.utils.train_utils import rate_scheduler

model_name = "PretrainedWholecellModel"
n_epoch = 66  


optimizer = Adam(lr=1e-5, clipnorm=0.001)
lr_sched = rate_scheduler(lr=1e-5, decay=0.99)


# Custom loss for semantic heads

from tensorflow.python.keras.losses import MSE
from deepcell import losses


def semantic_loss(n_classes):
    def _semantic_loss(y_pred, y_true):
        if n_classes > 1:
            return 0.01 * losses.weighted_categorical_crossentropy(
                y_pred, y_true, n_classes=n_classes)
        return MSE(y_pred, y_true)
    return _semantic_loss


loss = {}
for layer in new_model.layers:
    if layer.name.startswith('semantic_'):
        n_classes = layer.output_shape[-1]
        loss[layer.name] = semantic_loss(n_classes)

new_model.compile(loss=loss, optimizer=optimizer)


from deepcell.utils.train_utils import get_callbacks
from deepcell.utils.train_utils import count_gpus


model_path = os.path.join(MODEL_DIR, '{}.h5'.format(model_name))
loss_path = os.path.join(MODEL_DIR, '{}.npz'.format(model_name))

num_gpus = count_gpus()

print('Training on', num_gpus, 'GPUs.')

train_callbacks = get_callbacks(
    model_path,
    lr_sched=lr_sched,
    tensorboard_log_dir=LOG_DIR,
    save_weights_only=False,
    monitor='val_loss',
    verbose=1)

#Train
loss_history = new_model.fit_generator(
    train_data,
    steps_per_epoch=train_data.y.shape[0] // batch_size,
    epochs=n_epoch,
    validation_data=val_data,
    validation_steps=val_data.y.shape[0] // batch_size,
    callbacks=train_callbacks)

