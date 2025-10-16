import os
import errno
import numpy as np 
import deepcell
from deepcell.layers import Location2D
import tensorflow as tf
from deepcell.utils.data_utils import get_data
from skimage.segmentation import relabel_sequential
experiment_folder = "WholecellModel"
MODEL_DIR = os.path.join("models", experiment_folder)
LOG_DIR = os.path.join("models",experiment_folder,"logs")


#Load in BM data and augment into four channels for DAPI, CD71, MIP of all other membrane markers, CD61

NPZ_DIR_1 = "data/bm_data/MISI3527_P1_WCM1_061721_FUSED_THIS_refined202.npz"
NPZ_DIR_2 = "data/bm_data/MISI3527_P1_WCM36_FUSED2_refined263.npz"
NPZ_DIR_3 = "data/bm_data/MISI3527_P1_WCM33_FUSED_refined90.npz"
NPZ_DIR_4 = "data/bm_data/MISI3527_P1_WCM11_FUSED_refined185.npz"
NPZ_DIR_5 = "data/bm_data/MISI3527_P1_WCM31_FUSED_refined248.npz"
NPZ_DIR_6 = "data/bm_data/MISI3527_P1_WCM53_FUSED_refined35.npz"
alldata1 = np.load(NPZ_DIR_1)
alldata2 = np.load(NPZ_DIR_2)
alldata3 = np.load(NPZ_DIR_3)
alldata4 = np.load(NPZ_DIR_4)
alldata5 = np.load(NPZ_DIR_5)
alldata6 = np.load(NPZ_DIR_6)
X1, Y1, OY1 = alldata1['x'], alldata1['y'], alldata1['oy']
X2, Y2, OY2 = alldata2['x'], alldata2['y'], alldata2['oy']
X3, Y3, OY3 = alldata3['x'], alldata3['y'], alldata3['oy']
X4, Y4, OY4 = alldata4['x'], alldata4['y'], alldata4['oy']
X5, Y5, OY5 = alldata5['x'], alldata5['y'], alldata5['oy']
X6, Y6, OY6 = alldata6['x'], alldata6['y'], alldata6['oy']



X = np.concatenate((X1,X2,X3,X4,X5,X6),axis=0)
Y = np.concatenate((Y1,Y2,Y3,Y4,Y5,Y6),axis=0)

newX = np.zeros((X.shape[0],X.shape[1],X.shape[2],X.shape[3]))
for i in range(X.shape[0]):
    newX[i,:,:,0] = X[i,:,:,0] #DAPI
    newX[i,:,:,1] = X[i,:,:,2] #CD61
    newX[i,:,:,2] = X[i,:,:,1] #CD71
    newX[i,:,:,3] = X[i,:,:,3] #CD117
    newX[i,:,:,4] = X[i,:,:,4] #CD38
    newX[i,:,:,5] = X[i,:,:,5] #CD34
    newX[i,:,:,6] = X[i,:,:,6] #CD15
    newX[i,:,:,7] = X[i,:,:,7] #AF

MIP_X = np.zeros((X.shape[0],X.shape[1],X.shape[2],4))
for i in range(X.shape[0]):
    MIP_X[i,:,:,0] = newX[i,:,:,0]
    MIP_X[i,:,:,1] = newX[i,:,:,2]
    MIP_X[i,:,:,2] = np.max(newX[i,:,:,3:7],axis=2)
    MIP_X[i,:,:,3] = newX[i,:,:,1]


#Create training, validation data

np.random.seed(42)

indices = np.random.permutation(X.shape[0])
training_idx, test_idx = indices[:int(len(X)*.8)], indices[int(len(X)*.8):]
tr_X, val_X = MIP_X[training_idx,:], MIP_X[test_idx,:]
tr_Y, val_Y =Y[training_idx,:], Y[test_idx,:]



from deepcell import image_generators
from deepcell.utils import train_utils


batch_size = 8
min_objects = 5 
seed=0

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
    {'X': tr_X, 'y': np.expand_dims(tr_Y,axis=-1)},
    seed=seed,
    transforms=['inner-distance', 'outer-distance', 'fgbg', 'pixelwise'],
    transforms_kwargs={'pixelwise':{'dilation_radius': 1}, 
                      'inner-distance': {'erosion_width': 1, 'alpha': 'auto'},
                      'outer-distance': {'erosion_width': 1}},
    min_objects=0,
    batch_size=batch_size)

val_data = datagen_val.flow(
    {'X': val_X, 'y': np.expand_dims(val_Y,axis=-1)},
    seed=seed,
    transforms=['inner-distance', 'outer-distance', 'fgbg', 'pixelwise'],
    transforms_kwargs={'pixelwise':{'dilation_radius': 1}, 
                      'inner-distance': {'erosion_width': 1, 'alpha': 'auto'},
                      'outer-distance': {'erosion_width': 1}},
    min_objects=1,
    batch_size=batch_size)




#Instantiate model and load pretrained weights

from deepcell.model_zoo.panopticnet import PanopticNet
new_model = PanopticNet(
    backbone='resnet50',
    input_shape=(256, 256, 4),
    norm_method=None,
    num_semantic_heads=4,
    num_semantic_classes=[1, 1, 2, 3], # inner distance, outer distance, fgbg, pixelwise
    location=True,  
    include_top=True)



from tensorflow.keras.optimizers import SGD, Adam
from deepcell.utils.train_utils import rate_scheduler

model_name = "WholecellModel"
n_epoch = 100  

optimizer = Adam(lr=1e-4, clipnorm=0.001)
lr_sched = rate_scheduler(lr=1e-4, decay=0.99)



#Define custom loss for semantic heads

 
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

pretrained_model = "Wholecellpretrain/PretrainedWholecellModel.h5"
weights_model = tf.keras.models.load_model(os.path.join("models",pretrained_model),custom_objects={"Location2D":Location2D,"_semantic_loss":semantic_loss})



new_model.compile(loss=loss, optimizer=optimizer)
new_model.set_weights(weights_model.get_weights())

del weights_model
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

loss_history = new_model.fit_generator(
    train_data,
    steps_per_epoch=train_data.y.shape[0] // batch_size,
    epochs=n_epoch,
    validation_data=val_data,
    validation_steps=val_data.y.shape[0] // batch_size,
    callbacks=train_callbacks)

