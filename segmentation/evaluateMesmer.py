import os
import errno
import numpy as np 
import cv2
import deepcell
from timeit import default_timer
from tensorflow.python.keras.losses import MSE
from deepcell import losses
import pickle 

#Load in trained model weights

def semantic_loss(n_classes):
    def _semantic_loss(y_pred, y_true):
        if n_classes > 1:
            return 0.01 * losses.weighted_categorical_crossentropy(
                y_pred, y_true, n_classes=n_classes)
        return MSE(y_pred, y_true)
    return _semantic_loss


import tensorflow as tf
from deepcell.layers import Location2D
model_name = "WholecellModel/Model5.h5"
model = tf.keras.models.load_model(os.path.join("models",model_name),custom_objects={"Location2D":Location2D,"_semantic_loss":semantic_loss})


# Load in test data
alldata1 = np.load("data/bm_data/WHOLEIM_MISI3527_P1_WCM60_FUSED_refined52.npz")
alldata2 = np.load("data/bm_data/WHOLEIM_MISI3527_P1_WCM8_062921_FUSED_refined144.npz")
alldata3 = np.load("data/bm_data/MISI3527_P1_WCM57_FUSED_refined153.npz")
X1, Y1, OY1 = alldata1['x'], alldata1['y'], alldata1['oy']
X2, Y2, OY2 = alldata2['x'], alldata2['y'], alldata2['oy']
X3, Y3, OY3 = alldata2['x'], alldata2['y'], alldata2['oy']

X_test = np.concatenate((X1,X2,X3),axis=0)
y_test = np.concatenate((Y1,Y2,Y3),axis=0)

newX = np.zeros((X_test.shape[0],X_test.shape[1],X_test.shape[2],X_test.shape[3]))
for i in range(X_test.shape[0]):
    newX[i,:,:,0] = X_test[i,:,:,0]
    newX[i,:,:,1] = X_test[i,:,:,2]
    newX[i,:,:,2] = X_test[i,:,:,1]
    newX[i,:,:,3] = X_test[i,:,:,3]
    newX[i,:,:,4] = X_test[i,:,:,4]
    newX[i,:,:,5] = X_test[i,:,:,5]
    newX[i,:,:,6] = X_test[i,:,:,6]
    newX[i,:,:,7] = X_test[i,:,:,7]

MIP_X = np.zeros((X_test.shape[0],X_test.shape[1],X_test.shape[2],4))
for i in range(X_test.shape[0]):
    MIP_X[i,:,:,0] = newX[i,:,:,0]
    MIP_X[i,:,:,1] = newX[i,:,:,2]
    MIP_X[i,:,:,2] = np.max(newX[i,:,:,3:7],axis=2)
    MIP_X[i,:,:,3] = newX[i,:,:,1]


from deepcell import image_generators
from deepcell.utils import train_utils
batch_size = 1
min_objects = 0  
seed=0

datagen_val = image_generators.SemanticDataGenerator(
    rotation_range=0,
    shear_range=0,
    zoom_range=0,
    horizontal_flip=0,
    vertical_flip=0)

val_data = datagen_val.flow(
    {'X': MIP_X, 'y': np.expand_dims(y_test,axis=-1)},
    seed=seed,
    transforms=['inner-distance', 'outer-distance', 'fgbg', 'pixelwise'],
    transforms_kwargs={'pixelwise':{'dilation_radius': 1}, 
                      'inner-distance': {'erosion_width': 1, 'alpha': 'auto'},
                      'outer-distance': {'erosion_width': 1}},
    min_objects=min_objects,
    batch_size=batch_size,
     shuffle=False)


#Instantiate model

from deepcell.model_zoo.panopticnet import PanopticNet
new_model = PanopticNet(
    backbone='resnet50',
    input_shape=(256, 256, 4),
    norm_method=None,
    num_semantic_heads=4,
    num_semantic_classes=[1, 1, 2, 3], 
    location=True, 
    include_top=True)

new_model.set_weights(model.get_weights())



#Define custom loss

loss = {}
for layer in model.layers:
    if layer.name.startswith('semantic_'):
        n_classes = layer.output_shape[-1]
        loss[layer.name] = semantic_loss(n_classes)


from tensorflow.keras.optimizers import SGD, Adam
from deepcell.utils.train_utils import rate_scheduler

batch_end_loss = list()
class SaveBatchLoss(tf.keras.callbacks.Callback):
    def on_test_batch_end(self, batch, logs=None): 
        batch_end_loss.append(logs['loss'])


new_model.compile(loss=loss)


MASKS_OUTPUT_FILE = "eval/cell_masks.pkl"
MODEL_OUTPUT_FILE  = "eval/model_output.pkl"
OBJECT_STAT_FILE  = "eval/object_stat.pkl"
PIXEL_STAT_FILE  = "eval/pixel_stat.pkl"



output = new_model.predict(val_data)
pickle.dump(output,open(MODEL_OUTPUT_FILE,"wb")) #Raw semantic head output - use inner distance and pixelwise transformations for deep watershed

loss = new_model.evaluate(val_data, callbacks=SaveBatchLoss())

from deepcell_toolbox.deep_watershed import deep_watershed
masks = deep_watershed([output[0],output[3][:,:,:,1:2]],
                   radius=4,
                   maxima_threshold=0.1,
                   interior_threshold=0.3,
                   maxima_smooth=0,
                   interior_smooth=1,
                   maxima_index=0,
                   interior_index=1,
                   label_erosion=0,
                   small_objects_threshold=0,
                   fill_holes_threshold=0,
                   pixel_expansion=0,
                   maxima_algorithm='h_maxima',)


pickle.dump(masks,open(MASKS_OUTPUT_FILE,"wb")) #Cell M,sks


import numpy as np
from skimage.morphology import  remove_small_objects
from skimage.segmentation import clear_border,watershed
from deepcell.metrics import Metrics

y_pred = masks.copy()
y_true = np.expand_dims(y_test[:,:,:],axis=-1).copy()

m = Metrics('Segmentation', seg=False)
df = m.calc_object_stats(y_true, y_pred)


pickle.dump(df,open(OBJECT_STAT_FILE,"wb"))


df2 = m.calc_pixel_stats(y_true,y_pred)

pickle.dump(df2,open(PIXEL_STAT_FILE,"wb"))
