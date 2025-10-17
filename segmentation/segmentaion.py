
import os
import errno
import numpy as np 
import deepcell
from tensorflow.python.keras.losses import MSE
from deepcell import losses
import tifffile
import numpy as np
from deepcell_toolbox import utils
import tensorflow as tf
from deepcell.layers import Location2D
from tensorflow.keras.optimizers import SGD, Adam
from deepcell.utils.train_utils import rate_scheduler
import pickle
import gc
from deepcell_toolbox.deep_watershed import deep_watershed
from skimage import transform
import cv2
import skimage
import imageio as iio
import PIL
PIL.Image.MAX_IMAGE_PIXELS = 933120000
import relabel
import dask.array as da

import argparse
parser = argparse.ArgumentParser(description='GetFile')
parser.add_argument('--file', action="store", dest='file', default="")
parser.add_argument('--number', action="store", dest='number', default="")
parser.add_argument('--stage', action="store", dest='stage', default="")
parser.add_argument('--cond', action="store", dest='cond', default="")
args = parser.parse_args()
FILE = args.file
BASE = file.replace(".tif","")
FILE = os.path.join("../WSIMxIF",FILE)
NUMBER = args.number
COND = args.cond
STAGE = args.stage
if STAGE == "0":
    STAGE = ""
else:
    STAGE = "_" + str(STAGE) 

print("File: ", FILE)
print("Base: ", BASE)
print("Number: ", NUMBER)
print("Stage: ",STAGE)
im = imageio.imread(FILE)
im = np.moveaxis(im,0,-1)
msk_file = os.path.join("../output/",BASE,f"roi_{COND}{NUMBER}{STAGE}.tif")
msk = iio.imread(msk_file)
msk = msk > 0
stk = np.stack([msk]*8,axis=-1)
im = im*stk
im = np.moveaxis(im,-1,0)

def semantic_loss(n_classes):
    def _semantic_loss(y_pred, y_true):
        if n_classes > 1:
            return 0.01 * losses.weighted_categorical_crossentropy(
                y_pred, y_true, n_classes=n_classes)
        return MSE(y_pred, y_true)
    return _semantic_loss

model_path = os.path.join("models","WholecellModel/Model5.h5")
model = tf.keras.models.load_model(model_path,custom_objects={"Location2D":Location2D,"_semantic_loss":semantic_loss})


from deepcell.model_zoo.panopticnet import PanopticNet
new_model = PanopticNet(
    backbone='resnet50',
    input_shape=(256, 256, 4),
    norm_method=None,
    num_semantic_heads=4,
    num_semantic_classes=[1, 1, 2, 3], # inner distance, outer distance, fgbg, pixelwise
    location=True,  # should always be true
    include_top=True)
new_model.set_weights(model.get_weights())

loss = {}
# Give losses for all of the semantic heads
for layer in model.layers:
    if layer.name.startswith('semantic_'):
        n_classes = layer.output_shape[-1]
        loss[layer.name] = semantic_loss(n_classes)

new_model.compile(loss=loss)

def seg_fun(image_dask):
    imagedask_np = np.array(image_dask)
    input_shape = imagedask_np.shape
    empty_return = np.zeros_like(imagedask_np[0,:,:])
    if input_shape[1] < 256:
        empty_return = cv2.copyMakeBorder(empty_return,0,(256 - input_shape[1]),0,0,cv2.BORDER_CONSTANT, None, value = 0)
    if input_shape[2] < 256:
        empty_return = cv2.copyMakeBorder(empty_return,0,0,0,(256 - input_shape[2]),cv2.BORDER_CONSTANT, None, value = 0)
    if np.sum(imagedask_np) == 0:
        return empty_return
    MIP_imagedask = np.zeros((input_shape[1],input_shape[2],4))
    MIP_imagedask[:,:,0] = imagedask_np[0,:,:]
    MIP_imagedask[:,:,1] = imagedask_np[1,:,:]
    MIP_imagedask[:,:,2] = np.max(imagedask_np[3:7,:,:],axis=0)
    MIP_imagedask[:,:,3] = imagedask_np[2,:,:]
    if input_shape[1] < 256:
        MIP_imagedask = cv2.copyMakeBorder(MIP_imagedask,0,(256 - input_shape[1]),0,0,cv2.BORDER_CONSTANT, None, value = 0)
    if input_shape[2] < 256:
        MIP_imagedask = cv2.copyMakeBorder(MIP_imagedask,0,0,0,(256 - input_shape[2]),cv2.BORDER_CONSTANT, None, value = 0)
   
    output = new_model(np.expand_dims(MIP_imagedask,axis=0))
    
    mask = deep_watershed([output[0],output[3][:,:,:,1:2]],
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
            pixel_expansion=1,
            maxima_algorithm='h_maxima')
    

    return np.squeeze(mask[0])


im = da.from_array(im,chunks=(8,156,156))
block_labelled = relabel.relabeling.image2labels(im,seg_fun)
arr = np.array(block_labelled)

import tifffile
arr,  _ , _ = skimage.segmentation.relabel_sequential(arr)
tifffile.imwrite(os.path.join("../output/",BASE,f"wholecell_{COND}{NUMBER}{STAGE}.tif"),arr)