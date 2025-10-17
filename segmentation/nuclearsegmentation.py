
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
from deepcell_toolbox.processing import histogram_normalization

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


tiles, tiles_info = utils.tile_image(np.expand_dims(im,axis=0),model_input_shape=(256,256),stride_ratio=1.0)
X = histogram_normalization(tiles)

MIP_X = np.zeros((X.shape[0],X.shape[1],X.shape[2],2))
for i in range(X.shape[0]):
    MIP_X[i,:,:,0] = X[i,:,:,0]
    MIP_X[i,:,:,1] = np.max(X[i,:,:,1:],axis=2)
del X

MIP_X = utils.untile_image(MIP_X,tiles_info=tiles_info)
im = MIP_X[0]
del MIP_X

os.environ.update({"DEEPCELL_ACCESS_TOKEN":  "dTjvtWDJ.SAO5zQC7LAB1hDl5EHlaAO5lF4eOfOhC"})

app = Mesmer()
new_model = app.model

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
    imagedask_np = np.moveaxis(imagedask_np,0,-1)
    if input_shape[1] < 256:
        imagedask_np = cv2.copyMakeBorder(imagedask_np,0,(256 - input_shape[1]),0,0,cv2.BORDER_CONSTANT, None, value = 0)
    if input_shape[2] < 256:
        imagedask_np = cv2.copyMakeBorder(imagedask_np,0,0,0,(256 - input_shape[2]),cv2.BORDER_CONSTANT, None, value = 0)
   # imagedask_np = transform.resize(imagedask_np,(256,256,8))
    if imagedask_np.shape[0] != 256 or imagedask_np.shape[1] != 256:
        print(imagedask_np.shape)
    output = new_model(np.expand_dims(imagedask_np,axis=0))
    
    mask = deep_watershed([output[0],output[1][:,:,:,1:2]],
                   radius=4,
                   maxima_threshold=0.4,
                   interior_threshold=0.6,
                   maxima_smooth=0,
                   interior_smooth=1,
                   maxima_index=0,
                   interior_index=1,
                   label_erosion=0,
                   small_objects_threshold=0,
                   fill_holes_threshold=0,
                   pixel_expansion=0,
                   maxima_algorithm='h_maxima',)

    return np.squeeze(mask[0])



im = da.from_array(im,chunks=(2,156,156))
block_labelled = relabel.relabeling.image2labels(im,seg_fun)

arr = np.array(block_labelled)
import tifffile
arr,  _ , _ = skimage.segmentation.relabel_sequential(arr)
tifffile.imwrite(os.path.join("../output/",BASE,"nuclear_{COND}{NUMBER}{STAGE}.tif"))