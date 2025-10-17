#MUST BE RUN IN QUBVEL LD_LIBRARY_PATH=/media/redmondlab3/Onions/ryan/resnetenv/lib/python3.11/site-packages/nvidia/cudnn/lib
import numpy as np
import pickle
import cv2
import os
import gc
import sys
import subprocess
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
import numpy as np 
import tensorflow as tf
import pickle
import tensorflow.keras as keras
from tensorflow.keras import layers
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense

IMG_SHAPE = (128,128)
NUM_CLASSES = 2
NUM_CHANNELS = 6

from classification_models.keras import Classifiers

import imageio
import json
import timeit


import tifffile
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
image_clahe = cv2.createCLAHE(clipLimit=.01, tileGridSize=(8,8))


import argparse
parser = argparse.ArgumentParser(description='GetFile')
parser.add_argument('--file', action="store", dest='file', default="")
parser.add_argument('--cond', action="store", dest='cond', default="")
parser.add_argument('--number', action="store", dest='number', default="")
parser.add_argument('--stage', action="store", dest='stage', default="")
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

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()


full_im = imageio.imread(FILE)
full_im = np.moveaxis(full_im,0,-1)

for c in range(full_im.shape[2]):
    full_im[:,:,c] = ((full_im[:,:,c] - full_im[:,:,c].min()) / (full_im[:,:,c].max() - full_im[:,:,c].min()))*255



MASK_FILE =  os.path.join("../output",BASE,f"wholecell_{COND}{NUMBER}{STAGE}.tif")
full_mask = tifffile.imread(MASK_FILE)
full_mask = full_mask[50:,50:]
SHAPE = full_im.shape

NUC_MASK_FILE =  os.path.join("../output",BASE,f"nuclear_{COND}{NUMBER}{STAGE}.tif")
nuc_mask = tifffile.imread(NUC_MASK_FILE)
nuc_mask = nuc_mask[50:,50:]



yrange = [*range(0,full_mask.shape[0],256)]
xrange = [*range(0,full_mask.shape[1],256)]
rangedict = dict()
for i in range(len(yrange)-1):
    for j in range(len(xrange)-1):
        these = np.unique(full_mask[yrange[i]:yrange[i+1],xrange[j]:xrange[j+1]])
        if len(these) > 1:
            rangedict[(yrange[i],yrange[i+1],xrange[j],xrange[j+1])] = these
    nexthese = np.unique(full_mask[yrange[i]:yrange[i+1],xrange[-1]:])    
    if len(nexthese) > 1:
            rangedict[(yrange[i],yrange[i+1],xrange[-1],-1)] = nexthese
for j in range(len(xrange)-1):
    nexthese =  np.unique(full_mask[yrange[-1]:,xrange[j]:xrange[j+1]])
    if len(nexthese) > 1:
          rangedict[(yrange[-1],-1,xrange[j],xrange[j+1])] = nexthese
endthese = np.unique(full_mask[yrange[-1]:,xrange[-1]:])
if len(endthese) > 1:
      rangedict[(yrange[-1],-1,xrange[-1],-1)] = endthese


celldict = dict()
for rng in rangedict.keys():
    for cell in rangedict[rng]:
        celldict[cell] = rng

nl = []
for i, k in celldict.items():
    nl.append(k)

patch_list = []
for tup in nl:
    if tup not in patch_list:
        patch_list.append(tup)



def get_model():
    ResNet18, preprocess_input = Classifiers.get('resnet18')
    base_model = ResNet18(input_shape=(128, 128, 3), weights="imagenet",include_top=False)
    base_model.trainable = True
    x = keras.layers.GlobalAveragePooling2D()(base_model.output)
    output = keras.layers.Dense(1, activation='sigmoid')(x)
    model = keras.models.Model(inputs=[base_model.input], outputs=[output])
    model.compile(
        optimizer=keras.optimizers.Adam(lr=1e-5, clipnorm=0.001),
        loss='binary_crossentropy'
    )
    model_path = os.path.join("models","MembraneModel.h5")
    model.load_weights(model_path)
    return model, preprocess_input 


def create_composite(images, colors):
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

def createbounding(channel,centroid,newmaskpatch,newimpatch,cell,retbounds=False):
    xmin = int(np.round(centroid[0]-32))
    xmax = int(np.round(centroid[0]+32))
    ymin = int(np.round(centroid[1]-32))
    ymax = int(np.round(centroid[1]+32))

    bxmin = max(0,xmin)
    bxmax = min(SHAPE[1],xmax)
    bymin = max(0,ymin)
    bymax = min(SHAPE[0],ymax)


    patch = (newmaskpatch == cell).astype(np.uint8)
    channelpatch = newimpatch[:,:,channel]
    dapipatch = newimpatch[:,:,0]
    if ymin < 0:
        patch = cv2.copyMakeBorder(patch,(0 - ymin),0,0,0,cv2.BORDER_CONSTANT, None, value = 0)
        channelpatch = cv2.copyMakeBorder(channelpatch,(0 - ymin),0,0,0,cv2.BORDER_CONSTANT, None, value = 0)
        dapipatch = cv2.copyMakeBorder(dapipatch,(0 - ymin),0,0,0,cv2.BORDER_CONSTANT, None, value = 0)
    if xmin < 0:
        patch = cv2.copyMakeBorder(patch,0,0,(0 - xmin),0,cv2.BORDER_CONSTANT, None, value = 0)
        channelpatch = cv2.copyMakeBorder(channelpatch,0,0,(0 - xmin),0,cv2.BORDER_CONSTANT, None, value = 0)
        dapipatch = cv2.copyMakeBorder(dapipatch,0,0,(0 - xmin),0,cv2.BORDER_CONSTANT, None, value = 0)
    if xmax > SHAPE[1]:
        patch = cv2.copyMakeBorder(patch,0,0,0,(xmax - SHAPE[1]),cv2.BORDER_CONSTANT, None, value = 0)
        channelpatch = cv2.copyMakeBorder(channelpatch,0,0,0,(xmax - SHAPE[1]),cv2.BORDER_CONSTANT, None, value = 0)
        dapipatch = cv2.copyMakeBorder(dapipatch,0,0,0,(xmax - SHAPE[1]),cv2.BORDER_CONSTANT, None, value = 0)
    if ymax > SHAPE[0]:
        patch = cv2.copyMakeBorder(patch,0,(ymax - SHAPE[0]),0,0,cv2.BORDER_CONSTANT, None, value = 0)
        channelpatch = cv2.copyMakeBorder(channelpatch,0,(ymax - SHAPE[0]),0,0,cv2.BORDER_CONSTANT, None, value = 0)
        dapipatch = cv2.copyMakeBorder(dapipatch,0,(ymax - SHAPE[0]),0,0,cv2.BORDER_CONSTANT, None, value = 0)
        
    example = np.zeros((128,128,3))
    example[:,:,0] = cv2.resize(patch,(128,128),cv2.INTER_NEAREST)     #SEG MASK
    example[:,:,1] = cv2.resize(dapipatch,(128,128))                   #DAPI
    example[:,:,2] = cv2.resize(channelpatch,(128,128))                #CHANNEL OF INTEREST
    orig_example = example.copy()

    #Normalize DAPI
    example[:,:,1] = image_clahe.apply(np.expand_dims(((scaler.fit_transform(example[:,:,1].reshape(-1, 1))*255)).reshape((128,128)).astype(np.uint8),-1))
    example[:,:,1] = example[:,:,1]*.75

    #Normalize channel of interest
    if channel != 5:
        example[:,:,2] = image_clahe.apply(np.expand_dims(example[:,:,2].astype(np.uint8),-1)) #CD34 channel normalize to high background signal
    else:
        example[:,:,2] = example[:,:,2]*1.25 #All other channels increase intensity

    composite_example = create_composite(example.transpose(2,0,1),colors=[(50,50,50),(0,0,1),(0,1,0)]).astype(np.uint8)

    if np.sum(patch) > 2000:   #Return whole crop for large cells such as Megakaryocytes
        return composite_example, orig_example
    else:
        composite_example = cv2.resize(composite_example[16:112,16:112,:],(128,128)) #Crop closer to centroid of cell
        return composite_example, orig_example #Return composite image for deep network and raw image for MFI
        


def get_mfi_vector(c,allchannelsexample):
   locs = np.where((allchannelsexample[0,:,:,0] == 1).astype(np.uint8))
   mfi_list = []
   mfi_list.append(np.mean(allchannelsexample[0,:,:,1][locs])) #DAPI
   for i in range(NUM_CHANNELS):
        pixls = allchannelsexample[i,:,:,2] #Each channel
        mfi_list.append(np.mean(pixls))

   return mfi_list

def get_nuclei_info(c,newmaskpatch,newnucpatch):
    nucdict = {}
    mask = (newmaskpatch  == c).astype(np.uint8)
    nucs = mask * newnucpatch
    nuc_elements = [i for i in np.unique(nucs) if i != 0]
    count = 0
    for n in nuc_elements:
        fullnuc = (newnucpatch == n).astype(np.uint8)
        thisnuc = (nucs == n).astype(np.uint8)
        pc_nuc = np.sum(thisnuc) / np.sum(fullnuc)
        cont , h = cv2.findContours(thisnuc,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if len(cont[0]) < 5:
            eccent = -1.
            area = np.sum(thisnuc)
            nucdict[count] = {"cell":n,"area":area,"eccentricity":eccent}
            count += 1
            continue
        (_, axs, _) = cv2.fitEllipse(cont[0])
        eccent = np.sqrt(1-(min(axs)/max(axs))**2)
        area = np.sum(thisnuc)
        nucdict[count] = {"cell":n,"area":area,"eccentricity":eccent,"pct":pc_nuc}
        count += 1
    return nucdict


model, preprocessing = get_model()

THIS_DIR = f"../tmp/{COND}{NUMBER}{STAGE}"
if not os.path.exists(f"{THIS_DIR}/tmp"):
    os.makedirs(f"{THIS_DIR}/tmp")


import time
for i,p in enumerate(patch_list):
    time.sleep(.05)
    print("Patch: ", i)
    rng = p
    if rng[1] == -1 and rng[3] == -1:
         submask = full_mask[(rng[0]-256):,(rng[2]-256):]
         subim = full_im[(rng[0]-256):,(rng[2]-256):,:]
         subnuc = nuc_mask[(rng[0]-256):,(rng[2]-256):]
    elif rng[1] == -1:
        submask = full_mask[(rng[0]-256):,(rng[2]-256):(rng[3]+256)]
        subim = full_im[(rng[0]-256):,(rng[2]-256):(rng[3]+256),:]
        subnuc = nuc_mask[(rng[0]-256):,(rng[2]-256):(rng[3]+256)]
    elif rng[3] == -1:
        submask = full_mask[(rng[0]-256):(rng[1]+256),(rng[2]-256):]
        subim = full_im[(rng[0]-256):(rng[1]+256),(rng[2]-256):,:]
        subnuc = nuc_mask[(rng[0]-256):(rng[1]+256),(rng[2]-256):]
    else:
        submask = full_mask[(rng[0]-256):(rng[1]+256),(rng[2]-256):(rng[3]+256)]
        subim = full_im[(rng[0]-256):(rng[1]+256),(rng[2]-256):(rng[3]+256),:]
        subnuc = nuc_mask[(rng[0]-256):(rng[1]+256),(rng[2]-256):(rng[3]+256)]


    cells = rangedict[rng]
    tempdict = dict()
    tempdict["ind"] = []
    tempdict["area"] = []
    tempdict["eccent"] = []
    tempdict["centroid"] = []
    tempdict["mfivector"] = []
    tempdict["cellvector"] = []
    tempdict["nucinfo"] = []

    batchexample = np.zeros((len(cells),NUM_CHANNELS,128,128,3))
    usecells = []
    usecount = 0
    for cnt, c in enumerate(cells):
        thiscellmask = (submask == c).astype(np.uint8)
        area = np.sum(thiscellmask)
        if area == 0:
            continue

        M = cv2.moments(thiscellmask)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        
        centroid = cX, cY
        truecentroid = (cX+rng[2]-256), (cY+rng[0]-256)
        xmin = int(np.round(centroid[0]-32))
        xmax = int(np.round(centroid[0]+32))
        ymin = int(np.round(centroid[1]-32))
        ymax = int(np.round(centroid[1]+32))

        bxmin = max(0,xmin)
        bxmax = min(SHAPE[1],xmax)
        bymin = max(0,ymin)
        bymax = min(SHAPE[0],ymax)
        newmaskpatch = submask[bymin:bymax,bxmin:bxmax]
        newimpatch = subim[bymin:bymax,bxmin:bxmax,:]
        newnucpatch = subnuc[bymin:bymax,bxmin:bxmax]

        allchannelsexample = np.zeros((NUM_CHANNELS,128,128,3))
        mfiexample = np.zeros((NUM_CHANNELS,128,128,3))

        for imchannel in range(NUM_CHANNELS):
            ex, orig = createbounding(imchannel+1,centroid,newmaskpatch,newimpatch,c)
            allchannelsexample[imchannel,:,:,:] = ex.copy()
            mfiexample[imchannel,:,:,:] = orig.copy()


        mfivector = get_mfi_vector(c,mfiexample)
        cont , h = cv2.findContours((thiscellmask == 1).astype(np.uint8),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if len(cont) < 1:
            eccent = -1
            continue
        if len(cont[0]) < 5:
            eccent = -1.
            continue
        (_, axs, _) = cv2.fitEllipse(cont[0])
        eccent = np.sqrt(1-(min(axs)/max(axs))**2)
        nucinfo = get_nuclei_info(c,newmaskpatch,newnucpatch)    


        tempdict["ind"].append(c)
        tempdict["area"].append(area)
        tempdict["eccent"].append(eccent)
        tempdict["centroid"].append(truecentroid)
        tempdict["mfivector"].append(mfivector)
        tempdict["nucinfo"].append(nucinfo)

        batchexample[usecount,:,:,:,:] = allchannelsexample
        usecells.append(c)
        usecount += 1

    modelscores = []
    for imchannel in range(NUM_CHANNELS):
        score = model(preprocessing(batchexample[:,imchannel,:,:,:])).numpy()
        modelscores.append(score)
    for cnt, c in enumerate(usecells):
        dictionary = {
            "cell": int(c),
            "area": tempdict["area"][cnt],
            "eccent": tempdict["eccent"][cnt],
            "centroid": tempdict["centroid"][cnt],
            "mfivector": tempdict["mfivector"][cnt],
            "cellvector" : [sc[cnt][0] for sc in modelscores],
            "nucinfo" : tempdict["nucinfo"][cnt]
        }  
        
        
        with open(f"{THIS_DIR}/tmp/cell_{c}.json", "w") as outfile:
            json.dump(dictionary,outfile,default=str)



