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

qdict = pickle.load(open("./qdict.pkl","rb"))
thirty = pickle.load(open("./nbm30qs.pkl","rb"))
qdict["nbm30"] = thirty
qdict["avgs_"] = {}
for i in range(8):
    qdict["avgs_"][i] = []
for k in qdict.keys():
    for i in range(8):
        qdict["avgs_"][i].append(qdict[k][i][-1])

qdict["avgs"] = {}
for i in range(8):
    qdict["avgs"][i] = np.mean(qdict["avgs_"][i])


maxdict = pickle.load(open("./maxdict.pkl","rb"))
maxdict["avgs_"] = {}
for i in range(8):
    maxdict["avgs_"][i] = []
for k in maxdict.keys():
    for i in range(8):
        maxdict["avgs_"][i].append(maxdict[k][i][0])
maxdict["avgs"] = {}
for i in range(8):
    maxdict["avgs"][i] = np.mean(maxdict["avgs_"][i])


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

def preprocess(this3,q,c):
    qval = 0
    if q == -1 and c == -1:
        qval = 10
    elif q == -1:
        qval = qdict["avgs"][c]
    else:
        for n, h in enumerate(qdict.keys()):
            if n == q:
                qval = qdict[h][c][-1]
    this3[:,:,1] = image_clahe.apply(np.expand_dims(((scaler.fit_transform(this3[:,:,1].reshape(-1, 1))*255)).reshape((128,128)).astype(np.uint8),-1))
    this3[:,:,2] = ((this3[:,:,2]/max(np.percentile(this3[:,:,2],99),qval)))*255# * (ratio)
    #this3[:,:,2] = (this3[:,:,2]/max(this3[:,:,2].max(),10))*255
    #this3[:,:,2] = np.clip(this3[:,:,2],np.percentile(this3[:,:,2],1),np.percentile(this3[:,:,2],99))
    #this3[:,:,2] = this3[:,:,2] - this3[:,:,2].mean()
    #this3[:,:,2] = np.clip(this3[:,:,2],np.percentile(this3[:,:,2],1),np.percentile(this3[:,:,2],98))
    #this3[:,:,2] = this3[:,:,2] - this3[:,:,2].mean()
    #this3[:,:,2] = np.tanh((1/(2*this3[:,:,2].mean()))*this3[:,:,2])*255 
    #this3[:,:,2] = np.clip(this3[:,:,2],0,this3[:,:,1].max())
    #this3[:,:,2] = unsharp_mask(this3[:,:,2].astype(np.uint8))
    return create_composite(this3.transpose(2,0,1),colors=[(50,50,50),(0,0,1),(0,1,0)]).astype(int)


def preprocess(this3,q,c,opt=1):
    qval = 0
    if q == -1 and c == -1:
        qval = 10
        this3[:,:,1] = image_clahe.apply(np.expand_dims(((scaler.fit_transform(this3[:,:,1].reshape(-1, 1))*255)).reshape((128,128)).astype(np.uint8),-1))
        this3[:,:,2] = ((this3[:,:,2]/max(np.percentile(this3[:,:,2],99),qval)))*255# * (ratio)
        this3[:,:,2] = image_clahe.apply(np.expand_dims(this3[:,:,2].astype(np.uint8),-1))
        return np.zeros_like(this3)
		#return create_composite(this3.transpose(2,0,1),colors=[(50,50,50),(0,0,1),(0,1,0)]).astype(int)
    elif q == -1:
        dmv =  maxdict["avgs"][0]
        mval = maxdict["avgs"][c]
        return np.zeros_like(this3)
    else:
        for n, h in enumerate(maxdict.keys()):
            if n == q:
                dmv = maxdict[h][0][0]
                mval = maxdict[h][c][0]
    if opt ==1:
        this3[:,:,1] = (this3[:,:,1])*255/dmv
    this3[:,:,1] = image_clahe.apply(np.expand_dims(((scaler.fit_transform(this3[:,:,1].reshape(-1, 1))*255)).reshape((128,128)).astype(np.uint8),-1))
    if opt == 1:
        this3[:,:,2] = (this3[:,:,2])*255/mval
    this3[:,:,1] = this3[:,:,1]*.75
    if c != 5:
        this3[:,:,2] = image_clahe.apply(np.expand_dims(this3[:,:,2].astype(np.uint8),-1))
    else:
        this3[:,:,2] = this3[:,:,2]*1.25
    #this3[:,:,2] = ((this3[:,:,2]/max(np.percentile(this3[:,:,2],99),qval)))*255# * (ratio)
    #this3[:,:,2] = (this3[:,:,2]/max(this3[:,:,2].max(),10))*255
    #this3[:,:,2] = np.clip(this3[:,:,2],np.percentile(this3[:,:,2],1),np.percentile(this3[:,:,2],99))
    #this3[:,:,2] = this3[:,:,2] - this3[:,:,2].mean()
    #this3[:,:,2] = np.clip(this3[:,:,2],np.percentile(this3[:,:,2],1),np.percentile(this3[:,:,2],98))
    #this3[:,:,2] = this3[:,:,2] - this3[:,:,2].mean()
    #this3[:,:,2] = np.tanh((1/(2*this3[:,:,2].mean()))*this3[:,:,2])*255 
    #this3[:,:,2] = np.clip(this3[:,:,2],0,this3[:,:,1].max())
    #this3[:,:,2] = unsharp_mask(this3[:,:,2].astype(np.uint8))
    return create_composite(this3.transpose(2,0,1),colors=[(50,50,50),(0,0,1),(0,1,0)]).astype(int)
qdict = maxdict

###TEST
import numpy as np
import pickle
import cv2
alldata_out = []
alldata_zoom = []
alldata_y = []
qdata = []
labels1 = pickle.load(open("./X5/savelabels0_255.pkl","rb"))
data1 = pickle.load(open("./X5/savedata0_255.pkl","rb"))
labels2 = pickle.load(open("./X5/savelabels1_238.pkl","rb"))
data2 = pickle.load(open("./X5/savedata1_238.pkl","rb"))
labels3 = pickle.load(open("./X5/savelabels2_210.pkl","rb"))
data3 = pickle.load(open("./X5/savedata2_210.pkl","rb"))
labels4 = pickle.load(open("./X5/savelabels3_164.pkl","rb"))
data4 = pickle.load(open("./X5/savedata3_164.pkl","rb"))
labels5 = pickle.load(open("./X5/savelabels4_152.pkl","rb"))
data5 = pickle.load(open("./X5/savedata4_152.pkl","rb"))
labels6 = pickle.load(open("./X5/savelabels5_169.pkl","rb"))
data6 = pickle.load(open("./X5/savedata5_169.pkl","rb"))
labels7 = pickle.load(open("./X5/savelabels6_120.pkl","rb"))
data7 = pickle.load(open("./X5/savedata6_120.pkl","rb"))
labels8 = pickle.load(open("./X5/savelabels7_16.pkl","rb"))
data8 = pickle.load(open("./X5/savedata7_16.pkl","rb"))



olabels1 = pickle.load(open("./X5/savelabels11_20.pkl","rb"))
odata1 = pickle.load(open("./X5/savedata11_20.pkl","rb"))
olabels2 = pickle.load(open("./X5/savelabels10_26.pkl","rb"))
odata2 = pickle.load(open("./X5/savedata10_26.pkl","rb"))
olabels3 = pickle.load(open("./X5/savelabels9_25.pkl","rb"))
odata3 = pickle.load(open("./X5/savedata9_25.pkl","rb"))
olabels4 = pickle.load(open("./X5/savelabels8_25.pkl","rb"))
odata4 = pickle.load(open("./X5/savedata8_25.pkl","rb"))
olabels5 = pickle.load(open("./X5/savelabels12_64.pkl","rb"))
odata5 = pickle.load(open("./X5/savedata12_64.pkl","rb"))
olabels6 = pickle.load(open("./X5/savelabels13_42.pkl","rb"))
odata6 = pickle.load(open("./X5/savedata13_42.pkl","rb"))
np.random.seed(123)
for i in range(len(labels1)):
    newexample = data1[i][:,:,0:3]
    newexample_zoom = data1[i][:,:,3:6]


    alldata_out.append(newexample)
    alldata_zoom.append(newexample_zoom)
    alldata_y.append(labels1[i])

for j in range(len(labels2)):
    newexample = data2[j][:,:,0:3]
    newexample_zoom = data2[j][:,:,3:6]

    alldata_out.append(newexample)
    alldata_zoom.append(newexample_zoom)
    
    alldata_y.append(labels2[j])

for j in range(len(labels3)):
    newexample = data3[j][:,:,0:3]
    newexample_zoom = data3[j][:,:,3:6]

    alldata_out.append(newexample)
    alldata_zoom.append(newexample_zoom)
    
    alldata_y.append(labels3[j])

for j in range(len(labels4)):
    newexample = data4[j][:,:,0:3]
    newexample_zoom = data4[j][:,:,3:6]

    alldata_out.append(newexample)
    alldata_zoom.append(newexample_zoom)
    
    alldata_y.append(labels4[j])


for j in range(len(labels5)):
    newexample = data5[j][:,:,0:3]
    newexample_zoom = data5[j][:,:,3:6]

    alldata_out.append(newexample)
    alldata_zoom.append(newexample_zoom)
    
    alldata_y.append(labels5[j])

for j in range(len(labels6)):
    newexample = data6[j][:,:,0:3]
    newexample_zoom = data6[j][:,:,3:6]

    alldata_out.append(newexample)
    alldata_zoom.append(newexample_zoom)
    
    alldata_y.append(labels6[j])

for j in range(len(labels7)):
    newexample = data7[j][:,:,0:3]
    newexample_zoom = data7[j][:,:,3:6]

    alldata_out.append(newexample)
    alldata_zoom.append(newexample_zoom)
    
    alldata_y.append(labels7[j])

for j in range(len(labels8)):
    newexample = data8[j][:,:,0:3]
    newexample_zoom = data8[j][:,:,3:6]

    alldata_out.append(newexample)
    alldata_zoom.append(newexample_zoom)
    
    alldata_y.append(labels8[j])


for i in range(len(olabels1)):
    newexample = odata1[i][:,:,0:3]
    newexample_zoom = odata1[i][:,:,3:6]


    alldata_out.append(newexample)
    alldata_zoom.append(newexample_zoom)
    alldata_y.append(olabels1[i])

for j in range(len(olabels2)):
    newexample = odata2[j][:,:,0:3]
    newexample_zoom = odata2[j][:,:,3:6]

    alldata_out.append(newexample)
    alldata_zoom.append(newexample_zoom)
    
    alldata_y.append(olabels2[j])

for j in range(len(olabels3)):
    newexample = odata3[j][:,:,0:3]
    newexample_zoom = odata3[j][:,:,3:6]

    alldata_out.append(newexample)
    alldata_zoom.append(newexample_zoom)
    
    alldata_y.append(olabels3[j])

for j in range(len(olabels4)):
    newexample = odata4[j][:,:,0:3]
    newexample_zoom = odata4[j][:,:,3:6]

    alldata_out.append(newexample)
    alldata_zoom.append(newexample_zoom)
    
    alldata_y.append(olabels4[j])
    

for j in range(len(olabels5)):
    newexample = odata5[j][:,:,0:3]
    newexample_zoom = odata5[j][:,:,3:6]

    alldata_out.append(newexample)
    alldata_zoom.append(newexample_zoom)
    
    alldata_y.append(olabels5[j])

for j in range(len(olabels6)):
    newexample = odata6[j][:,:,0:3]
    newexample_zoom = odata6[j][:,:,3:6]

    alldata_out.append(newexample)
    alldata_zoom.append(newexample_zoom)
    
    alldata_y.append(olabels6[j])
#RESTORING ODATA7

import numpy as np
_ = np.array(alldata_out)
X = np.array(alldata_zoom)
Y = np.array(alldata_y)

newX = []
newY = []
removes = pickle.load(open("./removes1088.pkl","rb"))
removescomp = pickle.load(open("./removes_comp1088.pkl","rb"))
removessep = pickle.load(open("./removes_sep1088.pkl","rb"))
removesnext = [1101, 1109, 1116, 1201, 1255, 1358]
for i in range(len(X)):
    if i not in removes and i not in removessep and i not in removesnext:
        newX.append(X[i])
        newY.append(Y[i])

newX = np.array(newX)
newY = np.array(newY)



sizes = []
for i in range(len(newX)):
    if np.sum(newX[i,:,:,0]) > 2000:
        sizes.append(1)
    else:
        sizes.append(0)

def getval(img):
    if img[:,:,2].mean() > 10 and img[:,:,2].max() > 40:
        return 2
    elif img[:,:,2].mean() > 5:
        return 5
    elif img[:,:,2].mean() > 3:
        return 6
    elif img[:,:,2].mean() > 1:
        return 7
    else:
        return 8
    
import numpy as np

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

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()



newnewX = np.zeros_like(newX)
#CHANGING NORMALIZATION ON CHANNEL 2 from max10 to 1
for i in range(newX.shape[0]):
    this = preprocess(newX[i].copy(),-1,-1)
    #this3[:,:,1] = ((scaler.fit_transform(this3[:,:,1].reshape(-1, 1))*255)).reshape((128,128)).astype(np.uint8)
    #newestretex[:,:,2] = np.expand_dims(((newestretex[:,:,2]/max(newestretex[:,:,2].max(),10))*255),-1)
    #this3[:,:,2] = image_clahe.apply(np.expand_dims(((scaler.fit_transform(this3[:,:,2].reshape(-1, 1))*255)).reshape((128,128)).astype(np.uint8),-1))
    newnewX[i,:,:,:] = this#create_composite(this.transpose(2,0,1),colors=[(50,50,50),(0,0,1),(0,1,0)]).astype(int)

X = newnewX
Y = newY


removes = pickle.load(open("./removes_nov24.pkl","rb"))
firstsizes = []
newX = []
newY = []

for i in range(len(X)):
    if i not in removes:
        newX.append(X[i])
        newY.append(Y[i])
        firstsizes.append(sizes[i])

X = np.array(newX)
Y = np.array(newY)

newsizes = []

#NEW CD34
newcd34X = []
newy = []
import os
import pickle
for f in os.listdir("./newest_dataset_sep/CD34/pos"):
    rd = pickle.load(open(os.path.join("./newest_dataset_sep/CD34/pos",f),"rb"))
    newcd34X.append(rd)
    newy.append(1)
    newsizes.append(np.sum(rd[:,:,0]) > 2000)
for f in os.listdir("./newest_dataset_sep/CD34/neg"):
    rd = pickle.load(open(os.path.join("./newest_dataset_sep/CD34/neg",f),"rb"))
    newcd34X.append(rd)
    newy.append(0)
    newsizes.append(np.sum(rd[:,:,0]) > 2000)


newcd34X = np.array(newcd34X)
newcd34Y = np.array(newy)


newnewX = np.zeros_like(newcd34X)
#CHANGING NORMALIZATION ON CHANNEL 2 from max10 to 1
for i in range(newcd34X.shape[0]):
    this = preprocess(newcd34X[i].copy(),-1,5)
    #this3[:,:,1] = ((scaler.fit_transform(this3[:,:,1].reshape(-1, 1))*255)).reshape((128,128)).astype(np.uint8)
    #newestretex[:,:,2] = np.expand_dims(((newestretex[:,:,2]/max(newestretex[:,:,2].max(),10))*255),-1)
    #this3[:,:,2] = image_clahe.apply(np.expand_dims(((scaler.fit_transform(this3[:,:,2].reshape(-1, 1))*255)).reshape((128,128)).astype(np.uint8),-1))
    newnewX[i,:,:,:] = this#create_composite(this.transpose(2,0,1),colors=[(50,50,50),(0,0,1),(0,1,0)]).astype(int)


newaddcd34 = newnewX.copy()
newaddYcd34 = newcd34Y.copy()

#NEW CD117
newcd117X = []
newy = []
import os
import pickle
for f in os.listdir("./newest_dataset_sep/CD117/pos"):
    rd = pickle.load(open(os.path.join("./newest_dataset_sep/CD117/pos",f),"rb"))
    newcd117X.append(rd)
    newy.append(1)
    newsizes.append(np.sum(rd[:,:,0]) > 2000)
for f in os.listdir("./newest_dataset_sep/CD117/neg"):
    rd = pickle.load(open(os.path.join("./newest_dataset_sep/CD117/neg",f),"rb"))
    newcd117X.append(rd)
    newy.append(0)
    newsizes.append(np.sum(rd[:,:,0]) > 2000)


newcd117X = np.array(newcd117X)
newcd117Y = np.array(newy)


newnewX = np.zeros_like(newcd117X)
#CHANGING NORMALIZATION ON CHANNEL 2 from max10 to 1
for i in range(newcd117X.shape[0]):
    this = preprocess(newcd117X[i].copy(),-1,3)
    #this3[:,:,1] = ((scaler.fit_transform(this3[:,:,1].reshape(-1, 1))*255)).reshape((128,128)).astype(np.uint8)
    #newestretex[:,:,2] = np.expand_dims(((newestretex[:,:,2]/max(newestretex[:,:,2].max(),10))*255),-1)
    #this3[:,:,2] = image_clahe.apply(np.expand_dims(((scaler.fit_transform(this3[:,:,2].reshape(-1, 1))*255)).reshape((128,128)).astype(np.uint8),-1))
    newnewX[i,:,:,:] = this#create_composite(this.transpose(2,0,1),colors=[(50,50,50),(0,0,1),(0,1,0)]).astype(int)

newaddcd117 = newnewX.copy()
newaddYcd117 = newcd117Y.copy()

#NEW CD15
newcd15X = []
newy = []
import os
import pickle
for f in os.listdir("./newest_dataset_sep/CD15/pos"):
    rd = pickle.load(open(os.path.join("./newest_dataset_sep/CD15/pos",f),"rb"))
    newcd15X.append(rd)
    newy.append(1)
    newsizes.append(np.sum(rd[:,:,0]) > 2000)
for f in os.listdir("./newest_dataset_sep/CD15/neg"):
    rd = pickle.load(open(os.path.join("./newest_dataset_sep/CD15/neg",f),"rb"))
    newcd15X.append(rd)
    newy.append(0)
    newsizes.append(np.sum(rd[:,:,0]) > 2000)


newcd15X = np.array(newcd15X)
newcd15Y = np.array(newy)


newnewX = np.zeros_like(newcd15X)
#CHANGING NORMALIZATION ON CHANNEL 2 from max10 to 1
for i in range(newcd15X.shape[0]):
    this = preprocess(newcd15X[i].copy(),-1,6)
    #this3[:,:,1] = ((scaler.fit_transform(this3[:,:,1].reshape(-1, 1))*255)).reshape((128,128)).astype(np.uint8)
    #newestretex[:,:,2] = np.expand_dims(((newestretex[:,:,2]/max(newestretex[:,:,2].max(),10))*255),-1)
    #this3[:,:,2] = image_clahe.apply(np.expand_dims(((scaler.fit_transform(this3[:,:,2].reshape(-1, 1))*255)).reshape((128,128)).astype(np.uint8),-1))
    newnewX[i,:,:,:] = this#create_composite(this.transpose(2,0,1),colors=[(50,50,50),(0,0,1),(0,1,0)]).astype(int)

newaddcd15 = newnewX.copy()
newaddYcd15 = newcd15Y.copy()

X2 = np.concatenate((newaddcd34,newaddcd117,newaddcd15),axis=0)
Y2 = np.concatenate((newaddYcd34,newaddYcd117,newaddYcd15),axis=0)

newnewsizes = []
removes = [15,
 29,
 37,
 66,
 67,
 72,
 92,
 93,
 112,
 118,
 127,
 131,
 147,
 161,
 178,
 183,
 199,
 203,
 204,
 205,
 217,
 220,
 232,
 236,
 239,
 245,
 249,
 259,
 263,
 264,
 265,
 268,
 284,
 311,
 319,
 378,
 380,
 382]
newX = []
newY = []
newsizes2 = []
for i in range(len(X2)):
    if i not in removes:
        newX.append(X2[i])
        newY.append(Y2[i])
        newsizes2.append(newsizes[i])

X2 = np.array(newX)
Y2 = np.array(newY)

newsizes = []
#NEW CD61
newcd61X = []
newy = []
import os
import pickle
for f in os.listdir("./newest_dataset_sep/CD61/pos"):
    rd = pickle.load(open(os.path.join("./newest_dataset_sep/CD61/pos",f),"rb"))
    newcd61X.append(rd)
    newy.append(1)
    newsizes.append(np.sum(rd[:,:,0]) > 2000)
for f in os.listdir("./newest_dataset_sep/CD61/neg"):
    rd = pickle.load(open(os.path.join("./newest_dataset_sep/CD61/neg",f),"rb"))
    newcd61X.append(rd)
    newy.append(0)
    newsizes.append(np.sum(rd[:,:,0]) > 2000)


newcd61X = np.array(newcd61X)
newcd61Y = np.array(newy)


newnewX = np.zeros_like(newcd61X)
#CHANGING NORMALIZATION ON CHANNEL 2 from max10 to 1
for i in range(newcd61X.shape[0]):
    this = preprocess(newcd61X[i].copy(),-1,2)
    #this3[:,:,1] = ((scaler.fit_transform(this3[:,:,1].reshape(-1, 1))*255)).reshape((128,128)).astype(np.uint8)
    #newestretex[:,:,2] = np.expand_dims(((newestretex[:,:,2]/max(newestretex[:,:,2].max(),10))*255),-1)
    #this3[:,:,2] = image_clahe.apply(np.expand_dims(((scaler.fit_transform(this3[:,:,2].reshape(-1, 1))*255)).reshape((128,128)).astype(np.uint8),-1))
    newnewX[i,:,:,:] = this#create_composite(this.transpose(2,0,1),colors=[(50,50,50),(0,0,1),(0,1,0)]).astype(int)

newaddcd61 = newnewX.copy()
newaddYcd61 = newcd61Y.copy()
newsizes3 = newsizes2 + newsizes
X2 = np.concatenate((X2,newaddcd61),axis=0)
Y2 = np.concatenate((Y2,newaddYcd61),axis=0)


newestsizes = firstsizes + newsizes3
X = np.concatenate((X,X2),axis=0)
Y = np.concatenate((Y,Y2),axis=0)
removes2 = [32, 53, 81, 107, 165, 239, 345]
newX = []
newY = []
thesesizes = []
for i in range(len(X)):
    if i not in removes2:
        newX.append(X[i])
        newY.append(Y[i])
        thesesizes.append(newestsizes[i])

X = np.array(newX)
Y = np.array(newY)

thisX = []
if len(thesesizes) != len(X):
    print("bad")
for i in range(len(X)):
    if thesesizes[i] == 0:
        small = X[i][16:112,16:112,:]
        thisX.append(cv2.resize(small,(128,128)))
    else:
        thisX.append(X[i])

X = np.array(thisX)    #For small

qdata2 = []
#NEW CD34
newcd34X = []
newy = []
newsizes = []
import os
import pickle
for f in os.listdir("./newest_dataset2/CD34/pos"):
    rd = pickle.load(open(os.path.join("./newest_dataset_sep2/CD34/pos",f),"rb"))
    newcd34X.append(rd)
    newy.append(1)
    newsizes.append(0)
    fval = f.split("_")[-2]
    qdata2.append(fval)
for f in os.listdir("./newest_dataset2/CD34/neg"):
    rd = pickle.load(open(os.path.join("./newest_dataset_sep2/CD34/neg",f),"rb"))
    newcd34X.append(rd)
    newy.append(0)
    newsizes.append(0)
    fval = f.split("_")[-2]
    qdata2.append(fval)


newcd34X = np.array(newcd34X)
newcd34Y = np.array(newy)


newnewX = np.zeros_like(newcd34X)
#CHANGING NORMALIZATION ON CHANNEL 2 from max10 to 1
for i in range(newcd34X.shape[0]):
    if qdata2[i] not in qdict.keys():
        this = preprocess(newcd34X[i].copy(),-1,5)
    else:
        total_this_count += 1
        this = preprocess(newcd34X[i].copy(),[n for n,h in enumerate(qdict.keys()) if qdata2[i] in h][0],5)
    #this3[:,:,1] = ((scaler.fit_transform(this3[:,:,1].reshape(-1, 1))*255)).reshape((128,128)).astype(np.uint8)
    #newestretex[:,:,2] = np.expand_dims(((newestretex[:,:,2]/max(newestretex[:,:,2].max(),10))*255),-1)
    #this3[:,:,2] = image_clahe.apply(np.expand_dims(((scaler.fit_transform(this3[:,:,2].reshape(-1, 1))*255)).reshape((128,128)).astype(np.uint8),-1))
    newnewX[i,:,:,:] = this#create_composite(this.transpose(2,0,1),colors=[(50,50,50),(0,0,1),(0,1,0)]).astype(int)


newcd34X = newnewX.copy()
#newaddYcd34 = newcd34Y.copy()
qdata2 = []
#NEW CD117
newcd117X = []
newy = []
import os
import pickle
for f in os.listdir("./newest_dataset2/CD117/pos"):
    rd = pickle.load(open(os.path.join("./newest_dataset_sep2/CD117/pos",f),"rb"))
    newcd117X.append(rd)
    newy.append(1)
    newsizes.append(0)
    fval = f.split("_")[-2]
    qdata2.append(fval)
for f in os.listdir("./newest_dataset2/CD117/neg"):
    rd = pickle.load(open(os.path.join("./newest_dataset_sep2/CD117/neg",f),"rb"))
    newcd117X.append(rd)
    newy.append(0)
    newsizes.append(0)
    fval = f.split("_")[-2]
    qdata2.append(fval)

newcd117X = np.array(newcd117X)
newcd117Y = np.array(newy)


newnewX = np.zeros_like(newcd117X)
#CHANGING NORMALIZATION ON CHANNEL 2 from max10 to 1
for i in range(newcd117X.shape[0]):
    if qdata2[i] not in qdict.keys():
        this = preprocess(newcd117X[i].copy(),-1,3)
    else:
        total_this_count += 1
        this = preprocess(newcd117X[i].copy(),[n for n,h in enumerate(qdict.keys()) if qdata2[i] in h][0],3)
    #this3[:,:,1] = ((scaler.fit_transform(this3[:,:,1].reshape(-1, 1))*255)).reshape((128,128)).astype(np.uint8)
    #newestretex[:,:,2] = np.expand_dims(((newestretex[:,:,2]/max(newestretex[:,:,2].max(),10))*255),-1)
    #this3[:,:,2] = image_clahe.apply(np.expand_dims(((scaler.fit_transform(this3[:,:,2].reshape(-1, 1))*255)).reshape((128,128)).astype(np.uint8),-1))
    newnewX[i,:,:,:] = this#create_composite(this.transpose(2,0,1),colors=[(50,50,50),(0,0,1),(0,1,0)]).astype(int)

newcd117X = newnewX.copy()
#ewaddYcd117 = newcd117Y.copy()



qdata2 = []
#NEW CD38
newcd38X = []
newy = []
import os
import pickle
for f in os.listdir("./newest_dataset2/CD38/pos"):
    rd = pickle.load(open(os.path.join("./newest_dataset_sep2/CD38/pos",f),"rb"))
    newcd38X.append(rd)
    newy.append(1)
    newsizes.append(0)
    fval = f.split("_")[-2]
    qdata2.append(fval)
for f in os.listdir("./newest_dataset2/CD38/neg"):
    rd = pickle.load(open(os.path.join("./newest_dataset_sep2/CD38/neg",f),"rb"))
    newcd38X.append(rd)
    newy.append(0)
    newsizes.append(0)
    fval = f.split("_")[-2]
    qdata2.append(fval)


newcd38X = np.array(newcd38X)
newcd38Y = np.array(newy)


newnewX = np.zeros_like(newcd38X)
#CHANGING NORMALIZATION ON CHANNEL 2 from max10 to 1
for i in range(newcd38X.shape[0]):
    if qdata2[i] not in qdict.keys():
        this = preprocess(newcd38X[i].copy(),-1,4)
    else:
        total_this_count += 1
        this = preprocess(newcd38X[i].copy(),[n for n,h in enumerate(qdict.keys()) if qdata2[i] in h][0],4)
    #this3[:,:,1] = ((scaler.fit_transform(this3[:,:,1].reshape(-1, 1))*255)).reshape((128,128)).astype(np.uint8)
    #newestretex[:,:,2] = np.expand_dims(((newestretex[:,:,2]/max(newestretex[:,:,2].max(),10))*255),-1)
    #this3[:,:,2] = image_clahe.apply(np.expand_dims(((scaler.fit_transform(this3[:,:,2].reshape(-1, 1))*255)).reshape((128,128)).astype(np.uint8),-1))
    newnewX[i,:,:,:] = this#create_composite(this.transpose(2,0,1),colors=[(50,50,50),(0,0,1),(0,1,0)]).astype(int)

newcd38X = newnewX.copy()
#newaddYcd15 = newcd15Y.copy()


X2 = np.concatenate((newcd34X,newcd117X,newcd38X),axis=0)
Y2 = np.concatenate((newcd34Y,newcd117Y,newcd38Y),axis=0)




newsizes3 = thesesizes + newsizes
#X2 = np.concatenate((X2,newcd61X),axis=0)
#Y2 = np.concatenate((Y2,newcd61Y),axis=0)

from sklearn.utils import shuffle
X = np.concatenate((X,X2),axis=0)
Y = np.concatenate((Y,Y2),axis=0)

thisX = []
if len(newsizes3) != len(X):
    print("bad")
for i in range(len(X)):
    this = X[i]#cv2.filter2D(X[i].astype(np.uint8),-1,kernel)
    if newsizes3[i] == 0:
        small = this[16:112,16:112,:]
        thisX.append(cv2.resize(small,(128,128)))
    else:
        thisX.append(this)


from sklearn.utils import shuffle
X = np.array(thisX)

removes = [349,
 455,
 466,
 505,
 512,
 527,
 533,
 551,
 556,
 575,
 777,
 903,
 1069,
 1088,
 1089,
 1179,
 1198,
 1259,
 1275,
 1286,
 1288,
 1516,
 1519,
 1520,
 1527,
 1530,
 1533,
 1535,
 1538,
 1545,
 1555,
 1558,
 1560,
 1564,
 1565,
 1568,
 1570,
 1574,
 1665,
 1676,
 1728,
 1760,
 1761,
 1780,
 1782,
 1800,
 1821,
 2002,
 2003,
 2007,
 2011,
 2017,
 2018,
 2028,
 2034,
 2038,
 2040,
 2050,
 2052,
 2055,
 2056,
 2058,
 2063,
 2070,
 2072,
 2077,
 2078,
 2078,
 2079,
 2090,
 2113,
 2122,
 2123,
 2124,
 2125,
 2126,
 2128,
 2132,
 2138,
 2144,
 2145,
 2155,
 2159,
 2170,
 2171,
 2175,
 2176,
 2181,
 2184,
 2186,
 2188,
 2198,
 2201,
 2361,
 2380,
 2391,
 2394,
 2412]
thisnewestX = []
thisnewestY = []
for i in range(len(X)):
    if i not in removes:
        thisnewestX.append(X[i])
        thisnewestY.append(Y[i])
X = np.array(thisnewestX)
Y = np.array(thisnewestY)





qdata2 = []
#NEW CD38
newcd71X = []
newy = []
import os
import pickle
for f in os.listdir("./newest_dataset_sep3/CD71/pos"):
    rd = pickle.load(open(os.path.join("./newest_dataset_sep3/CD71/pos",f),"rb"))
    newcd71X.append(rd)
    newy.append(1)
    newsizes.append(0)
    fval = f.split("_")[-2]
    qdata2.append(fval)
for f in os.listdir("./newest_dataset_sep3/CD71/neg"):
    rd = pickle.load(open(os.path.join("./newest_dataset_sep3/CD71/neg",f),"rb"))
    newcd71X.append(rd)
    newy.append(0)
    newsizes.append(0)
    fval = f.split("_")[-2]
    qdata2.append(fval)


newcd71X = np.array(newcd71X)
newcd71Y = np.array(newy)


newnewX = np.zeros_like(newcd71X)
#CHANGING NORMALIZATION ON CHANNEL 2 from max10 to 1
for i in range(newcd71X.shape[0]):
    if qdata2[i] not in qdict.keys():
        this = preprocess(newcd71X[i].copy(),-1,4,0)
    else:
        total_this_count += 1
        this = preprocess(newcd71X[i].copy(),[n for n,h in enumerate(qdict.keys()) if qdata2[i] in h][0],1,0)
    #this3[:,:,1] = ((scaler.fit_transform(this3[:,:,1].reshape(-1, 1))*255)).reshape((128,128)).astype(np.uint8)
    #newestretex[:,:,2] = np.expand_dims(((newestretex[:,:,2]/max(newestretex[:,:,2].max(),10))*255),-1)
    #this3[:,:,2] = image_clahe.apply(np.expand_dims(((scaler.fit_transform(this3[:,:,2].reshape(-1, 1))*255)).reshape((128,128)).astype(np.uint8),-1))
    newnewX[i,:,:,:] = this#create_composite(this.transpose(2,0,1),colors=[(50,50,50),(0,0,1),(0,1,0)]).astype(int)

newcd71X = newnewX.copy()
#newaddYcd15 = newcd15Y.copy()




qdata2 = []
#NEW CD38
newcd15X = []
newy = []
import os
import pickle
for f in os.listdir("./newest_dataset_sep3/CD15/pos"):
    rd = pickle.load(open(os.path.join("./newest_dataset_sep3/CD15/pos",f),"rb"))
    newcd15X.append(rd)
    newy.append(1)
    newsizes.append(0)
    fval = f.split("_")[-2]
    qdata2.append(fval)
for f in os.listdir("./newest_dataset_sep3/CD15/neg"):
    rd = pickle.load(open(os.path.join("./newest_dataset_sep3/CD15/neg",f),"rb"))
    newcd15X.append(rd)
    newy.append(0)
    newsizes.append(0)
    fval = f.split("_")[-2]
    qdata2.append(fval)


newcd15X = np.array(newcd15X)
newcd15Y = np.array(newy)


newnewX = np.zeros_like(newcd15X)
#CHANGING NORMALIZATION ON CHANNEL 2 from max10 to 1
for i in range(newcd15X.shape[0]):
    if qdata2[i] not in qdict.keys():
        this = preprocess(newcd71X[i].copy(),-1,4,0)
    else:
        total_this_count += 1
        this = preprocess(newcd15X[i].copy(),[n for n,h in enumerate(qdict.keys()) if qdata2[i] in h][0],6,0)
    #this3[:,:,1] = ((scaler.fit_transform(this3[:,:,1].reshape(-1, 1))*255)).reshape((128,128)).astype(np.uint8)
    #newestretex[:,:,2] = np.expand_dims(((newestretex[:,:,2]/max(newestretex[:,:,2].max(),10))*255),-1)
    #this3[:,:,2] = image_clahe.apply(np.expand_dims(((scaler.fit_transform(this3[:,:,2].reshape(-1, 1))*255)).reshape((128,128)).astype(np.uint8),-1))
    newnewX[i,:,:,:] = this#create_composite(this.transpose(2,0,1),colors=[(50,50,50),(0,0,1),(0,1,0)]).astype(int)

newcd15X = newnewX.copy()




qdata2 = []
#NEW CD38
newcd34X = []
newy = []
import os
import pickle
for f in os.listdir("./newest_dataset_sep3/CD34/pos"):
    rd = pickle.load(open(os.path.join("./newest_dataset_sep3/CD34/pos",f),"rb"))
    newcd34X.append(rd)
    newy.append(1)
    newsizes.append(0)
    fval = f.split("_")[-2]
    qdata2.append(fval)
for f in os.listdir("./newest_dataset_sep3/CD34/neg"):
    rd = pickle.load(open(os.path.join("./newest_dataset_sep3/CD34/neg",f),"rb"))
    newcd34X.append(rd)
    newy.append(0)
    newsizes.append(0)
    fval = f.split("_")[-2]
    qdata2.append(fval)


newcd34X = np.array(newcd34X)
newcd34Y = np.array(newy)


newnewX = np.zeros_like(newcd34X)
#CHANGING NORMALIZATION ON CHANNEL 2 from max10 to 1
for i in range(newcd34X.shape[0]):
    if qdata2[i] not in qdict.keys():
        this = preprocess(newcd34X[i].copy(),-1,4,0)
    else:
        total_this_count += 1
        this = preprocess(newcd34X[i].copy(),[n for n,h in enumerate(qdict.keys()) if qdata2[i] in h][0],5,0)
    #this3[:,:,1] = ((scaler.fit_transform(this3[:,:,1].reshape(-1, 1))*255)).reshape((128,128)).astype(np.uint8)
    #newestretex[:,:,2] = np.expand_dims(((newestretex[:,:,2]/max(newestretex[:,:,2].max(),10))*255),-1)
    #this3[:,:,2] = image_clahe.apply(np.expand_dims(((scaler.fit_transform(this3[:,:,2].reshape(-1, 1))*255)).reshape((128,128)).astype(np.uint8),-1))
    newnewX[i,:,:,:] = this#create_composite(this.transpose(2,0,1),colors=[(50,50,50),(0,0,1),(0,1,0)]).astype(int)

newcd34X = newnewX.copy()


X2 = np.concatenate((newcd34X,newcd71X,newcd15X),axis=0)
Y2 = np.concatenate((newcd34Y,newcd71Y,newcd15Y),axis=0)


X = np.concatenate((X,X2),axis=0)
Y = np.concatenate((Y,Y2),axis=0)


thisnewestX = []
thisnewestY = []
for i in range(len(X)):
    if np.sum(X[i]) != 0:
        thisnewestX.append(X[i])
        thisnewestY.append(Y[i])
X = np.array(thisnewestX)
Y = np.array(thisnewestY)
X, Y = shuffle(X, Y, random_state=2)
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
"""
import imblearn
ovr = imblearn.over_sampling.RandomOverSampler(sampling_strategy=.73)
newX, newY = ovr.fit_resample(np.array(range(X_trn.shape[0])).reshape(-1,1),Y_trn)
balX = np.zeros((len(newX),128,128,3))
baly = newY
for i in range(len(newX)):
    balX[i,:,:,:] = X_trn[newX[i]]
X_trn = balX
Y_trn = baly

undr = imblearn.under_sampling.RandomUnderSampler(sampling_strategy=.95)
newX, newY = undr.fit_resample(np.array(range(X_trn.shape[0])).reshape(-1,1),Y_trn)
balX = np.zeros((len(newX),128,128,3))
baly = newY
for i in range(len(newX)):
    balX[i,:,:,:] = X_trn[newX[i]]
X_trn = balX
Y_trn = baly

print('number of images: %3d' % (len(X_trn) + len(X_val)))
print('- training:       %3d' % len(X_trn))
print('- validation:     %3d' % len(X_val))
print('- positive training:       %3d' % np.sum(Y_trn))
print('- positive val:       %3d' % np.sum(Y_val))
"""

print('number of images: %3d' % (len(X_trn) + len(X_val)))
print('- training:       %3d' % len(X_trn))
print('- positive training:       %3d' % np.sum(Y_trn))
print('- validation:     %3d' % len(X_val))
print('- positive val:       %3d' % np.sum(Y_val))
print('- testing:     %3d' % len(X_test))
print('- positive test:       %3d' % np.sum(Y_test))
#######         Train           ##############
model, preprocess2 = get_model("ResNet18_Pool")

train_gen = ImageDataGenerator(horizontal_flip=True,vertical_flip=True,rotation_range=360,zoom_range=(.8,1.2),brightness_range=(.95,1.05),preprocessing_function=preprocess2)

val_gen  = ImageDataGenerator(preprocessing_function=preprocess2)
print(X_trn.shape)
print(Y_trn.shape)
print(X_val.shape)
print(Y_val.shape)
train_generator  = train_gen.flow(X_trn,Y_trn,batch_size=BATCH_SIZE)
val_generator = val_gen.flow(X_val,Y_val,batch_size=BATCH_SIZE)



checkpoint_filepath = '/media/redmondlab3/Onions/ryan/celltyping/checkpoints/round7_ptrdec21_2.h5'
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
        
model.load_weights('/media/redmondlab3/Onions/ryan/celltyping/checkpoints/round7_norm2.h5')

history = model.fit(train_generator, validation_data=val_generator, epochs=25,callbacks=[model_checkpoint_callback,lr_sched_callback])
model.save_weights('/media/redmondlab3/Onions/ryan/celltyping/checkpoints/round7_ptrdec21_2_end.h5')


import pickle

#pickle.dump(history.history,open("./pickdir/round7_poolpluslayer_new.pkl","wb"))
del model
model, preprocessing = get_model("ResNet18_Pool")

model.load_weights("/media/redmondlab3/Onions/ryan/celltyping/checkpoints/round7_ptrdec21_2.h5")
for i in range(len(X_test)):
    if np.sum(X_test[i,:,:,0]) > 500000:
        print(np.sum(X_test[i,:,:,0]),i)
val_gen  = ImageDataGenerator(preprocessing_function=preprocess2)
val_generator = val_gen.flow(X_test,Y_test,batch_size=BATCH_SIZE,shuffle=False)
output  = model.evaluate(val_generator)
output  = model.predict(val_generator)
pickle.dump(output,open("./thisoutput_nov25117.pkl","wb"))
pickle.dump(X_test,open("./Xtest_nov25117.pkl","wb"))
pickle.dump(Y_test,open("./Ytest_nov25117.pkl","wb"))

#/media/redmondlab3/Onions/ryan/celltyping/checkpoints/round7_nov26_flat_sc.h5
#loss: 0.1559 - accuracy: 0.9393 - crossentropy: 0.1559 - precision: 0.9389 - recall: 0.904/ 70 cd34 precision 97 cd117 recall

# /media/redmondlab3/Onions/ryan/celltyping/checkpoints/round7_nov26_flat_resnet18_small.h5 is really good - can be better 
#1s 13ms/step - loss: 0.1314 - accuracy: 0.9470 - crossentropy: 0.1754 - precision: 0.9283 - recall: 0.9395 / 70 cd34 precision 89 cd117 recall


# /media/redmondlab3/Onions/ryan/celltyping/checkpoints/round7_nov26_flat_resnet18_small_pool.h5 is really good - can be better  -> will be in jsons_nov27
#loss: 0.1417 - accuracy: 0.9422 - crossentropy: 0.1417 - precision: 0.9203 - recall: 0.9338 - auc: 0.9890 / 78 cd34 precision 85 cd117 recall

#/media/redmondlab3/Onions/ryan/celltyping/checkpoints/round7_nov26_flat_resnet18_small_pool_minmax.h5  CLAHE .01  
# loss: 0.1408 - accuracy: 0.9567 - crossentropy: 0.2523 - precision: 0.9234 - recall: 0.9718 
#1 cd117 recell 70 cd34 precision

#/media/redmondlab3/Onions/ryan/celltyping/checkpoints/round7_nov26_flat_resnet18_small_pool_minmax2.h5 subtract mean and then minmax
#oss: 0.0999 - accuracy: 0.9470 - crossentropy: 0.1859 - precision: 0.9151 - recall: 0.9556 - auc: 0.9839 - prc: 0.9702
#1 cd117 recall 76 cd34 precision

#round7_sig_pool_far 1s 11ms/step - loss: 0.1480 - accuracy: 0.9398 - crossentropy: 0.1931 - precision: 0.9337 - recall: 0.9210 - auc: 0.9782 
#sig pool far -> 16:112
"""
def preprocess(this3):
    
    this3[:,:,1] = image_clahe.apply(np.expand_dims(((scaler.fit_transform(this3[:,:,1].reshape(-1, 1))*255)).reshape((128,128)).astype(np.uint8),-1))
    
    #this3[:,:,2] = (this3[:,:,2]/max(this3[:,:,2].max(),10))*255
    #this3[:,:,2] = np.clip(this3[:,:,2],np.percentile(this3[:,:,2],1),np.percentile(this3[:,:,2],99))
    #this3[:,:,2] = this3[:,:,2] - this3[:,:,2].mean()
    #this3[:,:,2] = np.clip(this3[:,:,2],np.percentile(this3[:,:,2],1),np.percentile(this3[:,:,2],98))
    #this3[:,:,2] = this3[:,:,2] - this3[:,:,2].mean()
    this3[:,:,2] = np.tanh((1/(2*this3[:,:,2].mean()))*this3[:,:,2])*255 
    this3[:,:,2] = np.clip(this3[:,:,2],0,this3[:,:,1].max())
    this3[:,:,2] = unsharp_mask(this3[:,:,2].astype(np.uint8))
    return create_composite(this3.transpose(2,0,1),colors=[(50,50,50),(0,0,1),(0,1,0)]).astype(int)
"""
