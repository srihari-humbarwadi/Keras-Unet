import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import random
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.layers import Input, Conv2D, MaxPooling2D, concatenate, UpSampling2D
from tensorflow.python.keras.optimizers import Adadelta, Nadam
from tensorflow.python.keras.models import Model, load_model
from tensorflow.python.keras.utils import multi_gpu_model, plot_model
from tensorflow.python.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.python.keras.preprocessing import image
import tensorflow as tf
from tensorflow.python.keras.losses import binary_crossentropy
from multiclassunet import Unet
import tqdm
import cv2
from tensorflow.python.keras.utils import Sequence
from tensorflow.python.keras.callbacks import Callback


# In[2]:


def dice_coeff(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    score = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return score

def dice_loss(y_true, y_pred):
    loss = 1 - dice_coeff(y_true, y_pred)
    return loss

def total_loss(y_true, y_pred):
    loss = binary_crossentropy(y_true, y_pred) + (3*dice_loss(y_true, y_pred))
    return loss



# In[4]:


image_dir = 'dataset/images'
mask_dir = 'dataset/masks'
image_list = os.listdir(image_dir)
mask_list = os.listdir(mask_dir)
image_list.sort()
mask_list.sort()
print(f'. . . . .Number of images: {len(image_list)}\n. . . . .Number of masks: {len(mask_list)}')

# sanity check
for i in range(len(image_list)):
    assert image_list[i][:-3] == mask_list[i][:-3]


batch_size = 8
samples = len(image_list)
steps = samples//batch_size
img_height, img_width = 256, 256
classes = 66
filters_n = 64
print(f'Batch size = {batch_size}\nSteps per epoch = {steps}')

class seg_gen(Sequence):
    def __init__(self, x_set, y_set, batch_size):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        idx = np.random.randint(0, samples, batch_size)
        batch_x, batch_y = [], []
        drawn = 0
        for i in idx:
            _image = image.img_to_array(image.load_img(f'{image_dir}/{image_list[i]}', target_size=(img_height, img_width)))/255.   
            mask = image.img_to_array(image.load_img(f'{mask_dir}/{mask_list[i]}', grayscale=True, target_size=(img_height, img_width)))
#             mask = np.resize(mask,(img_height*img_width, classes))
            batch_y.append(mask)
            batch_x.append(_image)
        return np.array(batch_x), np.array(batch_y)



# unet = DilatedNet(256, 256, classes,use_ctx_module=True, bn=True)
unet = Unet(256, 256, classes, filters_n)
print(unet.output_shape)
p_unet = multi_gpu_model(unet, 4)
p_unet.load_weights('models-dr/top_weights.h5')
p_unet.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
tb = TensorBoard(log_dir='logs', write_graph=True)
mc = ModelCheckpoint(mode='max', filepath='models-dr/top_weights.h5', monitor='acc', save_best_only='True', save_weights_only='True', verbose=1)
es = EarlyStopping(mode='max', monitor='acc', patience=6, verbose=1)
callbacks = [tb, mc, es]
train_gen = seg_gen(image_list, mask_list, batch_size)


p_unet.fit_generator(train_gen, steps_per_epoch=steps, epochs=13, callbacks=callbacks, workers=8)
print('Saving final weights')
unet.save_weights('unet.h5')

