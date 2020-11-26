# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 12:17:39 2020

@author: PCUser
"""


# %%
# The code have been made with inspiration from https://www.kaggle.com/koshirosato/bee-or-wasp-base-line-using-resnet50

import os
import cv2
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pathlib
import tensorflow as tf
from tensorflow.keras.layers import *
from tqdm import tqdm
import skimage.color
import skimage.io
import skimage.viewer


# %%
# Setting up constants
SEED = 5

# Reading in the lables file
df = pd.read_csv('labels.csv')
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# %%
def seed_everything(seed):
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    tf.random.set_seed(seed)

seed_everything(SEED)


# %%
# Replace \ with / to make the path work
for idx in tqdm(df.index):    
    df.loc[idx,'path']=df.loc[idx,'path'].replace('\\', '/') 

df.head()


# %%
# Show the procent of images that have the diffrent labels
labels = list(df['label'].unique())
y = list(df['label'].value_counts())
plt.pie(y, labels=labels, autopct='%1.2f%%')
plt.title('Unique values based in labels')
plt.show()


# %%
labels = list(df['photo_quality'].unique())
x = range(0, 2)
y = list(df['photo_quality'].value_counts())
plt.bar(x, y, tick_label=labels)
plt.title('High quality photos in original data')

plt.show()


# %%
def img_plot(df, label):
    df = df.query('label == @label')
    imgs = []
    for path in df['path'][:4]:
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        imgs.append(img)
    f, ax = plt.subplots(2, 2, figsize=(10,10))
    for i, img in enumerate(imgs):
        ax[i//2, i%2].imshow(img)
        ax[i//2, i%2].axis('off')
        ax[i//2, i%2].set_title('label: %s' % label)
    plt.show()


# %%
def hist_plot(df, label):
    colors = ("r", "g", "b")
    channel_ids = (0, 1, 2)
    df = df.query('label == @label')
    imgs = []
    for path in df['path'][:2200]:
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        imgs.append(img)

        plt.xlim([0, 256])

    imgs_resize = []

    for img in imgs:
        width = 40
        height = 40
        dim = (width, height)
        image = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
        imgs_resize.append(image)

    for channel_id, c in zip(channel_ids, colors):
        for image in imgs_resize:
            histogram, bin_edges = np.histogram(image[:, :, channel_id], bins=256, range=(0, 256))
            plt.plot(bin_edges[0:-1], histogram, color=c)

    plt.xlabel("Color value")
    plt.ylabel("Pixels")
    plt.show()


# %%
hist_plot(df, label='bee')


# %%
hist_plot(df, label='wasp')


# %%
img_plot(df, label='bee')


# %%
img_plot(df, label='wasp')


# %%
img_plot(df, label='insect')


# %%
img_plot(df, label='other')


# %%
# select only high quality photos
df = df.query('photo_quality == 1')
df['label'].value_counts()


# %%



