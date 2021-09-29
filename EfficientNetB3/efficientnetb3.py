import tensorflow as tf
import numpy as np
import random
import matplotlib.pyplot as plt

from zipfile import ZipFile
from tensorflow.keras import Sequential, Input, Model
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.data.experimental import cardinality
from tensorflow.keras.layers.experimental.preprocessing import RandomFlip, RandomZoom
from tensorflow.keras.applications import efficientnet, EfficientNetB3
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model

random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

def unzip(nm):
    with ZipFile(nm,"r") as zip:
        zip.extractall()

unzip("archive.zip")

train_ds = image_dataset_from_directory(
    directory = "train",
    shuffle = True,
    image_size = (128,128),
    batch_size = 32,
)

val_ds = image_dataset_from_directory(
    directory = "val",
    shuffle = True,
    image_size = (128,128),
    batch_size = 32,
)

class_names = train_ds.class_names

plt.figure(figsize=(10,10))
for images,labels in train_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3,3,i+1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis("off")
plt.show()

val_batches = cardinality(val_ds)
test_ds = val_ds.take(val_batches//5)
val_ds = val_ds.skip(val_batches//5)

data_augmentation = Sequential([
    RandomFlip("horizontal"),
])

plt.figure(figsize=(10,10))
for images,_ in train_ds.take(2):
    for i in range(9):
        ax = plt.subplot(3,3,i+1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.axis("off")
plt.show()

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.prefetch(buffer_size=AUTOTUNE)

rescale = efficientnet.preprocess_input
base_model = EfficientNetB3(input_shape=(128,128,3),include_top=False,weights="imagenet")
base_model.trainable = False
base_model.summary()

class Transfer_Model():
    def model(self,y):
        self.x = data_augmentation(y)
        self.x = rescale(self.x)
        self.x = base_model(self.x,training=False)
        self.x = GlobalAveragePooling2D()(self.x)
        self.x = Dropout(0.2,seed=42)(self.x)
        self.output = Dense(2,activation="sigmoid")(self.x)
        self.model = Model(y,self.output)
        return self.model

M = Transfer_Model()
model = M.model(Input(shape=(128,128,3)))
model.summary()
model.compile(Adam(),SparseCategoricalCrossentropy(),metrics=["accuracy"])

if __name__=="__main__":
    checkpoint = ModelCheckpoint("airbus.hdf5",save_weights_only=False,monitor="val_accuracy",save_best_only=True)
    model.fit(train_ds,epochs=3,validation_data=val_ds,callbacks=[checkpoint])
    best = load_model("airbus.hdf5")
    loss,accuracy = best.evaluate(test_ds)
    print("\nTest accuracy: {:.2f} %".format(100*accuracy))
    print("Test loss: {:.2f} %".format(100*loss))