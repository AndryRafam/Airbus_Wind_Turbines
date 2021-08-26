import tensorflow as tf
import numpy as np
import random
import matplotlib.pyplot as plt

from zipfile import ZipFile
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.data.experimental import cardinality
from tensorflow.keras.layers.experimental.preprocessing import RandomFlip

random.seed(123)
np.random.seed(123)
tf.random.set_seed(123)

with ZipFile("archive.zip","r") as zip:
    zip.extractall()

batch_size = 32
img_size = (128,128)

train_ds = image_dataset_from_directory(
    "train",
    shuffle = True,
    image_size = img_size,
    batch_size = batch_size,
)

val_ds = image_dataset_from_directory(
    "val",
    shuffle = True,
    image_size = img_size,
    batch_size = batch_size,
)

class_names = train_ds.class_names

val_batches = cardinality(val_ds)
test_ds = val_ds.take(val_batches//5)
val_ds = val_ds.skip(val_batches//5)

plt.figure(figsize=(10,10))
for images,labels in train_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3,3,i+1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis("off")
plt.show()

data_augmentation = tf.keras.Sequential([
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

from tensorflow.keras.applications import efficientnet, EfficientNetB3
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model

preprocess_input = efficientnet.preprocess_input
input_shape = img_size+(3,)
base_model = EfficientNetB3(input_shape=input_shape,include_top=False,weights="imagenet")
image_batch,label_batch = next(iter(train_ds))
feature_batch = base_model(image_batch)

base_model.trainable = False
base_model.summary()

feature_batch_average = GlobalAveragePooling2D()(feature_batch)
prediction_batch = Dense(2)(feature_batch_average)

def model():
    inputs = tf.keras.Input(shape=input_shape)
    x = data_augmentation(inputs)
    x = preprocess_input(x)
    x = base_model(x,training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.2,seed=123)(x)
    outputs = Dense(2)(x)
    model = tf.keras.Model(inputs,outputs)
    return model

model = model()
model.summary()
model.compile(Adam(),SparseCategoricalCrossentropy(from_logits=True),metrics=["accuracy"])

if __name__=="__main__":
    loss0,accuracy0 = model.evaluate(val_ds)
    checkpoint = ModelCheckpoint("airbus.hdf5",save_weights_only=False,monitor="val_accuracy",save_best_only=True)
    model.fit(train_ds,epochs=3,validation_data=val_ds,callbacks=[checkpoint])
    best = load_model("airbus.hdf5")
    loss,accuracy = best.evaluate(test_ds)
    print("\nTest accuracy: {:.2f} %".format(100*accuracy))
    print("Test loss {:.2f} %".format(100*loss))