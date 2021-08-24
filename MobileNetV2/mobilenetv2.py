import tensorflow as tf
import numpy as np
import random
import matplotlib.pyplot as plt
from zipfile import ZipFile

random.seed(1337)
np.random.seed(1337)
tf.random.set_seed(1337)

with ZipFile("archive.zip","r") as zip:
	zip.extractall()
	
BATCH_SIZE = 32
IMG_SIZE = (160,160)

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
	"train",
	shuffle = True,
	image_size = IMG_SIZE,
	batch_size = BATCH_SIZE,
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
	"val",
	shuffle = True,
	image_size = IMG_SIZE,
	batch_size = BATCH_SIZE,
)

val_batches = tf.data.experimental.cardinality(val_ds)
test_ds = val_ds.take(val_batches//5)
val_ds = val_ds.skip(val_batches//5)

class_names = train_ds.class_names

plt.figure(figsize=(10,10))
for images,labels in train_ds.take(1):
	for i in range(9):
		ax = plt.subplot(3,3,i+1)
		plt.imshow(images[i].numpy().astype("uint8"))
		plt.title(class_names[labels[i]])
		plt.axis("off")
plt.show()

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.prefetch(buffer_size=AUTOTUNE)

preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input

IMG_SHAPE = (160,160,3)
base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,include_top=False,weights="imagenet")
image_batch,label_batch = next(iter(train_ds))
feature_batch = base_model(image_batch)

base_model.trainable = False
base_model.summary()

global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
feature_batch_average = global_average_layer(feature_batch)

prediction_layer = tf.keras.layers.Dense(2)
prediction_batch = prediction_layer(feature_batch_average)

def get_model():
	inputs = tf.keras.Input(shape=(160,160,3))
	x = preprocess_input(inputs)
	x = base_model(x,training=False)
	x = global_average_layer(x)
	x = tf.keras.layers.Dropout(0.2)(x)
	outputs = prediction_layer(x)
	model = tf.keras.Model(inputs,outputs)
	return model

model = get_model()
model.summary()
model.compile(tf.keras.optimizers.Adam(),tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),metrics=["accuracy"])

if __name__=="__main__":
	initial_epochs = 1
	loss0,accuracy0 = model.evaluate(val_ds)
	print("Initial loss: {:.2f} %".format(100*loss0))
	print("Initial accuracy: {:.2f} %".format(100*accuracy0))
	checkpoint = tf.keras.callbacks.ModelCheckpoint("airbus.h5",save_weights_only=False,monitor="val_accuracy",save_best_only=True)
	model.fit(train_ds,epochs=initial_epochs,validation_data=val_ds,callbacks=[checkpoint])
	best = tf.keras.models.load_model("airbus.h5")
	loss,accuracy = best.evaluate(test_ds)
	print("\nTest accuracy: {:.2f} %".format(100*accuracy))
	print("Test loss: {:.2f} %".format(100*loss))
