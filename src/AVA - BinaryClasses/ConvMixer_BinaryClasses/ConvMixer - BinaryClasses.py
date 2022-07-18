# -*- coding: utf-8 -*-
"""pruebas-ava.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1pVXozWgfbcnmS8gmtHIU_QWHwWe6aiyk

**Distribución**: tuplas de la probabilidad de que sea buena y mala [[0.1,0.1,0.1,...(10 veces)],...] losses: EMD; 10 neuronas output con softmax
>(distribution, none)


**Clases (probabilidad)**: [[0.34,0.66],...] losses: categoricalCrossEntropy; 2 neuronas output con softmax
>(mean, binaryWeights)


**Regresión**: etiquetas entre 0-1 con la puntuacion media de que sea buena la foto [0.5,0.3,0.24,...] losses: MSE; 1 neurona output sin func activacion
>(mean, none)


**Clases (etiqueta)**: etiquetas o 1 o 0 si las fotos son buenas o malas [0,1,0,1,0,0,0,1]  losses: binary_crossentropy || sparse_categorical_crossentropy; 1 neurona output con sigmoid || 2 con softmax
>(mean, binaryClasses)
"""

import sys
sys.path.insert(1, 'AQA-framework-dev_coteach')

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_addons as tfa
import pickle

import tfimm
from datasets import AVA_generators
from tensorflow.keras.losses import MSE, MAE

model_list = tfimm.list_models(pretrained="timm")
str_match = [s for s in model_list if "convmixer" in s]
str_match

convmixer_model = tfimm.create_model("convmixer_768_32", pretrained="timm")
convmixer_model.summary()

"""image size de nuestro ConvMixer"""

image_size = 224

"""input shape de AVA"""

input_shape = (224, 224, 3)

avaBinaryClasses = AVA_generators(obj_class='mean', mod_class='binaryClasses')

def parse_image(filename, label):
  image = tf.io.read_file(filename)
  image = tf.io.decode_jpeg(image)
  image = tf.image.convert_image_dtype(image, tf.float32)
  image = tf.image.resize(image, [image_size, image_size])
  return image, label

preprocess_func = tfimm.create_preprocessing("convmixer_768_32", dtype="float32")
preprocess = lambda img, lab: (preprocess_func(img), lab)

x_train = tf.data.Dataset.from_tensor_slices((avaBinaryClasses.train_image_paths, avaBinaryClasses.train_scores)).map(parse_image).map(preprocess).shuffle(256).batch(16).prefetch(-1)
x_test = tf.data.Dataset.from_tensor_slices((avaBinaryClasses.test_image_paths, avaBinaryClasses.test_scores)).map(parse_image).map(preprocess).batch(16).prefetch(-1)
x_val = tf.data.Dataset.from_tensor_slices((avaBinaryClasses.val_image_paths, avaBinaryClasses.val_scores)).map(parse_image).map(preprocess).batch(16).prefetch(-1)

# data_augmentation = keras.Sequential(
#     [
#         layers.Normalization(),
#         layers.Resizing(image_size, image_size),
#         layers.RandomFlip("horizontal"),
#         layers.RandomRotation(factor=0.02),
#         layers.RandomZoom(
#             height_factor=0.2, width_factor=0.2
#         )
#     ],
#     name="data_augmentation",
# )

"""añadimos las capas de entrada y salida correspondiente a nuestro problema (AVA)"""

inputs = layers.Input(shape=input_shape)
# augmented = data_augmentation(inputs)
features = convmixer_model(inputs)
logits = keras.layers.Dense(2, activation='softmax')(features)
model = keras.Model(inputs=inputs, outputs=logits)
model.summary()

"""parámetros para compilar, entrenar y evaluar el modelo

a ajustar posteriormente
"""

learning_rate = 0.000001
weight_decay = 0.0001
num_epochs = 15

def run_experiment(model):
    optimizer = tfa.optimizers.AdamW(
        learning_rate=learning_rate, weight_decay=weight_decay
    )

    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    checkpoint_filepath = "checkpoint_convmixer/checkpoint"
    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        checkpoint_filepath,
        monitor="val_accuracy",
        save_best_only=True,
        save_weights_only=True,
    )

    history = model.fit(
        x= x_train,
        epochs=num_epochs,
        validation_data=x_val,
        callbacks=[checkpoint_callback],
    )

    model.load_weights(checkpoint_filepath)
    results = model.evaluate(x_test)
    print("test loss, test accuracy", results)

    return history.history

history = run_experiment(model)

with open('outputs_convmixer_ava.pickle', 'wb') as f:
    pickle.dump(history, f)
