import argparse
import os
import pathlib
import tensorflow 
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.layers import BatchNormalization, Activation
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import tensorflow as tf
from numpy import loadtxt
from PIL import Image, ImageOps
from sklearn.model_selection import train_test_split


def train(images_labels_csv, IMAGE_SIZE, BATCH_SIZE, EPOCHS, model_output_path):
  
  
  '''
  Parameters
  ----------
  input_csv_path (file_path) : Input CSV has column for image filenames and their labels
        
    EPOCHS (positive_int) : Epochs of training 
        
    BATCH_SIZE (positive_int) : Batch size of training

    IMAGE_SIZE (positive_int) : The size of image after resizing when training
        
    model_output_path (str) : path for saving the model

  Returns 
  -------
  '''
  

  # Reading the csv file including the images paths and lables
  train_data = pd.read_csv(images_labels_csv)
  train_data['label'] = train_data['label'].astype(str)
  # number of sampels
  n_samples = train_data.shape[0]
  Y = train_data[['label']].astype(str)
  # Splitting into training and test
  train_df, val_df = train_test_split(train_data, test_size=0.1)
  print ("train samples", len(train_df))
  print ("val samples", len(val_df))

  
  input_shape = (IMAGE_SIZE, IMAGE_SIZE, 3)

  # train generator 
  ## add different pixel ranges (min, max values of pixels for normalizaiton (robust normalization)) 
  ## hyperparametes put them in a json file ##
  train_datagen = ImageDataGenerator(rescale=1. / 255,
                    height_shift_range= 0.02, 
                    width_shift_range=0.02, 
                    rotation_range=0.02, 
                    shear_range = 0.01,
                    fill_mode='nearest',
                    zoom_range=0.01)

  train_generator = train_datagen.flow_from_dataframe(train_df, directory = None,
                    x_col = "filename", y_col = "label",
                    target_size=(IMAGE_SIZE, IMAGE_SIZE),
                    batch_size=BATCH_SIZE,
                    class_mode = "categorical", shuffle = True)

  print ("labels", train_generator.class_indices)

  # test generators 
  val_datagen = ImageDataGenerator(rescale=1. / 255)

  val_generator = val_datagen.flow_from_dataframe(val_df, directory = None,
                    x_col = "filename", y_col = "label",
                    target_size=(IMAGE_SIZE, IMAGE_SIZE),
                    batch_size=BATCH_SIZE,
                    class_mode = "categorical", shuffle = True)

  print ("--------------------------------------------------------------------")

  # Model

  pretrained_model = tf.keras.applications.Xception(weights='imagenet', include_top=False)
  #pretrained_model.summary()
  pretrained_model.trainable = False

  x = pretrained_model.output
  x = GlobalAveragePooling2D()(x)
  x = Dense(512)(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  x = Dropout(0.5)(x)
  predictions = Dense(2, activation='softmax')(x)
  model = Model(inputs=pretrained_model.input, outputs=predictions)

  # compiling the model
  model.compile(loss='categorical_crossentropy',
              optimizer='adam', 
              metrics=["accuracy"])

  filepath = "inceptionv3_best.h5"
  checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, 
                             save_best_only=True, mode='min')
  learning_rate_reduction = ReduceLROnPlateau(monitor='val_loss', 
                                            patience=10, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.00001)
  
  callbacks_list = [checkpoint, learning_rate_reduction]

  
  history = model.fit_generator(
      train_generator,
      steps_per_epoch=train_generator.n // BATCH_SIZE,
      validation_data=val_generator,
      validation_steps= val_generator.n // BATCH_SIZE,
      callbacks=callbacks_list,
      epochs=20,
      verbose=1)

  plt.plot(history.history['accuracy'], label='ACC on the training set')
  plt.plot(history.history['val_accuracy'], label='ACC on val set')
  plt.xlabel('Epoch')
  plt.ylabel('Score correct answers')
  plt.legend()
  plt.show()

  # fine tunning of the training loop
  model.load_weights("inceptionv3_best.h5")
  pretrained_model.trainable = False
  for layer in model.layers[:290]:
    layer.trainable = False
  for layer in model.layers[290:]:
    layer.trainable = True

  model.compile(loss='categorical_crossentropy',
              optimizer='adam', 
              metrics=["accuracy"])

  filepath=model_output_path
  checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
  callbacks_list = [checkpoint, learning_rate_reduction]

  history = model.fit_generator(
      train_generator,
      steps_per_epoch=train_generator.n // BATCH_SIZE,
      validation_data=val_generator,
      validation_steps=val_generator.n // BATCH_SIZE,
      callbacks=callbacks_list,
      epochs=EPOCHS,
      verbose=2)


  # evaluation 
  model = tf.keras.models.load_model(model_output_path)
  loss, acc = model.evaluate_generator(generator=val_generator)
  print ("acc", acc)

  # prediction
  val_generator.reset()
  pred=model.predict_generator(val_generator, verbose=1)
  predicted_class_indices=np.argmax(pred,axis=1)

  print("----------------------------------------------------------------------")

def file_path(path):
    p = pathlib.Path(path)
    if p.is_file():
        return p
    else:
        raise argparse.ArgumentTypeError(
            f"Invalid argument ({path}), not a file path or file does not exist."
        )

def nonnegative_int(i):
    I = int(i)
    if I >= 0:
        return I
    else:
        raise argparse.ArgumentTypeError(
            f"Invalid argument ({i}), expected value >= 0 ."
        )

def positive_int(i):
    I = int(i)
    if I > 0:
        return I
    else:
        raise argparse.ArgumentTypeError(
            f"Invalid argument ({i}), expected value > 0 ."
        )

def main():
    parser = argparse.ArgumentParser(description="Chest X-ray classification")
    parser.add_argument('images_labels_csv', type=file_path,help='Input CSV has column for image filenames and their labels')
    parser.add_argument('--IMAGE_SIZE', type=positive_int, default=299, help='The size of image after resizing when training')
    parser.add_argument('--BATCH_SIZE', type=positive_int, default=16)
    parser.add_argument('--EPOCHS', type=positive_int, default=40)
    parser.add_argument('--model_output_path', type=str, default='./weights/inceptionv3_fine_tuned.h5', help='path for saving the model')
    args = parser.parse_args()

    
    train(args.images_labels_csv, IMAGE_SIZE=args.IMAGE_SIZE, 
                  BATCH_SIZE=args.BATCH_SIZE, EPOCHS=args.EPOCHS, 
                  model_output_path=args.model_output_path)

if __name__ == '__main__':
    main()














