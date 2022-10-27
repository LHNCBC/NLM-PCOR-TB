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


def inference(images_labels_csv, IMAGE_SIZE, model_path, prediction_csv):
  '''
    Parameters
    ----------
    images_labels_csv (file_path) : Input CSV of image filenames for prediction

    IMAGE_SIZE (positive_int) : The size of image after resizing when training
        
    model_path (str) : The path for loading the trained model
    
    prediction_csv (file_path) : the output csv including predicted labels 

    Returns
    -------

    '''
  
  # Reading the csv file including the images paths 
  test_df = pd.read_csv(images_labels_csv)
  # number of sampels
  n_samples = test_df.shape[0]
  # samples in test data frame 
  print ("test samples", len(test_df))

  # parameters
  image_size = IMAGE_SIZE
  batch_size = 1
  input_shape = (IMAGE_SIZE, IMAGE_SIZE, 3)
  # test generators 
  test_datagen = ImageDataGenerator(rescale=1. / 255)

  test_generator = test_datagen.flow_from_dataframe(test_df, directory = None,
                    x_col = "filename",
                    target_size=(image_size, image_size),
                    batch_size=1, class_mode=None)

  print ("--------------------------------------------------------------------") 

  # Loading the model 
  model = tf.keras.models.load_model(model_path)
  # prediction
  pred=model.predict_generator(test_generator, verbose=1)
  predicted_class_indices=np.argmax(pred,axis=1)
  print (predicted_class_indices)
  # saving prediction in csv 
  filenames=test_generator.filenames
  results=pd.DataFrame({"Filename":filenames,
                        "Predictions":predicted_class_indices})
  results.to_csv(prediction_csv,index=False)

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
    parser.add_argument('images_labels_csv', type=file_path,help='Input CSV has column for image filenames for prediction')
    parser.add_argument('--IMAGE_SIZE', type=positive_int, default=299, help='The size of image after resizing')
    parser.add_argument('--model_path', type=str,default='../weights/inceptionv3_fine_tuned.h5', help='the path for loading the trained model')
    parser.add_argument('--prediction_csv', type=str,default='./prediction.csv', help='csv including predicted labels')
    args = parser.parse_args()

    
    inference(args.images_labels_csv, IMAGE_SIZE=args.IMAGE_SIZE,  
                  model_path=args.model_path, 
                  prediction_csv=args.prediction_csv)

if __name__ == '__main__':
    main()














