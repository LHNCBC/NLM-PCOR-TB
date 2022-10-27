'''
Purpose: Classifying Chest X-ray images as drug resistance or drug sensitive TB
'''
import os
import argparse
import h5py
import numpy as np
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Dense, Input, Concatenate, Dropout, GlobalAveragePooling2D
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model
from classification_models.keras import Classifiers
import pathlib
from augment import augment
from utils import *
import tempfile
from generate_data_file import read_files


def load_data(data_path,augment_images=True,random_seed=42,validation_ratio=0.2):
    """
    Load the preprocessed array containing cropped lung regions by shuffling
    and stratifying the labels

    Args:
        data_path:   path to data file (str)
        augment_images(bool): Whether to augment the images or not.
        random_seed(int): Seed to set the consistency of the data division betweeen train/valid
        validation_ratio(float): Ratio to divide training and validation
    Returns:
       train_X,train_Y,val_X,val_Y:  array of training data,array of training labels,
                                     array of validation data and array of validation labels
    """
    # Read X-ray data (cropped images, labels, country information and lungs masks
    data = h5py.File(data_path, 'r')  # preprocessed data, numpy arrays
    X = data['/dataset/X'][:]  # images array
    Y = data['/dataset/Y'][:]  # labels array
    data.close()

    # shuffle data
    np.random.seed(random_seed)
    perm = np.random.permutation(len(Y))
    X, Y = X[perm], Y[perm]

    # Select equal samples from each class
    Y,X  = balance(Y, X,random_seed)


    # Apply histogram equalization to each image
    X = hist_equal(X)

    # split train data into train-val (a % of data is used as validation)
    split = int(Y.shape[0] * validation_ratio)
    val_X, val_Y = X[:split], Y[:split]
    train_X, train_Y = X[split:], Y[split:]

    # For data augmentation:
    if augment_images:
        train_X, train_Y = augment(train_X, train_Y, AUG_FACTOR)

    return train_X,train_Y,val_X,val_Y

def build_model_pretrained():
    """
    Build pretrained model by concatenating the

    Args:
           ---

    Returns:
       model(keras.Model): Loaded model with 'pretrained' imagenet weights.
    """
    classifier, preprocess_input = Classifiers.get(MODEL_NAME)
    inputs = Input(shape=(IMAGE_WIDTH, IMAGE_HEIGHT, 1), name='image')
    img_conc = Concatenate()([inputs, inputs, inputs])  # pretrained model requires 3 channels
    input_model = Model(inputs=inputs, outputs=img_conc)
    base_model = classifier(input_tensor=input_model.output, weights='imagenet',
                                 input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, 3),
                                 include_top=False)

    # weight decay coefficient
    regularizer = l2(0.01)
    for layer in base_model.layers:
        for attr in ['kernel_regularizer', 'bias_regularizer']:
            if hasattr(layer, attr):
                setattr(layer, attr, regularizer)

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(16, activation='relu', kernel_regularizer=l2(0.02))(x)
    x = Dropout(.5)(x)
    x = Dense(NUM_CLASSES, activation='sigmoid', name='R_S')(x)  # R_S - classification branch

    model = Model(inputs=inputs, outputs=x)
    print('\n\nModels Combined!\n')
    # model.summary()
    return model


def train(model, epochs, batch_size, data, model_output_filename, lr,validation_ratio=0.2):
    """
    Trains Classification model, saves periodically and stops early if needed

    Args:
      model(keras.Model): Loaded model with pretrained 'imagenet' weights.
      epochs(int):  max number of epochs to train
      batch_size(int): number samples in a batch
      data(list): List of arrays containing arrays of training data,training labels, validation
                  data and validation labels as each item
      model_output_filename(string): Filename for output saving model
      lr(float): learning rate



    Returns:
       model(keras.Model): Loaded model with 'pretrained' imagenet weights.
    """

    train_X = data[0]
    train_Y = data[1]
    val_X = data[2]
    val_Y = data[3]
    # Callbacks to stop early and save model periodically
    earlystopper = EarlyStopping(monitor='val_loss', patience=7, verbose=1, restore_best_weights=True)
    modelSaver = ModelCheckpoint(model_output_filename, monitor='val_loss', verbose=0,
                                 save_best_only=True,
                                 save_weights_only=False, mode='auto', save_freq='epoch')

    # set optimization params and compile
    opt = optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics='accuracy')

    print('\n\nTraining... ')

    model.fit(train_X, train_Y, epochs=epochs, batch_size=batch_size,
              validation_data=(val_X, val_Y),
              callbacks=[earlystopper, modelSaver], shuffle=True)

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


# read arguments, build/initialize model, load data, train and save model, evaluate model.
def main(argv=None):
    parser = argparse.ArgumentParser(description="Drug Resistance Detection.")
    parser.add_argument('input_csv_path', type=file_path,help='Input CSV has column names image_file,dr_ds_label')
    parser.add_argument('model_output_filename', type=str)
    parser.add_argument('--lung_segmentation_model_path', type=file_path,default = '../../lung_segmentation/weights/Segmentation_resnet50_UNet.h5')
    parser.add_argument('--gpu_id', type=str,default='0')
    parser.add_argument('--batch_size', type=positive_int, default=8)
    parser.add_argument('--epochs', type=positive_int, default=100)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument("--augment_images", action="store_true", help= "If true, the images are augmented before the training.")
    args = parser.parse_args(argv)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    model = build_model_pretrained()

    #preprocess data
    with tempfile.TemporaryDirectory() as tempdir_name:
        preprocessed_data_path = pathlib.Path(tempdir_name).absolute() / 'preprocessed_data_path.h5'
        print("Segmenting the input images to lung regions..")
        read_files(args.input_csv_path,preprocessed_data_path, args.lung_segmentation_model_path)
        train_X,train_Y,val_X,val_Y = load_data(preprocessed_data_path,args.augment_images)

    # train
    print("Training on the segmented lung data..")
    train(model=model, epochs=args.epochs, batch_size=args.batch_size,
          data=[train_X,train_Y,val_X,val_Y], model_output_filename=args.model_output_filename,
          lr=args.lr)

if __name__ == '__main__':
    main()
