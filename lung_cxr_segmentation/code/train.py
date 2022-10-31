import argparse
import os
import h5py
import cv2
import numpy as np
import segmentation_models as sm
from skimage import exposure
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import pandas as pd
from global_constants import * 
import pathlib



sm.set_framework('tf.keras')
sm.framework()
preprocess_input = sm.get_preprocessing(BACKBONE)


def train(input_csv_path, lung_segmentation_model_output, EPOCHS, BATCH_SIZE, LEARNING_RATE):
    '''
    
    Parameters
    ----------
    input_csv_path (file_path) : The path to input CSV file which contains column 
                                names as 'images'and 'masks' for training step
        
    lung_segmentation_model_output (str) : The path to save the output of 
                                            lung segmentation model (weights)
        
    EPOCHS (positive_int) : Epochs of training 
        
    BATCH_SIZE (positive_int) : Batch size of training
        
    LEARNING_RATE (float) : Learning rate for training
        

    Returns
    -------
    

    '''

    df = pd.read_csv(input_csv_path)
    samples = df.shape[0]
    data = np.zeros((samples, SAVE_IMAGE_SIZE[0], SAVE_IMAGE_SIZE[1], 2))


    ctr = 0
    for index, row in df.iterrows():
        print(ctr)
        try: 
            img = cv2.imread(row["images"], 0)
            print ("reading images -->", row["images"])
            
            lbl = cv2.imread(row["masks"], 0)
            print ("reading masks -->", row["masks"])
            
            img = cv2.resize(img, SAVE_IMAGE_SIZE)
            lbl = cv2.resize(lbl, SAVE_IMAGE_SIZE, interpolation=cv2.INTER_NEAREST)
            data[ctr, :, :, 0] = img
            data[ctr, :, :, 1] = lbl
            ctr += 1
        except: 
            print ("check this image and its mask -->", row["images"])
        
        
    x = data[..., 0] / 255.
    y = (data[..., 1] / 255.)
    print (data.shape, "training mode...")
    print('Data Normalized!')
    
    # Histogram Equalization
    for k in range(x.shape[0]):
        x[k, ...] = exposure.equalize_adapthist(x[k, ...])
    
    x = np.stack((x, x, x), axis=3)
    x = preprocess_input(255. * x) / 255.
    
    
    # train-val split (Not cross validated)
    
    split = int(y.shape[0] * .1)
    val_X = x[:split, ...]
    val_Y = y[:split, :]
    train_X, train_Y = x[split:], y[split:]
    # preprocess input
    print('Data preprocessed and split!')
    



    # Build the model specified  (checks CONSTANTS at the top for details about the model)
    # Returns:
    #   model: compiled model
    model = sm.Unet(BACKBONE, encoder_weights=WEIGHTS, encoder_freeze=False)

    opt = optimizers.Adam(learning_rate=LEARNING_RATE)  # ,momentum=.8,nesterov=True) #.00001

    model.compile(optimizer=opt,
                  loss=sm.losses.bce_jaccard_loss,
                  metrics=[sm.metrics.iou_score],
                  )


    # Trains and saves model
    # Input:
    #   data: contains train_X, train_Y, val_X, val_Y
    #   model: initialized/compiled model to train

    earlystopper = EarlyStopping(monitor='val_loss', patience=7, verbose=1, restore_best_weights=True)
    modelSaver = ModelCheckpoint(lung_segmentation_model_output, monitor='val_loss',
                                 verbose=0,
                                 save_best_only=True,
                                 save_weights_only=False, mode='auto', period=1)

    print('\n\nTraining... ')

    model.fit(train_X, train_Y, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(val_X, val_Y),
              callbacks=[earlystopper, modelSaver])




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
    parser = argparse.ArgumentParser(description="Chest Xray Segmentaiton")
    parser.add_argument('input_csv_path', type=file_path,help='Input CSV has column names images and masks')
    parser.add_argument('--lung_segmentation_model_output', type=str,default = '../weights/Segmentation_resnet50_UNet.h5')
    parser.add_argument('--gpu_id', type=str,default='0')
    parser.add_argument('--BATCH_SIZE', type=positive_int, default=16)
    parser.add_argument('--EPOCHS', type=positive_int, default=100)
    parser.add_argument('--LEARNING_RATE', type=float, default=0.0001)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    
    # training 
    # input = input_csv_path ---> output = lung_segmentation_model_output 
    train(args.input_csv_path, lung_segmentation_model_output=args.lung_segmentation_model_output, 
          BATCH_SIZE=args.BATCH_SIZE, EPOCHS=args.EPOCHS, LEARNING_RATE=args.LEARNING_RATE)


if __name__ == '__main__':
    main()







