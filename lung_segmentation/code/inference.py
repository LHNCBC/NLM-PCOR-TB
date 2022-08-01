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
import matplotlib.pyplot as plt


sm.set_framework('tf.keras')
sm.framework()
preprocess_input = sm.get_preprocessing(BACKBONE)



def inference(input_csv_path, segmnetation_model_path, output_prediction_directory, threshold):
    '''
    
    Parameters
    ----------
    input_csv_path (file_path) : The path to input CSV file which contains 
                                column names as 'images' 
                                
    segmnetation_model_path (file_path) : The path for reading the saved wieghts 
                                from the training step to run the inference
        
    output_prediction_directory (str) : The folder for saving the predicted 
                                        binary images
        
    threshold (float) : The threshold for the inference prediction

    Returns
    -------

    '''
    
    df = pd.read_csv(input_csv_path)
    samples = df.shape[0]
    data = np.zeros((samples, SAVE_IMAGE_SIZE[0], SAVE_IMAGE_SIZE[1]))

    ctr = 0
    list_image_names = list()
    for index, row in df.iterrows():
        print(ctr)
        try:
            img = cv2.imread(row["images"], 0)
            image_name = os.path.basename(row["images"])
            list_image_names.append(image_name)
            print ("reading images for inference -->", row["images"])
            img = cv2.resize(img, SAVE_IMAGE_SIZE)
            data[ctr, :, :] = img
            ctr += 1
        except:
            print ("check this image -->", row["images"])
        
    print (list_image_names)
    # Normalization
    x = data / 255.
    print (data.shape, "inference mode...")
    print('Data Normalized!')

    # Histogram Equalization
    for k in range(x.shape[0]):
        x[k, ...] = exposure.equalize_adapthist(x[k, ...])

    x = np.stack((x, x, x), axis=3)
    x = preprocess_input(255. * x) / 255.
    
    data = x

    # Build the model specified  (checks CONSTANTS at the top for details about the model)
    # Returns:
    #    model: compiled model
    model = sm.Unet(BACKBONE, encoder_weights=WEIGHTS, encoder_freeze=False)
    opt = optimizers.Adam(learning_rate=0.001)  # ,momentum=.8,nesterov=True) #.00001
    model.compile(optimizer=opt,
                  loss=sm.losses.bce_jaccard_loss,
                  metrics=[sm.metrics.iou_score],
                  )
    
    # Loading segmentation model
    model.load_weights(segmnetation_model_path)
    
    # Does inference on trained model
    # Input:
    #   data:  array of images
    #   model: trained model for prediction
    seg = model.predict(data)
    
    
    seg = np.squeeze(seg)
    print (seg.shape)
    for i in range(seg.shape[0]):
        img_path = os.path.join(output_prediction_directory, list_image_names[i])
        print (img_path)
        plt.imsave(img_path, seg[i, :, :])
        
        originalImage = cv2.imread(img_path)
        grayImage = cv2.cvtColor(originalImage, cv2.COLOR_BGR2GRAY)
        (thresh, blackAndWhiteImage) = cv2.threshold(grayImage, threshold, 255, cv2.THRESH_BINARY)
        cv2.imwrite(img_path, blackAndWhiteImage)
        print ("inference images saved...")
    

    return

def file_path(path):
    p = pathlib.Path(path)
    if p.is_file():
        return p
    else:
        raise argparse.ArgumentTypeError(
            f"Invalid argument ({path}), not a file path or file does not exist."
        )

def main(argv=None):
    parser = argparse.ArgumentParser(description="Chest Xray Segmentation.")
    parser.add_argument('input_csv_path', type=file_path, help='Input CSV has column name image used for inference')
    parser.add_argument('--segmnetation_model_path', type=file_path, default='../weights/Segmentation_resnet50_UNet.h5')
    parser.add_argument('output_prediction_directory', type=str, help="the output folder path for saving the segmentated images")
    parser.add_argument('--threshold', type=float,default=127)
    args = parser.parse_args(argv)
    
    #inference
    # input = input csv path --> output = binary images 
    inference(input_csv_path=args.input_csv_path, segmnetation_model_path=args.segmnetation_model_path,
              output_prediction_directory=args.output_prediction_directory, threshold=args.threshold)

if __name__ == '__main__':
    main()


















