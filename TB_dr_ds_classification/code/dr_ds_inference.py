import argparse
import pathlib
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from generate_data_file import prep_inference_data

def inference(csv_path, dr_ds_model_path, lung_seg_model_path, output_csv_filename,
              threshold):
    """
    Generate predictions from the csv file. The CSV file should contain 'image_file'
    as the column name and it represents file paths of Chest X Rays.
    The inference from the each of the image happens in 3 stages.
        1.) Reading and Resampling of images from the image file paths.
        2.) Predict masks from the array of resampled image array
        3.) Segment lungs from the original images using the predicted masks.
    Args:
        csv_path(string): Path to CSV file containing column name as 'image_file'
        dr_ds_model_path(string): Model path for trained model DR/DS-TB.
        lung_segment_model_path(pathlib.Path): Pretrained Lung segmentation
                                               model path.
        output_csv_filename(string): Output CSV filename to save  predictions.
                                     The file contains'image_file','pred_label',
                                     'confidence' as columns.
    Returns:
       ----
    """

    model = load_model(dr_ds_model_path)

    filenames,test_X = prep_inference_data(csv_path,lung_seg_model_path,
                                           output_img_size=model.layers[0].input_shape[0][1:3])

    y_pred = model.predict(test_X)
    y_pred = np.squeeze(y_pred)
    labels = ['DR-TB' if val == 1 else 'DS-TB' for val in y_pred > threshold]

    df = pd.DataFrame({'image_file':filenames,'pred_label':labels,'confidence':y_pred})
    df.to_csv(output_csv_filename,index=None)

def file_path(path):
    """
    Returns and checks if the file path exists.

    Args:
        csv_path(string): Path to CSV file containing column name as 'image_file'
        dr_ds_model_path(string): Model path for trained model DR/DS-TB.
        lung_segment_model_path(pathlib.Path): Pretrained Lung segmentation
                                               model path.
        output_csv_filename(string): Output CSV filename to save  predictions.
                                     The file contains'image_file','pred_label',
                                     'confidence' as columns.
    Returns:
       ----
    """
    p = pathlib.Path(path)
    if p.is_file():
        return p
    else:
        raise argparse.ArgumentTypeError(
            f"Invalid argument ({path}), not a file path or file does not exist."
        )

def main(argv=None):
    parser = argparse.ArgumentParser(description="Drug Resistance Detection.")
    parser.add_argument('--dr_ds_model_path', type=file_path,default='../weights/dr_ds_tb.h5')
    parser.add_argument('--lung_segmentation_model_path', type=file_path,default='../../lung_segmentation/weights/Segmentation_resnet50_UNet.h5')
    parser.add_argument('input_filenames_csv_path', type=file_path)
    parser.add_argument('output_predictions_csv_path', type=str)
    parser.add_argument('--threshold', type=float,default=0.6323408)
    args = parser.parse_args(argv)

    inference(csv_path=args.input_filenames_csv_path,dr_ds_model_path=args.dr_ds_model_path,
              lung_seg_model_path=args.lung_segmentation_model_path,
              output_csv_filename=args.output_predictions_csv_path,
              threshold=args.threshold)


if __name__ == '__main__':
    main()