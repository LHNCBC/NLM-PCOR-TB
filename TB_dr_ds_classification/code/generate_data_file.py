import SimpleITK as sitk
import h5py
import numpy as np
import argparse
import segmentation_models as sm
from skimage import exposure
from crop_input import _resample_cxr_for_lung_segmentation_cnn, _segmented_lung_2_tb_cnn
from global_constants import *
import pandas as pd
import pathlib
from tqdm import tqdm
def load_seg_model(path):
    """
    Loads Segmentation model
    Args:
      path:   file path
    Returns:
      model : model with weights loaded
    """
    model = sm.Unet(SEGMENTATION_BACKBONE,  encoder_freeze=True)
    model.compile(optimizer='Adam',
                 loss=sm.losses.bce_jaccard_loss,
                 metrics=[sm.metrics.iou_score])
    model.load_weights(str(path))
    return model



def get_image(filepath, model):
    """
    Reads, normalizes and crops image around lungs, generates lung masks for each image
    Args:
          filepath(string): Image file path,
          model(keras.Model):    Trained Lung Segmentation Model
    Returns:
      out[...,0](np.ndarray): Cropped Image Array
      conf:       Segmentation Confidence
    """
    original_img, resampled_for_seg = _resample_cxr_for_lung_segmentation_cnn(SEGMENTATION_INPUT_SIZE, 0.5, filepath)
    img = sitk.GetArrayFromImage(resampled_for_seg)
    img = img - np.min(img)
    img = img / np.max(img)
    img = exposure.equalize_adapthist(img)

    if len(img.shape) == 2:
        c = np.expand_dims(img, axis=0)
        img = np.stack((c, c, c), axis=3)
    pred = model.predict(img, batch_size=1, verbose=0)[..., 0]

    pr = pred > 0.5 # converting probability map to mask where values >.5 is 1, 0 otherwise
    pr = pr[0, ...].astype(int)
    out, conf = _segmented_lung_2_tb_cnn((IMAGE_SIZE, IMAGE_SIZE), resampled_for_seg, pr, original_img, gaussian_sigma=0.5)
    return out[..., 0], conf



def read_files(input_csv_path,preprocessed_output_filename, seg_name):
    """
    Reads all image paths and corresponding demographic information
    Loads segmentation model, crops images and generates lung masks
    and saves them along with country of origin to a .h5 data file
    Input params:
          input_csv_path: CSV path containing column name 'image_file' with
                          filenames of Chest X Ray Images and/or column with 'dr_ds_label'
                          with label of 'R' or 'S' denoting either drug resistant TB
                          or drug sensitive TB pertaining to the corresponding Chest-X Ray
          seg_path: segmentation (path and) file name
          data_file: data filename (path and) file name
    """
    df = pd.read_csv(input_csv_path)

    samples = df.shape[0]
    X = np.zeros((samples, IMAGE_SIZE, IMAGE_SIZE))
    mask = np.zeros((samples, IMAGE_SIZE, IMAGE_SIZE))
    Y = np.zeros(samples)

    model = load_seg_model(seg_name)

    if 'dr_ds_label' in df.columns:
        df['dr_ds_label'].replace('R',1,inplace=True)
        df['dr_ds_label'].replace('S',0,inplace=True)

    for k in tqdm(range(samples)):
        actual_path = df['image_file'][k]
        X[k], _ = get_image(actual_path, model)
        if 'dr_ds_label' in df.columns:
            Y[k] = df['dr_ds_label'][k]

    hf = h5py.File(preprocessed_output_filename, 'w') #H5open
    grp = hf.create_group("dataset")
    grp.create_dataset('X', data=X, compression="gzip")
    grp.create_dataset('Y', data=Y, compression="gzip")
    grp.create_dataset('mask', data=mask, compression="gzip")
    hf.close()


def prep_inference_data(csv_path, lung_seg_model_path,output_img_size):
    """
    Preprocess segmentation model, crops images and generates lung masks
    Args:
          csv_path: CSV file path containing column name as image_file
          lung_segment_model_path(pathlib.Path): Pretrained Lung segmentation
                                                   model path.
          output_img_size: Output image size of the cropped lung-region containing image.
                           This size is calculated from the DR-TB/DS-TB model as
                           the script can predict only on the size that the
                           DR-TB/DS-TB has been trained on.
    Outputs:
        filenames(list): Filenames obtained from the 'image_file' column
        X(np.ndarray): preprocessed array with output_img_size containing only
                       lung regions from the original images
    """
    df = pd.read_csv(csv_path)
    filenames = df['image_file']

    samples = df.shape[0]

    X = np.zeros((samples, output_img_size[0], output_img_size[1]))

    lung_seg_model = load_seg_model(lung_seg_model_path)

    for k in tqdm(range(samples)):
        print("Segmenting the input images to lung regions..")
       	X[k], _ = get_image(filenames[k], lung_seg_model)

    return filenames,X

def file_path(path):
    p = pathlib.Path(path)
    if p.is_file():
        return p
    else:
        raise argparse.ArgumentTypeError(
            f"Invalid argument ({path}), not a file path or file does not exist."
        )

def main():

    parser = argparse.ArgumentParser(description="Generate preprocessed data.")
    parser.add_argument('input_csv_path', type=file_path,help="Input CSV file path contanining filepaths and/or label with column names 'image_file' and 'dr_ds_label' respectively")
    parser.add_argument('seg_path', type=file_path, help='trained model path for lung segmentation')
    parser.add_argument('preprocessed_data_output_filename', type=str, help='preprocessed data file name to save')

    args = parser.parse_args()
    read_files(input_csv_path=args.input_csv_path,
               preprocessed_output_filename=args.preprocessed_data_output_filename,
               seg_name=args.seg_path)


if __name__ == '__main__':
    main()
