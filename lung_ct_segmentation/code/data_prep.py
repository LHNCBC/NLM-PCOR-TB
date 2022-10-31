import pandas as pd
import argparse
import pathlib
from get_folds_datasets import (
    _get_all_folds_dataset,
    _get_train_val_test_files,
    save_fold_files_to_CSV,
)
import shutil
from nnunet.dataset_conversion.utils import generate_dataset_json
import json
"""
This script prepares train/val/test input CSV paths for just
NSCLC-Radiomics and Belarus datasets, useful for training and inference purposes.
"""


def _get_all_folds_specific_datasets(input_csv_path, num_folds):
    """

    This function prepares a dictionary for multiple datasets with key-value pair
    containing key as dataset name and value as list of {num_folds} dictionary objects
    with each dictionary object having key-value pairs as 'train':{train_files},
    'val':{valid_files},'test':{test_files} for each corresponding fold number

    Args:
        input_csv_path(string): Input CSV path contaning standardized CT and
                                binary lung masks and the correspnding dataset
                                they belong to in columns 'ct_file' , 'ref_seg_file'
                                and 'dataset' respectively

        num_folds(int): Total number of folds for Cross validation

    Returns:
        all_folds_all_datasets_dict(dict): Dictionary with a key value of dataset name
                                    and values containing list of 5 folds with
                                    each fold containing keys as 'train' and 'val'
                                    and 'test'


    """
    df = pd.read_csv(str(input_csv_path))
    df = df[df["dataset"].isin(["Belarus", "NSCLC-Radiomics"])]
    datasets = df["dataset"].unique()
    all_folds_all_datasets_dict = {}
    for dataset in datasets:

        dataset_cts = df[df["dataset"] == dataset]["ct_file"].tolist()
        dataset_segs = df[df["dataset"] == dataset]["ref_seg_file"].tolist()

        all_folds_dataset_dict = _get_all_folds_dataset(
            dataset, dataset_cts, dataset_segs, num_folds
        )

        all_folds_all_datasets_dict.update(all_folds_dataset_dict)

    return all_folds_all_datasets_dict

def create_folder(input_nnunet_task_folder, folder):
    """

    This function creates training and testing directories required for the nnunet network.

    Args:

        input_nnunet_task_folder(str): Input nnunet task directory
                                            (e.g: nnUNet_raw_data_base/nnUNet_raw_data/Task001_lungaxial)
                                            as required  for training the
                                            nnunet for the given task. The task
                                            folder must be in the format of
                                            Task{0-9}{0-9}{0-9}_taskname.
        folder(str):Output directory to be created for copying training and testing files.

    Returns:

        ----
    """
    # Create folder.
    if not (pathlib.Path(input_nnunet_task_folder) / folder).exists():
        (pathlib.Path(input_nnunet_task_folder) / folder).mkdir(
            parents=True, exist_ok=True
        )


def copy_files(df, output_ct_folder, output_seg_folder):
    """

    This function copies the files from input dataframe to created nnunet directories.

    Args:

        df(pd.Dataframe): Input dataframe containing columns as 'ct_file' and 'ref_seg_file'
                          representing files for CT file and it's corresponding label file
                          respectively.
        output_ct_folder(list):Output CT directory for copying the CT files.
        output_seg_folder(str): Output segmentation directory for copying the segmentation files.

    Returns:

        ----
    """
    for ct_file in df["ct_file"]:
        shutil.copyfile(
            ct_file,
            (pathlib.Path(output_ct_folder))
            / (pathlib.Path(ct_file).name.split(".nii.gz")[0] + "_0000.nii.gz"),
        )
    for ref_seg_file in df["ref_seg_file"]:
        shutil.copyfile(
            ref_seg_file,
            (pathlib.Path(output_seg_folder))
            / (pathlib.Path(ref_seg_file).name.split(".nii.gz")[0] + ".nii.gz"),
        )


def get_dataframe(dict_files,key):
    """

    This function gets the dataframe files from input dictionary to created train/val/test files.

    Args:

        dict_files(list): List of dictionary objects containing keys as 'img' and
                          'seg' for image path and the corresponding reference
                          path
        key(list):Output CT directory for copying the CT files.

    Returns:
        df_files(pd.DataFrame):  Dataframe containing columns as 'ct_file' and 'ref_seg_file'
                          representing files for CT file and it's corresponding label file
                          respectively.
    """
    imgs = [file["img"] for file in dict_files["train"]]
    segs = [file["seg"] for file in dict_files["train"]]

    df_files = pd.DataFrame({"ct_file": imgs, "ref_seg_file": segs})

    return df_files

def main(argv=None):
    parser = argparse.ArgumentParser(description=".")

    parser.add_argument(
        "input_csv_path",
        type=pathlib.Path,
        help="Input CSV path containing column names as 'ct_file' ,'ref_seg_file','dataset' which represent \
                                paths of  standardized CTs in NIFTI format \
                                paths of standardized binary masks in NIFTI format \
                                and the dataset they present respectively",
    )
    parser.add_argument(
        "sampling_factor_json_path",
        type=pathlib.Path,
        help="JSON containing dataset name as its key and the factor by which its corresponding \
           images are to be sampled during training..E,g: {'LCTSC':{'train':1,'val':1,'test'':1},\
                                                          'NSCLC-Radiomics':{'train':0.16666,'val':0.16666,'test':1},\
                                                          'covid19':{'train':1,'val':1,'test':1},\
                                                          'mdpi':{'train':1,'val':1,'test':1},\
                                                          'LUNA16':{'train':0.1,'val':0.1,'test':1},\
                                                          'VESSEL12':{'train':1,'val':1,'test':1}}",
    )
    parser.add_argument(
        "fold_num",
        type=int,
        help="Fold Number",
    )
    parser.add_argument(
        "folder_name",
        required=True,
        type=pathlib.Path,
        help="Output folder that user should give to save the files in \
             imagesTr,imagesTs,labelsTr,labelsTs subfolders as required per nnUNet training.This \
             folder name should be in the format of E.g : Task001_lungsegmentation.This folder \
             should exist in the directory stricture of nnUNet/",
    )

    parser.add_argument("--num_folds", type=int, default=5, help="Fold Number")
    args = parser.parse_args()

    with open(str(args.sampling_factor_json_path)) as f:
        sampling_factors = json.load(f)

    all_folds_specific_datasets_dict = _get_all_folds_specific_datasets(
        args.input_csv_path, args.num_folds
    )

    combined_single_fold_from_specific_datasets = _get_train_val_test_files(
        all_folds_specific_datasets_dict, args.fold_num,  sampling_factors
    )


    train_files = get_dataframe(combined_single_fold_from_specific_datasets,key="train")
    val_files = get_dataframe(combined_single_fold_from_specific_datasets,key="val")
    test_files = get_dataframe(combined_single_fold_from_specific_datasets,key="test")

    train_val_files = pd.concat([train_files, val_files])

    train_test_sub_folders = ["imagesTr", "labelsTr", "imagesTs", "labelsTs"]
    for folder in train_test_sub_folders:
        create_folder(args.folder_name, folder)

    # copy files from train and validation files to axial directory where nnUNet 2d network
    # trains the model on axial planes of 3D data.
    copy_files(train_val_files, train_test_sub_folders[0], train_test_sub_folders[1])
    # copy files from testing csv file to nnunet test directories.
    copy_files(test_files, train_test_sub_folders[2], train_test_sub_folders[3])

    generate_dataset_json(
        output_file=str(pathlib.Path(args.folder_name) / "dataset.json"),
        imagesTr_dir=str(pathlib.Path(args.folder_name) / "imagesTr"),
        imagesTs_dir=str(pathlib.Path(args.folder_name) / "imagesTs"),
        modalities=("CT"),
        labels={"0": "background", "1": "foreground"},
        dataset_name=pathlib.Path(args.folder_name).name,
    )

if __name__ == "__main__":
    main()
