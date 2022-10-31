import numpy as np
import pandas as pd
import pathlib
from sklearn.model_selection import KFold, train_test_split


def _get_all_folds_dataset(dataset, org_files, seg_files, num_folds=5):
    """

    This function prepares a dictionary for a single dataset with
    key-value pair containing key as dataset name and value as list of {num_folds}
    dictionary objects with each dictionary object having key-value pairs as
    'train':{train_files},'val':{valid_files},'test':{test_files} for each
    corresponding fold number

    Args:
        dataset(string): Dataset name
        org_files(string): CT directory containing standardized CTs
                                    in NIFTI format.

        seg_files(string): Segmentation mask directory containing s
                                    standardized lung binary masks in NIFTI format

        num_folds(int): Total number of folds for Cross validation

    Returns:
        all_folds_dataset_dict(dict): Dictionary with a key value of single dataset
                                     name and values containing list of 5 folds with
                                    each fold containing keys as 'train' and 'val'
                                    and 'test'


    """
    kf_train_test = KFold(n_splits=num_folds, shuffle=False)

    all_folds_dataset = []

    for train_val_idx, test_idx in kf_train_test.split(org_files):
        org_files_test = [org_files[idx] for idx in test_idx]
        seg_files_test = [seg_files[idx] for idx in test_idx]

        org_files_train_val = [org_files[idx] for idx in train_val_idx]
        seg_files_train_val = [seg_files[idx] for idx in train_val_idx]

        (
            org_files_train,
            org_files_val,
            seg_files_train,
            seg_files_val,
        ) = train_test_split(
            org_files_train_val, seg_files_train_val, test_size=0.2, random_state=42
        )

        dataset_train = [
            {"img": img_file, "seg": seg_file}
            for img_file, seg_file in zip(org_files_train, seg_files_train)
        ]
        dataset_val = [
            {"img": img_file, "seg": seg_file}
            for img_file, seg_file in zip(org_files_val, seg_files_val)
        ]

        dataset_test = [
            {"img": img_file, "seg": seg_file}
            for img_file, seg_file in zip(org_files_test, seg_files_test)
        ]

        all_folds_dataset.append(
            {"train": dataset_train, "val": dataset_val, "test": dataset_test}
        )

    all_folds_dataset_dict = {dataset: all_folds_dataset}

    return all_folds_dataset_dict


def _get_all_folds_all_datasets(input_csv_path, num_folds):
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


def _get_train_val_test_files(
    all_folds_all_datasets_dict, fold_num, sampling_rate=None
):
    """

    After all the folds each containing train/val/test sets are obtained for
    each of the dataset, this function then combines 'train' set from all
    the datasets for a given fold so that the training files contain
    CTs ranging from diffused to focused abnormalities which could be
    generally useful to improve the robustness of the model for lung segmentation.


    Args:

        all_folds_all_datasets_dict(dict): Dictionary with a key value of dataset name
                                    and values containing list of 5 folds with
                                    each fold containing keys as 'train' and 'val'
                                    and 'test'
        sampling_rate(dict): Dictionary containing key value of dataset name
                                 and the value with a dictionary of "train","val,"test",
                                  keys and the values in [0,1]. The fraction of the data is used
                                  for each of the dataset in train/val/test.
        fold_num(int): fold number for the cross validation




    Returns:
        combined_single_fold_from_all_datasets(dict): Dictionary object containing key-value pairs
                              as 'train':{train_files},'val':{valid_files},
                              'test':{test_files}

    """
    combined_single_fold_from_all_datasets = {"train": [], "val": [], "test": []}

    for dataset in all_folds_all_datasets_dict.keys():

        # Combine train/val/test files accordingly for each of the datasets
        # obtained from the corresponding fold number
        # When repitition factors is less than 1 we choose the fraction of
        # random samples from the trainingg set of dataset list.

        for key in ["train", "val", "test"]:

            if sampling_rate is not None and sampling_rate[dataset][key] >= 1:
                combined_single_fold_from_all_datasets[key] = (
                    combined_single_fold_from_all_datasets[key]
                    + all_folds_all_datasets_dict[dataset][fold_num][key]
                    * sampling_rate[dataset][key]
                )

            elif sampling_rate[dataset][key] is not None:
                # For when repitition factors are fractions.
                np.random.seed(42)
                num_samples = round(
                    sampling_rate[dataset][key]
                    * len(all_folds_all_datasets_dict[dataset][fold_num][key])
                )

                dataset_samples = np.random.choice(
                    all_folds_all_datasets_dict[dataset][fold_num][key],
                    num_samples,
                    replace=False,
                ).tolist()
                combined_single_fold_from_all_datasets[key] = (
                    combined_single_fold_from_all_datasets[key] + dataset_samples
                )

            else:
                # For when repitition factors are not given
                combined_single_fold_from_all_datasets[key] = (
                    combined_single_fold_from_all_datasets[key]
                    + all_folds_all_datasets_dict[dataset][fold_num][key]
                )

    return combined_single_fold_from_all_datasets


def save_fold_files_to_CSV(
    combined_single_fold_from_all_datasets, input_csv_path, output_filename
):
    """
    Save CSV file with column names as 'ct_file' and 'ref_seg_file' representing the
    filepaths for CTs and reference binary masks respectively for each of the
    train/val/test sets saves it in the same folder as the input csv path.

    Args:

        combined_single_fold_from_all_datasets(dict): List of dictionary objects containing keys as 'img' and
                          'seg' for image path and the corresponding reference
                          path
        input_csv_path(string): Input CSV path contaning standardized CT and
                                binary lung masks and the correspnding dataset
                                they belong to in columns 'ct_file' , 'ref_seg_file'
                                and 'dataset' respectively
        output_filename(str): Output filename to save the test files of a given fold

    Returns:
        ----
    """

    for key in combined_single_fold_from_all_datasets.keys():

        imgs = [file["img"] for file in combined_single_fold_from_all_datasets[key]]
        segs = [file["seg"] for file in combined_single_fold_from_all_datasets[key]]

        fold_files = pd.DataFrame({"ct_file": imgs, "ref_seg_file": segs})
        fold_files.to_csv(
            str(
                pathlib.Path(input_csv_path).parent
                / (key + "_" + str(pathlib.Path(output_filename).name))
            ),
            index=False,
        )
