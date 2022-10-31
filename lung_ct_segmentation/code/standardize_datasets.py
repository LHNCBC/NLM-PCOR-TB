import os
import pandas as pd
import pathlib
import glob
import SimpleITK as sitk
import subprocess
import shutil
import argparse
import json
import pydicom


"""
This script reads CT series and RTSTRUCT/mhd files (for labels) for each
of the dataset and standardizes all of the files in all the dataset
in the same format i.e: NIFTI.If the segmentation file for any dataset is not
in regular readable format through SimpleITK, this file checks for RTSTRUCT
format and converts those files to NIFTI format through plastimatch
program(https://plastimatch.org/). After plastimatch converts RTSTRUCT image
to NIFTI format, this file then filters and combines lung labels and saves the
segmentation file in their respective output segmentation directory. If a
dataset contains label file in mhd format, this file reads and saves
that file in NIFTI format through SimpleITK.

"""


def convert_rtstruct_to_nifti(rtstruct_file, output_seg_filename):

    """
    This function  converts a RTSTRUCT file to NIFTI using plastimatch program  and
    saves it in {output_seg_filename}.nii.gz format

    Args:
        rtstruct_file(string): Label file in RTSTRUCT format
        output_seg_filename(str): Output label file name in NIFTI format
    Returns:
        --
    """

    subprocess.run(
        [
            "plastimatch",
            "convert",
            "--input",
            str(rtstruct_file),
            "--output-ss-img",
            output_seg_filename,
        ]
    )


def _save_data_to_nifti(input_dir, output_seg_filename, output_ct_filename):

    """
    This function traverses the input directory of a  dataset and
    filters the file types (CT or RTSTRUCT ) to save them in the NIFTI
    format.

    Args:
        input_dir(string):  Input directory of a dataset E.g: /input/path/to/LCTSC
        output_seg_filename(string): Output filename for the binary mask with
                                      lung regions labelled as 1
        output_ct_filename(string): Output CT filename to save the series in NIFTI
                                    format
    Returns:
        ---
    """

    for dr, sub_dir, files in os.walk(input_dir):

        if len(files) > 0:
            ds = pydicom.dcmread(files[0])

            if ds.Modality == "CT":

                reader = sitk.ImageSeriesReader()

                dicom_names = reader.GetGDCMSeriesFileNames(dr)
                reader.SetFileNames(dicom_names)

                img = reader.Execute()

                sitk.WriteImage(img, output_ct_filename)

            elif ds.Modality == "RTSTRUCT":

                convert_rtstruct_to_nifti(
                    pathlib.Path(dr) / files[0], output_seg_filename
                )
            else:
                pass


def _filter_lunglabels_from_LCTSC_RTStruct_to_NIFTI_output(
    plastimatch_converted_nifti_seg_file,
):

    """

    Plastimatch saved output in NIFTI format contains labels for all the organs
    from the RTSTRUCT file. This function filters the lung labels (right lung -4,
    left lung -2) from that plastimatch generated output and combines both the
    regions and saves them with the save filename

    Args:

        plastimatch_converted_nifti_seg_file(string):  Plastimatch converted
                                                        output segmented file
                                                        (This file contains labels
                                                         from all the organs)
    Returns:
        ---
    """

    img = sitk.ReadImage(plastimatch_converted_nifti_seg_file)

    # Left lung has label 2 and right lung has label 4
    img = (img == 2) + (img == 4)

    sitk.WriteImage(img, plastimatch_converted_nifti_seg_file)


def _filter_lunglabels_from_NSCLC_RTStruct_to_NIFTI_output(
    plastimatch_converted_nifti_seg_file,
):
    """

    Plastimatch saved output in NIFTI format contains labels for all the organs
    from the RTSTRUCT file. This function filters the lung labels (right lung -2,
    left lung -1) from that plastimatch generated output and combines both the
    regions and saves them with the save filename

    Args:

        plastimatch_converted_nifti_seg_file(string):  Plastimatch converted
                                                        output segmented file
                                                        (This file contains labels
                                                         from all the organs)
    Returns:
        -
    """
    img = sitk.ReadImage(plastimatch_converted_nifti_seg_file)

    # Left lung has label 1 and right lung has label 2
    img = (img == 1) + (img == 2)

    sitk.WriteImage(img, plastimatch_converted_nifti_seg_file)


def _prepare_LCTSC_data(dataset_info, output_segmentation_dir, output_ct_dir):
    """

    Standardize LCTSC data by saving the CT volumes and binary lung masks
    in NIFTI.Filenames of each of the CT and their corresponding binary lung mask
    are saved with the same unique sub directory names which are under the
    hierarchy structure of root dataset directory.
    (E.g: Each CT volume and label are saved by the same
                            filenames of 'LCTSC-Test-S1-101.nii.gz'
                            ,'LCTSC-Test-S1-102.nii.gz' ..
                            which are the sub directory names under the root
                            dataset directory of /input/path/to/LCTSC)

    Args:

        dataset_info(dict): path to json file in the format. The json
                                    file must contain
                                    'LCTSC':{'input_dir': /input/path/to/LCTSC}
                                    as one of the key-value pair in the above format
                                    if user wants to standardize this dataset

        output_segmentation_dir(string): Output segmentation directory to
                                        save the binary lung mask niftis
                                        Images will be saved in the below format
                                        {output_segmentation_dir} / 'LCTSC' /
                                        {sub_directory_name}.nii.gz
        output_ct_dir(string): Output CT directory to save the CT volume in niftis
                                        Images will be saved in the below format
                                        {output_ct_dir} / 'LCTSC' /
                                        {sub_directory_name}.nii.gz
    Returns:
        ---

    """
    if not pathlib.Path(output_ct_dir).is_dir():
        os.mkdir(output_ct_dir)
    if not pathlib.Path(output_segmentation_dir).is_dir():
        os.mkdir(output_segmentation_dir)

    patientIDs = os.listdir(dataset_info["input_dir"])
    for patientID in patientIDs:

        input_patient_dir = str(
            pathlib.Path(dataset_info["input_dir"]).absolute() / patientID
        )
        output_ct_filename = (
            str(pathlib.Path(output_ct_dir).absolute() / patientID) + ".nii.gz"
        )

        output_seg_filename = (
            str(pathlib.Path(output_segmentation_dir).absolute() / patientID)
            + ".nii.gz"
        )

        _save_data_to_nifti(input_patient_dir, output_seg_filename, output_ct_filename)

        _filter_lunglabels_from_LCTSC_RTStruct_to_NIFTI_output(output_seg_filename)


def _prepare_covid19_data(dataset_info, output_segmentation_dir, output_ct_dir):
    """

    Standardize covid19 data by saving the CT volumes and binary lung masks
    in NIFTI. This function simply copies the CTs from the input directory to
    the output CT directory,as the original image format already exists in NIFTI.
    This function also filters and combines only the labels from lung regions
    (1 & 2 - for right and left lung respectively) from all the labels and
    saves them in the NIFTI format. Standardized CT volumes and labels are saved
    with the same filenameas their original filename.

    Args:

        dataset_info(dict): path to json file in the format. The json
                                    file must contain
                                    'covid19':{'input_dir': /input/path/to/covid19,
                                               'ct_dir':{ct_directory_name},
                                               'seg_dir':{seg_directory_name}}
                                    as one of the key-value pair in the above format
                                    if user wants to standardize this dataset

        output_segmentation_dir(string): Output segmentation directory to
                                        save the binary lung mask niftis
                                        Images will be saved in the below format
                                        {output_segmentation_dir} / 'covid19' /
                                        {filename}.nii.gz
        output_ct_dir(string): Output CT directory to save the CT volume in niftis
                                        Images will be saved in the below format
                                        {output_ct_dir} / 'covid19' /
                                        {filename}.nii.gz
    Returns:
        ---

    """
    if not pathlib.Path(output_ct_dir).is_dir():
        os.mkdir(output_ct_dir)
    if not pathlib.Path(output_segmentation_dir).is_dir():
        os.mkdir(output_segmentation_dir)

    cts = glob.glob(
        str(
            pathlib.Path(dataset_info["input_dir"])
            / dataset_info["ct_dir"]
            / "*.nii.gz"
        )
    )
    segs = glob.glob(
        str(
            pathlib.Path(dataset_info["input_dir"])
            / dataset_info["seg_dir"]
            / "*.nii.gz"
        )
    )

    for ct in cts:
        shutil.copy(ct, str(pathlib.Path(output_ct_dir) / pathlib.Path(ct).name))

    for seg in segs:
        img = sitk.ReadImage(seg)
        #  Label for left lung = 2 and Label for right lung = 1
        img = (img == 2) + (img == 1)
        sitk.WriteImage(
            seg, str(pathlib.Path(output_segmentation_dir) / pathlib.Path(seg).name)
        )


def _prepare_LUNA16_data(dataset_info, output_segmentation_dir, output_ct_dir):
    """

    Standardize LUNA16 data by saving the CT volumes and labels of lung regions
    in NIFTI. This function grabs mhd format CTs from the input directory and
    converts them through SimpleITK and saves them in NIFTI format in the output
    CT directory.This function also filters and combines only the labels from lung
    regions(4 & 3 - for right and left lung respectively) from all the labels and
    saves them in the NIFTI format.CTs and labels are saved under the same filename
    based on the directories named under {ct_dirs_prefix} (E.g: Standardized CTs and
    labels under the sub directory subset0 will be saved as
    '1.3.6.1.4.1.14519.5.2.1.6279.6001.105756658031515062000744821260.nii.gz',
    '1.3.6.1.4.1.14519.5.2.1.6279.6001.108197895896446896160048741492.nii.gz' etc..
    in the output ct directory and output segmentation directory respectively).

    Args:

        dataset_info(dict): path to json file in the format. The json
                                    file must contain
                                    'covid19':{'input_dir': /input/path/to/LUNA16,
                                               'ct_dirs_prefix':subset,
                                               'seg_dir':{seg_directory_name}}
                                    as one of the key-value pair in the above format
                                    if user wants to standardize this dataset

        output_segmentation_dir(string): Output segmentation directory to
                                        save the binary lung mask niftis
                                        Images will be saved in the below format
                                        {output_segmentation_dir} / 'LUNA16' /
                                        {filename}.nii.gz
        output_ct_dir(string): Output CT directory to save the CT volume in niftis
                                        Images will be saved in the below format
                                        {output_ct_dir} / 'LUNA16' /
                                        {filename}.nii.gz
    Returns:
        ---

    """
    if not pathlib.Path(output_ct_dir).is_dir():
        os.mkdir(output_ct_dir)
    if not pathlib.Path(output_segmentation_dir).is_dir():
        os.mkdir(output_segmentation_dir)

    # For CTs
    ct_dirs_prefixes = str(
        pathlib.Path(dataset_info["input_dir"]) / dataset_info["ct_dirs_prefix"] / "*"
    )
    subsets = glob.glob(ct_dirs_prefixes)

    cts = []
    for subset in subsets:
        subset_cts = glob.glob(str(pathlib.Path(subset) / "*.mhd"))
        cts.append(subset_cts)

    for ct in cts:
        img = sitk.ReadImage(ct)
        sitk.WriteImage(
            ct, str(pathlib.Path(output_ct_dir) / pathlib.Path(ct).stem) + ".nii.gz"
        )

    # For Lung masks
    segs = glob.glob(
        str(pathlib.Path(dataset_info["input_dir"]) / dataset_info["seg_dir"] / "*.mhd")
    )
    for seg in segs:
        img = sitk.ReadImage(seg)

        #   Label for left lung = 3 and Label for right lung = 4
        img = (img == 3) + (img == 4)
        filename = (
            str(pathlib.Path(output_segmentation_dir) / pathlib.Path(seg).stem)
            + ".nii.gz"
        )
        sitk.WriteImage(seg, filename)


def _prepare_mdpi_data(dataset_info, output_segmentation_dir, output_ct_dir):
    """

    Standardize 'MDPI' data by saving the CT volumes and binary lung masks
    in NIFTI. This function grabs nrrd format CTs from the input directory and
    converts them through SimpleITK and saves them in NIFTI format in the output
    CT directory.The labels for this dataset are generated through gaussian
    mixture model for all the lung tissues present in lungs. As such the binary
    mask based on lungs is generated if any label > 0.All CTs and binary lung
    masks are saved under the same filename based on the directories named under
    {ct_dirs_prefix} (E.g:  CTs and  labels from under the sub directory subset0 will be
     saved as '1.nii.gz','2.nii.gz' etc..in the output ct directory and output
     segmentation directory respectively).

    Args:

        dataset_info(dict): path to json file in the format. The json
                                    file must contain
                                    'mdpi':{'input_dir': /input/path/to/mdpi,
                                               'ct_dirs_prefix':patients_,
                                               'ct_filename':CT.nrrd,
                                               'seg_filename':GMM_LABELS.nrrd}
                                    as one of the key-value pair in the above format
                                    if user wants to standardize this dataset

        output_segmentation_dir(string): Output segmentation directory to
                                        save the binary lung mask niftis
                                        Images will be saved in the below format
                                        {output_segmentation_dir} / 'mdpi' /
                                        {filename}.nii.gz
        output_ct_dir(string): Output CT directory to save the CT volume in niftis
                                        Images will be saved in the below format
                                        {output_ct_dir} / 'mdpi' /
                                        {filename}.nii.gz
    Returns:
        ---

    """
    if not pathlib.Path(output_ct_dir).is_dir():
        os.mkdir(output_ct_dir)
    if not pathlib.Path(output_segmentation_dir).is_dir():
        os.mkdir(output_segmentation_dir)

    ct_dirs_prefixes = str(
        pathlib.Path(dataset_info["input_dir"]) / dataset_info["ct_dirs_prefix"] / "*"
    )
    grouped_patient_dirs = glob.glob(ct_dirs_prefixes)
    cts = []
    segs = []
    for grouped_patient_dir in grouped_patient_dirs:
        patientIDs = os.listdir(grouped_patient_dir)
        cts.append(
            [
                pathlib.Path(grouped_patient_dir) / dataset_info["ct_filename"]
                for patientID in patientIDs
            ]
        )
        segs.append(
            [
                pathlib.Path(grouped_patient_dir) / dataset_info["seg_filename"]
                for patientID in patientIDs
            ]
        )

    for ct in cts:
        img = sitk.ReadImage(str(ct))
        sitk.WriteImage(
            img, str(pathlib.Path(output_ct_dir) / pathlib.Path(ct).stem) + ".nii.gz"
        )
    for seg in segs:
        img = sitk.ReadImage(str(seg))
        img = img > 0
        seg_filename = (
            str(pathlib.Path(output_segmentation_dir) / pathlib.Path(seg).stem)
            + ".nii.gz"
        )
        sitk.WriteImage(img, seg_filename)


def _prepare_NSCLC_data(dataset_info, output_segmentation_dir, output_ct_dir):
    """

    Standardize 'NSCLC-Radiomics' data by saving the CT volumes and binary lung masks
    of lung regions in NIFTI.Filenames of each of the CT and their corresponding
    binary lung mask are saved with the same unique sub directory names which
    are under the heirarchy structure of root dataset directory (E.g: Each CT \
    volume and binary lung mask are saved by the same filenames of 'LUNG1-001.nii.gz',
    'LUNG1-002.nii.gz' .. which are the sub directory names under the root
     dataset directory of /input/path/to/NSCLC-Radiomics)

    Args:

        dataset_info(dict): path to json file in the format. The json
                                    file must contain
                                    'NSCLC-Radiomics':
                                        {'input_dir':/input/path/to/NSCLC-Radiomics,
                                         'ct_dirs_prefix':LUNG1-}
                                    as one of the key-value pair in the above format
                                    if user wants to standardize this dataset

        output_segmentation_dir(string): Output segmentation directory to
                                        save the binary lung mask niftis
                                        Images will be saved in the below format
                                        {output_segmentation_dir} / 'NSCLC-Radiomics' /
                                        {sub_directory_name}.nii.gz
        output_ct_dir(string): Output CT directory to save the CT volume in niftis
                                        Images will be saved in the below format
                                        {output_ct_dir} / 'NSCLC-Radiomics' /
                                        {sub_directory_name}.nii.gz
    Returns:
        ---

    """
    if not pathlib.Path(output_ct_dir).is_dir():
        os.mkdir(output_ct_dir)
    if not pathlib.Path(output_segmentation_dir).is_dir():
        os.mkdir(output_segmentation_dir)

    ct_dirs_prefixes = str(
        pathlib.Path(dataset_info["input_dir"]) / dataset_info["ct_dirs_prefix"] / "*"
    )

    patientIDs = glob.glob(ct_dirs_prefixes)

    for patientID in patientIDs:
        input_patient_dir = (
            pathlib.Path(dataset_info["input_dir"]).absolute() / patientID
        )
        output_ct_filename = str(output_ct_dir.absolute() / patientID) + ".nii.gz"

        output_seg_filename = (
            str(output_segmentation_dir.absolute() / patientID) + ".nii.gz"
        )

        _save_data_to_nifti(input_patient_dir, output_seg_filename, output_ct_filename)

        _filter_lunglabels_from_NSCLC_RTStruct_to_NIFTI_output(output_seg_filename)


def _prepare_VESSEL12_data(dataset_info, output_segmentation_dir, output_ct_dir):
    """

    Standardize 'VESSEL12' data by saving the CT volumes and binary lung masks
    in NIFTI. This function grabs mhd format of CTs and lung binary masks from
    their repsective input directories and  converts them through SimpleITK and
    saves them in NIFTI format.CTs and binary lung
    masks are saved under the same filename based on their original filename.

    Args:

        dataset_info(dict): path to json file in the format. The json
                                    file must contain
                                    'VESSEL12':{'input_dir':/hpcdata/bcbb/ct_data/VESSEL12,
                                                 'ct_dir':cts,
                                                 'seg_dir':lung_masks}
                                    as one of the key-value pair in the above format
                                    if user wants to standardize this dataset

        output_segmentation_dir(string): Output segmentation directory to
                                        save the binary lung mask niftis
                                        Images will be saved in the below format
                                        {output_segmentation_dir} / 'VESSEL12' /
                                        {filename}.nii.gz
        output_ct_dir(string): Output CT directory to save the CT volume in niftis
                                        Images will be saved in the below format
                                        {output_ct_dir} / 'VESSEL12' /
                                        {filename}.nii.gz
    Returns:
        ---

    """
    if not pathlib.Path(output_ct_dir).is_dir():
        os.mkdir(output_ct_dir)
    if not pathlib.Path(output_segmentation_dir).is_dir():
        os.mkdir(output_segmentation_dir)

    # For CTs
    cts = glob.glob(
        str(pathlib.Path(dataset_info["input_dir"]) / dataset_info["ct_dir"] / "*.mhd")
    )

    for ct in cts:
        img = sitk.ReadImage(ct)
        sitk.WriteImage(img, str(output_ct_dir / pathlib.Path(ct).stem) + ".nii.gz")
    # For Lung masks
    segs = glob.glob(
        str(pathlib.Path(dataset_info["input_dir"]) / dataset_info["seg_dir"] / "*.mhd")
    )
    for seg in segs:
        img = sitk.ReadImage(seg)
        seg_filename = str(output_segmentation_dir / pathlib.Path(seg).stem) + ".nii.gz"
        sitk.WriteImage(img, seg_filename)


def _prepare_tbportals_internal_data(
    dataset_info, output_segmentation_dir, output_ct_dir
):
    """

    Standardize TB portals internal ('Belarus') data by saving the CT volumes and binary lung masks
    in NIFTI. This function grabs CTs and lesions containing labels from
    their repsective input directories and  converts them through SimpleITK and
    saves them in NIFTI format.

    Args:

        dataset_info(dict): path to json file in the format. The json
                                    file must contain
                                    'Belarus':{'input_dir':/path/to/ManualLesionSegmentationsBelarusTeam,
                                                 'ct_dir':'CT',
                                                 'seg_dir':'labelImages'}
                                    as one of the key-value pair in the above format
                                    if user wants to standardize this dataset

        output_segmentation_dir(string): Output segmentation directory to
                                        save the binary lung mask niftis
                                        Images will be saved in the below format
                                        {output_segmentation_dir} / 'Belarus' /
                                        {filename}.nii.gz
        output_ct_dir(string): Output CT directory to save the CT volume in niftis
                                        Images will be saved in the below format
                                        {output_ct_dir} / 'Belarus' /
                                        {filename}_lesions.nii.gz
    Returns:
        ---

    """
    if not pathlib.Path(output_ct_dir).is_dir():
        os.mkdir(output_ct_dir)
    if not pathlib.Path(output_segmentation_dir).is_dir():
        os.mkdir(output_segmentation_dir)

    # For CTs
    cts = glob.glob(
        str(
            pathlib.Path(dataset_info["input_dir"])
            / dataset_info["ct_dir"]
            / "*.nii.gz"
        )
    )

    for ct in cts:
        img = sitk.ReadImage(ct)
        sitk.WriteImage(img, str(output_ct_dir / pathlib.Path(ct).stem) + ".nii.gz")
    # For Lung masks
    segs = glob.glob(
        str(
            pathlib.Path(dataset_info["input_dir"])
            / dataset_info["seg_dir"]
            / "*.nii.gz"
        )
    )
    for seg in segs:
        img = sitk.ReadImage(seg)
        img = img > 0
        seg_filename = str(output_segmentation_dir / pathlib.Path(seg).stem) + ".nii.gz"
        sitk.WriteImage(img, seg_filename)


def standardize_datasets(
    dataset_information: dict, output_segmentation_dir, output_ct_dir
):

    if not pathlib.Path(output_ct_dir).is_dir():
        os.mkdir(output_ct_dir)
    if not pathlib.Path(output_segmentation_dir).is_dir():
        os.mkdir(output_segmentation_dir)

    _prepare_LCTSC_data(
        dataset_information["LCTSC"],
        pathlib.Path(output_segmentation_dir) / "LCTSC",
        pathlib.Path(output_ct_dir) / "LCTSC",
    )
    _prepare_covid19_data(
        dataset_information["covid19"],
        pathlib.Path(output_segmentation_dir) / "covid19",
        pathlib.Path(output_ct_dir) / "covid19",
    )
    _prepare_LUNA16_data(
        dataset_information["LUNA16"],
        pathlib.Path(output_segmentation_dir) / "LUNA16",
        pathlib.Path(output_ct_dir) / "LUNA16",
    )
    _prepare_mdpi_data(
        dataset_information["mdpi"],
        pathlib.Path(output_segmentation_dir) / "mdpi",
        pathlib.Path(output_ct_dir) / "mdpi",
    )
    _prepare_NSCLC_data(
        dataset_information["NSCLC-Radiomics"],
        pathlib.Path(output_segmentation_dir) / "NSCLC-Radiomics",
        pathlib.Path(output_ct_dir) / "NSCLC-Radiomics",
    )
    _prepare_VESSEL12_data(
        dataset_information["VESSEL12"],
        pathlib.Path(output_segmentation_dir) / "VESSEL12",
        pathlib.Path(output_ct_dir) / "VESSEL12",
    )
    _prepare_tbportals_internal_data(
        dataset_information["Belarus"],
        pathlib.Path(output_segmentation_dir) / "Belarus",
        pathlib.Path(output_ct_dir) / "Belarus",
    )


def _save_csv_img_paths(
    dataset_info, cts_niftis_dir, segs_niftis_dir, output_csv_filename
):

    all_cts = []
    all_segs = []
    all_datasets = []
    for dataset in dataset_info.keys():
        cts = glob.glob(str(pathlib.Path(cts_niftis_dir) / dataset / "*"))
        segs = glob.glob(str(pathlib.Path(segs_niftis_dir) / dataset / "*"))

        all_cts = all_cts + cts
        all_segs = all_segs + segs
        all_datasets = all_datasets + [dataset] * len(cts)

    relative_path_all_cts = [
        pathlib.Path(ct).relative_to(pathlib.Path(cts_niftis_dir).parent)
        for ct in all_cts
    ]
    relative_path_all_segs = [
        pathlib.Path(seg).relative_to(pathlib.Path(segs_niftis_dir).parent)
        for seg in all_segs
    ]
    df = pd.DataFrame(
        {
            "ct_file": relative_path_all_cts,
            "ref_file": relative_path_all_segs,
            "dataset": all_datasets,
        }
    )

    df.to_csv(output_csv_filename, index=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Data preparation for TB Portals CTs.")
    parser.add_argument(
        "dataset_info_json_path",
        type=pathlib.Path,
        help="Path to JSON file containing each dataset's keys and their respective input paths for \
                cts and binary masks as values . E.g: {'LCTSC':{'input_dir': /input/path/to/dir/LCTSC},\
                                      'covid19':{'input_dir':/input/path/to/dir/covid19, \
                                               'ct_dir': ct_directory_name,  \
                                               'seg_dir': seg_directory_name },\
                                       'LUNA16':{'input_dir':/input/path/to/dir/LUNA16,\
                                                 'ct_dirs_prefix':'subset',\
                                                 'seg_dir':'seg-lungs-LUNA16'},\
                                        'mdpi':{'input_dir':/input/path/to/dir/mdpi,\
                                                'ct_dirs_prefix':'patients_',\
                                                'ct_filename':CT.nrrd,\
                                                 'seg_filename':GMM_LABELS.nrrd},\
                                        'NSCLC-Radiomics':{'input_dir':/input/path/to/dir/NSCLC-Radiomics,\
                                                           'ct_dirs_prefix':LUNG1-},\
                                        'VESSEL12':{'input_dir':/input/path/to/dir/VESSEL12,\
                                                    'ct_dir':cts,\
                                                    'seg_dir':lung_masks}}",
    )

    parser.add_argument(
        "output_segmentation_dir",
        type=str,
        help="Output lung segmentation directory to save each dataset's binary lung masks\
              in NIFTI format",
    )
    parser.add_argument(
        "output_ct_dir",
        type=str,
        help="Output CT directory to save each dataset's CT images in NIFTI format",
    )
    parser.add_argument(
        "output_csv_filename",
        type=str,
        help="CSV filename containing column names as 'ct_file' , 'seg_file' and \
              'dataset' as columns,representing CT path and the corresponding\
              binary lung masks and the corresponding dataset each CT and \
              binary segmentation belonging to respectively.",
    )

    args = parser.parse_args()

    with open(args.dataset_info_json_path) as f:
        dataset_info = json.load(f)

    standardize_datasets(
        dataset_information=dataset_info,
        output_segmentation_dir=args.output_segmentation_dir,
        output_ct_dir=args.output_ct_dir,
    )

    _save_csv_img_paths(
        args.dataset_info_json_path,
        cts_niftis_dir=args.output_ct_dir,
        segs_niftis_dir=args.output_segmentation_dir,
        output_csv_filename=args.output_csv_filename,
    )


if __name__ == "__main__":
    main()
