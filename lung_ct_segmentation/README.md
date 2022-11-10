
## Lung Segmentation in CT Scans

This directory contains codes for performing lung segmentation in CT volumes using a deep learning model.

## License

/*==============================================================================

Copyright 2022 The PCOR project. All Rights Reserved.
This  was developed under contract funded by the National Library of Medicine (NLM),
which is part of the National Institutes of Health, an agency of the Department of Health and Human
Services, United States Government.
Licensed under GNU General Public License v3.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    https://www.gnu.org/licenses/gpl-3.0.html
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

==============================================================================*/

## Authors

Karthik Kantipudi (karthik.kantipudi@nih.gov)<br/>
Babak Haghighi (babak.haghighi@nih.gov)<br/>
Hang Yu (hang.yu@nih.gov)<br/>
Ziv Rafael Yaniv (zivrafael.yaniv@nih.gov)<br/>
Stefan Jaeger (stefan.jaeger@nih.gov)<br/>

## Contact information

For questions about the software and research, please contact Karthik Kantipudi (karthik.kantipudi@nih.gov) or Dr. Ziv Yaniv (zivrafael.yaniv@nih.gov)


## Setting up virtual environment and installing necessary python packages

YML file is provided to install neccessary packages for users. Use the terminal or an Anaconda Prompt for the following steps:
Create the environment from the environment.yml file:
```
conda env create -f environment.yml
```

Activate the new environment:
```
conda activate myenv
```

Verify that the new environment was installed correctly:
```
conda env list
```

For more information please see (https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html)


It deals with training a 2d nnUNet model for lung segmentation.

For training lung segmentation models, we use all the available data from the publicly available datasets , i.e,
[LCTSC](https://wiki.cancerimagingarchive.net/display/Public/Lung+CT+Segmentation+Challenge+2017) ,
[NSCLC-Radiomics](https://wiki.cancerimagingarchive.net/display/Public/NSCLC-Radiomics) ,
[LUNA16](https://luna16.grand-challenge.org/),[mdpi](https://www.imagenglab.com/newsite/covid-19/) ,
[covid19](https://doi.org/10.5281/zenodo.3757476),[VESSEL12](https://vessel12.grand-challenge.org/Download/) , with
focal datasets being 'covid19' and 'mdpi' and diffused datasets being 'VESSEL12','LCTSC','NSCLC-Radiomics' and
'LUNA16'.
Prepare the inputs for this experiment by following the below steps.

```
python -m segment_lung_ct.data_preparation_code.data_prep segment_lung_ct/data/inputs/lung_segmentation_files.csv segment_lung_ct/data/inputs/sampling_factors_for_lung_segmentation.json 0 lung_segmentation
```

In the above command, complete_list_of_lung_segment_files.csv is the input CSV file containing column names as 'ct_file' and 'ref_seg_file' of all the CTs and their respective lung segmentations. 0 indicates the fold number in the command line and lung_segmentation indicates the main directory under which sub folders of "imagesTr","imagesTs","labelsTr","labelsTs" are created.These subfolders are required as per nnUNet training .

```
python -m segment_lung_ct.data_preparation_code.data_prep segment_lung_ct/data/inputs/lesion_segmentation_files.csv segment_lung_ct/data/inputs/sampling_factors_for_lesion_segmentation.json 0 lesion_segmentation
```

The similar command(above) has to be run again for generating imagesTr/imagesTs/labelsTr/labelsTs for lesion segmentation files.


After preparing the data for nnUNet training from above two commands, user will be seeing the following number of images for each dataset.

1. Lung Segmentation
Total Number of Train and Validation Images: 245
    * LCTSC - 48
    * NSCLC-Radiomics - 54
    * LUNA16 - 71
    * mdpi - 40
    * covid19 - 16
    * VESSEL12 - 16

    * Diffused(covid19+mdpi): 56
    * Focal(LCTSC+NSCLC-Radiomics+LUNA16+VESSEL12): 189

Total Number of Test Images: 290
    * LCTSC - 12
    * NSCLC-Radiomics - 82
    * LUNA16 - 178
    * mdpi - 10
    * covid19 - 4
    * VESSEL12 - 4

    * Diffused(covid19+mdpi): 14
    * Focal(LCTSC+NSCLC-Radiomics+LUNA16+VESSEL12): 276


User can also refer to [this](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/dataset_conversion.md) link for any further reference for setting up their databases.

Add below paths after preparing the data for nnunet.

```
export nnUNet_raw_data_base="/path/to/nnUNet_raw_data_base"
export nnUNet_preprocessed="/path/to/nnUNet_preprocessed"
export RESULTS_FOLDER='/path/to/nnUNet_trained_models'
```

Each of the lung/lesion directories for each task must contain a dataset.json file which is generated from the above commands.


User can also refer to (this)(https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/dataset_conversion.md) link for any further reference for setting up their databases.

## Preprocess the CT volumes:
```
python  code/preprocess_lung_segment_nnunet.py -task_number 001
```
Above command will preprocess the dataset using the task number provided while creating the

## Train the nnUNet model:
```
python code/train_lung_ct.py -task_number 001 -configuration 2d -cv_fold_number 0
```
Above command will train a lung/lesion segmentation model where the user must provide the cross validation number in [0,1,2,3,4] to train the model on a given fold.You can also provide the value 'all' for '-cv_fold_number' argument if the user wants to train the model on all of the available images instead of a fold-based paritioned dataset.User can also provide various configurations(2d,3dfull_res,3d_lowres,3d_cascade_fullres) for training their nnUNet.
For more info on this, read (this)[https://github.com/MIC-DKFZ/nnUNet] repository.

## Predict lung regions from sample CT volume:
After training the model user can then implement the below command to predict the segmentations.
The weight files are trained based on the concept of [nnUnet](https://arxiv.org/pdf/1809.10486.pdf) where the algorithm is  self adapting and contains minor modifications from original [Unet](https://arxiv.org/abs/1505.04597) architecture.

User can run below inference command that requires inputs such as input CT volume , trained segmentation model(lung or lesion) and the output segmentation filename. It is also highly recommended to run the inference command line on a GPU device to get a quicker prediction results.
```
python code/inference_lung_ct.py sample_volume.nii.gz  weights/lung_segment.model sample_volume_lung_preds.nii.gz --post_process
```

Note: User has to make sure that the pickle files present in the 'weights/' directory should be in the same directory as the segmentation model files. These pickle files that are present along with the weight files in the 'weights' directory contain dataset and model characteristics  that the  training model used while training.
In the command line above, user can provide an additional argument in the command line by just adding a flag '--post_process', if user wants postprocessing done to the combined prediction by the models. If user does not wish to post process the predicted mask, they can remove the argument '--post_process' from the command line.
