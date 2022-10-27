# Classification of a Chest-X Ray as Drug Resistant TB (DR-TB) or Drug Sensitive TB (DS-TB)

This directory contains code for training and inference using Chest X Rays to predict DR-TB vs DS-TB

License
/*==============================================================================

Copyright 2022 The PCOR project. All Rights Reserved. This was developed under contract funded by the National Library of Medicine (NLM), which is part of the National Institutes of Health, an agency of the Department of Health and Human Services, United States Government. Licensed under GNU General Public License v3.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at https://www.gnu.org/licenses/gpl-3.0.html Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

==============================================================================*/

Authors
Manohar Karki (mkarki2@gmail.com)
Karthik Kantipudi (karthik.kantipudi@nih.gov)
Feng Yang (feng.yang2@nih.gov)
Hang Yu (hang.yu@nih.gov)
Ziv Rafael Yaniv (zivrafael.yaniv@nih.gov)
Stefan Jaeger (stefan.jaeger@nih.gov)

Contact information
For questions about the software and research, please contact Karthik Kantipudi (karthik.kantipudi@nih.gov), Ziv Yaniv (ziv.yaniv@nih.gov)  or Dr. Stefan Jaeger (stefan.jaeger@nih.gov)

Setting up virtual environment and installing necessary python packages
YML file is provided to install neccessary packages for users. Use the terminal or an Anaconda Prompt for the following steps: Create the environment from the environment.yml file:

```
conda env create -f environment.yml
```
Activate the new environment:

```
conda activate dr_ds_tb
```

Verify that the new environment was installed correctly:
```
conda env list
```

### Using the codes includes two steps: (1) training and (2) inference.

## Training

Training has been done using various network architectures with pretrained 'imagenet' weights as the initialization. From all the models that were trained, InceptionV3 model was finally selected based on their performances. For more details, user can refer to [this](https://pubmed.ncbi.nlm.nih.gov/34891867/#:~:text=Early%20diagnosis%20of%20drug%20resistance,resistant%20and%20drug%2Dsensitive%20tuberculosis.) paper for the training strategies that were implemented.

When user provides input csv path(with columns names as 'image_file' and 'dr_ds_label') as in input in the training command below, the script first preprocesses the Chest X Ray image file paths using below steps.

This process happens in 3 stages:
    1.) Reading and Resampling of images from the image file paths.
    2.) Predict masks from the array of resampled image array
    3.) Segment lungs from the original images using the predicted masks

Segmented lung image array that contains the cropped lung images and labels(1- For Drug Resistant / 0- Drug Sensitive) are saved in 'preprocessed_data_path.h5' file in a temporary directory.

This file is then automatically read from the temporary directory by the training script,
which then prepares the data to be fed to the 'train' function for training the model.

Example of training:

```
python dr_ds_train.py input_csv_path model_output_filename
```


Description of arguments above:


(1) [required] input_csv_path: (CSV Path) Input CSV file path with column names as 'image_file','dr_ds_label'. 'image_file' represents the file path of the image
and 'dr_ds_label' represents the label(R - Drug Resistant TB/ S- Drug Sensitive TB)

(2) [required] model_output_filename:: (Pathlib path) Output model file path to save

(3) [optional] lung_segmentation_model_path: (Pathlib path) Input lung segmentation model path to segment Chest X Rays to lung regions.
Note: More adjustments are possible by changing the global_constants.py file

(4) [optional] gpu_id:: (string) GPU id to train the model on.

(5) [optional] batch_size: (int) Batch size for the model to train in a single iteration.(Default: 8)

(5) [optional] epochs: (int) Number of epochs to train the model (Default: 100)

(5) [optional] lr: (float) Learning Rate (Default: 0.0001)

(5) [optional] random_seed: (int) Random seed to split the data into training and validation (Default: 42)

Augmentation parameters can be changed to decide the amount of augmentation data to produce and the parameters for image transformations.

## Inference

To infer on the files, user has to provide input_filenames.csv (with 'image_file'
as file path for each Chest-X Ray image) and output_predictions.csv (Output CSV
File that is saved with 'image_file', 'pred_label','confidence' as columns.
When the given command below is executed, the image files undergo the same preprocessing steps as described in the training section , before the trained model generates the predictions
on the preprocessed array.

'weights' folder in this repository contains InceptionV3 model weights (7th fold) and is selected for this repository based on its results on corresponding test set.It generated highest AUC on its test set when compared to the rest of 10-fold trained models.

 In the output_predictions.csv file,
  'pred_label' values are either DR/DS based on the probability('Confidence')
  and the default threshold that we gave(if probability > threshold -> 'DR' else 'DS').
  User can change the threshold by adding argument as --threshold in the command line.
  By default, the below command utilizes the weights file(dr_ds_tb.h5), stored in the 'weights' subfolder.

```
python dr_ds_inference.py input_filenames.csv output_predictions.csv --threshold 0.5
```
Description of arguments above:


(1) [required] input_csv_path: (CSV Path) Input CSV file path with column names as 'image_file'. 'image_file' represents the file path of the image


(2) [required] model_output_filename:: (Pathlib path) Output model file path to save

(3) [optional] lung_segmentation_model_path: (Pathlib path) Input lung segmentation model path to segment Chest X Rays to lung regions.
Note: More adjustments are possible by changing the global_constants.py file

(4) [optional] gpu_id:: (string) GPU id to train the model on.

(5) [optional] batch_size: (int) Batch size for the model to train in a single iteration.(Default: 8)

(5) [optional] epochs: (int) Number of epochs to train the model (Default: 100)

(5) [optional] lr: (float) Learning Rate (Default: 0.0001)

(5) [optional] random_seed: (int) Random seed to split the data into training and validation (Default: 42)


## Dataset
The model was trained on [TB Portals](https://tbportals.niaid.nih.gov/download-data), [Shenzen](https://lhncbc.nlm.nih.gov/LHC-downloads/downloads.html), [Montgomery dataset](https://lhncbc.nlm.nih.gov/LHC-downloads/downloads.html) and [TBX11K](https://mmcheng.net/tb/) datasets. CSV file containing sample (n=24 images) file paths from TB Portals data is provided for users in the data folder for train and inference example.

## References
Please see the below reference for more information:

[1] Karki M, Kantipudi K, Yu H, Yang F, Kassim YM, Yaniv Z, Jaeger S. Identifying Drug-Resistant Tuberculosis in Chest Radiographs: Evaluation of CNN Architectures and Training Strategies. Annu Int Conf IEEE Eng Med Biol Soc. 2021 Nov;2021:2964-2967. doi: 10.1109/EMBC46164.2021.9630189. PMID: 34891867.

[2] Karki M, Kantipudi K, Yang F, Yu H, Wang YXJ, Yaniv Z, Jaeger S. Generalization Challenges in Drug-Resistant Tuberculosis Detection from Chest X-rays. Diagnostics (Basel). 2022 Jan 13;12(1):188. doi: 10.3390/diagnostics12010188. PMID: 35054355; PMCID: PMC8775073.
