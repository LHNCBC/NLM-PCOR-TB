
## Lung Segmentation of Chest-X Ray Images

This directory contains codes and insturction for performing lung segmentation using a UNET model.

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

Babak Haghighi (babak.haghighi@nih.gov)<br/>
Karthik Kantipudi (karthik.kantipudi@nih.gov)<br/>
Ziv Rafael Yaniv (zivrafael.yaniv@nih.gov)<br/>
Hang Yu (hang.yu@nih.gov)<br/>
Stefan Jaeger (stefan.jaeger@nih.gov)<br/>

## Contact information

For questions about the software and research, please contact Dr. Babak Haghighi (babak.haghighi@nih.gov) or Dr. Stefan Jaeger (stefan.jaeger@nih.gov)


## Setting up virtual environment and installing necessary python packages

YML file is provided to install neccessary packages for users. Use the terminal or an Anaconda Prompt for the following steps:
Create the environment from the environment.yml file:

conda env create -f environment.yml

Activate the new environment: 

conda activate myenv

Verify that the new environment was installed correctly:

conda env list

For more information please see (https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html)
 

### Using the codes includes two steps: (1) training and (2) inference.
## Training
To train a lung segmentation model:

Training requires an input CSV file which contains column names as 'images'
and 'masks' where 'images' represents the file path of the image
and 'masks' represents the file path of the related masks for training

A pretrained UNet with a ResNet50 backbone was utilized. The initial weights for the backbone were obtained from [this GitHub repository](https://github.com/qubvel/segmentation_models).


Example of training:
```
python train.py input_csv_path --lung_segmentation_model_output  model_output_filename 

```
(1) [required] input_csv_path: the path to input CSV file which contains column names as 'images'and 'masks' where 'images' represents the file path of the image and 'masks' for training
(2) [optional]--lung_segmentation_model_output: indicates the path to save the output of lung segmentation model (weights).
If not specified, the defalut path for saving the model is '../weights/Segmentation_resnet50_UNet.h5'
Also, the resulting weight file (*Segmentation_resnet50_UNet.h5*) is available in the weights folder.
The default hyper-parameters for the training are BATCH_SIZE=16, EPOCHS=100 and LEARNING_RATE=0.0001. A user can change and train a model with different hyper-parameter values.
The model was trained on NIH ShenZhen X-ray images (https://openi.nlm.nih.gov/faq#collection)
NVIDIA Tesla K80 GPU with 24 GB of GDDR5 memory was used for training. 


## Inference
Inference requires an input CSV file which contains column names as 'images' used for inference and output_path for saving output binary images (segmented images)


Example of inference:
```
python inference.py input_csv_path output_prediction_directory --segmnetation_model_path model_path

```
(1) [required] input_csv_path: the path to input CSV file which contains column names as 'images' for inference
(2) [required] output_prediction_directory: the folder for saving the predicted binary images 
(3) [optional] --segmnetation_model_path: the path for reading the saved wieghts from the training step to run the inference. 
If not specified, the defalut path for loading is '../weights/Segmentation_resnet50_UNet.h5'

## Samples images and the related masks from NIH ShenZhen X-ray images are provided for users in the data folder for train and inference example. 
The whole data set can be downloaded from (https://openi.nlm.nih.gov/faq#collection) 

##Please see the below reference for more information:
[1] S. Jaeger, S. Candemir, S. Antani, Y.-X. J. Wáng, P.-X. Lu, and G. Thoma, "Two public chest X-ray datasets for computer-aided screening of pulmonary diseases," 
Quantitative imaging in medicine and surgery, vol. 4 (6), p. 475(2014)





