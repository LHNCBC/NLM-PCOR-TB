
## TB classification of Chest-X Ray Images

This directory contains codes for performing lung TB classification using a deep learning model.

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

Babak Haghighi (babak.haghighi@nih.gov)<br/>
Karthik Kantipudi (karthik.kantipudi@nih.gov)<br/>
Hang Yu (hang.yu@nih.gov)<br/>
Ziv Rafael Yaniv (zivrafael.yaniv@nih.gov)<br/>
Stefan Jaeger (stefan.jaeger@nih.gov)<br/>

## Contact information

For questions about the software and research, please contact Dr. Babak Haghighi (babak.haghighi@nih.gov) or Dr. Stefan Jaeger (stefan.jaeger@nih.gov)


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
 

### Using the codes includes two steps: (1) training and (2) inference.
## Training
To train a classification model:

Training requires an input CSV file which contains column names of 'images' and their "labels"

A pretrained model with Xception backbone was utilized. The initial weights for the backbone were obtained from Keras application module.


Example of training:
```
python train.py images_labels_csv --model_output_path  model_output_filename 

```
(1) [required] input_csv_path: the path to input CSV file which contains column names as 'images'and 'labels' for training

(2) [optional]--model_output_path: indicates the path to save the output of classification model (weights).
If not specified, the defalut path for saving the model is '../weights/inceptionv3_fine_tuned.h5'


## Inference
Inference requires an input CSV file which contains column names as 'images' used for inference and output_path for saving output labels


Example of inference:
```
python inference.py input_csv_path prediction_csv --model_path model_path

```
(1) [required] input_csv_path: the path to input CSV file which contains column names as 'images' for inference

(2) [required] prediction_csv: the csv file path for saving the predicted labels 

(3) [optional] --model_path: the path for reading the saved wieghts from the training step to run the inference. 
If not specified, the defalut path for loading is '../weights/inceptionv3_fine_tuned.h5'

## Dataset

The model was trained on NIH ShenZhen X-ray images. Some sample images are provided for users in the data folder for train and inference example. 
Samples for CSV files for traing and inference are also provided. A user can modify them based on their data set. 
The whole data set can be downloaded from (https://openi.nlm.nih.gov/faq#collection). 

For any new data set, please copy images and masks to the data folder, then create the related csv files and perform the traing or inference.

## References
Please see the below reference for more information:

[1] S. Jaeger, S. Candemir, S. Antani, Y.-X. J. Wáng, P.-X. Lu, and G. Thoma, "Two public chest X-ray datasets for computer-aided screening of pulmonary diseases," 
Quantitative imaging in medicine and surgery, vol. 4 (6), p. 475(2014)

[2] M. Karki et al., “Generalization Challenges in Drug-Resistant Tuberculosis Detection from Chest X-rays,” Diagnostics, vol. 12, no. 1, Art. no. 1, Jan. 2022

