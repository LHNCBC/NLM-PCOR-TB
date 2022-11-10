# Understanding the relationship of radiological and genomic information on drug-resistant TB, drug-sensitive TB, and TB treatment period.

License
/*==============================================================================

Copyright 2022 The PCOR project. All Rights Reserved. This was developed under contract funded by the National Library of Medicine (NLM), which is part of the National Institutes of Health, an agency of the Department of Health and Human Services, United States Government. Licensed under GNU General Public License v3.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at https://www.gnu.org/licenses/gpl-3.0.html Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

==============================================================================*/

## Authors
Vy Bui (vy.bui@nih.gov)
Feng Yang (feng.yang2@nih.gov)
Babak Haghighi (babak.haghighi@nih.gov)
Hang Yu (hang.yu@nih.gov)
Karthik Kantipudi (karthik.kantipudi@nih.gov)
Ziv Rafael Yaniv (zivrafael.yaniv@nih.gov)
Stefan Jaeger (stefan.jaeger@nih.gov)

## Contact information
For questions about the software and research, please contact Vy Bui (vy.bui@nih.gov), Ziv Yaniv (ziv.yaniv@nih.gov)  or Dr. Stefan Jaeger (stefan.jaeger@nih.gov)

## Setting up virtual environment and installing necessary python packages
YML file is provided to install neccessary packages for users. Use the terminal or an Anaconda Prompt for the following steps: Create the environment from the environment.yml file:

```
conda env create -f environment.yml
```
Activate the new environment:

```
conda activate radgentb
```

Verify that the new environment was installed correctly:
```
conda env list
```

### Using the codes includes two parts: (1) Investigating the relationship between radiological and genomic features on drug-resistant TB, drug-sensitive TB prediction and (2) Investigating the associations between radiological and genomic features with the TB treatment period.

## (1) Investigating the relationship between radiological and genomic features on drug-resistant TB, drug-sensitive TB prediction

Description of arguments:
usage: F_rad_gen_drds_classification.py 
                                [-h] 
                                [-rad Add radiological features]
                                [-gen Add genomic features] 
                                [-same Use the same samples] 
                                [-rp Radiological csv file location] 
                                [-gp Genomic csv file location]
                                [-tp TB Profiler csv file location]
                                [-db debug path]    
                                             
DR DS Classifier using radiological and/or genomic features

optional arguments:
  -h, --help    show this help message and exit
  -rad Add radiological features
            Specify if users want to use radiological features.
  -gen Add genomic features
            Specify if users want to use genomic features.
  -same Use the same samples
            Specify if users want to use the same samples 
            which have both radiological and genomic information.
  -rp Radiological csv file location
            Path to radiological csv file. 
            Default:../data/TB_Portals_Patient_Cases_January_2022.csv
  -gp Genomic csv file location
            Path to genomic csv file. 
            Default:../data/TB_Portals_Genomics_January_2022.csv
  -tp TB Profiler csv file location
            Path to TB Profiler csv file. 
            Default:../data/TB2258-variantDetail.csv
  -db debug path    
            Debug Path. 
            Default:../out/

Example 1: performing classification using both radiological and genomic features:

```
python F_rad_gen_drds_classification.py -rad 1 -gen 1 -same 1
```

Example 2: performing classification using only radiological features on the same sample size in Example 1. Without setting the -same flag, the program uses all the samples that have radiological information in the entire dataset.

```
python F_rad_gen_drds_classification.py -rad 1 -same 1
```

Example 3: performing classification using only genomic features on the same sample size in Example 1. Without setting the -same flag, the program uses all the samples that have genomic information in the entire dataset. 

```
python F_rad_gen_drds_classification.py -gen 1 -same 1
```

Example 4: The following example is shown to point to the radiological csv file located in /data/TB\_Portals\_Patient\_Cases.csv by setting up the -rp flag. Similar, one can point to the desired genomic or TB Profiler data locations using -fp and -tp flags.

```
python F_rad_gen_drds_classification.py -rad 1 -gen 1 -same 1 
                -rp '/data/TB_Portals_Patient_Cases.csv'
```



## (2) Investigating the associations between radiological and genomic features with the TB treatment period.

Description of arguments:
usage: F_rad_gen_TP_regression.py 
                            [-h] 
			    [-rad Add radiological features]
                            [-gen Add genomic features] 
                            [-same Use the same samples] 
                            [-pdrug Predict treatment period of specific popular drug combination]
                            [-rp Radiological csv file location]
                            [-gp Genomic csv file location]
                            [-tp TB Profiler csv file location]
                            [-dstp DST csv file location]
                            [-regimenp Regimen csv file location]
                            [-db debug path]

Predicting treatment period using radiological 
                       and/or genomic features

optional arguments:
  -h, --help    show this help message and exit
  -rad Add radiological features
            Specify if users want to use radiological features.
  -gen Add genomic features
            Specify if users want to use genomic features.
  -same Use the same samples
            Specify if users want to use the same samples 
            which have both radiological and genomic information.
  -pdrug Predict treatment period of specific drug combination
            Specify if users want to predict treatment period 
            of specific drug combination
            Path to TB Profiler csv file. 
            Default:../data/TB2258-variantDetail.csv
  -dstp DST csv file location
            Path to TB Profiler csv file. 
            Default:../data/TB_Portals_DST_January_2022.csv
  -regimenp Regimen csv file location
            Path to TB Profiler csv file. 
            Default:../data/TB_Portals_Regimens_January_2022.csv
  -db debug path        
            Debug Path. Default:../out/

Example 1: performing TB treatment period regression using both radiological and genomic features to predict treatment period of all drug combinations (-pdrug flag is 0).

```
python F_rad_gen_TP_regression.py -rad 1 -gen 1 -same 1 -pdrug 0
```

Example 2: performing TB treatment period regression using both radiological and genomic features to predict treatment period of the most popular drug combination (-pdrug flag is 1).

```
python F_rad_gen_TP_regression.py -rad 1 -gen 1 -same 1 -pdrug 1
```

Example 3: performing TB treatment period regression using only radiological features on the same sample size in Example 1. Without setting the -same flag, the program uses all the samples that have radiological information in the entire dataset.

```
python F_rad_gen_TP_regression.py -rad 1 -same 1 -pdrug 1
```

Example 4: performing TB treatment period regression using only genomic features on the same sample size in Example 1. Without setting the -same flag, the program uses all the samples that have genomic information in the entire dataset. 
```
python F_rad_gen_TP_regression.py -gen 1 -same 1 -pdrug 1
```

## Dataset
The model was trained on [TB Portals](https://tbportals.niaid.nih.gov/download-data)

## References
Please see the below reference for more information:

[1] Yang, F., Yu, H., Kantipudi, K., Karki, M., Kassim, Y. M., Rosenthal, A., ... & Jaeger, S. (2022). Differentiating between drug-sensitive and drug-resistant tuberculosis with machine learning for clinical and radiological features. Quantitative Imaging in Medicine and Surgery, 12(1), 675.

[2] Yang, F., Yu, H., Kantipudi, K., Rosenthal, A., Hurt, D. E., Yaniv, Z., & Jaeger, S. (2021, October). Automated Drug-Resistant TB Screening: Importance of Demographic Features and Radiological Findings in Chest X-Ray. In 2021 IEEE Applied Imagery Pattern Recognition Workshop (AIPR) (pp. 1-4). IEEE.
