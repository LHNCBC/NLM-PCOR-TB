__author__ = "Vy Bui, Ph.D."
__email__ = "vy.bui@nih.gov"

import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing, metrics, model_selection, ensemble
import numpy as np
import os, re, datetime, argparse
from helpers import INVALID_INPUT_MSG, validate_file, correlation, chi2

CT_fields = ['image_body_site', 'dissemination', 'lungcavity_size',
       'anomaly_of_mediastinum_vessels_develop', 'affect_pleura',
       'shadow_pattern', 'affect_level', 'pneumothorax', 'plevritis',
       'affected_segments', 'nodicalcinatum', 'process_prevalence',
       'thromboembolism_of_the_pulmonaryartery', 'posttbresiduals',
       'lung_capacity_decrease', 'bronchial_obstruction',
       'anomaly_of_lungdevelop', 'accumulation_of_contrast', 'limfoadenopatia',
       'totalcavernum']
CXR_fields = ['overall_percent_of_abnormal_volume',
              'pleural_effusion_percent_of_hemithorax_involved',
              'ispleuraleffusionbilateral', 'other_non_tb_abnormalities',
              'are_mediastinal_lymphnodes_present', 'collapse',
              'smallcavities', 'mediumcavities', 'largecavities',
              'isanylargecavitybelongtoamultisextantcavity',
              'canmultiplecavitiesbeseen', 'infiltrate_lowgroundglassdensity',
              'infiltrate_mediumdensity', 'infiltrate_highdensity', 'smallnodules',
              'mediumnodules', 'largenodules', 'hugenodules',
              'isanycalcifiedorpartiallycalcifiednoduleexist',
              'isanynoncalcifiednoduleexist', 'isanyclusterednoduleexists',
              'aremultiplenoduleexists', 'lowgroundglassdensityactivefreshnodules',
              'mediumdensitystabalizedfibroticnodules',
              'highdensitycalcifiedtypicallysequella']
TCS_fields = ['period_start_x', 'period_end_x', 'treatment_status', 'stemcell_dose', 'social_risk_factors',
              'specimen','country','education','employment','number_of_children','lung_localization',
              'number_of_daily_contacts','bmi']
GENOMIC_fields = ['gene_name', 'high_confidence', 'hain', 'genexpert']
QURE_fields = ['qure_bluntedcp', 'qure_abnormal', 'qure_consolidation',
               'qure_fibrosis', 'qure_opacity', 'qure_peffusion', 'qure_tuberculosis',
               'qure_nodule', 'qure_cavity', 'qure_hilarlymphadenopathy', 'qure_atelectasis']
TBP_variants_fields = ['ahpC', 'ald', 'alr', 'ddn', 'eis', 'embA', 'embB', 'embC', 'embR',
                       'ethA', 'fabG1', 'fbiA', 'folC', 'gid', 'gyrA', 'gyrB', 'ethR',
                       'inhA', 'kasA', 'katG', 'mmpR5', 'panD', 'pncA', 'ribD', 'rplC',
                       'rpoB', 'rpoC', 'rpsA', 'rpsL', 'rrl', 'rrs', 'thyA', 'thyX', 'tlyA']
DST_merged_fields = ['specimen_id_x', 'specimen_id_y', 'specimen_collection_date_x', 'specimen_collection_date_y',
       'specimen_collection_site_x', 'specimen_collection_site_y', 'observation_id', 'microscopyresults',
       'firstmicroscopyresults','microscopytype','bactecresults','leresults','lpaotherresults','hainresults','genexpertresults',
       'firstcultureresults','cultureresults']
RAD_merged_fields = ['identifier','case_definition', 'age_of_onset_x', 'age_of_onset_y', 'gender_x', 'gender_y', 'type_of_resistance_y',
          'specimen_identifier', 'sra_id', 'ncbi_sra', 'ncbi_sourceorganism_x', 'ncbi_sourceorganism_y', 'ncbi_bioproject_x',
          'ncbi_bioproject_y', 'ncbi_biosample_x', 'ncbi_biosample_y', 'lineage_y', 'sit_designation',
          'high_confidence_snp_mutations', 'hain_snp_mutations', 'patient_id_x', 'registration_date',
          'diagnosis_code', 'x_ray_count', 'status', 'organization', 'x_ray_exists', 'ct_exists', 'genomic_data_exists',
          'comorbidity','outcome_y','microscopy','culturetype_x','rater','culture','culturetype_y']
CLINICAL_fields = ['identifier', 'case_definition', 'age_of_onset', 'gender', 'ncbi_sourceorganism', 'ncbi_bioproject',
              'ncbi_biosample', 'lineage', 'registration_date', 'diagnosis_code', 'x_ray_count', 'status',
              'organization', 'x_ray_exists', 'ct_exists', 'genomic_data_exists', 'comorbidity','microscopy',
              'rater','culture','patient_id_x', 'specimen_id', 'specimen_collection_date',
              'specimen_collection_site','culturetype_x','culturetype_y']
RAD_fields = ['age_of_onset', 'gender', 'ncbi_sourceorganism', 'ncbi_bioproject',
              'ncbi_biosample', 'lineage', 'specimen_identifier', 'sra_id', 'ncbi_sra',
              'high_confidence_snp_mutations','culturetype','patient_id','sit_designation']
DST_fields = ['observation_id', 'microscopyresults',
       'firstmicroscopyresults','microscopytype','bactecresults','leresults','lpaotherresults','hainresults','genexpertresults',
       'firstcultureresults','cultureresults']
REGIMEN_merged_fields = ['period_start_y', 'period_end_y', 'period_span_y', 'outcome_cd', 'activities_period_end',
           'activities_statusreason_cd','dose','collected','reinfusioned','patient_id_y','regimen_drug_y']
REGIMEN_fields = ['outcome_cd', 'activities_period_end','period_start','period_end',
           'activities_statusreason_cd','dose','collected','reinfusioned']

TOR = 'type_of_resistance'
OUTCOME = 'outcome'
RD = 'regimen_drug'
PS = 'period_span'

def preprocess_csv(args, r_df, g_df, t_df, dst_df, regi_df, out_folder):
    """
    This function preprocess the csv files, it merges the csv rows and remove unused columns
    Args:
        args (parser.parse_args()): inputs from the comand line
        r_df, g_df, t_df, dst_df, regi_df (pandas.DataFrame): pandas dataframes read from
            clinical, genomic, TB Profiler, DST profile, and Regimen csv spreadsheet.
        out_folder (string): path of folder to output the results
    Returns:
        df (pandas.DataFrame): return pandas dataframe
    """
    global TOR, OUTCOME, RD, PS

    if (args.gen[0] == 1 and args.rad[0] == 1) or args.same[0] == 1:
        # Merge two TB Portal Genomics
        df = pd.merge(g_df, t_df, how="inner", left_on='sra_id', right_on='sample')
        df.reset_index(drop=True, inplace=True)
        # Merge two TB Portal Genomics and Clinical based on condition_id
        df = pd.merge(df, r_df, how="inner", on="condition_id")
        df.reset_index(drop=True, inplace=True)
    elif args.gen[0] == 1 and args.rad[0] != 1 and args.same[0] != 1:
        # Merge two TB Portal Genomics
        df = pd.merge(g_df, t_df, how="inner", left_on='sra_id', right_on='sample')
        df.reset_index(drop=True, inplace=True)
    elif args.gen[0] != 1 and args.rad[0] == 1 and args.same[0] != 1:
        df = r_df

    # Merge DST
    df = pd.merge(df, dst_df, how="inner", on="condition_id")
    df.reset_index(drop=True, inplace=True)

    # Merge Regimen
    df = pd.merge(df, regi_df, how="inner", on="condition_id")
    df.reset_index(drop=True, inplace=True)

    df.to_csv(out_folder + '/merged.csv', index=True)

    TOR = [col for col in df if col.startswith(TOR)][0]
    OUTCOME = [col for col in df if col.startswith(OUTCOME)][0]
    RD = [col for col in df if col.startswith(RD)][0]
    PS = [col for col in df if col.startswith(PS)][0]

    # drop rows that contain any value in the list
    drop_values = ['Poly DR', 'Pre-XDR', 'Not Reported', 'Negative']
    df = df[df[TOR].isin(drop_values) == False]

    drop_values = ['Not Reported']
    df = df[df[RD].isin(drop_values) == False]

    drop_values = ['Unknown', 'Palliative Care', 'Lost to follow up', 'Still on treatment']
    df = df[df[OUTCOME].isin(drop_values) == False]

    df = df[df[PS] >= 30]

    if (args.gen[0] == 1 and args.rad[0] == 1) or args.same[0] == 1:
        # Preprocess: drop unuse columns
        df = df.drop(columns=CT_fields, axis=1)
        df = df.drop(columns=DST_merged_fields, axis=1)
        df = df.drop(columns=TCS_fields, axis=1)
        df = df.drop(columns=QURE_fields, axis=1)
        df = df.drop(columns=GENOMIC_fields, axis=1)
        df = df.drop(columns=RAD_merged_fields, axis=1)
        df = df.drop(columns=REGIMEN_merged_fields, axis=1)

    if args.gen[0] != 1 and args.rad[0] == 1 and args.same[0] != 1:
        # Preprocess: drop unuse columns
        df = df.drop(columns=CT_fields, axis=1)
        df = df.drop(columns=TCS_fields, axis=1)
        df = df.drop(columns=QURE_fields, axis=1)
        df = df.drop(columns=CLINICAL_fields, axis=1)
        df = df.drop(columns=DST_fields, axis=1)
        df = df.drop(columns=GENOMIC_fields, axis=1)
        df = df.drop(columns=REGIMEN_merged_fields, axis=1)

    if args.gen[0] == 1 and args.rad[0] != 1 and args.same[0] != 1:
        # Preprocess: drop unuse columns
        df = df.drop(columns=DST_merged_fields, axis=1)
        df = df.drop(columns=REGIMEN_fields, axis=1)
        df = df.drop(columns=RAD_fields, axis=1)

    df = df[df.columns.drop(list(df.filter(regex='genexpert_')))]
    df = df[df.columns.drop(list(df.filter(regex='hain_')))]
    df = df[df.columns.drop(list(df.filter(regex='lpaother_')))]
    df = df[df.columns.drop(list(df.filter(regex='le_')))]
    df = df[df.columns.drop(list(df.filter(regex='bactec_')))]
    df = df.drop(columns=['test_date'], axis=1)

    if args.rad[0] == 1 or args.same[0] == 1:
        drop_values = ['Not Reported']
        for column_name in CXR_fields:
            df = df[df[column_name].isin(drop_values) == False]

    # Filter out first treatment
    df = df.sort_values(by=['condition_id', 'activities_period_start'])
    df = df.drop_duplicates(subset=['condition_id'], keep='first')
    df = df.drop(columns=['condition_id'], axis=1)
    df = df.drop(columns=['activities_period_start'], axis=1)

    df = df[df.columns].replace({'MDR non XDR': 'DR', 'XDR': 'DR', 'Mono DR': 'DR'})
    df.to_csv(out_folder + '/pre_processed.csv', index=True)
    return df

def popular_drug(args, df, out_folder):
    """
    This function finds the top 20 most popular drug combinations
    and computes the conditional probabilities between the outcome, regimen_count, and regimen_drug
    Args:
        args (parser.parse_args()): inputs from the comand line
        df (pandas.DataFrame): pandas dataframes to use
        out_folder (string): path of folder to output the results
    Returns:
        df (pandas.DataFrame): return pandas dataframe
    """
    global OUTCOME, RD

    """Finding the total categories"""
    dfo = pd.DataFrame()
    dfo = pd.concat([dfo, df[RD].value_counts().sort_values(ascending=False)])
    dfo.to_csv(out_folder + '/drug_combination.csv')

    """Keep the most porpular regimen"""
    popular_regimen = df[RD].value_counts().index.tolist()[0] # 0: the most popular, 1: the second most popular, etc.
    df = df[df[RD].isin([popular_regimen]) == True]
    # popular_regimen = df[RD].value_counts().index.tolist()[:3]
    # df = df[df[RD].isin(popular_regimen) == False]
    df.to_csv(out_folder + '/popular_regimen.csv', index=True)

    """Confitional Probabilities"""
    if args.rad[0] == 1 or args.same[0] == 1:
        pdf = df.filter([OUTCOME, 'regimen_count', RD], axis=1)
        pdf.to_csv(out_folder + '/pdf.csv', index=True)

        cp = pd.crosstab(pdf['regimen_count'], pdf[OUTCOME], normalize="columns")
        cp.to_csv(out_folder + '/cond_prob_regimen_count_outcome.csv', index=True)
        cp = pd.crosstab(pdf[OUTCOME], pdf['regimen_count'], normalize="columns")
        cp.to_csv(out_folder + '/cond_prob_outcome_regimen_count.csv', index=True)

        cp = pd.crosstab(pdf[RD], pdf[OUTCOME], normalize="columns")
        cp.to_csv(out_folder + '/cond_prob_regimen_drug_outcome.csv', index=True)
        cp = pd.crosstab(pdf[OUTCOME], pdf[RD], normalize="columns")
        cp.to_csv(out_folder + '/cond_prob_outcome_regimen_drug.csv', index=True)

        cp = pd.crosstab(pdf[RD], pdf['regimen_count'], normalize="columns")
        cp.to_csv(out_folder + '/cond_prob_regimen_drug_regimen_count.csv', index=True)
        cp = pd.crosstab(pdf['regimen_count'], pdf[RD], normalize="columns")
        cp.to_csv(out_folder + '/cond_prob_regimen_count_regimen_drug.csv', index=True)
    else:
        pdf = df.filter([OUTCOME, RD], axis=1)
        pdf.to_csv(out_folder + '/pdf.csv', index=True)

        cp = pd.crosstab(pdf[RD], pdf[OUTCOME], normalize="columns")
        cp.to_csv(out_folder + '/cond_prob_regimen_drug_outcome.csv', index=True)
        cp = pd.crosstab(pdf[OUTCOME], pdf[RD], normalize="columns")
        cp.to_csv(out_folder + '/cond_prob_outcome_regimen_drug.csv', index=True)

    return df

def encoding(args, df, r_df, g_df, v_df, out_folder):
    """
    This function converts the categorical features to numeric type as the machine learning models
    only received numeric inputs.
    Args:
        args (parser.parse_args()): inputs from the comand line
        df, r_df, g_df, v_df (pandas.DataFrame): pandas dataframes to use
        out_folder (string): path of folder to output the results
    Returns:
        df (pandas.DataFrame): return pandas dataframe
    """

    global TOR, OUTCOME, RD
    df = df.drop(columns=[OUTCOME], axis=1)
    df = df.drop(columns=[RD], axis=1)
    if args.same[0] == 1 or args.rad[0] == 1:
        df = df.drop(columns=['regimen_count'], axis=1)

    # Encoding
    enc = preprocessing.OneHotEncoder()
    categorical = df[[TOR]]

    # Missing values in categorical features
    for item in categorical.columns[categorical.isnull().any()].tolist():
        categorical[item].fillna('None', inplace=True)

    encoded_categ = pd.DataFrame(enc.fit_transform(categorical).toarray())
    encoded_categ.columns = enc.get_feature_names_out(categorical.columns)
    df = df.reset_index()
    encoded_categ = encoded_categ.reset_index()
    df = df.drop(columns=['index'], axis=1)
    encoded_categ = encoded_categ.drop(columns=['index'], axis=1)
    df = pd.concat([df, encoded_categ], axis=1)

    if args.rad[0] == 1:
        df = df.fillna("None")
        df.replace(re.compile('.*Yes.*'), 'Yes', inplace=True)
        df.replace(re.compile('Upper.*'), 'No', inplace=True)
        df.replace(re.compile('Middle.*'), 'No', inplace=True)
        df.replace(re.compile('Lower.*'), 'No', inplace=True)
        df.replace(re.compile('None'), 'No', inplace=True)
        df['other_non_tb_abnormalities'] =  df['other_non_tb_abnormalities']\
            .replace({'No, Yes': 'Yes'})
        df['pleural_effusion_percent_of_hemithorax_involved'] =  \
            df['pleural_effusion_percent_of_hemithorax_involved']\
                .replace({'0, Less than 50': 'Greater than 0'})
        df['pleural_effusion_percent_of_hemithorax_involved'] =  \
            df['pleural_effusion_percent_of_hemithorax_involved']\
                .replace({'Less than 50': 'Greater than 0'})
        df['pleural_effusion_percent_of_hemithorax_involved'] =  \
            df['pleural_effusion_percent_of_hemithorax_involved']\
                .replace({'Greater than or equal to 50, Less than 50': 'Greater than 0'})
        df['pleural_effusion_percent_of_hemithorax_involved'] =  \
            df['pleural_effusion_percent_of_hemithorax_involved']\
                .replace({'Greater than or equal to 50': 'Greater than 0'})
        df['overall_percent_of_abnormal_volume'] =  \
            df['overall_percent_of_abnormal_volume']\
                .replace({'0, Less than 50': 'Greater than 0'})
        df['overall_percent_of_abnormal_volume'] =  \
            df['overall_percent_of_abnormal_volume']\
                .replace({'Less than 50': 'Greater than 0'})
        df['overall_percent_of_abnormal_volume'] =  \
            df['overall_percent_of_abnormal_volume']\
                .replace({'Greater than or equal to 50, Less than 50': 'Greater than 0'})
        df['overall_percent_of_abnormal_volume'] =  \
            df['overall_percent_of_abnormal_volume']\
                .replace({'Greater than or equal to 50': 'Greater than 0'})
        df = df.assign(cavities = df['smallcavities'].astype(str) + ', ' +
                                  df['mediumcavities'].astype(str) + ', ' +
                                  df['largecavities'].astype(str))
        df = df.drop(columns=['smallcavities'], axis=1)
        df = df.drop(columns=['mediumcavities'], axis=1)
        df = df.drop(columns=['largecavities'], axis=1)
        df = df.assign(infiltrate =
                       df['infiltrate_lowgroundglassdensity'].astype(str) + ', ' +
                       df['infiltrate_mediumdensity'].astype(str) + ', ' +
                       df['infiltrate_highdensity'].astype(str))
        df = df.drop(columns=['infiltrate_lowgroundglassdensity'], axis=1)
        df = df.drop(columns=['infiltrate_mediumdensity'], axis=1)
        df = df.drop(columns=['infiltrate_highdensity'], axis=1)
        df = df.assign(nodules = df['smallnodules'].astype(str) + ', ' +
                                 df['mediumnodules'].astype(str) + ', ' +
                                 df['largenodules'].astype(str) + ', ' +
                                 df['hugenodules'].astype(str))
        df = df.drop(columns=['smallnodules'], axis=1)
        df = df.drop(columns=['mediumnodules'], axis=1)
        df = df.drop(columns=['largenodules'], axis=1)
        df = df.drop(columns=['hugenodules'], axis=1)
        df.replace(re.compile('.*Yes.*'), 'Yes', inplace=True)
        df.replace(re.compile('.*No.*'), 'No', inplace=True)
        lst_to_remove = ['smallcavities', 'mediumcavities', 'largecavities',
                         'infiltrate_lowgroundglassdensity',
                         'infiltrate_mediumdensity', 'infiltrate_highdensity',
                         'smallnodules', 'mediumnodules', 'largenodules', 'hugenodules']
        CXR_grp_fields = list(set(CXR_fields) - set(lst_to_remove))
        CXR_grp_fields.extend(['cavities', 'infiltrate', 'nodules'])

    """Prepare high cardinality gene variants features""" ### Individual variants
    if args.gen[0] == 1:
        temp = df['gene_snp_mutations'].str.get_dummies(', ').add_prefix('gene_snp_mutations'+'_')
        df = df.reset_index()
        temp = temp.reset_index()
        df = df.drop(columns=['index'], axis=1)
        temp = temp.drop(columns=['index'], axis=1)
        df = pd.concat([df, temp], axis=1)
        df = df.drop(columns=['gene_snp_mutations'], axis=1)

    """Prepare high cardinality radiological features"""
    if args.rad[0] == 1:
        for item in CXR_grp_fields:
            temp = df[item].str.get_dummies(', ').add_prefix(item+'_')
            df = df.reset_index()
            temp = temp.reset_index()
            df = df.drop(columns=['index'], axis=1)
            temp = temp.drop(columns=['index'], axis=1)
            df = pd.concat([df, temp], axis=1)
            df = df.drop(columns=[item], axis=1)

    """Prepare high cardinality tbprofiler features"""
    if args.gen[0] == 1:
        for item in v_df.columns:
            if item != 'sample':
                df.loc[df[item] >= 0.5, item] = 1.0
                df.loc[df[item] < 0.5, item] = 0.0
        for item in TBP_variants_fields:
            df[item] = df[list(df.filter(regex=item))].sum(axis=1)
            df.loc[df[item] > 1, item] = 1.0
        df = df.drop(columns=v_df.columns.tolist(), axis=1)

    df.dropna(how='all', axis=1, inplace=True) #remove empty columns
    df.to_csv(out_folder + '/encoded.csv', index=True)

    #Process
    df = df[df.columns.drop(list(df.filter(regex='No')))]
    df = df[df.columns.drop(list(df.filter(regex='_0')))]
    df = df[df.columns.drop(list(df.filter(regex='lineage')))]
    df = df[df.columns.drop(list(df.filter(regex='main_lineage')))]
    df = df[df.columns.drop(list(df.filter(regex='sub_lineage')))]
    df.columns = df.columns.str.replace(r'_Yes$', '')
    df.columns = df.columns.str.replace(r'_Greater than 0$', '')
    df = df[df.columns.drop(list(df.filter(regex='_Sensitive')))]
    df = df[df.columns.drop(list(df.filter(regex='_DR')))]

    # Remove empty columns
    empty_cols = df.columns[(df == 0).all()]
    df = df.drop(columns=empty_cols, axis=1)

    if args.same[0] == 1:
        if args.gen[0] != 1:
            for gene in TBP_variants_fields:
                df = df[df.columns.drop(list(df.filter(regex=gene)))] #tbp
            df = df[df.columns.drop(list(df.filter(regex='gene')))]
            df = df.drop(columns=['sample'], axis=1)
        if args.rad[0] != 1:
            df = df.drop(columns=df.columns[1:26], axis=1)

    df.to_csv(out_folder + '/filtered.csv', index=True)
    return df

def regressor(df, out_folder):
    """
    This function predicts the treatment period using Gradient Boosting Regressor technique,
    A 5-fold cross validation is used.
    Mean/median and relative errors are computed to evaluate the model
    Args:
        df (pandas.DataFrame): pandas dataframes to use
        out_folder (string): path of folder to output the results
    Returns:
        None
    """

    global TOR, PS

    """Select Features"""
    df = df.reset_index()
    features = df
    """Select Target"""
    target = df[PS]

    """Cross-validation"""
    num_fold = 5
    cv = model_selection.KFold(n_splits=num_fold, shuffle=True)

    """Classifiers & Evaluation"""
    params = {
        "n_estimators": 1000,
        "max_depth": 4,
        "min_samples_split": 5,
        "learning_rate": 0.01,
        "loss": "squared_error",
    }
    model = ensemble.GradientBoostingRegressor(**params)

    plt.rcParams['font.size'] = 16
    plt.rcParams['font.family'] = "Times New Roman"

    mean_ae_lst = []
    med_ae_lst = []
    relative_err_lst = []
    for i, (train, test) in enumerate(cv.split(features)):
        X_train = features.iloc[train]
        X_test = features.iloc[test]
        y_train = target[train]
        y_test = target[test]

        X_train = X_train.drop(columns=[TOR], axis=1)
        X_test = X_test.drop(columns=[TOR], axis=1)
        X_train = X_train.drop(columns=[PS], axis=1)
        X_test = X_test.drop(columns=[PS], axis=1)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Eval
        mean_ae_lst.append(metrics.mean_absolute_error(y_test, y_pred))
        med_ae_lst.append(metrics.median_absolute_error(y_test, y_pred))
        relative_err_lst.append(np.mean((abs(y_pred - y_test) / y_test)) * 100)

    with open(out_folder + 'eval.txt', 'w') as f:
        f.write("Mean AE: " + str(np.mean(mean_ae_lst)) + '  ' + str(np.std(mean_ae_lst)))
        f.write("\n")
        f.write("Med AE: " + str(np.mean(med_ae_lst)) + '  ' + str(np.std(med_ae_lst)))
        f.write("\n")
        f.write("Relative Error: " + str(np.mean(relative_err_lst)) + '  ' + str(np.std(relative_err_lst)))

    print("Mean AE: ", np.mean(mean_ae_lst), np.std(mean_ae_lst))
    print("Med AE: ", np.mean(med_ae_lst), np.std(med_ae_lst))
    print("Relative Error: ", np.mean(relative_err_lst), np.std(relative_err_lst))

def main():
    global TOR
    parser = argparse.ArgumentParser(description="Predicting treatment period using radiological and/or genomic features")
    parser.add_argument("-rad", type=int, nargs=1, metavar="Add radiological features",
                        default=0,
                        help="Specify if users want to use radiological features.")
    parser.add_argument("-gen", type=int, nargs=1, metavar="Add genomic features",
                        default=0,
                        help="Specify if users want to use genomic features.")
    parser.add_argument("-same", type=int, nargs=1, metavar="Use the same samples",
                        default=0,
                        help="Specify if users want to use the same samples which have both radiological "
                             "and genomic information.")
    parser.add_argument("-pdrug", type=int, nargs=1, metavar="Predict treatment period of specific "
                                                             "popular drug combination",
                        default=[0],
                        help="Specify if users want to predict treatment period of specific popular drug combination")
    parser.add_argument("-rp", type=str, nargs=1, metavar="Radiological csv file location",
                        default=['../data/TB_Portals_Patient_Cases_January_2022.csv'],
                        help="Path to radiological csv file. Default: ../data/TB_Portals_Patient_Cases_January_2022.csv")
    parser.add_argument("-gp", type=str, nargs=1, metavar="Genomic csv file location",
                        default=['../data/TB_Portals_Genomics_January_2022.csv'],
                        help="Path to genomic csv file. Default: ../data/TB_Portals_Genomics_January_2022.csv")
    parser.add_argument("-tp", type=str, nargs=1, metavar="TB Profiler csv file location",
                        default=['../data/TB2258-variantDetail.csv'],
                        help="Path to TB Profiler csv file. Default: ../data/TB2258-variantDetail.csv")
    parser.add_argument("-dstp", type=str, nargs=1, metavar="DST csv file location",
                        default=['../data/TB_Portals_DST_January_2022.csv'],
                        help="Path to TB Profiler csv file. Default: ../data/TB_Portals_DST_January_2022.csv")
    parser.add_argument("-regimenp", type=str, nargs=1, metavar="Regimen csv file location",
                        default=['../data/TB_Portals_Regimens_January_2022.csv'],
                        help="Path to TB Profiler csv file. Default: ../data/TB_Portals_Regimens_January_2022.csv")
    parser.add_argument("-resultp", type=str, nargs=1, metavar="results path",
                        default=r'../results/',
                        help="Results Path. Default:../results/")
    args = parser.parse_args()

    if args.rad[0] == 1 and args.same[0] != 1:
        radiological_file = args.rp[0]
        validate_file(radiological_file)
        r_df = pd.read_csv(radiological_file)
        if args.gen[0] != 1:
            g_df = None
            t_df = None
    if args.gen[0] == 1 and args.same[0] != 1:
        genomic_file = args.gp[0]
        validate_file(genomic_file)
        g_df = pd.read_csv(genomic_file)

        tbp_file = args.tp[0]
        validate_file(tbp_file)
        t_df = pd.read_csv(tbp_file)

        if args.rad[0] != 1:
            r_df = None
    if args.same[0] == 1:
        radiological_file = args.rp[0]
        validate_file(radiological_file)
        r_df = pd.read_csv(radiological_file)

        genomic_file = args.gp[0]
        validate_file(genomic_file)
        g_df = pd.read_csv(genomic_file)

        tbp_file = args.tp[0]
        validate_file(tbp_file)
        t_df = pd.read_csv(tbp_file)

    dst_file = args.dstp[0]
    validate_file(dst_file)
    dst_df = pd.read_csv(dst_file)

    regimen_file = args.regimenp[0]
    validate_file(regimen_file)
    regi_df = pd.read_csv(regimen_file)


    if args.rad[0] != 1 and args.gen[0] != 1:
        print(INVALID_INPUT_MSG)
        quit()

    now = datetime.datetime.now()
    out_folder = r'../results/' + now.strftime("%Y-%m-%d") + '/'
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    df = preprocess_csv(args, r_df, g_df, t_df, dst_df, regi_df, out_folder)
    if args.pdrug[0] == 1:
        df = popular_drug(args, df, out_folder)
    df = encoding(args, df, r_df, g_df, t_df, out_folder)

    """Correlation"""
    correlation(df, out_folder)

    """Chi2 test"""
    chi2(df, TOR, out_folder)

    """Regressor"""
    regressor(df, out_folder)

if __name__ == '__main__':
   main()