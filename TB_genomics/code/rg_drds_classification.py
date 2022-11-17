__author__ = "Vy Bui, Ph.D."
__email__ = "vy.bui@nih.gov"

import os, re, datetime, argparse
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing, metrics, model_selection, ensemble
import numpy as np
from itertools import cycle
from helpers import INVALID_INPUT_MSG,  validate_file, correlation, chi2

# column names
CT_fields = ['image_body_site', 'dissemination', 'lungcavity_size',
             'anomaly_of_mediastinum_vessels_develop', 'affect_pleura',
             'shadow_pattern', 'affect_level', 'pneumothorax', 'plevritis',
             'affected_segments', 'nodicalcinatum', 'process_prevalence',
             'thromboembolism_of_the_pulmonaryartery', 'posttbresiduals',
             'lung_capacity_decrease', 'bronchial_obstruction',
             'anomaly_of_lungdevelop', 'accumulation_of_contrast',
             'limfoadenopatia', 'totalcavernum']
CXR_fields = ['overall_percent_of_abnormal_volume',
              'pleural_effusion_percent_of_hemithorax_involved',
              'ispleuraleffusionbilateral', 'other_non_tb_abnormalities',
              'are_mediastinal_lymphnodes_present', 'collapse',
              'smallcavities', 'mediumcavities', 'largecavities',
              'isanylargecavitybelongtoamultisextantcavity',
              'canmultiplecavitiesbeseen', 'infiltrate_lowgroundglassdensity',
              'infiltrate_mediumdensity', 'infiltrate_highdensity',
              'smallnodules', 'mediumnodules', 'largenodules', 'hugenodules',
              'isanycalcifiedorpartiallycalcifiednoduleexist',
              'isanynoncalcifiednoduleexist', 'isanyclusterednoduleexists',
              'aremultiplenoduleexists', 'lowgroundglassdensityactivefreshnodules',
              'mediumdensitystabalizedfibroticnodules',
              'highdensitycalcifiedtypicallysequella']
CLINICAL_merged_fields = ['country', 'education', 'employment', 'lung_localization',
                   'number_of_children', 'number_of_daily_contacts',
                   'gender_x', 'bmi', 'age_of_onset_x', 'case_definition',
                   'age_of_onset_y', 'gender_y', 'type_of_resistance_y',
                   'specimen_identifier', 'sra_id', 'ncbi_sra',
                   'ncbi_sourceorganism_x', 'ncbi_sourceorganism_y',
                   'ncbi_bioproject_x', 'ncbi_bioproject_y', 'ncbi_biosample_x',
                   'ncbi_biosample_y', 'lineage_y', 'sit_designation',
                   'high_confidence_snp_mutations', 'hain_snp_mutations',
                   'genexpert_snp_mutations','gene_name','high_confidence','hain','genexpert',
                   'patient_id', 'identifier', 'registration_date',
                   'diagnosis_code', 'x_ray_count', 'status', 'organization',
                   'x_ray_exists', 'ct_exists', 'genomic_data_exists', 'comorbidity', 'outcome_x',
                   'outcome_y', 'regimen_count', 'period_span', 'rater','condition_id',
                   'specimen_id','specimen_collection_site']
CLINICAL_fields = ['country', 'education', 'employment', 'lung_localization',
                   'number_of_children', 'number_of_daily_contacts',
                   'gender', 'bmi','age_of_onset', 'case_definition',
                   'ncbi_sourceorganism','ncbi_bioproject','lineage',
                   'ncbi_biosample','patient_id', 'identifier', 'registration_date',
                   'diagnosis_code', 'x_ray_count', 'status', 'organization',
                   'x_ray_exists', 'ct_exists', 'genomic_data_exists', 'comorbidity', 'outcome',
                   'regimen_count', 'period_span', 'rater','condition_id',
                   'gene_name','high_confidence','hain','genexpert']
TCS_fields = ['period_start', 'period_end', 'treatment_status', 'stemcell_dose',
              'social_risk_factors', 'specimen', 'regimen_drug']
GENOMIC_fields = ['condition_id','gender','age_of_onset','outcome',
                  'specimen_id','specimen_identifier','specimen_collection_date',
                  'specimen_collection_site','sra_id','ncbi_sra','ncbi_sourceorganism',
                  'ncbi_bioproject','ncbi_biosample','sit_designation',
                  'high_confidence_snp_mutations', 'hain_snp_mutations',
                  'genexpert_snp_mutations']
QURE_fields = ['qure_bluntedcp', 'qure_abnormal', 'qure_consolidation',
               'qure_fibrosis', 'qure_opacity', 'qure_peffusion',
               'qure_tuberculosis', 'qure_nodule', 'qure_cavity',
               'qure_hilarlymphadenopathy', 'qure_atelectasis']
TBP_variants_fields = ['ahpC', 'ald', 'alr', 'ddn', 'eis', 'embA', 'embB', 'embC', 'embR',
                       'ethA', 'fabG1', 'fbiA', 'folC', 'gid', 'gyrA', 'gyrB', 'ethR',
                       'inhA', 'kasA', 'katG', 'mmpR5', 'panD', 'pncA', 'ribD', 'rplC',
                       'rpoB', 'rpoC', 'rpsA', 'rpsL', 'rrl', 'rrs', 'thyA', 'thyX', 'tlyA']
DST_fields = ['culture', 'culturetype', 'microscopy', 'bactec_test', 'le_test',
              'hain_test', 'lpaother_test', 'genexpert_test', 'bactec_isoniazid',
              'bactec_rifampicin', 'bactec_streptomycin', 'bactec_ethambutol',
              'bactec_ofloxacin', 'bactec_capreomycin', 'bactec_amikacin',
              'bactec_kanamycin', 'bactec_pyrazinamide', 'bactec_levofloxacin',
              'bactec_moxifloxacin', 'bactec_p_aminosalicylic_acid',
              'bactec_prothionamide', 'bactec_cycloserine',
              'bactec_amoxicillin_clavulanate', 'bactec_mycobutin',
              'bactec_delamanid', 'bactec_bedaquiline', 'bactec_imipenem_cilastatin',
              'bactec_linezolid', 'bactec_clofazimine', 'bactec_clarithromycin',
              'bactec_fluoroquinolones', 'bactec_aminoglycosides_injectible_agents',
              'le_isoniazid', 'le_rifampicin', 'le_streptomycin', 'le_ethambutol',
              'le_ofloxacin', 'le_capreomycin', 'le_amikacin', 'le_kanamycin',
              'le_pyrazinamide', 'le_levofloxacin', 'le_moxifloxacin',
              'le_p_aminosalicylic_acid', 'le_prothionamide', 'le_cycloserine',
              'le_amoxicillin_clavulanate', 'le_mycobutin', 'le_delamanid',
              'le_bedaquiline', 'le_imipenem_cilastatin', 'le_linezolid',
              'le_clofazimine', 'le_clarithromycin', 'le_fluoroquinolones',
              'le_aminoglycosides_injectible_agents', 'hain_isoniazid',
              'hain_rifampicin', 'hain_streptomycin', 'hain_ethambutol',
              'hain_ofloxacin', 'hain_capreomycin', 'hain_amikacin', 'hain_kanamycin',
              'hain_pyrazinamide', 'hain_levofloxacin', 'hain_moxifloxacin',
              'hain_p_aminosalicylic_acid', 'hain_prothionamide', 'hain_cycloserine',
              'hain_amoxicillin_clavulanate', 'hain_mycobutin', 'hain_delamanid',
              'hain_bedaquiline', 'hain_imipenem_cilastatin', 'hain_linezolid',
              'hain_clofazimine', 'hain_clarithromycin', 'hain_fluoroquinolones',
              'hain_aminoglycosides_injectible_agents', 'lpaother_isoniazid',
              'lpaother_rifampicin', 'lpaother_streptomycin', 'lpaother_ethambutol',
              'lpaother_ofloxacin', 'lpaother_capreomycin', 'lpaother_amikacin',
              'lpaother_kanamycin', 'lpaother_pyrazinamide', 'lpaother_levofloxacin',
              'lpaother_moxifloxacin', 'lpaother_p_aminosalicylic_acid',
              'lpaother_prothionamide', 'lpaother_cycloserine',
              'lpaother_amoxicillin_clavulanate', 'lpaother_mycobutin',
              'lpaother_delamanid', 'lpaother_bedaquiline',
              'lpaother_imipenem_cilastatin', 'lpaother_linezolid', 'lpaother_clofazimine',
              'lpaother_clarithromycin', 'lpaother_fluoroquinolones',
              'lpaother_aminoglycosides_injectible_agents',
              'genexpert_isoniazid', 'genexpert_rifampicin']

TOR = 'type_of_resistance'

def preprocess_csv(args, r_df, g_df, t_df, out_folder):
    """
    This function preprocess the csv files, it merges the csv rows and remove unused columns
    Args:
        args (parser.parse_args()): inputs from the comand line
        r_df, g_df, t_df (pandas.DataFrame): pandas dataframes read from
            clinical, genomic, and TB Profiler csv spreadsheet.
        out_folder (string): path of folder to output the results
    Returns:
        df (pandas.DataFrame): return pandas dataframe
    """
    global TOR

    if (args.gen == [1] and args.rad == [1]) or args.same == [1]:
        # Merge two TB Portal Genomics
        df = pd.merge(g_df, t_df, how="inner", left_on='sra_id', right_on='sample')
        df.reset_index(drop=True, inplace=True)
        # Merge two TB Portal Genomics and Clinical based on condition_id
        df = pd.merge(df, r_df, how="inner", on="condition_id")
        df.reset_index(drop=True, inplace=True)
    elif args.gen == [1] and args.rad != [1] and args.same != [1]:
        # Merge two TB Portal Genomics
        df = pd.merge(g_df, t_df, how="inner", left_on='sra_id', right_on='sample')
        df.reset_index(drop=True, inplace=True)
    elif args.gen != [1] and args.rad == [1] and args.same != [1]:
        df = r_df

    df.to_csv(out_folder + '/merged.csv', index=True)

    TOR = [col for col in df if col.startswith(TOR)][0]

    # drop rows that contain any value in the list
    drop_values = ['Poly DR', 'Pre-XDR']
    df = df[df[TOR].isin(drop_values) == False]

    if (args.gen == [1] and args.rad == [1]) or args.same == [1]:
        # Keep new patients only
        df = df.loc[df['specimen_collection_date'] == 0]
        # Preprocess: drop unuse columns
        df = df.drop(columns=CT_fields, axis=1)
        df = df.drop(columns=DST_fields, axis=1)
        df = df.drop(columns=TCS_fields, axis=1)
        df = df.drop(columns=QURE_fields, axis=1)
        df = df.drop(columns=CLINICAL_merged_fields, axis=1)
        df = df.drop(columns=['specimen_collection_date'], axis=1)

    if args.gen != [1] and args.rad == [1] and args.same != [1]:
        # Preprocess: drop unuse columns
        df = df.drop(columns=CT_fields, axis=1)
        df = df.drop(columns=DST_fields, axis=1)
        df = df.drop(columns=TCS_fields, axis=1)
        df = df.drop(columns=QURE_fields, axis=1)
        df = df.drop(columns=CLINICAL_fields, axis=1)

    if args.gen == [1] and args.rad != [1] and args.same != [1]:
        # Preprocess: drop unuse columns
        df = df.drop(columns=GENOMIC_fields, axis=1)

    if args.rad == [1] or args.same == [1]:
        drop_values = ['Not Reported']
        for column_name in CXR_fields:
            df = df[df[column_name].isin(drop_values) == False]

    df = df[df.columns].replace({'MDR non XDR': 'DR', 'XDR': 'DR', 'Mono DR': 'DR'})
    df.to_csv(out_folder + '/pre_processed.csv', index=True)
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

    global TOR

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

    if args.rad == [1]:
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
    if args.gen == [1]:
        temp = df['gene_snp_mutations'].str.get_dummies(', ').add_prefix('gene_snp_mutations'+'_')
        df = df.reset_index()
        temp = temp.reset_index()
        df = df.drop(columns=['index'], axis=1)
        temp = temp.drop(columns=['index'], axis=1)
        df = pd.concat([df, temp], axis=1)
        df = df.drop(columns=['gene_snp_mutations'], axis=1)

    """Prepare high cardinality radiological features"""
    if args.rad == [1]:
        for item in CXR_grp_fields:
            temp = df[item].str.get_dummies(', ').add_prefix(item+'_')
            df = df.reset_index()
            temp = temp.reset_index()
            df = df.drop(columns=['index'], axis=1)
            temp = temp.drop(columns=['index'], axis=1)
            df = pd.concat([df, temp], axis=1)
            df = df.drop(columns=[item], axis=1)

    """Prepare high cardinality tbprofiler features"""
    if args.gen == [1]:
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

    if args.same == [1]:
        if args.gen[0] != 1:
            for gene in TBP_variants_fields:
                df = df[df.columns.drop(list(df.filter(regex=gene)))] #tbp
            df = df[df.columns.drop(list(df.filter(regex='gene')))]
            df = df.drop(columns=['sample'], axis=1)
        if args.rad != [1]:
            df = df.drop(columns=df.columns[1:26], axis=1)

    df.to_csv(out_folder + '/filtered.csv', index=True)
    return df

def classifier(df, out_folder):
    """
    This function predicts the DR, DS type using Random Forest Classifier technique,
    A 5-fold cross validation is used.
    ROC and accuracy are computed to evaluate the model
    Args:
        df (pandas.DataFrame): pandas dataframes to use
        out_folder (string): path of folder to output the results
    Returns:
        None
    """

    global TOR

    """Select Features"""
    features = df.drop(columns =[TOR], axis = 1)
    """Select Target"""
    target = df[TOR]
    classes = df[TOR].unique()
    target = preprocessing.label_binarize(target, classes=classes)

    """Cross-validation"""
    num_fold = 5
    cv = model_selection.KFold(n_splits=num_fold, shuffle=True)
    """Classifiers & Evaluation"""
    classifier = ensemble.RandomForestClassifier(max_depth=10, n_estimators=100)

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    acc = dict()
    cm = dict()
    for i, (train, test) in enumerate(cv.split(features)):
        X_train = features.iloc[train]
        X_test = features.iloc[test]
        y_train = target[train]
        y_test = target[test]
        classifier.fit(X_train, y_train)

        """Feature Importance"""
        importances = classifier.feature_importances_
        feature_scores = pd.Series(importances, index=X_train.columns,
                    name="Importance Score").sort_values(ascending=False)
        feature_scores.to_csv(out_folder +
                    '/feature_scores_fold0' + str(i) + '.csv', index=True)

        """Train on important features"""
        drop_features = feature_scores[20:].index
        X_train = X_train.drop(drop_features, axis=1)
        X_test = X_test.drop(drop_features, axis=1)
        classifier.fit(X_train, y_train)

        importances = classifier.feature_importances_
        feature_scores = pd.Series(importances, index=X_train.columns,
                    name="Importance Score").sort_values(ascending=False)
        feature_scores.to_csv(out_folder +
                    '/feature_scores_fold0' + str(i) + '.csv', index=True)
        plt.figure()
        y=feature_scores[:20]
        plt.bar(x=y.index,height=y,align="center")
        plt.xticks(rotation=45, ha='right', rotation_mode='anchor')
        plt.ylabel("Feature importance score")
        plt.tight_layout()
        plt.savefig(out_folder + 'FI_fold0' + str(i) + '.jpg')

        # Predict Test Data
        y_probas = classifier.predict_proba(X_test)[:, 1]
        y_pred = classifier.predict(X_test)

        #Accuracy
        acc[i] = metrics.accuracy_score(y_test, y_pred)

        #ROC
        fpr[i], tpr[i], threshold = metrics.roc_curve(y_test, y_probas)
        roc_auc[i] = metrics.auc(fpr[i], tpr[i])

    aucs = []
    for value in roc_auc.values():
        aucs.append(value)
    roc_auc["avg"] = np.mean(aucs)
    roc_auc["std"] = np.std(aucs)

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(len(classes))]))
    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(len(classes)):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    # Finally average it and compute AUC
    mean_tpr /= len(classes)

    plt.rcParams['font.size'] = 11
    fsize = (12, 6)
    plt.figure(figsize=fsize)
    colors = cycle(["red", "yellow", "blue", "orange", "deeppink"])
    plt.plot(all_fpr, mean_tpr, color="green", lw=2, label="ROC-avg (area = {0:0.2f})".format(roc_auc["avg"]))
    plt.plot([0, 1], [0, 1], "k--", lw=2)
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC to type_of_resistance using Random Forest")
    plt.legend(loc="lower right")
    plt.savefig(out_folder + 'ROC.jpg')

    # Accuracy
    stds = []
    for value in acc.values():
        stds.append(value)
    print("ROC: ", roc_auc["avg"], roc_auc["std"])
    print("Accuracy: ", np.mean(stds), np.std(stds))

def main():
    global TOR
    parser = argparse.ArgumentParser(description="DR DS Classifier using radiological and/or genomic features")
    parser.add_argument("-rad", type=int, nargs=1, metavar="Add radiological features",
                        default=None,
                        help="Specify if users want to use radiological features.")
    parser.add_argument("-gen", type=int, nargs=1, metavar="Add genomic features",
                        default=None,
                        help="Specify if users want to use genomic features.")
    parser.add_argument("-same", type=int, nargs=1, metavar="Use the same samples",
                        default=None,
                        help="Specify if users want to use the same samples which have both radiological "
                             "and genomic information.")
    parser.add_argument("-up", type=int, nargs=1, metavar="Include unpublished data",
                        default=None,
                        help="Specify if users want to include unpublished data.")
    parser.add_argument("-rp", type=str, nargs=1, metavar="Radiological csv file location",
                        default=['../data/TB_Portals_Patient_Cases_January_2022.csv'],
                        help="Path to radiological csv file. Default: ../data/TB_Portals_Patient_Cases_January_2022.csv")
    parser.add_argument("-gp", type=str, nargs=1, metavar="Genomic csv file location",
                        default=['../data/TB_Portals_Genomics_January_2022.csv'],
                        help="Path to genomic csv file. Default: ../data/TB_Portals_Genomics_January_2022.csv")
    parser.add_argument("-tp", type=str, nargs=1, metavar="TB Profiler csv file location",
                        default=['../data/TB2258-variantDetail.csv'],
                        help="Path to TB Profiler csv file. Default: ../data/TB2258-variantDetail.csv")
    parser.add_argument("-urp", type=str, nargs=1, metavar="Unpublished Radiological csv file location",
                        default=['../data/unpublished/TB_Portals_Unpublished_Patient_Cases_January_2022.csv'],
                        help="Path to radiological csv file. Default: ../data/unpublished/TB_Portals_Unpublished_Patient_Cases_January_2022.csv")
    parser.add_argument("-ugp", type=str, nargs=1, metavar="Unpublished Genomic csv file location",
                        default=['../data/unpublished/TB_Portals_Unpublished_Genomics_January_2022.csv'],
                        help="Path to genomic csv file. Default: ../data/unpublished/TB_Portals_Unpublished_Genomics_January_2022.csv")
    parser.add_argument("-resultp", type=str, nargs=1, metavar="results path",
                        default=r'../results/',
                        help="Results Path. Default:../results/")
    args = parser.parse_args()

    if args.rad == [1] and args.same != [1]:
        radiological_file = args.rp[0]
        validate_file(radiological_file)
        r_df = pd.read_csv(radiological_file)

        if args.up == [1]:
            unpub_radiological_file = args.urp[0]
            validate_file(unpub_radiological_file)
            ur_df = pd.read_csv(unpub_radiological_file)

            # Merge radiomic tb portal & unpublished
            r_df = r_df.append(ur_df, ignore_index=True)

        if args.gen != [1]:
            g_df = None
            t_df = None
    if args.gen == [1] and args.same != [1]:
        genomic_file = args.gp[0]
        validate_file(genomic_file)
        g_df = pd.read_csv(genomic_file)

        if args.up == [1]:
            unpub_genomic_file = args.ugp[0]
            validate_file(unpub_genomic_file)
            ug_df = pd.read_csv(unpub_genomic_file)

            # Merge radiomic tb portal & unpublished
            g_df = g_df.append(ug_df, ignore_index=True)

        tbp_file = args.tp[0]
        validate_file(tbp_file)
        t_df = pd.read_csv(tbp_file)

        if args.rad != [1]:
            r_df = None
    if args.same == [1]:
        radiological_file = args.rp[0]
        validate_file(radiological_file)
        r_df = pd.read_csv(radiological_file)

        if args.up == [1]:
            unpub_radiological_file = args.urp[0]
            validate_file(unpub_radiological_file)
            ur_df = pd.read_csv(unpub_radiological_file)

            # Merge radiomic tb portal & unpublished
            r_df = r_df.append(ur_df, ignore_index=True)

        genomic_file = args.gp[0]
        validate_file(genomic_file)
        g_df = pd.read_csv(genomic_file)

        if args.up == [1]:
            unpub_genomic_file = args.ugp[0]
            validate_file(unpub_genomic_file)
            ug_df = pd.read_csv(unpub_genomic_file)

            # Merge radiomic tb portal & unpublished
            g_df = g_df.append(ug_df, ignore_index=True)

        tbp_file = args.tp[0]
        validate_file(tbp_file)
        t_df = pd.read_csv(tbp_file)

    if args.rad != [1] and args.gen != [1]:
        print(INVALID_INPUT_MSG)
        quit()

    now = datetime.datetime.now()
    out_folder = r'../results/' + now.strftime("%Y-%m-%d") + '/'
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    df = preprocess_csv(args, r_df, g_df, t_df, out_folder)
    df = encoding(args, df, r_df, g_df, t_df, out_folder)

    """Correlation"""
    correlation(df, out_folder)

    """Chi2 test"""
    chi2(df, TOR, out_folder)

    """Select balanced data"""
    TYPE = 'balanced'
    if TYPE == 'balanced':
        S = df.loc[df[TOR] == 'Sensitive']
        DR = df.loc[df[TOR] == 'DR']
        DR = DR[:S.shape[0]]
        #DR = DR[S.shape[0]*3:S.shape[0]*4+1]
        df = pd.concat([S, DR])
        df.to_csv(out_folder + '/balanced.csv', index=True)
    else:
        df.to_csv(out_folder + '/unbalanced.csv', index=True)

    """Classifier"""
    classifier(df, out_folder)

if __name__ == '__main__':
   main()