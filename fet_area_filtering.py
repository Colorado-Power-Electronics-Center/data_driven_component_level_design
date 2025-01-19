'''
    This file contains the info needed to turn the sizes of components into areas, to be used to optimize for area on
    the board or figure out if optimized topology meets area constraints.
'''

from joblib import dump
from csv_conversion import csv_to_df, df_to_csv
from collections import Counter
from numpy import where
import matplotlib.pyplot as plt
from sklearn import neighbors, ensemble, tree
from sklearn.model_selection import RepeatedStratifiedKFold,KFold, cross_val_score, cross_validate
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import RandomOverSampler,SMOTE
from tabulate import tabulate
from sklearn.linear_model import LinearRegression, RANSACRegressor, HuberRegressor, TheilSenRegressor, ElasticNet, LogisticRegression
from sklearn.ensemble import RandomForestRegressor



from fet_regression import preproc
import numpy as np
import pandas as pd

# dimensions of package case sizes in mm
package_dict = \
    {
        'TO-252-3, DPak (2 Leads + Tab), SC-63': (7.0+2.7)*6.6,
        'TO-220-3': 10*4,
        'TO-263-3, D²Pak (2 Leads + Tab), TO-263AB': 14.6*10.1,
        'TO-220-3 Full Pack': 4.7*10.3,
        'TO-247-3': 4.9*15.5,
        '8-PowerTDFN': 6.15*5.35,
        'TO-236-3, SC-59, SOT-23-3': 2.9*2.6,
        '8-SOIC (0.154", 3.90mm Width)': 4.9*6.0,
        '8-PowerVDFN':3.3*3.3,
        'TO-251-3 Short Leads, IPak, TO-251AA':5.55*6.5,
        'TO-262-3 Long Leads, I²Pak, TO-262AA':2.25*6.5,
        '8-PowerTDFN, 5 Leads':5.15*6.15,
        '8-PowerWDFN':3.3*3.3,
        'PowerPAK® SO-8':6.61*4.42,
        'SOT-23-6 Thin, TSOT-23-6':2.85*3.05,
        'SC-100, SOT-669':4.9*6,
        'TO-261-4, TO-261AA':6.35*7,
        'TO-3P-3, SC-65-3': 16*4.8,
        '6-UDFN Exposed Pad':2*2,
        'SOT-227-4, miniBLOC':38.1*25.3,
        'TO-264-3, TO-264AA': 20*5,
        'TO-268-3, D³Pak (2 Leads + Tab), TO-268AA':16*5,
        'TO-263-7, D²Pak (6 Leads + Tab)':10.1*15.2,
        'TO-251-3 Stub Leads, IPak':4.19*10,
        'PowerPAK® 1212-8':3.3*3.3,
        '8-PowerSFN':9.8*11.6,
        'TO-247-4':4.9*15.5,
        'SC-70, SOT-323':2*2.3,
        'TO-220-3 Full Pack, Isolated Tab':10.1*4.7,
        '3-XFDFN':0.6*1,
        '6-SMD, Flat Leads':2*2,
        'SC-96':2.9*1.65,
        '8-SMD, Flat Lead':2.36*2.03,
        '8-PowerSMD, Flat Leads':5*6.15,
        'TO-263-8, D²Pak (7 Leads + Tab), TO-263CA':10.05*15.3,
        'SOT-1210, 8-LFPAK33':3.3*3.3,
        'SC-74, SOT-457':2.9*2.75,
        'SOT-23-6':2.9*1.62,
        '6-WDFN Exposed Pad':2*2,                               
        'SOT-23-3 Flat Leads': 2.9*2.4,                              
        '3-SMD, Flat Lead':2*2.1,
        '4-PowerTSFN': 8*8,
        'TO-3P-3 Full Pack': 15.5*5.5,
        'PowerPAK® 1212-8S': 3.3*3.3,
        '6-TSSOP, SC-88, SOT-363':2*2.1,
        'TO-263-7, D²Pak (6 Leads + Tab), TO-263CB':8.15*10.8,
        'TO-247-3 Variant':5*15.85,
        'TO-262-3 Full Pack, I²Pak':2.3*2.28,
        'PowerPAK® SC-70-6':2.05*2.05,
        'Die':0.9*0.9,
        'SOT-563, SOT-666':0.051*0.069,
        'SOT-1023, 4-LFPAK':6.15*5.15,
        'PDFN-6':8*8,
        'PDFN-9':8*8,
        'DIE':5.01*6.56,
        'GaNPX-4':7.55*4.59,
        'TO-274AA': 15.6*5,
        '24-PowerSMD, 21 Leads': 25*23,
        'TO-204AA, TO-3': 25.5*39.3,
        'SC-101, SOT-883': 0.6*1,
        '3-UFDFN': 0.98*0.61,
        '3-SMD, Flat Leads': 2*2.1,
        'SOT-523': 1.6*1.6,
        'SC-89, SOT-490': 1.6*1.6,
        'TO-226-3, TO-92-3 (TO-226AA) Formed Leads': 0,
        '3-SMD, SOT-23-3 Variant': 0,
        'SOT-723': 0,
        '8-TSSOP (0.173", 4.40mm Width)': 0,
        'PowerPAK® 1212-8SH': 0,
        'SOT-23-5 Thin, TSOT-23-5': 0,
        'SOT-1210, 8-LFPAK33 (5-Lead)': 0,
        '6-VDFN Exposed Pad': 0,
        'TO-243AA': 0,
        '8-PowerUDFN': 0,
        '8-VDFN Exposed Pad': 0,
        '3-SMD, No Lead': 0,
        'TO-220-3 Full Pack, Formed Leads': 0,
        'SC-75, SOT-416': 0,
        '6-PowerVDFN': 0,
        'DirectFET™ Isometric MX': 0,
        '3-WDSON': 0,
        '5-PowerSFN': 0,
        '10-PowerSOP Module': 0,
        '4-XFBGA, WLCSP': 0,
        '6-PowerUDFN': 0,
        '6-PowerWDFN': 0,
        'TO-226-3, TO-92-3 (TO-226AA)': 0,
        '4-DIP (0.300", 7.62mm)': 0,
        '4-VSFN Exposed Pad': 0,
        'SOT-1235': 0,
        '3-UDFN': 0,
        'SC-85': 0,
        '3-XDFN Exposed Pad': 0,
        '6-PowerUFDFN': 0,
        'DirectFET™ Isometric MZ': 0,
        'TO-253-4, TO-253AA': 0,
        '4-SMD, No Lead': 0,
        'TO-263-4, D²Pak (3 Leads + Tab), TO-263AA': 0,
        '8-WFDFN Exposed Pad': 0
    }
'''
        'ISOPLUS247™':,                                      37
        'TO-243AA':,                                         37
        '4-DIP (0.300", 7.62mm)':,                           37
        'SOT-1210, 8-LFPAK33 (5-Lead)':,                     35
        'SC-101, SOT-883':,                                  35
        'DirectFET™ Isometric L8':,                          34
        '6-PowerUDFN':,                                      34
        'SOT-523':                                          32
        }
'''

'''
    This function filters the values currently in the df column 'Pack_size', which are strings of package names,
    and replaces them with the associated area size in mm.
'''
def area_filter(fet_df):

    # First get rid of any components that don't have those packages
    boolean_series = fet_df.loc[:,'Pack_case'].isin(package_dict.keys())
    filtered_df = fet_df[boolean_series]

    # Next turn the package sizes into their appropriate areas using the dictionary 'package_dict'
    filtered_df = filtered_df.replace({"Pack_case": package_dict})
    return filtered_df

def area_filter_gd(fet_df):

    # First get rid of any components that don't have those packages
    boolean_series = fet_df.loc[:,'Pack_case'].isin(package_dict_gd.keys())
    filtered_df = fet_df[boolean_series]

    # Next turn the package sizes into their appropriate areas using the dictionary 'package_dict'
    filtered_df = filtered_df.replace({"Pack_case": package_dict_gd})
    return filtered_df

# dimensions of gate drivers in mm
package_dict_gd =\
    {
        '8-SOIC (0.154", 3.90mm Width)': 6*4.9,
        '8-DIP (0.300", 7.62mm)': 9.65*7.62,
        # 'Module': 61*40,
        '8-SOIC (0.154", 3.90mm Width) Exposed Pad': 4*5,
        '14-SOIC (0.154", 3.90mm Width)': 8.74*6,
        '16-SOIC (0.295", 7.50mm Width)': 10.35*10.3,
        '10-VFDFN Exposed Pad': 3*3,
        '8-TSSOP, 8-MSOP (0.118", 3.00mm Width) Exposed Pad': 5.23*2.794,
        '8-VDFN Exposed Pad':3.1*3.1,
        '8-TSSOP, 8-MSOP (0.118", 3.00mm Width)':2*3,
        '8-WDFN Exposed Pad':4.9*6,
        '28-SOIC (0.295", 7.50mm Width)':19.9*7.5,
        '16-VQFN Exposed Pad':5*5,
        '10-WDFN Exposed Pad':4*4,
        'SOT-23-6':3*3,
        '14-DIP (0.300", 7.62mm)':7.5*10.3,
        '44-LCC (J-Lead), 32 Leads':37*10.3,
        '8-CDIP (0.300", 7.62mm)': 9.5*8.5,
        'SC-74A, SOT-753':3*3,
        '8-WFDFN Exposed Pad':2*2,
        'TO-220-5': 10.1*4.15,
        '8-VQFN Exposed Pad':3*3,
        '8-VFDFN Exposed Pad':2*2,
        '20-SOIC (0.295", 7.50mm Width)':12.75*10.3 #stopped here
        # '14-TSSOP (0.173", 4.40mm Width) Exposed Pad':3.3*3.3,
        # '10-TFSOP, 10-MSOP (0.118", 3.00mm Width)':9.8*11.6,
        # '6-WDFN Exposed Pad':4.9*15.5,
        # '10-WFDFN Exposed Pad':2*2.3,
        # '10-TFSOP, 10-MSOP (0.118", 3.00mm Width) Exposed Pad':10.1*4.7,
        # '6-VDFN Exposed Pad':0.6*1,
        # '8-PowerSOIC (0.154", 3.90mm Width)':2*2,
        # '16-DIP (0.300", 7.62mm)':2.9*1.65,
        # '14-CDIP (0.300", 7.62mm)':2.36*2.03,
        # '16-TFSOP (0.118", 3.00mm Width) Exposed Pad':5*6.15,
        # 'TO-99-8 Metal Can':10.05*15.3,
        # 'TO-263-6, D²Pak (5 Leads + Tab), TO-263BA':3.3*3.3,
        # '28-DIP (0.600", 15.24mm)':2.9*2.75,
        # '24-SSOP (0.209", 5.30mm Width)':2.9*1.62,
        # '16-TFSOP (0.118", 3.00mm Width), 12 Leads, Exposed Pad':2*2,
        # '12-VFDFN Exposed Pad': 2.9*2.4,
        # '8-PowerTSSOP, 8-MSOP (0.118", 3.00mm Width)':2*2.1,
        # '24-SOIC (0.295", 7.50mm Width)': 8*8,
        # '20-DIP (0.300", 7.62mm)': 15.5*5.5,
        # '28-VFQFN Exposed Pad': 3.3*3.3,
        # '36-DIP Module, 24 Leads':2*2.1,
        # '16-SSOP (0.154", 3.90mm Width)':8.15*10.8,
        # '28-SSOP (0.209", 5.30mm Width)':5*15.85,
        # '8-TSSOP (0.173", 4.40mm Width)':2.3*2.28,
        # '20-CLCC':2.05*2.05,
        # '18-SOIC (0.295", 7.50mm Width)':0.9*0.9,
        # '16-VFQFN Exposed Pad':0.051*0.069,
        # '9-VFDFN Exposed Pad':6.15*5.15,
        # '9-WDFN Exposed Pad':8*8
    }
'''
    This function filters the values currently in the df column 'Pack_size', which are strings of package names,
    and replaces them with the associated area size in mm.
'''
def manual_area_filter(fet_df):

    # First get rid of any components that don't have those packages, for any components whose 'Pack_case' is a string
    for index,row in fet_df.iterrows():
        try:
            fet_df.loc[index, 'Pack_case'] = eval(row['Pack_case'])
        except:
            try:
                fet_df.loc[index, 'Pack_case'] = package_dict[fet_df.loc[index, 'Pack_case']]
            except:
                fet_df.loc[index, 'Pack_case'] = np.nan

    # drop the np.nan values
    fet_df = fet_df.dropna(subset=['Pack_case'])
    return fet_df


'''
    This function filters into segments based on area, and produces a list of dataframe subsets broken into bins by 
    areas. This is useful for looking at individual area size ranges.
'''
def area_segment_filter(df, num_chunks):
    # line to see all the different areas
    #   ready_df['Pack case'].value_counts()

    # Break up area values into specified number of chunks
    unique_areas = np.sort(df['Pack_case'].unique())
    area_segments_list = np.array_split(unique_areas, num_chunks)
    area_complete_list = [df[df['Pack_case'].isin(area_segment)] for area_segment in area_segments_list]
    return area_complete_list

''' 
    This function sets the X (input) and y (output) variables based on the ones we care about and to get a linear
    relationship. Note that here, the y-axis is NOT on log scale.   
'''
def X_y_set(df, output_param, attributes, log_x=False, pareto=False,more_inputs=False):

    # Case where we want to use multiple input features to predict another value, beyond just Vdss and unit price
    if more_inputs:
        attributes = ['V_dss', 'Unit_price', 'R_ds', 'Q_g']

    # Input features
    if log_x:
        X = np.log10(df.loc[:, attributes])
    else:
        X = df.loc[:, attributes]
    y = df.loc[:, output_param].astype(int)
    #y = np.log10(df.loc[:, output_param].astype(float))
    return X,y

'''
    This function performs classification on the fet areas, using multiple classification model techniques.
'''
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def area_training(df, file_name, degree=1, conf_matrix=False):
    # train on area with the discrete classification models, view performance
    X, y = X_y_set(df, 'Pack_case', ['V_dss', 'R_ds', 'Unit_price', 'Q_g'], False, True, False)
    X = preproc(X, degree)
    # cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=1)
    # clf = svm.SVC(gamma=0.001)
    # scores = cross_val_score(clf, X_over, y_over, scoring='f1_micro', cv=cv, n_jobs=-1)
    # print('Model: KNN')
    # print('Accuracy: %.3f (%.3f)' % (np.mean(scores), np.std(scores)))
    # reg_knn = model.fit(X, y)

    # assign labels for classification
    # y = LabelEncoder().fit_transform(y)
    # print(Counter(y))

    scores_df = pd.DataFrame(columns=['knn1','knn2','knn3','rf','grad_boost','dec_tree'])
    # remove areas with less than 5 values in the entry, and implement oversampling on the minorities in the
    # unbalanced data set
    oversample = SMOTE(random_state=1, k_neighbors=1)


    oversample = RandomOverSampler(random_state=1)
    X_over, y_over = oversample.fit_resample(X, y)

    # X_over = X
    # y_over = y
    # y_over = LabelEncoder().inverse_transform(y_over)
    # print(Counter(y_over))

    # use knn regression instead
    trained_models = []
    cv = KFold(n_splits=5, random_state=1, shuffle=True)
    attributes = ['N', 'P', 'MOSFET', 'SiCFET','GaNFET']
    log_list = ['V_dss', 'R_ds', 'Unit_price', 'Q_g']
    df[['V_dss', 'R_ds', 'Unit_price', 'Q_g']] = df[['V_dss', 'R_ds', 'Unit_price', 'Q_g']].replace(0.0, np.NaN)
    df = df.dropna(subset=['V_dss', 'R_ds', 'Unit_price', 'Q_g'])
    for variable in log_list:
        new_variable_name = 'log10[' + str(variable) + ']'
        df[new_variable_name] = np.log10(df[variable].astype(float))
        attributes.append(new_variable_name)
    # if df['Q_rr'].any() > 0:
    #     X, y = X_y_set(df, 'Pack_case', ['V_dss', 'R_ds', 'Unit_price', 'Q_g','N','P','MOSFET','SiCFET','GaNFET'], True, True, False)
    # else:
    #     X, y = X_y_set(df, 'Pack_case', ['V_dss', 'R_ds', 'Unit_price', 'Q_g', 'N','P','MOSFET','SiCFET','GaNFET'], True, True, False)


    X = df.loc[:, attributes]
    X = preproc(X, degree)
    y = np.log10(df.loc[:, 'Pack_case'].astype(float))

    # write the trained models to a joblib file to be saved
    model = neighbors.KNeighborsRegressor(n_neighbors=8, weights='distance')
    reg_knn = model.fit(X, y)
    trained_models.append(reg_knn)

    dump(reg_knn, 'full_dataset_Pack_case.joblib')
    return

    model = neighbors.KNeighborsRegressor(n_neighbors=8, weights='distance')
    scores = cross_validate(model, X, y, cv=cv,scoring=['r2','neg_mean_absolute_error'], return_train_score=True)
    model = LinearRegression()
    scores = cross_validate(model, X, y, cv=cv,scoring=['r2','neg_mean_absolute_error'], return_train_score=True)
    model = RandomForestRegressor(min_samples_split=2, random_state=1)
    scores = cross_validate(model, X, y, cv=cv,scoring=['r2','neg_mean_absolute_error'], return_train_score=True)

    scores = cross_val_score(model, X, y, scoring='r2', cv=cv, n_jobs=-1)
    print('r2')
    print('Accuracy: %.3f (%.3f)' % (np.mean(scores), np.std(scores)))
    scores = cross_val_score(model, X, y, scoring='neg_mean_squared_error', cv=cv, n_jobs=-1)
    print('MSE')
    print('Accuracy: %.3f (%.3f)' % (np.mean(scores), np.std(scores)))
    scores = cross_val_score(model, X, y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
    print('MAE')
    print('Accuracy: %.3f (%.3f)' % (np.mean(scores), np.std(scores)))
    scores_df.loc[0, 'knn1'] = np.mean(scores)

    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    pd.set_option('display.colheader_justify', 'center')
    pd.set_option('display.precision', 3)
    fet_column_list = ['knn1'
                       ]
    print('Training results:')
    print(
        tabulate(scores_df.drop_duplicates(inplace=False), headers=fet_column_list, showindex=False,
                 tablefmt='fancy_grid',
                 floatfmt=".3f"))
    # write the trained models to a joblib file to be saved
    reg_knn = model.fit(X,y)
    trained_models.append(reg_knn)

    dump(trained_models, str(file_name) + '_' + 'Pack_case' + '.joblib')
    return

    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=1)
    model = neighbors.KNeighborsRegressor(n_neighbors=3)
    scores = cross_val_score(model, X_over, y_over, scoring='r2', cv=cv, n_jobs=-1)
    print('r2')
    print('Accuracy: %.3f (%.3f)' % (np.mean(scores), np.std(scores)))
    scores = cross_val_score(model, X_over, y_over, scoring='neg_mean_squared_error', cv=cv, n_jobs=-1)
    print('MSE')
    print('Accuracy: %.3f (%.3f)' % (np.mean(scores), np.std(scores)))
    scores = cross_val_score(model, X_over, y_over, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
    print('MAE')
    print('Accuracy: %.3f (%.3f)' % (np.mean(scores), np.std(scores)))
    scores_df.loc[0, 'knn1'] = np.mean(scores)


    trained_models = []
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=1)
    model = neighbors.KNeighborsClassifier(n_neighbors=2)
    scores = cross_val_score(model, X_over, y_over, scoring='f1_micro', cv=cv, n_jobs=-1)
    print('Model: KNN')
    print('Accuracy: %.3f (%.3f)' % (np.mean(scores), np.std(scores)))
    scores_df.loc[0, 'knn1'] = np.mean(scores)
    scores = cross_val_score(model, X_over, y_over, scoring='balanced_accuracy', cv=cv, n_jobs=-1)
    print('Model: KNN')
    print('balanced Accuracy: %.3f (%.3f)' % (np.mean(scores), np.std(scores)))
    scores_df.loc[0,'knn2'] = np.mean(scores)
    scores = cross_val_score(model, X_over, y_over, scoring='accuracy', cv=cv, n_jobs=-1)
    print('Model: KNN')
    print('scoring Accuracy: %.3f (%.3f)' % (np.mean(scores), np.std(scores)))
    scores_df.loc[0,'knn3'] = np.mean(scores)
    reg_knn = model.fit(X_over, y_over)
    trained_models.append(reg_knn)

    if conf_matrix:
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X_over, y_over, random_state=0)
        model = neighbors.KNeighborsClassifier(n_neighbors=2)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        cm = confusion_matrix(y_test, predictions, labels=model.classes_)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
        disp.plot()
        plt.xticks(rotation=45)
        plt.title('Confusion matrix of area classification')

    # for train_index, test_index in cv.split(X_over, y_over):
    #     X_train, X_test = X_over[train_index], X_over[test_index]
    #     y_train, y_test = y_over[train_index], y_over[test_index]
    #     y_pred = model.predict(X_test)
    #     cm = confusion_matrix(y_test, y_pred)
    #
    #     cm_display = ConfusionMatrixDisplay(cm).plot()
    #     reg = model.fit(X_train, y_train)


    model = ensemble.RandomForestClassifier(max_depth=2, random_state=0)
    scores = cross_val_score(model, X_over, y_over, scoring='f1_micro', cv=cv, n_jobs=-1)
    print('Model: Random Forest')
    print('Accuracy: %.3f (%.3f)' % (np.mean(scores), np.std(scores)))
    scores_df.loc[0,'rf'] = np.mean(scores)
    reg_rf = model.fit(X, y)
    trained_models.append(reg_rf)

    model = ensemble.GradientBoostingClassifier(random_state=1)
    scores = cross_val_score(model, X_over, y_over, scoring='f1_micro', cv=cv, n_jobs=-1)
    print('Model: Gradient boosting')
    print('Accuracy: %.3f (%.3f)' % (np.mean(scores), np.std(scores)))
    scores_df.loc[0,'grad_boost'] = np.mean(scores)
    reg_gradBoost = model.fit(X, y)
    trained_models.append(reg_gradBoost)

    model = tree.DecisionTreeClassifier(random_state=1)
    scores = cross_val_score(model, X_over, y_over, scoring='f1_micro', cv=cv, n_jobs=-1)
    print('Model: Decision Tree')
    print('Accuracy: %.3f (%.3f)' % (np.mean(scores), np.std(scores)))
    scores_df.loc[0,'dec_tree'] = np.mean(scores)
    reg_decTree = model.fit(X, y)
    trained_models.append(reg_decTree)

    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    pd.set_option('display.colheader_justify', 'center')
    pd.set_option('display.precision', 3)
    fet_column_list = ['knn1', 'knn2', 'knn3', 'rf', 'grad boost', 'decision tree',
                       ]
    print('Training results:')
    print(
        tabulate(scores_df.drop_duplicates(inplace=False), headers=fet_column_list, showindex=False, tablefmt='fancy_grid',
                 floatfmt=".3f"))
    # write the trained models to a joblib file to be saved
    dump(trained_models, str(file_name) + '_' + 'Pack_case' + '.joblib')


if __name__ == '__main__':

    #
    # #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~with Qrr
    # # Implement classification on area, with Vdss, Rds, cost, Qg, Qrr as inputs, area label as the output
    # import pandas as pd
    # pd.set_option("display.max_rows", None, "display.max_columns", None)
    #
    # file_name = 'pareto_opt_Si_final.csv'
    # fet_df = csv_to_df(file_name)
    # fet_df.columns = ['idx', 'V_dss', 'Unit_price', 'R_ds', 'Q_g', 'Q_rr','FET_type', 'Technology','Pack_case']
    # fet_df = fet_df.dropna(axis='rows', subset=fet_df.columns)
    # fet_df = fet_df.drop('idx', axis=1)
    #
    # # work with the areas
    # area_df = area_filter(fet_df)
    #
    # # write area_df to a csv
    # file_name = 'area_df_Si_wQrr.csv'
    # df_to_csv(area_df, file_name, 'fet', pareto=True)
    #

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~no Qrr

    pd.set_option("display.max_rows", None, "display.max_columns", None)

    file_name = 'csv_files/pareto_opt_Si_noQrr_final.csv'
    fet_df = csv_to_df(file_name)
    fet_df.columns = ['idx', 'V_dss', 'Unit_price', 'R_ds', 'Q_g', 'FET_type', 'Technology', 'Pack_case']
    fet_df = fet_df.dropna(axis='rows', subset=fet_df.columns)
    fet_df = fet_df.drop('idx', axis=1)


    # work with the areas
    area_df = area_filter(fet_df)

    # write area_df to a csv
    file_name = 'csv_files/area_df_Si.csv'
    df_to_csv(area_df, file_name, 'fet', pareto=True)


    trained_models = []
    threshold = 6  # Anything that occurs less than this will be removed.
    allowable_vals = area_df['Pack_case'].value_counts().loc[lambda x : x>5]
    area_df = area_df[area_df['Pack_case'].isin(allowable_vals.index)]

    # keep only components with a certain Vdss value, and Rds*cost and Rds*Qg ranges
    area_df = area_df[area_df['V_dss'].isin([100])]
    area_df['RdsCost_product'] = area_df['R_ds']*area_df['Unit_price']
    area_df['RdsQg_product'] = area_df['R_ds']*area_df['Q_g']
    area_df['RdsQgCost_product'] = area_df['Q_g']*area_df['Unit_price']*area_df['R_ds']
    plt.scatter(np.log10(area_df['Unit_price']), np.log10(area_df['Pack_case']))
    plt.show()

    temp_df = area_df[area_df['Pack_case'].between(10**1.4, 10**1.6)]
    temp_df['RdsQgCost_product'] = temp_df['R_ds']**2*temp_df['Q_g']*temp_df['Unit_price']
    plt.scatter(np.log10(temp_df['R_ds']), np.log10(temp_df['RdsQgCost_product']))

    temp1_df = area_df[area_df['Pack_case'].between(10 ** 1.7, 10 ** 1.9)]
    temp1_df['RdsQgCost_product'] = temp1_df['R_ds']**2 * temp1_df['Q_g'] * temp1_df['Unit_price']
    plt.scatter(np.log10(temp1_df['R_ds']), np.log10(temp1_df['RdsQgCost_product']))

    temp2_df = area_df[area_df['Pack_case'].between(10 ** 2, 10 ** 2.3)]
    temp2_df['RdsQgCost_product'] = temp2_df['R_ds']**2 * temp2_df['Q_g'] * temp2_df['Unit_price']
    plt.scatter(np.log10(temp2_df['R_ds']), np.log10(temp2_df['RdsQgCost_product']))
    plt.show()

    for i in temp_df.index:
        print(temp_df.loc[i,'R_ds']*temp_df.loc[i,'Unit_price']*temp_df.loc[i,'Q_g'])
    #area_df['RdsCost_product'].value_counts(bins=30)
    RdsCost_sizes = {'size1':(0.006, 0.0197), 'size2':(0.006, 0.008), 'size3':(0.004,0.006)}
    RdsQg_sizes = {'size1': (.1*10**-10,2.8*10**-10),'size2': (2.9*10**-10,6.9*10**-10), 'size3': (7*10**-10,30*10**-10),'size4':(31*10**-10,100*10**-10)}

    temp_df = area_df[area_df['RdsCost_product'].between(RdsCost_sizes['size1'][0], RdsCost_sizes['size1'][1])]
    temp_df = temp_df[temp_df['RdsQg_product'].between(RdsQg_sizes['size1'][0], RdsQg_sizes['size1'][1])]
    plt.scatter(np.log10(temp_df['R_ds']), np.log10(temp_df['Pack_case']))
    temp_df = area_df[area_df['RdsCost_product'].between(RdsCost_sizes['size1'][0], RdsCost_sizes['size1'][1])]
    temp_df = temp_df[temp_df['RdsQg_product'].between(RdsQg_sizes['size2'][0], RdsQg_sizes['size2'][1])]
    plt.scatter(np.log10(temp_df['R_ds']), np.log10(temp_df['Pack_case']))
    temp_df = area_df[area_df['RdsCost_product'].between(RdsCost_sizes['size1'][0], RdsCost_sizes['size1'][1])]
    temp_df = temp_df[temp_df['RdsQg_product'].between(RdsQg_sizes['size3'][0], RdsQg_sizes['size3'][1])]
    plt.scatter(np.log10(temp_df['R_ds']), np.log10(temp_df['Pack_case']))
    plt.legend(['size1','size2','size3'])
    plt.title('Area classifications vs. Rds at set Rds*Qg and Rds*Cost FOMs')
    plt.xlabel('Log[Rds [mΩ]')
    plt.ylabel('Log[Area] [mm^2]')
    plt.show()

    X, y = X_y_set(area_df, 'Pack_case', ['V_dss', 'R_ds'], True, True, False)
    counter = Counter(y)
    print(counter)
    for i in range(10):
        print(X.iloc[i + 1], y.iloc[i + 1])
    for label, _ in counter.items():
        row_ix = where(y == label)[0].tolist()
        plt.scatter(X.iloc[row_ix, 0], X.iloc[row_ix, 1], label=str(round(label, 2)), s=10, alpha=0.7)
    Legend = plt.legend()
    Legend.set_title('Area [mm^2]')
    plt.title('Area classifications')
    plt.xlabel('Log[V_dss [V]')
    plt.ylabel('Log[R_ds] [mΩ]')
    # change to truncate the area size to 2 decimals before setting as label
    # also consider using on full pareto-optimized dataset, not the smaller one w/ Qrr, but first check the performance
    # w/ this one
    plt.show()




    # steps = [('over', RandomOverSampler()),('model', DecisionTreeClassifier())]
    steps = [('over', RandomOverSampler()), ('model', neighbors.KNeighborsClassifier())]

    pipeline = Pipeline(steps=steps)
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=1,shuffle=True)
    scores = cross_val_score(pipeline, X, y, scoring='f1_micro',cv=cv,n_jobs=-1)
    print('Model: Stratified K-fold')
    print('Accuracy: %.3f (%.3f)' % (np.mean(scores), np.std(scores)))

    ######################### from here down is with the Q_rr data as well
    # Get the pareto-optimized data
    file_name = 'csv_files/pareto_opt_Si_final.csv'
    fet_df = csv_to_df(file_name)
    fet_df.columns = ['idx','V_dss', 'Unit_price', 'R_ds', 'Q_g', 'Q_rr','FET_type','Technology','Pack_case']
    # Give areas to the df
    area_df = area_filter(fet_df)

    # visualized below. here, implement various classification algorithms on the data, w/ all available input parameters
    # and Pack_case as the output parameter
    trained_models = []
    X,y = X_y_set(area_df, 'Pack_case',['V_dss','R_ds','Unit_price','Q_g','Q_rr'],True,True,False)
    degree = 1
    X = preproc(X, degree)

    y=np.floor(y*1000)
    # steps = [('over', RandomOverSampler()),('model', DecisionTreeClassifier())]
    steps = [('over', RandomOverSampler()),('model', neighbors.KNeighborsClassifier())]

    pipeline = Pipeline(steps=steps)
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3,random_state=1,shuffle=True)
    scores = cross_val_score(pipeline, X,y,cv=cv)
    print('Model: Stratified K-fold')
    print('Accuracy: %.3f (%.3f)' % (np.mean(scores), np.std(scores)))


    model = neighbors.KNeighborsClassifier(n_neighbors = 2)
    model = ensemble.RandomForestClassifier()
    cv = KFold(n_splits=2, random_state=1, shuffle=True)
    scores = cross_val_score(model, X, y, cv=cv)
    print('Model: Linear regression')
    print('Accuracy: %.3f (%.3f)' % (np.mean(scores), np.std(scores)))
    reg_lin = model.fit(X, y)
    # y_pred = reg_lin.predict(X)
    trained_models.append(reg_lin)



    # use these areas as the assigned values for the classification
    # first plot the data based on Vdss to Rds, showing area as the label
    X,y = X_y_set(area_df, 'Pack_case',['V_dss','R_ds'],True,True,False)
    counter = Counter(y)
    print(counter)
    for i in range(10):
        print(X.iloc[i+1], y.iloc[i+1])
    for label,_ in counter.items():
        row_ix = where(y==label)[0].tolist()
        plt.scatter(X.iloc[row_ix,0],X.iloc[row_ix,1],label=str(round(label,2)),s=1)
    plt.legend()
    plt.title('Area classifications')
    plt.xlabel('Log[V_dss [V]')
    plt.ylabel('Log[R_ds] [mΩ]')
    # change to truncate the area size to 2 decimals before setting as label
    # also consider using on full pareto-optimized dataset, not the smaller one w/ Qrr, but first check the performance
    # w/ this one
    plt.show()
