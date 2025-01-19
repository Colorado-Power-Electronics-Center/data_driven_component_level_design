'''
    This file contains the work used for the APEC digest--experimental plotting, capturing Coss data, and the script for
    inductor ac losses.
'''


from csv import reader

import matplotlib.pyplot as plt
from csv_conversion import csv_to_mosfet
from fet_regression import *
from sklearn.covariance import EllipticEnvelope
import numpy as np
from fet_best_envelope import pareto_optimize
from fet_data_parsing import initial_fet_parse
import matplotlib.lines as mlines
import matplotlib.colors as colors
from fet_price_filtering import outlier_removal
from fet_pdf_scraper import parse_pdf_param
from fet_area_filtering import area_filter, area_training, manual_area_filter, area_filter_gd
from sklearn import preprocessing
from sklearn.model_selection import cross_validate
from sklearn.metrics import mean_absolute_error
import pandas as pd
import re
import oapackage
from csv_conversion import df_to_csv, csv_to_df
import seaborn as sn
from fet_visualization import correlation_matrix, FOM_scatter, Visualization_parameters
from tabulate import tabulate
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import LabelEncoder
import numpy as np
from matplotlib.pyplot import cm

# This function goes through the various physics-based groupings and plots and determines MAE between predicted
# and actual quantities. Grouping options: 'lowVolt_Si', 'highVolt_Si', 'lowVolt_GaN', 'highVolt_GaN', 'allVolt_SiC'
def physics_groupings():
    grouping = 'lowVolt_Si'
    Ron_Coss_MAE(grouping)
    Ron_plotting(grouping)
    Coss_plotting(grouping)

# Using the data with calculated ac loss parameters, train models with fb, b, Kfe, alpha, beta, Nturns, Ac as outputs,
# and current rating, sat current, cost, as inputs. Note only looking at powder cores here.
def ind_ac_training():
    # csv_file = 'csv_files/inductor_info.csv'
    # ind_df = pd.read_csv(csv_file)

    xlsx_file = 'csv_files/inductor_training_updatedAlgorithm.csv'
    ind_df = pd.read_csv(xlsx_file)


    # first clean the data, drop rows without necessary info
    attr_list = ['Mfr_part_no', 'Unit_price', 'fb', 'b', 'Kfe', 'alpha', 'beta', 'Nturns', 'Ac']
    # fet_df = initial_fet_parse(fet_df, attr_list)
    # fet_df = fet_df.replace(0, np.nan)
    #
    # fet_df = fet_df.dropna(subset=attr_list)

    # make a copy of ind_df to use later
    ind_df_copy = ind_df.copy(deep=True)

    # Pareto-optimize data on dimensions of interest
    ind_df['Volume [mm^3]'] = ind_df['Length [mm]']*ind_df['Width [mm]']*ind_df['Height [mm]']
    data_dims = ['Mfr_part_no', 'Unit_price [USD]', 'Current_Rating [A]', 'DCR [Ohms]', 'Volume [mm^3]', 'Inductance [H]',
                 'Current_Sat [A]', 'fb [Hz]',  'b', 'Kfe', 'Alpha', 'Beta', 'Nturns', 'Ac [m^2]']
    data_dims_paretoFront_dict = {'Ind_params': ['Unit_price [USD]', 'Current_Rating [A]', 'DCR [Ohms]', 'Volume [mm^3]']}
    pareto_opt_data = pareto_optimize(ind_df, data_dims, data_dims_paretoFront_dict['Ind_params'], technology='ind',component = 'ind')
    pareto_opt_df = pd.DataFrame(pareto_opt_data,
                                 columns=data_dims)
    pareto_opt_df = pareto_opt_df.astype({'Unit_price [USD]':float, 'Current_Rating [A]':float, 'DCR [Ohms]':float, 'Volume [mm^3]':float, 'Inductance [H]':float,
                 'Current_Sat [A]':float, 'fb [Hz]':float,  'b':float, 'Kfe':float, 'Alpha':float, 'Beta':float, 'Nturns':float, 'Ac [m^2]':float})

    # Train with ac inductor parameters as the output:

    pareto_opt_df = pareto_opt_df.drop_duplicates(subset='Mfr_part_no')

    inputs = ['Unit_price [USD]', 'Current_Rating [A]', 'Current_Sat [A]', 'Inductance [H]']
    outputs = ['fb [Hz]', 'b', 'Kfe', 'Alpha', 'Beta', 'Nturns', 'Ac [m^2]', 'Volume [mm^3]']
    # fet_df = fet_df[np.all(fet_df[inputs] != 0, axis=1)]
    for output in outputs:
        X = np.log10(pareto_opt_df.loc[:, inputs])
        y = np.log10(pareto_opt_df.loc[:, output])
        scores_df = pd.DataFrame({'scoring_type': ['r^2', 'RMSE', 'MAE']}).set_index('scoring_type')
        model = LinearRegression(fit_intercept=True, normalize=True)
        # model = RandomForestRegressor(min_samples_split=3, random_state=1)

        rmse_scorer = make_scorer(mean_squared_error, squared=False)
        cv = KFold(n_splits=3, random_state=1, shuffle=True)
        scores = cross_val_score(model, X, y, cv=cv, scoring='r2')
        scores_df.loc['r^2', 'linear'] = np.mean(scores)
        scores = cross_val_score(model, X, y, cv=cv, scoring=rmse_scorer)
        scores_df.loc['RMSE', 'linear'] = np.mean(scores)
        scores = cross_val_score(model, X, y, cv=cv, scoring='neg_mean_absolute_error')
        scores_df.loc['MAE', 'linear'] = np.mean(scores)
        # print('Model: Linear regression')
        # print('Accuracy: %.3f (%.3f)' % (np.mean(scores), np.std(scores)))
        # scores_df.loc[0, 'linear'] = np.mean(scores)
        print('attribute: {}, scores: {}'.format(output, scores_df))

        reg_lin = model.fit(X, y)

        # save the model
        trained_models = []
        trained_models.append(reg_lin)
        dump(trained_models, str('Ind_ac_models') + '_' + output + '.joblib')

    # X = preproc(X, 2)
    # bad_index_list = []
    # for index, row in X.iterrows():
    #     if 0 in row.values or np.nan in row.values or row.isnull().values.any():
    #         print(row)
    #         bad_index_list.append(index)
    #
    # for index, row in y.iterrows():
    #     if 0 in row.values or np.nan in row.values or row.isnull().values.any():
    #         print(row)
    #         bad_index_list.append(index)
    #
    # # X = X[np.all(X != 0, axis=1)]
    # fet_df.drop(bad_index_list, axis=0, inplace=True)

    # Pareto-optimize data on dimensions of interest
    data_dims = ['Mfr_part_no', 'V_dss', 'Unit_price', 'R_ds', 'Q_g', 'C_oss', 'C_ossp1', 'Vds_meas']
    data_dims_paretoFront_dict = {'C_oss': ['V_dss', 'Unit_price', 'R_ds', 'Q_g', 'C_ossp1']}
    pareto_opt_data = pareto_optimize(fet_df, data_dims, data_dims_paretoFront_dict['C_oss'], technology=test_tech)
    pareto_opt_df = pd.DataFrame(np.transpose(pareto_opt_data),
                                 columns=data_dims)
    pareto_opt_df['R_ds'] = pareto_opt_df['R_ds'].astype(float)
    pareto_opt_df['V_dss'] = pareto_opt_df['V_dss'].astype(float)
    pareto_opt_df['Unit_price'] = pareto_opt_df['Unit_price'].astype(float)
    pareto_opt_df['C_oss'] = pareto_opt_df['C_oss'].astype(float)
    pareto_opt_df['C_ossp1'] = pareto_opt_df['C_ossp1'].astype(float)
    pareto_opt_df['Q_g'] = pareto_opt_df['Q_g'].astype(float)
    pareto_opt_df['Vds_meas'] = pareto_opt_df['Vds_meas'].astype(float)

    X = np.log10(pareto_opt_df.loc[:, inputs])
    # X = preproc(X, 3)

    y = np.log10(pareto_opt_df.loc[:, outputs])
    # y = y[np.all(y != 0, axis=1)]

    scores_df = pd.DataFrame({'scoring_type': ['r^2', 'RMSE', 'MAE']}).set_index('scoring_type')
    model = LinearRegression(fit_intercept=True, normalize=True)
    model = RandomForestRegressor(min_samples_split=3, random_state=1)

    rmse_scorer = make_scorer(mean_squared_error, squared=False)
    cv = KFold(n_splits=3, random_state=1, shuffle=True)
    scores = cross_val_score(model, X, y, cv=cv, scoring='r2')
    scores_df.loc['r^2', 'linear'] = np.mean(scores)
    scores = cross_val_score(model, X, y, cv=cv, scoring=rmse_scorer)
    scores_df.loc['RMSE', 'linear'] = np.mean(scores)
    scores = cross_val_score(model, X, y, cv=cv, scoring='neg_mean_absolute_error')
    scores_df.loc['MAE', 'linear'] = np.mean(scores)
    # print('Model: Linear regression')
    # print('Accuracy: %.3f (%.3f)' % (np.mean(scores), np.std(scores)))
    # scores_df.loc[0, 'linear'] = np.mean(scores)
    reg_lin = model.fit(X, y)

    # save the model
    trained_models = []
    trained_models.append(reg_lin)
    dump(trained_models, str('Si_models') + '_' + 'C_ossp1' + '.joblib')


# Use the scraped (Coss, Vds) pairs to determine C_{oss,0.1}, and then train with C_{oss,0.1} as output and all other
# variables (Vdss, Ron, Cost, Qg) as inputs

def Coss_Vds_training(output_param = 'C_ossp1'):
    # Get the data after preprocessing
    csv_file = 'csv_files/postPreprocessing.csv'
    # csv_file = 'csv_files/wManual.csv'

    fet_df = pd.read_csv(csv_file)
    fet_df = fet_df.iloc[:, 1:]

    # Change the technology we are looking at for training
    test_tech = 'MOSFET'

    # Filter the Vds values
    csv_file = 'csv_files/FET_pdf_tables_wVds.csv'
    add_df = pd.read_csv(csv_file)
    add_df.columns = ['Mfr_part_no', 'Q_rr', 'C_oss','V_ds_coss']
    filtered_df = pd.DataFrame()
    for part in add_df['Mfr_part_no'].unique():
        # print(part)
        # print(add_df[add_df['Mfr_part_no'] == part])
        filt_df = add_df[add_df['Mfr_part_no'] == part]
        filt_df = filt_df.drop(filt_df[filt_df['C_oss'].isna() & filt_df['Q_rr'].isna()].index).drop_duplicates()
        filt_df = filt_df.drop(filt_df[(filt_df['C_oss'] == 0) & (filt_df['Q_rr'] == 0)].index).drop_duplicates()
        # print(filt_df)
        filtered_df = filtered_df.append(filt_df)

    # new csv, with all the info.
    # steps:
    #   1. clean it
    #   2. drop duplicate part numbers
    #   3. determine grouping category based on material and Vdss rating
    #   4. train with Coss,0.1 as output after using appropriate equation from paper w/ gamma prediction
    csv_file = 'csv_files/FET_pdf_complete_database.csv'
    fet_df = pd.read_csv(csv_file)

    fet_df = fet_df.iloc[:, 1:]
    fet_df.columns = ['Mfr_part_no', 'Unit_price', 'Mfr', 'Series', 'FET_type', 'Technology', 'V_dss', 'I_d', 'V_drive',
                      'R_ds', 'V_thresh', 'Q_g', 'V_gs', 'Input_cap', 'P_diss', 'Op_temp', 'Mount_type', 'Supp_pack',
                      'Pack_case', 'Q_rr', 'C_oss', 'I_F', 'Vds_meas']

    # first clean the data, drop rows without necessary info
    attr_list = ['Mfr_part_no', 'Unit_price', 'FET_type', 'Technology', 'V_dss', 'R_ds', 'Q_g']
    fet_df = initial_fet_parse(fet_df, attr_list)
    fet_df = area_filter(fet_df)
    fet_df = fet_df.replace(0, np.nan)

    fet_df = fet_df.dropna(subset=attr_list)

    # Find the Vds value: filter the Vds values in the V_ds_coss column by finding the first number after VDS
    for index, row in fet_df.iterrows():
        try:
            text = fet_df.loc[index, 'Vds_meas']
            # print(text)
            vds_match_list = ['VDS', 'V DS']
            for word in vds_match_list:
                if word in text:
                    word = word
                    break
            match = re.compile(word).search(text)
            vds_break = [text[:match.start()], text[match.start():]]
            match2 = re.compile("V").search(vds_break[1])
            V_break = [vds_break[1][:match2.start()], vds_break[1][match2.start() + 1:]]
            match3 = re.compile("V").search(V_break[1])
            V_break2 = [V_break[1][:match3.start()], V_break[1][match3.start():]]

            Vds_value = float(re.sub('[^0-9.-]', '', V_break2[0]))
            fet_df.loc[index, 'Vds_meas'] = Vds_value

        except:
            Vds_value = np.nan
            fet_df.loc[index, 'Vds_meas'] = Vds_value
        # print(Vds_value)

    for index, row in fet_df.iterrows():

        # Find the I_F value: find instance of 'IF', then find unit letter A, then filter out any other characterse
        try:
            text = fet_df.loc[index, 'I_F']
            print(text)
            if_match_list = ['IF', 'I F', 'IS', 'I S', 'ISD', 'I SD']
            for word in if_match_list:
                if word in text:
                    word = word
                    break
            match = re.compile(word).search(text)
            i_f_break = [text[:match.start()], text[match.start():]]
            match2 = re.compile("A").search(i_f_break[1])
            A_break = [i_f_break[1][:match2.start()], i_f_break[1][match2.start() + 1:]]

            I_F_value = float(re.sub('[^0-9.-]', '', A_break[0]))
            fet_df.loc[index, 'I_F'] = I_F_value

        except:
            I_F_value = np.nan
            fet_df.loc[index, 'I_F'] = I_F_value
        print(I_F_value)

    # make a copy of fet_df to use later
    fet_df_copy = fet_df.copy(deep=True)

    if output_param == 'C_ossp1':
        # Train with Coss,0.1 as the output:
        # Compute Coss,0.1, based on 1. computing gamma depending on relevant equation for Vdss and technology,
        # 2. use Vdss, Coss, and Vds,meas to compute Coss,0.1, 3. Use all parameters as inputs to train with Coss,0.1 as
        # output. Also, plot Coss,0.1 w.r.t. Vdss or correlation matrix or something
        gamma_eqn_dict = {'Si_low_volt': [-0.0021, -0.251], 'Si_high_volt': [-0.000569, -0.579], 'GaN_low_volt': [-0.00062, -0.355],
                          'GaN_high_volt': [-0.000394, -0.353], 'SiC': [0, -0.4509]}
        fet_df = fet_df.dropna(subset=['C_oss', 'Vds_meas'])
        fet_df = fet_df[fet_df['C_oss'] != 0]
        fet_df = fet_df[fet_df['Vds_meas'] != 0]

        for index, row in fet_df.iterrows():
            if fet_df.loc[index, 'Technology'] == 'MOSFET' and fet_df.loc[index, 'V_dss'] <= 200:
                category = 'Si_low_volt'
            elif fet_df.loc[index, 'Technology'] == 'MOSFET' and fet_df.loc[index, 'V_dss'] > 200:
                category = 'Si_high_volt'
            elif fet_df.loc[index, 'Technology'] == 'GaNFET' and fet_df.loc[index, 'V_dss'] <= 100:
                category = 'GaN_low_volt'
            elif fet_df.loc[index, 'Technology'] == 'GaNFET' and fet_df.loc[index, 'V_dss'] <= 200:
                category = 'GaN_high_volt'
            elif fet_df.loc[index, 'Technology'] == 'SiCFET':
                category = 'SiC'

            gamma = gamma_eqn_dict[category][0]*fet_df.loc[index, 'V_dss'] + gamma_eqn_dict[category][1]

            # now compute Coss,0.1
            fet_df.loc[index, 'C_ossp1'] = fet_df.loc[index, 'C_oss'] / ((0.1*fet_df.loc[index, 'V_dss'] / fet_df.loc[index, 'Vds_meas'])**gamma)


        # drop duplicates by part no.
        fet_df = fet_df.drop_duplicates(subset = 'Mfr_part_no')

        # drop columns without Coss, for training, filter based on FET type and technology, and develop model based on
        # Vdss, Ron, Qg, Cost
        fet_df = fet_df.dropna(subset = ['C_ossp1'])
        fet_df = fet_df[fet_df['C_oss'] > 0]
        fet_df = fet_df[fet_df['FET_type']=='N']

        # make a copy of this df to use for the test cases
        with open('fet_df_test_values', 'wb') as fet_df_file:
            # Step 3
            pickle.dump(fet_df, fet_df_file)

        fet_df = fet_df[fet_df['Technology']==test_tech]

        fet_df = fet_df[['Mfr_part_no','Unit_price', 'V_dss', 'R_ds','Q_g', 'C_oss','Vds_meas','C_ossp1']]
        fet_df = fet_df[np.all(fet_df != 0, axis=1)]

        inputs = ['Unit_price', 'V_dss', 'R_ds', 'Q_g']
        # inputs = ['V_dss', 'R_ds', 'Q_g']

        outputs = ['C_ossp1']
        # fet_df = fet_df[np.all(fet_df[inputs] != 0, axis=1)]

        X = np.log10(fet_df.loc[:, inputs])
        y = np.log10(fet_df.loc[:, outputs])

        # X = preproc(X, 2)
        bad_index_list = []
        for index, row in X.iterrows():
            if 0 in row.values or np.nan in row.values or row.isnull().values.any():
                print(row)
                bad_index_list.append(index)

        for index, row in y.iterrows():
            if 0 in row.values or np.nan in row.values or row.isnull().values.any():
                print(row)
                bad_index_list.append(index)

        # X = X[np.all(X != 0, axis=1)]
        fet_df.drop(bad_index_list, axis=0, inplace=True)

        # Pareto-optimize data on dimensions of interest
        data_dims = ['Mfr_part_no', 'V_dss', 'Unit_price', 'R_ds', 'Q_g', 'C_oss','C_ossp1','Vds_meas']
        data_dims_paretoFront_dict = {'C_oss': ['V_dss', 'Unit_price', 'R_ds', 'Q_g', 'C_ossp1']}
        pareto_opt_data = pareto_optimize(fet_df, data_dims, data_dims_paretoFront_dict['C_oss'], technology=test_tech)
        pareto_opt_df = pd.DataFrame(np.transpose(pareto_opt_data),
                                     columns=data_dims)
        pareto_opt_df['R_ds'] = pareto_opt_df['R_ds'].astype(float)
        pareto_opt_df['V_dss'] = pareto_opt_df['V_dss'].astype(float)
        pareto_opt_df['Unit_price'] = pareto_opt_df['Unit_price'].astype(float)
        pareto_opt_df['C_oss'] = pareto_opt_df['C_oss'].astype(float)
        pareto_opt_df['C_ossp1'] = pareto_opt_df['C_ossp1'].astype(float)
        pareto_opt_df['Q_g'] = pareto_opt_df['Q_g'].astype(float)
        pareto_opt_df['Vds_meas'] = pareto_opt_df['Vds_meas'].astype(float)

        X = np.log10(pareto_opt_df.loc[:, inputs])
        # X = preproc(X, 3)



        y = np.log10(pareto_opt_df.loc[:, outputs])
        # y = y[np.all(y != 0, axis=1)]


        scores_df = pd.DataFrame({'scoring_type':['r^2','RMSE','MAE']}).set_index('scoring_type')
        model = LinearRegression(fit_intercept=True, normalize=True)
        # model = RandomForestRegressor(min_samples_split=3, random_state=1)

        rmse_scorer = make_scorer(mean_squared_error, squared=False)
        cv = KFold(n_splits=3, random_state=1, shuffle=True)
        scores = cross_val_score(model, X, y, cv=cv, scoring='r2')
        scores_df.loc['r^2', 'linear'] = np.mean(scores)
        scores = cross_val_score(model, X, y, cv=cv, scoring=rmse_scorer)
        scores_df.loc['RMSE', 'linear'] = np.mean(scores)
        scores = cross_val_score(model, X, y, cv=cv, scoring='neg_mean_absolute_error')
        scores_df.loc['MAE', 'linear'] = np.mean(scores)
        # print('Model: Linear regression')
        # print('Accuracy: %.3f (%.3f)' % (np.mean(scores), np.std(scores)))
        # scores_df.loc[0, 'linear'] = np.mean(scores)
        reg_lin = model.fit(X, y)

        # save the model
        trained_models = []
        trained_models.append(reg_lin)
        dump(trained_models, str(test_tech) + '_models' + '_' + 'C_ossp1' + '.joblib')

        # Visualize the model on top of the data
        max_volt = 1000
        min_volt = 10.0
        T = np.linspace(min_volt, max_volt, 1000)
        T = np.log10(T)
        model_list = ['RdsCossp1_product']
        # model = 'random_forest'
        fet_reg_models_df = load(str(test_tech) + '_models' + '_' + 'C_ossp1' + '.joblib')
        y_ = []
        T = np.linspace(min_volt, max_volt, 1000)
        for volt in T:
            # print(volt)
            y_.append(10 ** (fet_reg_models_df[0].predict(np.array(
                [np.log10(pareto_opt_df['Unit_price'].mean()), np.log10(volt), np.log10(pareto_opt_df['R_ds'].mean()),
                 np.log10(pareto_opt_df['Q_g'].mean())]).reshape(1, -1)))[0])

        # filter the dataset we are plotting
        opt_df = pareto_opt_df.copy()
        opt_df = opt_df[opt_df['Unit_price'] < 3 * pareto_opt_df['Unit_price'].mean()]
        opt_df = opt_df[opt_df['Unit_price'] > 0.6 * pareto_opt_df['Unit_price'].mean()]
        opt_df = opt_df[opt_df['R_ds'] < 3 * pareto_opt_df['R_ds'].mean()]
        opt_df = opt_df[opt_df['R_ds'] > 0.6 * pareto_opt_df['R_ds'].mean()]
        opt_df = opt_df[opt_df['Q_g'] < 3 * pareto_opt_df['Q_g'].mean()]
        opt_df = opt_df[opt_df['Q_g'] > 0.6 * pareto_opt_df['Q_g'].mean()]

        # plt.scatter(opt_df.loc[:, 'V_dss'],
        #             opt_df.loc[:, 'C_ossp1'] * opt_df.loc[:, 'R_ds'], color='g', s=0.8)
        plt.scatter(opt_df.loc[:, 'V_dss'],
                    opt_df.loc[:, 'C_ossp1'], color='g', s=0.8)
        plt.plot(T, y_, color='navy', label=model, linewidth=.9)
        l = plt.legend()
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('Log[%s]' % 'vdss')
        plt.ylabel('Log[%s * %s]' % ('Cossp1', 'Ron'))
        plt.xlim((200,1000))


    elif output_param == 'Q_rr':
        # Train with Qrr as the output, then I_F:
        fet_df = fet_df.dropna(subset=['Q_rr'])
        fet_df = fet_df[fet_df['Q_rr'] > 0]
        fet_df = fet_df[fet_df['FET_type'] == 'N']
        fet_df = fet_df[fet_df['Technology'] == test_tech]
        # drop duplicates by part no.
        fet_df = fet_df.drop_duplicates(subset='Mfr_part_no')

        fet_df = fet_df[['Mfr_part_no', 'Unit_price', 'V_dss', 'R_ds', 'Q_g', 'Q_rr', 'I_F']]
        fet_df = fet_df[np.all(fet_df != 0, axis=1)]

        inputs = ['Unit_price', 'V_dss', 'R_ds', 'Q_g']
        outputs = ['Q_rr']
        # fet_df = fet_df[np.all(fet_df[inputs] != 0, axis=1)]

        X = np.log10(fet_df.loc[:, inputs])
        y = np.log10(fet_df.loc[:, outputs])

        # X = preproc(X, 2)
        bad_index_list = []
        for index, row in X.iterrows():
            if 0 in row.values or np.nan in row.values or row.isnull().values.any():
                print(row)
                bad_index_list.append(index)

        for index, row in y.iterrows():
            if 0 in row.values or np.nan in row.values or row.isnull().values.any():
                print(row)
                bad_index_list.append(index)

        # X = X[np.all(X != 0, axis=1)]
        fet_df.drop(bad_index_list, axis=0, inplace=True)

        # Pareto-optimize data on dimensions of interest
        data_dims = ['Mfr_part_no', 'V_dss', 'Unit_price', 'R_ds', 'Q_g', 'Q_rr', 'I_F']
        data_dims_paretoFront_dict = {'Q_rr': ['V_dss', 'Unit_price', 'R_ds', 'Q_g', 'Q_rr']}
        pareto_opt_data = pareto_optimize(fet_df, data_dims, data_dims_paretoFront_dict['Q_rr'], technology=test_tech)
        pareto_opt_df = pd.DataFrame(np.transpose(pareto_opt_data),
                                     columns=data_dims)
        pareto_opt_df['R_ds'] = pareto_opt_df['R_ds'].astype(float)
        pareto_opt_df['V_dss'] = pareto_opt_df['V_dss'].astype(float)
        pareto_opt_df['Unit_price'] = pareto_opt_df['Unit_price'].astype(float)
        pareto_opt_df['Q_rr'] = pareto_opt_df['Q_rr'].astype(float)
        pareto_opt_df['Q_g'] = pareto_opt_df['Q_g'].astype(float)
        pareto_opt_df['I_F'] = pareto_opt_df['I_F'].astype(float)

        if test_tech == 'MOSFET':
            pareto_opt_df = pareto_opt_df[pareto_opt_df['Q_rr'] >= 10]
            pareto_opt_df[pareto_opt_df['V_dss'] >= 200] = pareto_opt_df[pareto_opt_df['Q_rr'] >= 100]



        X = np.log10(pareto_opt_df.loc[:, inputs])
        # X = preproc(X, 3)

        y = np.log10(pareto_opt_df.loc[:, outputs])
        # y = y[np.all(y != 0, axis=1)]

        bad_index_list = []
        for index, row in X.iterrows():
            if 0 in row.values or np.nan in row.values or row.isnull().values.any():
                print(row)
                bad_index_list.append(index)

        for index, row in y.iterrows():
            if 0 in row.values or np.nan in row.values or row.isnull().values.any():
                print(row)
                bad_index_list.append(index)

        pareto_opt_df.drop(bad_index_list, axis=0, inplace=True)

        X = np.log10(pareto_opt_df.loc[:, inputs])
        # X = preproc(X, 3)

        y = np.log10(pareto_opt_df.loc[:, outputs])
        # y = y[np.all(y != 0, axis=1)]

        scores_df = pd.DataFrame({'scoring_type': ['r^2', 'RMSE', 'MAE']}).set_index('scoring_type')
        model = LinearRegression(fit_intercept=True, normalize=True)
        model = RandomForestRegressor(min_samples_split=3, random_state=1)

        rmse_scorer = make_scorer(mean_squared_error, squared=False)
        cv = KFold(n_splits=3, random_state=1, shuffle=True)
        scores = cross_val_score(model, X, y, cv=cv, scoring='r2')
        scores_df.loc['r^2', 'linear'] = np.mean(scores)
        scores = cross_val_score(model, X, y, cv=cv, scoring=rmse_scorer)
        scores_df.loc['RMSE', 'linear'] = np.mean(scores)
        scores = cross_val_score(model, X, y, cv=cv, scoring='neg_mean_absolute_error')
        scores_df.loc['MAE', 'linear'] = np.mean(scores)
        # print('Model: Linear regression')
        # print('Accuracy: %.3f (%.3f)' % (np.mean(scores), np.std(scores)))
        # scores_df.loc[0, 'linear'] = np.mean(scores)
        reg_lin = model.fit(X, y)

        # save the model
        trained_models = []
        trained_models.append(reg_lin)
        dump(trained_models, str('Si_models') + '_' + 'Q_rr' + '.joblib')

        # Visualize the model on top of the data
        max_volt = 1000
        min_volt = 10.0
        T = np.linspace(min_volt, max_volt, 1000)
        T = np.log10(T)
        model_list = ['RdsCossp1_product']
        # model = 'random_forest'
        fet_reg_models_df = load(str('Si_models') + '_' + 'Q_rr' + '.joblib')
        y_ = []
        T = np.linspace(min_volt, max_volt, 1000)
        for volt in T:
            # print(volt)
            y_.append(10 ** (fet_reg_models_df[0].predict(np.array(
                [np.log10(pareto_opt_df['Unit_price'].mean()), np.log10(volt), np.log10(pareto_opt_df['R_ds'].mean()),
                 np.log10(pareto_opt_df['Q_g'].mean())]).reshape(1, -1))))

        # # filter the dataset we are plotting
        opt_df = pareto_opt_df.copy()
        # opt_df = opt_df[opt_df['Unit_price'] < 3 * pareto_opt_df['Unit_price'].mean()]
        # opt_df = opt_df[opt_df['Unit_price'] > 0.6 * pareto_opt_df['Unit_price'].mean()]
        # opt_df = opt_df[opt_df['R_ds'] < 3 * pareto_opt_df['R_ds'].mean()]
        # opt_df = opt_df[opt_df['R_ds'] > 0.6 * pareto_opt_df['R_ds'].mean()]
        # opt_df = opt_df[opt_df['Q_g'] < 3 * pareto_opt_df['Q_g'].mean()]
        # opt_df = opt_df[opt_df['Q_g'] > 0.6 * pareto_opt_df['Q_g'].mean()]

        # plt.scatter(opt_df.loc[:, 'V_dss'],
        #             opt_df.loc[:, 'C_ossp1'] * opt_df.loc[:, 'R_ds'], color='g', s=0.8)
        plt.scatter(opt_df.loc[:, 'V_dss'],
                    opt_df.loc[:, 'Q_rr'], color='g', s=0.8)
        plt.plot(T, y_, color='navy', label='Random Forest regressor', linewidth=.9)
        l = plt.legend()
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('$V_{dss}$ [V]', fontsize=15)
        plt.ylabel('$Q_{rr}$ [nC]', fontsize=15)
        plt.xlim((20, 1000))

        plt.scatter(pareto_opt_df.loc[:, 'V_dss'],
                    pareto_opt_df.loc[:, 'Q_rr'], color='g', s=0.8)
        plt.grid(color='lightgrey', linewidth=1, alpha=0.4)

        plt.xscale('log')
        plt.yscale('log')

        inputs = ['Unit_price', 'V_dss', 'R_ds', 'Q_g']
        outputs = ['I_F']
        X = np.log10(pareto_opt_df.loc[:, inputs])
        # X = preproc(X, 3)

        y = np.log10(pareto_opt_df.loc[:, outputs])
        # y = y[np.all(y != 0, axis=1)]

        bad_index_list = []
        for index, row in X.iterrows():
            if 0 in row.values or np.nan in row.values or row.isnull().values.any():
                print(row)
                bad_index_list.append(index)

        for index, row in y.iterrows():
            if 0 in row.values or np.nan in row.values or row.isnull().values.any():
                print(row)
                bad_index_list.append(index)

        pareto_opt_df.drop(bad_index_list, axis=0, inplace=True)

        X = np.log10(pareto_opt_df.loc[:, inputs])
        # X = preproc(X, 3)

        y = np.log10(pareto_opt_df.loc[:, outputs])
        # y = y[np.all(y != 0, axis=1)]

        scores_df = pd.DataFrame({'scoring_type': ['r^2', 'RMSE', 'MAE']}).set_index('scoring_type')
        model = LinearRegression(fit_intercept=True, normalize=True)
        model = RandomForestRegressor(min_samples_split=3, random_state=1)

        rmse_scorer = make_scorer(mean_squared_error, squared=False)
        cv = KFold(n_splits=3, random_state=1, shuffle=True)
        scores = cross_val_score(model, X, y, cv=cv, scoring='r2')
        scores_df.loc['r^2', 'linear'] = np.mean(scores)
        scores = cross_val_score(model, X, y, cv=cv, scoring=rmse_scorer)
        scores_df.loc['RMSE', 'linear'] = np.mean(scores)
        scores = cross_val_score(model, X, y, cv=cv, scoring='neg_mean_absolute_error')
        scores_df.loc['MAE', 'linear'] = np.mean(scores)
        # print('Model: Linear regression')
        # print('Accuracy: %.3f (%.3f)' % (np.mean(scores), np.std(scores)))
        # scores_df.loc[0, 'linear'] = np.mean(scores)
        reg_lin = model.fit(X, y)

    # Now train with I_F as the output
    elif output_param == 'I_F':

        # Train with Qrr as the output, then I_F:
        fet_df = fet_df.dropna(subset=[output_param])
        fet_df = fet_df[fet_df[output_param] > 0]
        fet_df = fet_df[fet_df['FET_type'] == 'N']
        fet_df = fet_df[fet_df['Technology'] == test_tech]
        # drop duplicates by part no.
        fet_df = fet_df.drop_duplicates(subset='Mfr_part_no')

        fet_df = fet_df[['Mfr_part_no', 'Unit_price', 'V_dss', 'R_ds', 'Q_g', 'Q_rr', 'I_F']]
        fet_df = fet_df[np.all(fet_df != 0, axis=1)]

        inputs = ['Unit_price', 'V_dss', 'R_ds', 'Q_g']
        outputs = ['I_F']
        # fet_df = fet_df[np.all(fet_df[inputs] != 0, axis=1)]

        fet_df['I_F'] = fet_df['I_F'].astype(float)
        X = np.log10(fet_df.loc[:, inputs])
        y = np.log10(fet_df.loc[:, outputs])

        # X = preproc(X, 2)
        bad_index_list = []
        for index, row in X.iterrows():
            if 0 in row.values or np.nan in row.values or row.isnull().values.any():
                print(row)
                bad_index_list.append(index)

        for index, row in y.iterrows():
            if 0 in row.values or np.nan in row.values or row.isnull().values.any():
                print(row)
                bad_index_list.append(index)

        # X = X[np.all(X != 0, axis=1)]
        fet_df.drop(bad_index_list, axis=0, inplace=True)

        # Pareto-optimize data on dimensions of interest
        data_dims = ['Mfr_part_no', 'V_dss', 'Unit_price', 'R_ds', 'Q_g', 'Q_rr', 'I_F']
        data_dims_paretoFront_dict = {'Q_rr': ['V_dss', 'Unit_price', 'R_ds', 'Q_g', 'Q_rr']}
        pareto_opt_data = pareto_optimize(fet_df, data_dims, data_dims_paretoFront_dict['Q_rr'], technology=test_tech)
        pareto_opt_df = pd.DataFrame(np.transpose(pareto_opt_data),
                                     columns=data_dims)
        pareto_opt_df['R_ds'] = pareto_opt_df['R_ds'].astype(float)
        pareto_opt_df['V_dss'] = pareto_opt_df['V_dss'].astype(float)
        pareto_opt_df['Unit_price'] = pareto_opt_df['Unit_price'].astype(float)
        pareto_opt_df['Q_rr'] = pareto_opt_df['Q_rr'].astype(float)
        pareto_opt_df['Q_g'] = pareto_opt_df['Q_g'].astype(float)
        pareto_opt_df['I_F'] = pareto_opt_df['I_F'].astype(float)

        if test_tech == 'MOSFET':
            pareto_opt_df = pareto_opt_df[pareto_opt_df['Q_rr'] >= 10]
            pareto_opt_df[pareto_opt_df['V_dss'] >= 200] = pareto_opt_df[pareto_opt_df['Q_rr'] >= 100]

        X = np.log10(pareto_opt_df.loc[:, inputs])
        # X = preproc(X, 3)

        y = np.log10(pareto_opt_df.loc[:, outputs])
        # y = y[np.all(y != 0, axis=1)]

        bad_index_list = []
        for index, row in X.iterrows():
            if 0 in row.values or np.nan in row.values or row.isnull().values.any():
                print(row)
                bad_index_list.append(index)

        for index, row in y.iterrows():
            if 0 in row.values or np.nan in row.values or row.isnull().values.any():
                print(row)
                bad_index_list.append(index)

        pareto_opt_df.drop(bad_index_list, axis=0, inplace=True)

        X = np.log10(pareto_opt_df.loc[:, inputs])
        # X = preproc(X, 3)

        y = np.log10(pareto_opt_df.loc[:, outputs])
        # y = y[np.all(y != 0, axis=1)]

        scores_df = pd.DataFrame({'scoring_type': ['r^2', 'RMSE', 'MAE']}).set_index('scoring_type')
        model = LinearRegression(fit_intercept=True, normalize=True)
        # model = RandomForestRegressor(min_samples_split=3, random_state=1)

        rmse_scorer = make_scorer(mean_squared_error, squared=False)
        cv = KFold(n_splits=3, random_state=1, shuffle=True)
        scores = cross_val_score(model, X, y, cv=cv, scoring='r2')
        scores_df.loc['r^2', 'linear'] = np.mean(scores)
        scores = cross_val_score(model, X, y, cv=cv, scoring=rmse_scorer)
        scores_df.loc['RMSE', 'linear'] = np.mean(scores)
        scores = cross_val_score(model, X, y, cv=cv, scoring='neg_mean_absolute_error')
        scores_df.loc['MAE', 'linear'] = np.mean(scores)
        # print('Model: Linear regression')
        # print('Accuracy: %.3f (%.3f)' % (np.mean(scores), np.std(scores)))
        # scores_df.loc[0, 'linear'] = np.mean(scores)
        reg_lin = model.fit(X, y)

        # save the model
        trained_models = []
        trained_models.append(reg_lin)
        dump(trained_models, str('Si_models') + '_' + 'I_F' + '.joblib')

        fet_df = fet_df_copy.copy(deep=True)
        fet_df = fet_df_copy.dropna(subset=['I_F'])
        fet_df = fet_df[fet_df['I_F' != 0]]

        # drop columns without Coss, for training, filter based on FET type and technology, and develop model based on
        # Vdss, Ron, Qg, Cost
        fet_df = fet_df[fet_df['FET_type'] == 'N']
        fet_df = fet_df[fet_df['Technology'] == test_tech]

        inputs = ['Unit_price', 'V_dss', 'R_ds', 'Q_g']
        outputs = ['I_F']
        X = np.log10(fet_df.loc[:, inputs])
        X = preproc(X, 2)
        y = np.log10(fet_df.loc[:, outputs])

        scores_df = pd.DataFrame({'scoring_type': ['r^2', 'RMSE', 'MAE']}).set_index('scoring_type')
        model = LinearRegression(fit_intercept=True, normalize=True)
        rmse_scorer = make_scorer(mean_squared_error, squared=False)
        cv = KFold(n_splits=3, random_state=1, shuffle=True)
        scores = cross_val_score(model, X, y, cv=cv, scoring='r2')
        scores_df.loc['r^2', 'linear'] = np.mean(scores)
        scores = cross_val_score(model, X, y, cv=cv, scoring=rmse_scorer)
        scores_df.loc['RMSE', 'linear'] = np.mean(scores)
        scores = cross_val_score(model, X, y, cv=cv, scoring='neg_mean_absolute_error')
        scores_df.loc['MAE', 'linear'] = np.mean(scores)

        reg_lin = model.fit(X, y)

        # save the model
        trained_models = []
        trained_models.append(reg_lin)
        dump(trained_models, str('Si_models') + '_' + 'I_F' + '.joblib')


    ################# stop here
    attr_list = ['Mfr_part_no', 'Unit_price', 'FET_type', 'Technology', 'V_dss', 'R_ds', 'Q_g']
    # fet_df = fet_df.dropna(subset = ['C_oss'])
    #
    # fet_df = initial_fet_parse(fet_df, attr_list), attr_list
    # fet_df = fet_df.dropna(subset = attr_list)
    #
    #
    #
    # # Filter the Vds values in the V_ds_coss column by finding the first number after VDS
    # for index, row in filtered_df.iterrows():
    #     try:
    #         text = filtered_df.loc[index, 'V_ds_coss']
    #         match = re.compile("VDS ").search(text)
    #         vds_break = [text[:match.start()], text[match.start():]]
    #         match2 = re.compile("V").search(vds_break[1])
    #         V_break = [vds_break[1][:match2.start()], vds_break[1][match2.start() + 1:]]
    #         match3 = re.compile("V").search(V_break[1])
    #         V_break2 = [V_break[1][:match3.start()], V_break[1][match3.start():]]
    #
    #         Vds_value = float(re.sub('[^0-9.]', '', V_break2[0]))
    #     except:
    #         Vds_value = np.nan
    #
    #     # Add the Vds values to the overall df in postPreprocessing.csv, now in fet_df dataframe
    #     print(filtered_df.loc[index, 'Mfr_part_no'])
    #     fet_df_match = fet_df[fet_df['Mfr_part_no'].isin([filtered_df.loc[index, 'Mfr_part_no']]) == True].index
    #     if fet_df_match.empty:
    #         print('empty')
    #     fet_df.loc[fet_df_match, 'V_ds_coss'] = Vds_value
    #
    # both_df = fet_df.dropna(subset=['V_ds_coss'])
    # both_df = both_df[(both_df['C_oss'] != 0) & (both_df['V_ds_coss'] != 0)]
    # both_df = both_df[both_df['Technology'] == 'MOSFET']
    #
    # # Now determine C_{oss,0.1} for all quantities, using equation from paper. Gotten from the investigation of low-voltage
    # # Si components, first need to figure out the slope based on the Vdss for each component, then use this slope in Eq. (2)
    # # in the APEC digest
    # both_df['V_dss_slope'] = 0.0021*both_df['V_dss'] + 0.251
    # both_df['C_oss_p1'] = both_df['C_oss'] / ((0.1 * both_df['V_dss'] / both_df['V_ds_coss']) ** both_df['V_dss_slope'])
    #
    # # Now that we have the C_{oss,0.1} values for the datapoints, visualize them, and use the other set of 4 inputs
    # # to train with C_{oss,0.1} as the output
    #
    # # First visualize the distribution of Vdss and Coss,p1 values
    # plt.scatter(both_df['V_dss'], both_df['C_oss_p1'])
    # plt.xscale('log')
    # plt.yscale('log')
    #
    # # Train a simple ML model on the data we have
    # reg_score_and_dump(both_df, 'C_oss_p1', ['V_dss','Unit_price','Q_g','R_ds'], 'test_csv.csv', pareto=True, chained=False)
    # degree = 1
    #
    # rmse_scorer = make_scorer(mean_squared_error, squared=False)
    # # Get the X,y for model training
    # (X, y) = X_y_set(df, output_param, attributes, log_x=True, more_inputs=False, pareto=pareto)
    #
    # model = LinearRegression(fit_intercept=True, normalize=True)
    #
    # # model.fit(X, y)
    # # bef_df = before_df[before_df[output_param] != 0]
    # # bef_df = bef_df[bef_df[output_param] != np.nan]
    # # (X_before, y_before) = X_y_set(bef_df, output_param, attributes, log_x=True, more_inputs=False, pareto=pareto)
    # # X_before = preproc(X_before, degree)
    # # y_pred = model.predict(X_before)
    # # mae_score = mean_absolute_error(y_before, y_pred)
    #
    # scores = cross_validate(model, X, y, cv=cv, scoring=['r2', 'neg_mean_absolute_error'], return_train_score=True)
    # scores = cross_val_score(model, X, y, cv=cv, scoring='r2')
    # scores_df.loc['r^2', 'linear'] = np.mean(scores)
    # scores = cross_val_score(model, X, y, cv=cv, scoring=rmse_scorer)
    # scores_df.loc['RMSE', 'linear'] = np.mean(scores)
    # scores = cross_val_score(model, X, y, cv=cv, scoring='neg_mean_absolute_error')
    # scores_df.loc['MAE', 'linear'] = np.mean(scores)
    # # print('Model: Linear regression')
    # # print('Accuracy: %.3f (%.3f)' % (np.mean(scores), np.std(scores)))
    # # scores_df.loc[0, 'linear'] = np.mean(scores)
    # reg_lin = model.fit(X, y)


# compute kT given the proper coefficients of the linear equation for that category/grouping
def kt_compute(Vdss, category):
    kT_eqn_dict = {'Si_low_volt': [0.233, 1.15], 'Si_high_volt': [0.353, 0.827],
                   'GaN_low_volt': [0.066, 1.34],
                   'GaN_high_volt': [0.148, 1.145], 'SiC': [0.572, -0.323]}

    kT = kT_eqn_dict[category][0] * np.log10(Vdss) + kT_eqn_dict[category][1]
    return kT

# compute gamma given the proper coefficients of the linear equation for that category/grouping
def gamma_compute(Vdss, category):
    gamma_eqn_dict = {'Si_low_volt': [-0.0021, -0.251], 'Si_high_volt': [-0.000887, -0.428],
                      'GaN_low_volt': [-0.00062, -0.355],
                      'GaN_high_volt': [-0.000394, -0.353], 'SiC': [0, -0.4509]}
    gamma = gamma_eqn_dict[category][0] * Vdss + gamma_eqn_dict[category][1]
    return gamma

# Compute the true gamma value given the manually digitized points for that component
def trueGamma_compute(part_no, Vdss, Cossp1, Vds, Coss):
    file_name = 'datasheet_graph_info/' + part_no + '_' + str(Vdss) + 'V.csv'
    Coss_dataset = pd.read_csv(file_name)
    Coss_dataset.columns = ['Vds', 'Coss']

    # p = np.polyfit(np.log10(Coss_dataset['Vds']), np.log10(Coss_dataset['Coss']), deg=1)
    Coss_dataset['Vds'] = (0.1 * Vdss) / Coss_dataset['Vds']
    Coss_dataset['Coss'] = Coss_dataset['Coss'] / Cossp1
    # c = next(color)
    # plt.plot(Coss_dataset['Vdss'], Coss_dataset['Coss'], c=c)
    p = np.polyfit(np.log10(Coss_dataset['Vds']), np.log10(Coss_dataset['Coss']), deg=1)
    slope= p[0]
    return slope


# This function looks at out-of-sample components, and based on their corresponding equation based on Vdss,
# calculates the coefficient and compares with the manually-digitized coefficient value, ultimately computing the
# MAE for each grouping
def Ron_Coss_MAE(category):
    # coefficients for the linear equation for each grouping


    # decide which technology to look at
    # category = 'Si_high_volt'
    # for each grouping dictionary, a dictionary of the component and a list of [predicted kT, datasheet-reported kT]
    datasheet_validation_dict = {
                        'lowVolt_Si':
                            {'AO3414': [kt_compute(20, 'Si_low_volt'), 1.27],
                             'BSS670S2LH6327XTSA1': [kt_compute(55, 'Si_low_volt'), 935/650],
                             'BSN20BKR': [kt_compute(60, 'Si_low_volt'), 1.52],
                             'DMG3420U7': [kt_compute(20, 'Si_low_volt'), 1.31],
                             # 'RYM002N05T2CL': [kt_compute(50, 'Si_low_volt'), 'truekT'],
                             'TSM300NB06DCR-RL': [kt_compute(60, 'Si_low_volt'), 1.7]},
                      'highVolt_Si':
                            {'AOD2N60': [kt_compute(600, 'Si_high_volt'), 1.7],
                             'IPD70R360P7SAUMA1': [kt_compute(700, 'Si_high_volt'), 588/360],
                             'FDB14N30TM': [kt_compute(300, 'Si_high_volt'), 1.57],
                             'STD15N50M2AG': [kt_compute(500, 'Si_high_volt'), 1.8],
                             # 'IPD80R450P7ATMA1': [kt_compute(800, 'Si_high_volt'), 'truekT'],
                             'IPD60R380C6': [kt_compute(600, 'Si_high_volt'), 683/380]},
                      'lowVolt_GaN':
                            {'EPC2202': [kt_compute(80, 'GaN_low_volt'), 1.51],
                             'EPC2212': [kt_compute(100, 'GaN_low_volt'), 1.55],
                             'EPC8004': [kt_compute(40, 'GaN_low_volt'), 1.4],
                             'EPC2066': [kt_compute(40, 'GaN_low_volt'), 1.38],
                             'EPC2031': [kt_compute(60, 'GaN_low_volt'), 1.47]},
                      'highVolt_GaN':
                            {'EPC2054': [kt_compute(200, 'GaN_high_volt'), 1.46],
                             'EPC2033': [kt_compute(150, 'GaN_high_volt'), 1.5],
                             'TP65H050WS': [kt_compute(650, 'GaN_high_volt'), 1.62],
                             'TPH3206PD': [kt_compute(600, 'GaN_high_volt'), 1.52],
                             # 'EPC2025': [kt_compute(300, 'GaN_high_volt'), 'truekT'],
                             'IGT60R070D1': [kt_compute(600, 'GaN_high_volt'), 93/55]},
                        'allVolt_SiC':
                            {'UF3C120150B7S': [kt_compute(1200, 'SiC'), 1.45],
                             'C3M0280090J': [kt_compute(900, 'SiC'), 1.1],
                             'MSC035SMA170S': [kt_compute(1700, 'SiC'), 1.25],
                             'C3M0065100J': [kt_compute(1000, 'SiC'), 1.2],
                             # 'NVH4L020N120SC1': [kt_compute(1200, 'SiC'), 'truekT'],
                             'C3M0045065K': [kt_compute(650, 'SiC'), 52/49]
                                }
                        }

    # compute the MAE for each grouping
    kT_predicted = [kT[0] for kT in datasheet_validation_dict[category].values()]
    kT_true = [kT[1] for kT in datasheet_validation_dict[category].values()]

    MAE_score = mean_absolute_error(kT_true, kT_predicted)

    # Now compute the MAEs for the gamma of Coss out-of-sample components, using a similar approach, but computing each
    # gamma, where the predicted gamma is based on the corresponding category equation, and the
    # actual gamma is found by plotting the log-log curve.
    datasheet_validation_dict = {
        'lowVolt_Si':
                            {'AO3414': [gamma_compute(20, 'Si_low_volt'), trueGamma_compute('AO3414',20, 79, 'Vds', 'Coss')],
                             'BSS670S2LH6327XTSA1': [gamma_compute(55, 'Si_low_volt'), trueGamma_compute('BSS670S2LH6327XTSA1',55, 44, 'Vds', 'Coss')],
                             'BSN20BKR': [gamma_compute(60, 'Si_low_volt'), trueGamma_compute( 'BSN20BKR',60, 5.4, 'Vds', 'Coss')],
                             'DMG3420U7': [gamma_compute(20, 'Si_low_volt'), trueGamma_compute('DMG3420U7',20, 128, 'Vds', 'Coss')],
                             # 'RYM002N05T2CL': [kt_compute(50, 'Si_low_volt'), 'truekT'],
                             'TSM300NB06DCR-RL': [gamma_compute(60, 'Si_low_volt'), trueGamma_compute('TSM300NB06DCR-RL',60, 129, 'Vds', 'Coss')]},
                      'highVolt_Si':
                            {
                                # 'AOD2N60': [gamma_compute(600, 'Si_high_volt'), trueGamma_compute('Vdss', 19, 'Vds', 'Coss')],

                                'IPL65R115CFD7': [gamma_compute(700, 'Si_high_volt'),
                                               trueGamma_compute('IPL65R115CFD7', 700, 62, 'Vds', 'Coss')],
                                'SiHS36N50D': [gamma_compute(500, 'Si_high_volt'),
                                               trueGamma_compute('SiHS36N50D', 500, 430, 'Vds', 'Coss')],
                                'IRFR320': [gamma_compute(400, 'Si_high_volt'), trueGamma_compute('IRFR320',400, 56, 'Vds', 'Coss')],
                                'RCJ331N25': [gamma_compute(250, 'Si_high_volt'),
                                            trueGamma_compute('RCJ331N25', 250, 226, 'Vds', 'Coss')],

                                'STD15N50M2AG': [gamma_compute(500, 'Si_high_volt'), trueGamma_compute('STD15N50M2AG',500, 111, 'Vds', 'Coss')],
                             # 'IPD80R450P7ATMA1': [kt_compute(800, 'Si_high_volt'), 'truekT'],
                             'IPD60R380C6': [gamma_compute(600, 'Si_high_volt'), trueGamma_compute('IPD60R380C6',600, 109, 'Vds', 'Coss')]},
                      'lowVolt_GaN':
                            {'EPC2202': [gamma_compute(80, 'GaN_low_volt'), trueGamma_compute('EPC2202',80, 550, 'Vds', 'Coss')],
                             'EPC2212': [gamma_compute(100, 'GaN_low_volt'), trueGamma_compute('EPC2212',100, 537, 'Vds', 'Coss')],
                             'EPC8004': [gamma_compute(40, 'GaN_low_volt'), trueGamma_compute('EPC8004',40, 38, 'Vds', 'Coss')],
                             'EPC2066': [gamma_compute(40, 'GaN_low_volt'), trueGamma_compute('EPC2066',40, 3640, 'Vds', 'Coss')],
                             'EPC2031': [gamma_compute(60, 'GaN_low_volt'), trueGamma_compute('EPC2031',60, 2030, 'Vds', 'Coss')]},
                      'highVolt_GaN':
                            {'EPC2054': [gamma_compute(200, 'GaN_high_volt'), trueGamma_compute('EPC2054',200, 168, 'Vds', 'Coss')],
                             'EPC2033': [gamma_compute(150, 'GaN_high_volt'), trueGamma_compute('EPC2033',150, 1564, 'Vds', 'Coss')],
                             'TP65H050WS': [gamma_compute(650, 'GaN_high_volt'), trueGamma_compute('TP65H050WS',650, 422, 'Vds', 'Coss')],
                             'TPH3206PD': [gamma_compute(600, 'GaN_high_volt'), trueGamma_compute('TPH3206PD',600, 187, 'Vds', 'Coss')],
                             # 'EPC2025': [kt_compute(300, 'GaN_high_volt'), 'truekT'],
                             'IGT60R070D1': [gamma_compute(600, 'GaN_high_volt'), trueGamma_compute('IGT60R070D1',600, 157, 'Vds', 'Coss')]},
                        'allVolt_SiC':
                            {'UF3C120150B7S': [gamma_compute(1200, 'SiC'), trueGamma_compute('UF3C120150B7S',1200, 52, 'Vds', 'Coss')],
                             'C3M0280090J': [gamma_compute(900, 'SiC'), trueGamma_compute('C3M0280090J',900, 51, 'Vds', 'Coss')],
                             'MSC035SMA170S': [gamma_compute(1700, 'SiC'), trueGamma_compute('MSC035SMA170S',1700, 375, 'Vds', 'Coss')],
                             'C3M0065100J': [gamma_compute(1000, 'SiC'), trueGamma_compute('C3M0065100J',1000, 153, 'Vds', 'Coss')],
                             # 'NVH4L020N120SC1': [kt_compute(1200, 'SiC'), 'truekT'],
                             'C3M0045065K': [gamma_compute(650, 'SiC'), trueGamma_compute('C3M0045065K',650, 233, 'Vds', 'Coss')]
                                }
    }

    # compute the MAE for each grouping
    gamma_predicted = [-gamma[0] for gamma in datasheet_validation_dict[category].values()]
    gamma_true = [gamma[1] for gamma in datasheet_validation_dict[category].values()]

    MAE_score = mean_absolute_error(gamma_true, gamma_predicted)
    print('done')




def Ron_plotting(grouping):
    # low voltage Si
    values_dict = {'lowVolt_Si':
                       {'AOSS62934': [100, 1.57], 'DMN63D8L': [30,1.6], 'DMN67D8LW': [60,1.54], 'IRFRU220NPbF': [200,1.64],
                        'NX7002AK': [60,1.56], 'PSMN017': [80, 1.66], '2N7002E': [60, 1.44], 'BSS138W': [50, 1.48],
                        'BSS123NH6327XTSA1': [100, 3.89/2.4], 'DMN10H120SFG': [100, 1.69], 'FDT86244': [150, 1.68],
                        'IRF7465PbF': [150, 1.63], 'FDB52N20': [200, 1.76]},
                   'lowVolt_GaN':
                       {'EPC2040': [15, 1.41], 'EPC2023': [30, 1.46],'EPC2055':[40,1.44],
                        'EPC2039': [80, 1.47],'EPC2214': [80,1.44],
                        'EPC2038': [100, 1.49], 'EPC2036': [100,1.47]},
                   'highVolt_Si':
                       {'STN3N40K3': [400, 1.76], 'IPD50R3K0CE': [500, 4.89/2.7],
                        'AOD4S60': [600, 1.86], 'IPD60R400CE': [600, 0.58/0.34],
                        'R8002CND3FRATL': [800, 1.8], 'APT9M100B': [1000,2.06],
                        'STH6N95K5': [950, 1.83],
                        'STD2N105K5': [1050, 1.84], 'STH12N120K5': [1200, 1.9]},
                   'highVolt_GaN':
                       {'EPC2018': [150, 1.47],
                        'EPC2010': [200, 1.49], 'EPC2034': [200, 1.5],
                        'TPH3206LS': [600, 1.56],
                        'TP65H070L': [650, 1.59], 'TPH3208LD': [650, 1.55],
                        'TP90H050WS': [900, 1.59], 'IGT40R070D1': [400, .082/.055]},
                   'allVolt_SiC':
                       {'UF3C065080B7S': [650, 1.18],
                        'MSC035SMA070S': [700, 1.04],
                        'C3M0120090J': [900, 1.08],
                        'NVBG160N120SC1': [1200, 1.18],
                        'C2M1000170J': [1700, 1.51],
                        'IMBF170R450M1': [1700, 580 / 450],
                        'G3R450MT17J': [1700, 609/452],

                        'G3R40MT12J': [1700, 45/39.8],
                        'UF3C120080B7S': [1200, 1.4],
                        'C3M0060065J': [650, 1.07],
                        'SCT3060AW7TL': [650, 69/63]


                        },
                   'allVolt_GaN':
                       {'EPC2040': [15, 1.41], 'EPC2023': [30, 1.46],
                        'EPC2039': [80, 1.47],
                        'EPC2038': [100, 1.49], 'EPC2018': [150, 1.47],
                        'EPC2010': [200, 1.49],'EPC2010': [200, 1.49], 'EPC2034': [200, 1.5],
                        'TPH3206LS': [600, 1.56],
                        'TP65H070L': [650, 1.59], 'TPH3208LD': [650, 1.55],
                        'TP90H050WS': [900, 1.59], 'IGT40R070D1': [400, .082/.055]},
                   'allVolt_Si':
                       {'AOSS62934': [100, 1.57], 'DMN63D8L': [30, 1.6], 'DMN67D8LW': [60, 1.54],
                        'IRFRU220NPbF': [200, 1.64],
                        'NX7002AK': [60, 1.56], 'PSMN017': [80, 1.66], '2N7002E': [60, 1.44], 'BSS138W': [50, 1.48],
                        'BSS123NH6327XTSA1': [100, 3.89 / 2.4], 'DMN10H120SFG': [100, 1.69], 'FDT86244': [150, 1.68],
                        'IRF7465PbF': [150, 1.63], 'FDB52N20': [200, 1.76],
                        'STN3N40K3': [400, 1.76], 'IPD50R3K0CE': [500, 4.89 / 2.7],
                        'AOD4S60': [600, 1.86], 'IPD60R400CE': [600, 0.58 / 0.34],
                        'R8002CND3FRATL': [800, 1.8], 'APT9M100B': [1000, 2.06],
                        'STH6N95K5': [950, 1.83],
                        'STD2N105K5': [1050, 1.84], 'STH12N120K5': [1200, 1.9]}
                   }


    # , 'RE1C002UNTCL': [20], 'RUM002N05': [50, 1.03]}

    # low voltage GaN

    # high voltage Si
    # values_dict =

    # high voltage GaN
    # values_dict =

    # all voltage SiC
    # values_dict =

    # low voltage

    # grouping = 'lowVolt_Si'
    slope_list = []
    Vdss_vals = []
    Ron_vals = []
    for key in values_dict[grouping].keys():
        # file_name = 'csv_files/' + list(values_dict.keys())[0] + '_' + str(
        #     values_dict[list(values_dict.keys())[0]][0]) + 'V.csv'
        # file_name = 'csv_files/' + key + '_' + str(
        #     values_dict[key][0]) + 'V.csv'
        # Coss_dataset = pd.read_csv(file_name)
        # Coss_dataset.columns = ['Vdss', 'Coss']
        color = iter(cm.rainbow(np.linspace(0, 1, len(values_dict[grouping]))))
        c=next(color)
        plt.scatter(values_dict[grouping][key][0], values_dict[grouping][key][1], c='blue')
        Vdss_vals.append(values_dict[grouping][key][0])
        Ron_vals.append(values_dict[grouping][key][1])

        # p = np.polyfit(np.log10(Coss_dataset['Vdss']), np.log10(Coss_dataset['Coss']), deg=1)
        # Coss_dataset['Vdss'] = (0.1*values_dict[key][0]) / Coss_dataset['Vdss']
        # Coss_dataset['Coss'] = Coss_dataset['Coss']/values_dict[key][1]
        # # c = next(color)
        # # plt.plot(Coss_dataset['Vdss'], Coss_dataset['Coss'], c=c)

    # Vdss_vals = [lis[0] for lis in values_dict[grouping]]
    # Ron_vals = [lis[1] for lis in values_dict[grouping]]

    p = np.polyfit(np.log10(Vdss_vals), Ron_vals, deg=1)

    plt.xlabel('$V_{dss}$ [V]', fontsize=15)
    plt.ylabel('$k_T$', fontsize = 15)
    plt.xscale('log')
    # add best-fit line
    a, b = np.polyfit(np.log10(Vdss_vals), Ron_vals, deg=1)
    # add line of best fit to plot
    T = np.linspace(min(Vdss_vals), max(Vdss_vals), 1000)
    plt.plot(T, a * np.log10(T) + b, c='blue')

    legend_list = []
    for key in values_dict[grouping].keys():
        legend_list.append((values_dict[grouping][key][0], key))

 # p = np.polyfit(np.log10(Coss_dataset['Vdss']), np.log10(Coss_dataset['Coss']), deg=1)
 #        Coss_dataset['Vdss'] = (0.1*values_dict[grouping][key][0]) / Coss_dataset['Vdss']
 #        Coss_dataset['Coss'] = Coss_dataset['Coss']/values_dict[grouping][key][1]
 #        # c = next(color)
 #        # plt.plot(Coss_dataset['Vdss'], Coss_dataset['Coss'], c=c)
 #        # p = np.polyfit(np.log10(Coss_dataset['Vdss']), np.log10(Coss_dataset['Coss']), deg=1)
 #        slope_list.append((values_dict[grouping][key][0], p[0]))
 #
 #    legend_list = []
 #    for key in values_dict[grouping].keys():
 #        legend_list.append((values_dict[grouping][key][0], key))
 #    # for key in values_dict.keys():
 #    #     legend_list.append(values_dict[key][0])
 #    # l = plt.legend(legend_list, prop={'size': 8}, title='$V_{dss}$ device rating [V]', title_fontsize='small')
 #    # plt.xlabel('$0.1V_{dss}/V_{ds}$')
 #    # plt.ylabel('$C_{oss}/C_{oss,0.1}$')
 #
 #    plt.xscale('log')
 #    plt.yscale('log')
 #    plt.grid(color='lightgrey', linewidth=1, alpha=0.4)
 #
 #    plt.xlabel('$V_{ds}/V_{dss}$', fontsize=15)
 #    plt.ylabel('$C_{oss}/C_{oss,0.1}$', fontsize=15)
 #    l = plt.legend(legend_list, prop={'size': 8}, title='$V_{dss}$ device rating [V]', title_fontsize=12, ncol=2)
 #
 #    plt.show()
 #    plt.close()
 #    plt.scatter(*zip(*slope_list))
 #    plt.xlabel('Vdss')
 #    plt.ylabel('Slope of Coss vs. Vdss plot')
 #    g = np.polyfit(*zip(*slope_list), deg=1)

# load and gather the Coss data for each category, normalize, and piece together
def Coss_plotting(grouping):
    xlsx_file = 'csv_files/STN1HNK60_650V_23p5p_25V.csv'
    xlsx_file = 'csv_files/IPD95R2K0P7_950V_5p_400V.csv'
    xlsx_file = 'csv_files/FDB14N30_300V_150p_25V.csv'
    xlsx_file = 'csv_files/TSM70N380_E1706_700V_58p_100V.csv'
    # xlsx_file = 'csv_files/STL15N65M5_650V.csv'
    #
    # Coss_dataset = pd.read_csv(xlsx_file)
    # Coss_dataset.columns = ['Vdss', 'Coss']

    # Now normalize the data values, voltage to rated, Coss to

    # turn the following into a dataframe, with: voltage rating [V], Coss value [pF] at 10% voltage,
    # Vdss at Coss reported measurement [V], Coss reported measurement [pF], Ron factor at 100*C
    # values_dict = {'FDB14N30': [300, 121, 25, 150], 'STL15N65M5': [650, 31, 100, 23], 'HB21N65EF': [650, 213, 100, 105], 'TSM70N380': [700, 67, 100, 58],
    #                'IPD95R2K0P7': [950, 7.6, 400, 5]}

    # Use the grouping_MAE entries to determine the MAE between the reported values and what is predicted by using the equations
    # for each grouping (the estimated versions) gotten using the components in values_dict

    # low voltage Si

    # grouping = 'lowVolt_Si'
    values_dict = {'lowVolt_Si': {'CSD13381F4': [12, 88, 6, 47], 'RE1C002UNTCL': [20, 14, 10, 10], 'DMN63D8L': [30, 5.4, 25, 3],
                   'RUM002N05': [50, 6.7, 10, 6], 'DMN67D8LW': [60, 6.5, 25, 4.1], 'NX7002AK': [60, 0.81, 10, 3.4],'BSN20BK': [60, 5.5, 30, 3.1],
                    'SL3607PbF': [75, 720, 50, 280], 'PSMN017': [80, 283, 40, 154], 'AOSS62934': [100, 89, 50, 19], 'FDD86252': [150, 370, 75, 78], 'IRFRU220NPbF': [200, 50, 25, 53]
                   },
                   'lowVolt_GaN':
                       {
                        'EPC2023': [30, 2301, 15, 1530],
                        'FBG04N30B': [40, 1247, 20, 650], 'EPC2055': [40, 824, 20, 408], 'EPC2066': [40, 3639, 20, 1670],
                        'EPC2030': [40, 1900, 20, 1120], 'EPC8004': [40, 37, 20, 23], 'EPC2031': [60, 2041, 30, 980], 'EPC2020': [60, 2100, 30, 1020],
                        'EPC8002': [65, 12.7, 32.5, 6.7],
                        'EPC2039': [80, 276, 40, 115], 'EPC2214': [80, 302, 40, 129], 'EPC2051': [100, 223, 50, 86],
                        'EPC2038': [100, 3.6, 50, 1.6], 'EPC8010': [100, 56.5, 50, 0.3],
                        },
                   'highVolt_Si':
                       {'BSP126': [250, 16, 25, 21], 'STD17NF25': [250,107,25,178], 'STN3N40K3': [400, 20, 50, 17],
                        'AOD4S60': [600, 30, 100, 21], 'IPD60R400CE': [600, 110, 100, 46],
                        'TK31V60W5': [600, 137, 300, 70],'IPD60R1K5CE': [600,37,100,16], 'IPD70R1K4P7S': [700,5.9,400,3],
                        'R8002CND3FRATL': [800, 20, 25, 125], 'SiHD6N80E': [850,46,100,37],
                        'IPD70R360P7SAUMA1': [700, 18, 'Vds', 'Coss'],
                        'FDB14N30TM': [300, 134, 'Vds', 'Coss'],
                        },
                        # 'STH6N95K5': [950, 33, 100, 30], 'APT9M100B': [1000, 97, 25, 220],
                        # 'STP7N105K5': [1050, 47, 100, 40],
                        # 'STD2N105K5': [1050, 16, 100, 15], 'STH12N120K5': [1200, 91, 100, 110]},
                   'highVolt_GaN':
                       {'EPC8010': [100, 56.5, 50, 0.3], 'EPC2018': [150, 0.59, 100, 270],
                       'EPC2054': [200, 169, 100, 89],
                        'EPC2010': [200, 0.4, 100, 270], 'EPC2034': [200, 708, 100, 450],
                                       'EPC2034C': [200, 1000, 100, 641],
                                       'IGOT60R070D1': [600, 156, 400, 72],
                                       'TPH3206LS': [600, 186, 480, 44], 'GPI65010DF56': [650, 71, 400, 25],'GPI65015DFN': [650, 153, 400, 34],
                                       'P1H06300D8': [650, 83, 400, 27.3],'TP65H070L': [650, 320, 400, 88],'TP65H480G4JSG': [650, 27, 400, 9],
                        'TP90H050WS': [900, 454, 600, 115],'GPIHV30DFN': [1200, 127, 400, 25]},
                   'allVolt_SiC':
                       {'SCT3120AW7': [650, 116, 500, 35], 'UF3C065080B7S': [650, 122, 100, 98],
                        'MSC035SMA070S': [700, 431, 700, 247],
                        'C3M0120090J': [900, 109, 600, 48], 'C3M0065100J': [1000, 152, 600, 70],
                        'NVBG160N120SC1': [1200, 96, 800, 50.7],
                        'SCT3105KW7': [1200, 103, 800, 59], 'C2M1000170J': [1700, 28, 1000, 19],
                        'IMBF170R450M1': [1700, 41, 1000, 16],
                        'MSC035SMA170S': [1700, 422, 1000, 150],
                        },
                   'lowVolt_GaN_experimental': {'EPC2036': [100, 19, 50, 50], 'EPC2035': [60, 19.3, 30, 60]},
                   'allVolt_SiC_experimental': {'UF3C120150B7S': [1200, 55, 100, 58], 'C3M0280090J': [900, 52, 600, 26]},
                   'highVolt_Si_experimental': {'IPx60R380C6': [600, 116, 100, 46], 'STB13N60M2': [600, 57, 100, 32]},
                   'highVolt_GaN_experimental': {'TPH3208L': [650,227,400,56], 'IGT60R070D1':[600, 143, 400,72]},
                   'allVolt_GaN': {'EPC2040': [15, 73, 6, 67], 'EPC2216': [15, 73, 7.5, 66], 'EPC2023': [30, 2301, 15, 1530],
                        'FBG04N30B': [40, 1247, 20, 650],
                        'EPC2039': [80, 276, 40, 115], 'EPC2214': [80, 302, 40, 129], 'EPC2051': [100, 223, 50, 86],
                        'EPC2038': [100, 3.6, 50, 1.6],
                        'EPC8010': [100, 56.5, 50, 0.3], 'EPC2018': [150, 0.59, 100, 270],
                        'EPC2054': [200, 169, 100, 89],
                        'EPC2010': [200, 0.4, 100, 270], 'EPC2034': [200, 708, 100, 450],
                                       'EPC2034C': [200, 1000, 100, 641],
                                       'IGOT60R070D1': [600, 156, 400, 72],
                                       'TPH3206LS': [600, 186, 480, 44], 'GPI65010DF56': [650, 71, 400, 25],'GPI65015DFN': [650, 153, 400, 34],
                                       'P1H06300D8': [650, 83, 400, 27.3],'TP65H070L': [650, 320, 400, 88],'TP65H480G4JSG': [650, 27, 400, 9],
                        'TP90H050WS': [900, 454, 600, 115],'GPIHV30DFN': [1200, 127, 400, 25]},
                   'allVolt_Si': {'CSD13381F4': [12, 88, 6, 47], 'RE1C002UNTCL': [20, 14, 10, 10], 'DMN63D8L': [30, 5.4, 25, 3],
                   'RUM002N05': [50, 6.7, 10, 6], 'DMN67D8LW': [60, 6.5, 25, 4.1], 'NX7002AK': [60, 0.81, 10, 3.4],'BSN20BK': [60, 5.5, 30, 3.1],
                    'SL3607PbF': [75, 720, 50, 280], 'PSMN017': [80, 283, 40, 154], 'AOSS62934': [100, 89, 50, 19], 'FDD86252': [150, 370, 75, 78], 'IRFRU220NPbF': [200, 50, 25, 53],
                                  'BSP126': [250, 16, 25, 21], 'STD17NF25': [250, 107, 25, 178],
                                  'STN3N40K3': [400, 20, 50, 17],
                                  'AOD4S60': [600, 30, 100, 21], 'IPD60R400CE': [600, 110, 100, 46],
                                  'TK31V60W5': [600, 137, 300, 70], 'IPD60R1K5CE': [600, 37, 100, 16],
                                  'IPD70R1K4P7S': [700, 5.9, 400, 3],
                                  'R8002CND3FRATL': [800, 20, 25, 125], 'SiHD6N80E': [850, 46, 100, 37]},
                   'lowVolt_Si_MAE': {'AO3414': [20, 79, 'Vds', 'Coss'],
                             'BSS670S2LH6327XTSA1': [55, 44, 'Vds', 'Coss'],
                             'BSN20BKR': [60, 5.4, 'Vds', 'Coss'],
                             'DMG3420U7': [20, 128, 'Vds', 'Coss'],
                             'TSM300NB06DCR-RL': [60, 129, 'Vds', 'Coss']},
                   'highVolt_Si_MAE': {
                             'IPD70R360P7SAUMA1': [700, 18, 'Vds', 'Coss'],
                             'FDB14N30TM': [300, 134, 'Vds', 'Coss'],
                             'STD15N50M2AG': [500, 111, 'Vds', 'Coss'],
                             'IPD60R380C6': [600, 109, 'Vds', 'Coss']},
                   'lowVolt_GaN_MAE': {'EPC2202': [80, 550, 'Vds', 'Coss'],
                             'EPC2212': [100, 537, 'Vds', 'Coss'],
                             'EPC8004': [40, 38, 'Vds', 'Coss'],
                             'EPC2066': [40, 3640, 'Vds', 'Coss'],
                             'EPC2031': [60, 2030, 'Vds', 'Coss']},
                   'highVolt_GaN_MAE': {'EPC2054': [200, 168, 'Vds', 'Coss'],
                             'EPC2033': [150, 1564, 'Vds', 'Coss'],
                             'TP65H050WS': [650, 422, 'Vds', 'Coss'],
                             'TPH3206PD': [600, 187, 'Vds', 'Coss'],
                             'IGT60R070D1': [600, 157, 'Vds', 'Coss']},
                   'allVolt_SiC_MAE': {'UF3C120150B7S': [1200, 52, 'Vds', 'Coss'],
                             'C3M0280090J': [900, 51, 'Vds', 'Coss'],
                             'MSC035SMA170S': [1700, 375, 'Vds', 'Coss'],
                             'C3M0065100J': [1000, 153, 'Vds', 'Coss'],
                             'C3M0045065K': [650, 233, 'Vds', 'Coss']
                                }


                   }

                        # components used in low voltage Si build
    # values_dict = {'TSM300NB06DCR': [60, 132, 30, 68], 'DMT6015LSS': [60, 487, 30, 251]}


    color = iter(cm.rainbow(np.linspace(0, 1, len(values_dict[grouping]))))
    slope_list = []
    for key in values_dict[grouping].keys():
        # file_name = 'csv_files/' + list(values_dict.keys())[0] + '_' + str(
        #     values_dict[list(values_dict.keys())[0]][0]) + 'V.csv'
        file_name = 'datasheet_graph_info/' + key + '_' + str(
            values_dict[grouping][key][0]) + 'V.csv'
        Coss_dataset = pd.read_csv(file_name)
        Coss_dataset.columns = ['Vdss', 'Coss']
        c=next(color)
        plt.plot(Coss_dataset['Vdss']/(0.1*values_dict[grouping][key][0]), Coss_dataset['Coss']/values_dict[grouping][key][1], c=c)

        p = np.polyfit(np.log10(Coss_dataset['Vdss']/(0.1*values_dict[grouping][key][0])), np.log10(Coss_dataset['Coss']/values_dict[grouping][key][1]), deg=1)
        Coss_dataset['Vdss'] = (0.1*values_dict[grouping][key][0]) / Coss_dataset['Vdss']
        Coss_dataset['Coss'] = Coss_dataset['Coss']/values_dict[grouping][key][1]
        # c = next(color)
        # plt.plot(Coss_dataset['Vdss'], Coss_dataset['Coss'], c=c)
        # p = np.polyfit(np.log10(Coss_dataset['Vdss']), np.log10(Coss_dataset['Coss']), deg=1)
        slope_list.append((values_dict[grouping][key][0], -p[0]))

    legend_list = []
    # for key in values_dict[grouping].keys():
    #     legend_list.append((values_dict[grouping][key][0], key))
    for key in values_dict[grouping].keys():
        legend_list.append(values_dict[grouping][key][0])
    # l = plt.legend(legend_list, prop={'size': 8}, title='$V_{dss}$ device rating [V]', title_fontsize='small')
    # plt.xlabel('$0.1V_{dss}/V_{ds}$')
    # plt.ylabel('$C_{oss}/C_{oss,0.1}$')


    # Plot the Coss vs. Vds normalized curves for all components in each grouping based on the e.g. 'highVolt_Si'
    # grouping, also determine the slope for each of those lines, and then plot the slopes of each of those lines
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(color='lightgrey', linewidth=1, alpha=0.4)

    plt.xlabel('$V_{ds}/(0.1V_{dss})$', fontsize=15)
    plt.ylabel('$C_{oss}/C_{oss,0.1}$', fontsize=15)
    l = plt.legend(legend_list, prop={'size': 8}, title='$V_{dss}$ device rating [V]', title_fontsize=12, ncol=2)

    plt.show()
    plt.close()
    plt.scatter(*zip(*slope_list))
    plt.xlabel('$V_{dss}$ [V]', fontsize=15)
    plt.ylabel('$\gamma$', fontsize=15)
    a, b = np.polyfit(*zip(*slope_list), deg=1)
    # add line of best fit to plot
    T = np.linspace(min([i[0] for i in legend_list]), max([i[0] for i in legend_list]), 1000)
    plt.plot(T, a * T + b)

    plt.show()

def loss_model(Pout, fsw, Vout, Vin, Vgate, QgQ1, CossQ1, RonQ1, QgQ2, QrrQ2, CossQ2, RonQ2, Rdc, L):
    # First calculate the new Iout
    Vsw = 1
    Vd = 0.5
    ripple = 0.2
    Iout = (Vin - Vsw - Vout)*(Vout + Vd) / (L * fsw * (Vin - Vsw + Vd) * ripple)
    Iout = 1

    dc = Vout/Vin
    Q1_loss = (np.sqrt(dc)*(Iout))**2 * RonQ1 + fsw*QgQ1*Vgate + 0.5*fsw*CossQ1*Vin**2
    Q2_loss = (np.sqrt(1-dc)*(Iout)**2 * RonQ2 + fsw*QgQ2*Vgate + 0.5*fsw*CossQ2*Vin**2
                 + fsw*QrrQ2*Vin)
    L1_loss = (Iout)**2 * Rdc
    loss = (1-((Q1_loss + Q2_loss + L1_loss)/Pout))*100
    return loss

def experimental_curves():

    # Read in the data from experimental results spreadsheet
    xlsx_file = 'csv_files/Efficiency_Si_Buck.xlsx'

    xls = pd.ExcelFile(xlsx_file)

    # Now you can list all sheets in the file

    # to read just one sheet to dataframe:
    si_200k = pd.read_excel(xlsx_file, sheet_name=xls.sheet_names[4])
    si_300k = pd.read_excel(xlsx_file, sheet_name=xls.sheet_names[3])
    si_400k = pd.read_excel(xlsx_file, sheet_name=xls.sheet_names[1])
    si_500k = pd.read_excel(xlsx_file, sheet_name=xls.sheet_names[0])


    # Now plot the loss-model equations

    datapoints_dfs = [si_200k, si_300k, si_400k, si_500k]
    fsw_list = [2 * 10 ** 5, 3 * 10 ** 5, 4 * 10 ** 5, 5 * 10 ** 5]
    color_list = ['cornflowerblue', 'green', 'red', 'purple']
    Vin = 48
    Vout = 12
    Vgate = 10
    QgQ1 = 18 * 10 ** -9
    CossQ1 = 71 * 10 ** -12
    RonQ1 = 25 * 10 ** -3
    QgQ2 = 8.9 * 10 ** -9
    QrrQ2 = 13.2 * 10 ** -9
    CossQ2 = 251 * 10 ** -12
    RonQ2 = 12.4 * 10 ** -3
    Rdc = 15.6 * 10 ** -3
    L = 10*10**-6
    i = 0
    for fsw in fsw_list:
        y_ = []
        T = np.linspace(si_500k['Pout'].min(), si_500k['Pout'].max(), 1000)
        # plot the datapoints
        plt.scatter(datapoints_dfs[i]['Pout'], datapoints_dfs[i]['Eff'], c=color_list[i], s=8)

        for Pout in T:
            y_.append(loss_model(Pout, fsw, Vout, Vin, Vgate, QgQ1, CossQ1, RonQ1, QgQ2, QrrQ2, CossQ2, RonQ2, Rdc, L))

        plt.plot(T, y_, c=color_list[i])
        i += 1
    plt.xlabel('Pout [W]')
    plt.ylabel('Efficiency [%]')
    _ = plt.legend(['200kHz', '300kHz', '400kHz', '500kHz'])
    plt.show()



    # Read in the data from experimental results spreadsheet
    xlsx_file = 'csv_files/Efficiency_GaN_Buck.xlsx'

    xls = pd.ExcelFile(xlsx_file)

    # Now you can list all sheets in the file

    # to read just one sheet to dataframe:
    gan_1M = pd.read_excel(xlsx_file, sheet_name=xls.sheet_names[3])
    gan_2M = pd.read_excel(xlsx_file, sheet_name=xls.sheet_names[2])
    gan_2p94M = pd.read_excel(xlsx_file, sheet_name=xls.sheet_names[1])



    # Now plot the loss-model equations


    datapoints_dfs = [gan_1M, gan_2M, gan_2p94M]
    fsw_list = [1*10**6, 2*10**6, 2.94*10**6]
    color_list = ['cornflowerblue', 'green', 'red']
    Vin = 48
    Vout = 12
    Vgate = 5
    QgQ1 = 700 * 10 ** -12
    CossQ1 = 50 * 10 ** -12
    RonQ1 = 73 * 10 ** -3
    QgQ2 = 880 * 10 ** -12
    QrrQ2 = 0 * 10 ** -9
    CossQ2 = 60 * 10 ** -12
    RonQ2 = 45 * 10 ** -3
    Rdc = 10.6 * 10 ** -3
    L = 1*10**-6
    i = 0
    for fsw in fsw_list:
        y_ = []
        T = np.linspace(gan_2M['Pout'].min(), gan_2M['Pout'].max(), 1000)
        # plot the datapoints
        plt.scatter(datapoints_dfs[i]['Pout'], datapoints_dfs[i]['Efficiency'], c=color_list[i])

        for Pout in T:
            y_.append(loss_model(Pout, fsw, Vout, Vin, Vgate, QgQ1, CossQ1, RonQ1, QgQ2, QrrQ2, CossQ2, RonQ2, Rdc, L))

        plt.plot(T, y_, c=color_list[i])
        i+=1
    plt.xlabel('Pout [W]')
    plt.ylabel('Efficiency [%]')
    _ = plt.legend(['1MHz', '2MHz', '2.94MHz'])
    plt.show()




if __name__ == '__main__':
    pd.set_option("display.max_rows", 100, "display.max_columns", 100)
    # ac_inductor_loss()
    Ron_Coss_MAE(category = 'highVolt_GaN')
    # ind_ac_training()
    Coss_Vds_training(output_param='C_ossp1')
    # Ron_plotting()
    Coss_plotting()
    experimental_curves()