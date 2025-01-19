'''
    This file contains the functions used for taking the predicted parameters after running the optimization, and
    looking through the database of components for a component with similar parameters.
'''

import fet_regression
import numpy as np
from csv_conversion import csv_to_df
import pandas as pd
from fet_area_filtering import area_filter
# from IPython.display import display
# from pandas.io.formats.style import Styler
from tabulate import tabulate

'''
    Using the optimized values x = [Rds1, Rds2, fsw], make predictions about the other quantities of interest
'''
def make_component_predictions(x, x_predicts_global, converter_df, fets_df, inds_df, constraints_df, fet_technology, degree):
    # we know we can use x_predicts, which is a dataframe that has the info we care about
    pd.set_option("display.max_rows", None, "display.max_columns", None)

    total_cost = x_predicts_global.loc[0, 'fet_Unit_price'] + x_predicts_global.loc[1, 'fet_Unit_price'] + x_predicts_global.loc[
                                                                                   0, 'ind_Unit_price']
    total_area = x_predicts_global.loc[0, 'fet_Area'] + x_predicts_global.loc[1, 'fet_Area'] + x_predicts_global.loc[0, 'ind_Area']
    dc = converter_df.loc[0, 'Vout'] / converter_df.loc[0, 'Vin']
    I_Q1 = np.sqrt(dc) * converter_df.loc[0, 'Iout']
    I_Q2 = np.sqrt(1 - dc) * converter_df.loc[0, 'Iout']

    total_power_loss = I_Q1 ** 2 * x[0] + x[2] * x_predicts_global.loc[0, 'fet_Qg'] * fets_df.loc[0, 'Vgate'] + \
    I_Q2 ** 2 * x[1] + x[2] * x_predicts_global.loc[1, 'fet_Qg'] * fets_df.loc[1, 'Vgate'] + x[2] * x_predicts_global.loc[
        1, 'fet_Qrr'] * \
    converter_df.loc[0, 'Vin'] + \
    converter_df.loc[0, 'Iout'] ** 2 * x_predicts_global.loc[0, 'ind_Rdc']

    print('Component parameters: \nQ1:\nVdss: %d, Rds: %f, Qg: %f, Cost: %f, Area: %f\n'
          'Q2:\nVdss: %d, Rds: %f, Qg: %f, Cost: %f, Area: %f\n'
          'L1:\nInductance: %f, Current_rating: %f, Rdc: %f, Cost: %f, Area: %f\n'
          'General:\nFrequency: %d, Total cost: %f, Total area: %f, Total power loss/Pout: %f\n' % (fets_df.loc[0]['Vdss'], x[0],
                x_predicts_global.loc[0, 'fet_Qg']*10e8, x_predicts_global.loc[0, 'fet_Unit_price'], x_predicts_global.loc[0, 'fet_Area'],
                                                                   fets_df.loc[1]['Vdss'], x[1],
                                                                   x_predicts_global.loc[1, 'fet_Qg'],
                                                                   x_predicts_global.loc[1, 'fet_Unit_price'],
                                                                   x_predicts_global.loc[1, 'fet_Area'],
                                                                     inds_df.loc[0,'Ind_fsw']/x[2],
                                                                     inds_df.loc[0,'Current_rating'],
                                                                     x_predicts_global.loc[0, 'ind_Rdc'],
                                                                     x_predicts_global.loc[
                                                                                   0, 'ind_Unit_price'],
                                                                     x_predicts_global.loc[
                                                                                   0, 'ind_Area'],
                                                                    x[2], total_cost, total_area, total_power_loss/(converter_df.loc[0, 'Vout'] * converter_df.loc[0, 'Iout'])
                                                                   ))
    #print('Recommended optimized parameters for the components:\n')
    Q1_rec_df = pd.DataFrame.from_dict(
        {'Vdss': [fets_df.loc[0]['Vdss']], 'Rds': [x[0]*10**3], 'Qg': [x_predicts_global.loc[0, 'fet_Qg']*10**9],
         'Cost': [x_predicts_global.loc[0, 'fet_Unit_price']],
         'Area': [x_predicts_global.loc[0, 'fet_Area']]})#, columns=pd.MultiIndex.from_product([['Q1']], names=['Component:']))

    #Q1_rec_df.style.set_caption("Q1 Recommended Parameters")
    Q2_rec_df = pd.DataFrame({'Vdss': [fets_df.loc[1]['Vdss']], 'Rds': [x[1]*10**3] , 'Qg': [x_predicts_global.loc[1, 'fet_Qg']*10**9], 'Cost': [x_predicts_global.loc[1, 'fet_Unit_price']], 'Area': [x_predicts_global.loc[1, 'fet_Area']]})
    Q2_rec_df.style.set_caption("Q2 Recommended Parameters")
    L1_rec_df = pd.DataFrame(
        {'Inductance': [10**6*inds_df.loc[0,'Ind_fsw']/x[2]], 'Current_rating': [inds_df.loc[0,'Current_rating']], 'Rdc': [x_predicts_global.loc[0, 'ind_Rdc']*10**3],
         'Cost':  [x_predicts_global.loc[0, 'ind_Unit_price']], 'Area': [x_predicts_global.loc[0, 'ind_Area']]})
    L1_rec_df.style.set_caption("Q2 Recommended Parameters")
    converter_rec_df = pd.DataFrame.from_dict({'frequency': [x[2]], 'Total cost': [total_cost], 'Total area': [total_area], 'Total Ploss/Pout': [total_power_loss/(converter_df.loc[0, 'Vout'] * converter_df.loc[0, 'Iout'])]})

    # set the variables so they are in the units we want
    Q1_rec_df['Rds'] = Q1_rec_df['Rds']
    Q2_rec_df['Rds'] = Q2_rec_df['Rds']
    Q1_rec_df['Qg'] = Q1_rec_df['Qg']
    Q2_rec_df['Qg'] = Q2_rec_df['Qg']
    L1_rec_df['Rdc'] = L1_rec_df['Rdc']
    L1_rec_df['Inductance'] = L1_rec_df['Inductance']

    fet_column_list = ['V_dss [V]',  'R_ds [mΩ]', 'Q_g [nC]', 'Unit_price [$]',
                       'Area [mm^2]']
    ind_column_list = ['Inductance [µH]', 'Current_rating [A]', 'R_dc [mΩ]', 'Unit_price [$]','Area [mm^2]'
                       ]
    performance_column_list = ['frequency [Hz]', 'Total cost [$]', 'Total area [mm^2]', 'Total Ploss/Pout']
    print('Q1 recommended parameters:')
    print(tabulate(Q1_rec_df.drop_duplicates(inplace=False), headers=fet_column_list, showindex=False, tablefmt='fancy_grid',
                   floatfmt=".3f"))
    print('Q2 recommended parameters:')
    print(tabulate(Q2_rec_df.drop_duplicates(inplace=False), headers=fet_column_list, showindex=False,
                   tablefmt='fancy_grid',
                   floatfmt=".3f"))
    print('L1 recommended parameters:')
    print(tabulate(L1_rec_df.drop_duplicates(inplace=False), headers=ind_column_list, showindex=False,
                   tablefmt='fancy_grid',
                   floatfmt=".3f"))
    print('Converter expected performance using recommended parameters:')
    print(tabulate(converter_rec_df, headers=performance_column_list, showindex=False,
                   tablefmt='fancy_grid',
                   floatfmt=".3f"))


    # now search the database for parts with similar parameters
    if fet_technology == 'GaNFET':
        Vdss_tol_percent = 3
        Rds_tol_percent = 3
        Qg_tol_percent = 3
        Cost_tol_percent = 4
        Cost_ind_tol_percent = 0.2
        Current_tol_percent = 1.5
        Rdc_tol_percent = 0.3
        Dimension_tol_percent = 0.3
        Inductance_tol_percent = 0.5

    if fet_technology == 'MOSFET':
        Vdss_tol_percent = 5
        Rds_tol_percent = 1.5
        Qg_tol_percent = 1.5
        Cost_tol_percent = 1.1
        Cost_ind_tol_percent = 0.9
        Current_tol_percent = 0.9
        Rdc_tol_percent = 0.5
        Dimension_tol_percent = 0.9
        Inductance_tol_percent = 0.6

    csv_file = '../mosfet_data/csv_files/' + fet_technology + '_data_noPDF.csv'
    fets_database_df = csv_to_df(csv_file)
    fets_database_df = fets_database_df.reset_index()
    data_dims = ['V_dss', 'Unit_price', 'R_ds', 'Q_g', 'FET_type', 'Technology', 'Pack_case']
    fets_database_df = fets_database_df.iloc[:, 2:]
    fets_database_df.columns = data_dims

    csv_file = '../mosfet_data/csv_files/inductor_data_noPDF.csv'
    inds_database_df = csv_to_df(csv_file)
    inds_database_df = inds_database_df.reset_index()
    inds_database_df = inds_database_df.iloc[:, 2:]
    attr_list = ['Unit_price', 'Current_rating', 'R_dc', 'Dimension', 'Height', 'Inductance','Mfr_part_no']
    inds_database_df.columns = attr_list


    # filter the fet df to find a component with the following parameters we care about (within an x% range):
    #   Vdss, cost, Rds, Qg
    # for Q1:
    Q1_df = fets_database_df[fets_database_df['V_dss'].between(fets_df.loc[0]['Vdss'], (1+Vdss_tol_percent)*fets_df.loc[0]['Vdss'])]
    Q1_df = Q1_df[Q1_df['R_ds'].between(0.001*x[0], (1+Rds_tol_percent)*x[0])]
    Q1_df = Q1_df[Q1_df['Q_g'].between(0.001*x_predicts_global.loc[0, 'fet_Qg'], (1+Qg_tol_percent)*x_predicts_global.loc[0, 'fet_Qg'])]
    Q1_df = Q1_df[Q1_df['Unit_price'].between(0.001*x_predicts_global.loc[0, 'fet_Unit_price'], (1+Cost_tol_percent)*x_predicts_global.loc[0, 'fet_Unit_price'])]

    Q2_df = fets_database_df[fets_database_df['V_dss'].between(fets_df.loc[1]['Vdss'],
                                                               (1 + Vdss_tol_percent) * fets_df.loc[1]['Vdss'])]
    Q2_df = Q2_df[Q2_df['R_ds'].between(0.001 * x[1], (1+Rds_tol_percent) * x[1])]
    Q2_df = Q2_df[Q2_df['Q_g'].between(0.001 * x_predicts_global.loc[1, 'fet_Qg'],
                                       (1+Qg_tol_percent) * x_predicts_global.loc[1, 'fet_Qg'])]
    Q2_df = Q2_df[Q2_df['Unit_price'].between(0.001 * x_predicts_global.loc[1, 'fet_Unit_price'],
                                              (1+Cost_tol_percent) * x_predicts_global.loc[1, 'fet_Unit_price'])]
    L1_df = inds_database_df[inds_database_df['Current_rating'].between(inds_df.loc[0]['Current_rating'],
                                                               (1 + Current_tol_percent) * inds_df.loc[0]['Current_rating'])]
    L1_df = L1_df[L1_df['Inductance'].between((1-Inductance_tol_percent) * (inds_df.loc[0, 'Ind_fsw']/x[2]),
                                        (1 + Inductance_tol_percent) * (inds_df.loc[0, 'Ind_fsw']/x[2]))]
    L1_df = L1_df[L1_df['R_dc'].between(0.001 * x_predicts_global.loc[0, 'ind_Rdc'], (1 + Rdc_tol_percent) * x_predicts_global.loc[0, 'ind_Rdc'])]
    L1_df = L1_df[L1_df['Dimension'].between(0.001 * x_predicts_global.loc[0, 'ind_Area'],
                                             (1 + Dimension_tol_percent) * x_predicts_global.loc[0, 'ind_Area'])]
    L1_df = L1_df[L1_df['Unit_price'].between(0.001 * x_predicts_global.loc[0, 'ind_Unit_price'],
                                              (1 + Cost_ind_tol_percent) * x_predicts_global.loc[0, 'ind_Unit_price'])]

    # reset the units for better display purposes
    Q1_dup = Q1_df.copy(deep=True)
    Q2_dup = Q2_df.copy(deep=True)
    L1_dup = L1_df.copy(deep=True)

    Q1_dup['R_ds'] = Q1_dup['R_ds']*10**3
    Q2_dup['R_ds'] = Q2_dup['R_ds']*10**3
    Q1_dup['Q_g'] = Q1_dup['Q_g']*10**9
    Q2_dup['Q_g'] = Q2_dup['Q_g']*10**9
    L1_dup['R_dc'] = L1_dup['R_dc']*10**3
    L1_dup['Inductance'] = L1_dup['Inductance']*10**6

    if fet_technology == 'GaNFET':
        Q1_dup = area_filter(Q1_dup)
        Q2_dup = area_filter(Q2_dup)
    # then the user chooses which components they want to go with, and this tells them what the expected loss is given
    # these choices. For Qrr, if not present, give the expected Qrr in the sum term.
    # also, find some way to put all this information onto a nice info-graphic from within the program

    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    pd.set_option('display.colheader_justify', 'center')
    pd.set_option('display.precision', 3)
    fet_column_list = ['V_dss [V]', 'Unit_price [$]', 'R_ds [mΩ]', 'Q_g [nC]', 'FET_type', 'Technology', 'Pack_case [mm^2]']
    ind_column_list = ['Unit_price [$]', 'Current_rating [A]', 'R_dc [mΩ]', 'Dimension [mm^2]', 'Height [mm]', 'Inductance [µH]', 'Mfr_part_no']
    print('Available Q1 components:')
    print(tabulate(Q1_dup.drop_duplicates(inplace=False), headers=fet_column_list, showindex=False, tablefmt='fancy_grid',floatfmt=".3f"))
    print('Available Q2 components:')
    print(tabulate(Q2_dup.drop_duplicates(inplace=False), headers=fet_column_list, showindex=False, tablefmt='fancy_grid',floatfmt=".3f"))
    print('Available L1 components:')
    print(tabulate(L1_dup.drop_duplicates(inplace=False), headers=ind_column_list, showindex=False, tablefmt='fancy_grid',floatfmt=".3f"))

    # for MOSFET do 4,5,1
    [user_Q1_index, user_Q2_index, user_L1_index] = user_COTS_choice(Q1_df, Q2_df, L1_df)
    user_prefs = {'Q1': int(user_Q1_index), 'Q2': int(user_Q2_index), 'L1': int(user_L1_index)}
    COTS_total_cost = Q1_df.iloc[user_prefs['Q1']]['Unit_price'] + Q2_df.iloc[user_prefs['Q2']]['Unit_price'] + L1_df.iloc[user_prefs['L1']]['Unit_price']
    # to get area of FETs need to convert the package sizes to actual areas using the dictionary
    if fet_technology == 'GaNFET':
        Q1_df = area_filter(Q1_df)
        Q2_df = area_filter(Q2_df)
    COTS_total_area = Q1_df.iloc[user_prefs['Q1']]['Pack_case'] + Q2_df.iloc[user_prefs['Q2']]['Pack_case'] + L1_df.iloc[user_prefs['L1']]['Dimension']
    COTS_total_powerfrac = (I_Q1 ** 2 * Q1_df.iloc[user_prefs['Q1']]['R_ds'] + x[2] * Q1_df.iloc[user_prefs['Q1']]['Q_g'] * fets_df.loc[0, 'Vgate'] + \
                I_Q2 ** 2 * Q2_df.iloc[user_prefs['Q2']]['R_ds'] + x[2] * Q2_df.iloc[user_prefs['Q2']]['Q_g'] * fets_df.loc[1, 'Vgate'] + x[2] * x_predicts_global.loc[
                1, 'fet_Qrr'] * converter_df.loc[0, 'Vin'] + \
                converter_df.loc[0, 'Iout'] ** 2 * L1_df.iloc[user_prefs['L1']]['R_dc'])/(converter_df.loc[0, 'Vout'] * converter_df.loc[0, 'Iout'])
    # print('Using the parameters of the chosen COTS components:\nTotal cost = %f, Total area = %f, Total power loss fraction = %f\n' % (COTS_total_cost, COTS_total_area, COTS_total_powerfrac))
    COTS_performance_df = pd.DataFrame.from_dict({'Total cost': [COTS_total_cost], 'Total area': [COTS_total_area], 'Total Ploss/Pout': [COTS_total_powerfrac]})
    performance_column_list = ['Total cost [$]', 'Total area [mm^2]', 'Total Ploss/Pout']
    print('Expected performance given these component choices:')
    print(
        tabulate(COTS_performance_df.drop_duplicates(inplace=False), headers=performance_column_list, showindex=False, tablefmt='fancy_grid',
                 floatfmt=".3f"))
    print('searching in parts database')

'''
    This function asks the user for input values corresponding to the number of desired parameters needed to 
    characterize the component, e.g. the mosfet loss equation. Returns a list of these user-input parameters.
'''
def user_COTS_choice(Q1_df, Q2_df, L1_df):
    Q1_index, Q2_index, L1_index = input("\nEnter choices for components as Index of Q1 [0-indexed], Index of Q2, "
                         "Index of L1  --use commas not brackets").split(",",3)

    return [Q1_index, Q2_index, L1_index]

'''
    model = 'linear'
    file_name = '../mosfet_data/joblib_files/' + fet_technology + '_models_noPDF'
    fet_models_nopdf_list = ['Pack_case']
    fet_reg_models_nopdf_df = fet_regression.load_models(fet_models_nopdf_list, file_name, param='area')
    fet_reg_models_nopdf_df = fet_reg_models_nopdf_df.set_index('output_param')

    if fet_technology == 'MOSFET':
        file_name = '../mosfet_data/joblib_files/' + fet_technology + '_models_wPDF'
        fet_models_wpdf_list = ['RdsCost_product', 'RdsQg_product', 'RdsQrr_product']
        fet_reg_models_wpdf_df = fet_regression.load_models(fet_models_wpdf_list, file_name)
        fet_reg_models_wpdf_df = fet_reg_models_wpdf_df.set_index('output_param')
    else:
        file_name = '../mosfet_data/joblib_files/' + fet_technology + '_models_wPDF'
        fet_models_wpdf_list = ['RdsCost_product', 'RdsQg_product']
        fet_reg_models_wpdf_df = fet_regression.load_models(fet_models_wpdf_list, file_name)
        fet_reg_models_wpdf_df = fet_reg_models_wpdf_df.set_index('output_param')

    file_name = '../mosfet_data/joblib_files/' + 'inductor_models_noPDF'
    ind_models_nopdf_list = ['DCR_Inductance_product', 'DCR_Cost_product', 'Dimension', 'Energy']
    ind_reg_models_nopdf_df = fet_regression.load_models(ind_models_nopdf_list, file_name)
    ind_reg_models_nopdf_df = ind_reg_models_nopdf_df.set_index('output_param')

    # Make a prediction on cost, area, and efficiency to compare with the constraints
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    costs = {}
    c_pred = 0
    # print(x)
    # RdsCost_df = fet_reg_models_wpdf_df if fet_technology == 'MOSFET' else RdsCost_df = fet_reg_models_nopdf_df
    for fet_index in fets_df.index:
        single_cost_pred_fet += (10 ** (fet_reg_models_wpdf_df.loc['RdsCost_product', 'linear'].predict(
            fet_regression.preproc(
                np.array([np.log10(fets_df.loc[fet_index]['Vdss']), np.log10(x[fet_index])]).reshape(1, -1),
                degree))[0])) / x[fet_index]
        c_pred += single_cost_pred_fet
        costs[('Rds', 0)] = single_cost_pred_fet

    for ind_index in inds_df.index:
        ind1 = inds_df.loc[ind_index]['Ind_fsw'] / x[ind_index + offset]
        single_Rdc_pred = 10 ** (ind_reg_models_nopdf_df.loc['DCR_Inductance_product', model].predict(
            fet_regression.preproc(
                np.array([np.log10(ind1), np.log10(1 / inds_df.loc[ind_index]['Current_rating'])]).reshape(1, -1),
                degree))[0]) / ind1

        # we have inductance -- use this to predict DCR*inductance to then use DCR to get cost
        c_pred += (10 ** (ind_reg_models_nopdf_df.loc['DCR_Cost_product', 'linear'].predict(
            fet_regression.preproc(np.array([np.log10(ind1), np.log10(single_Rdc_pred)]).reshape(1, -1), degree))[
            0]) / single_Rdc_pred)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    areas = []
    area_pred = 0
    for fet_index in fets_df.index:  # inputs for area are ['V_dss', 'R_ds', 'Unit_price', 'Q_g']
        # make individual prediction for the cost and Qg given the other parameters
        # for cost:
        single_cost_pred_fet = (10 ** (fet_reg_models_wpdf_df.loc['RdsCost_product', 'linear'].predict(
            fet_regression.preproc(
                np.array([np.log10(fets_df.loc[fet_index]['Vdss']), np.log10(x[fet_index])]).reshape(1, -1),
                degree))[0])) / x[fet_index]

        # for Qg:
        single_Qg_pred = (10 ** fet_reg_models_wpdf_df.loc['RdsQg_product', model].predict(
            fet_regression.preproc(
                np.array([np.log10(fets_df.loc[fet_index]['Vdss']), np.log10(x[fet_index]),
                          np.log10(single_cost_pred_fet)]).reshape(1, -1),
                degree))[0]) / x[fet_index]

        area_pred += ((fet_reg_models_nopdf_df.loc['Pack_case', 'knn'].predict(
            fet_regression.preproc(np.array(
                [np.log10(fets_df.loc[fet_index]['Vdss']), np.log10(x[fet_index]), np.log10(single_c_pred),
                 np.log10(single_Qg_pred)]).reshape(1, -1), degree))[0]))



    for ind_index in range(len(inds_df.index)):  # inputs for area are ['Energy', 'Unit_price']
        ind1 = inds_df.loc[ind_index]['Ind_fsw'] / x[ind_index + offset]
        # Energy prediction
        single_Energy_pred = ind1 * (inds_df.loc[ind_index]['Current_rating'] ** 2)
        # Rdc prediction
        single_Rdc_pred = 10 ** (ind_reg_models_nopdf_df.loc['DCR_Inductance_product', model].predict(
            fet_regression.preproc(
                np.array([np.log10(ind1), np.log10(1 / inds_df.loc[ind_index]['Current_rating'])]).reshape(1, -1),
                degree))[0]) / ind1
        # Cost prediction
        single_Cost_pred_ind = (10 ** (ind_reg_models_nopdf_df.loc['DCR_Cost_product', model].predict(
            fet_regression.preproc(np.array([np.log10(ind1), np.log10(single_Rdc_pred)]).reshape(1, -1),
                                   degree))[0])) / single_Rdc_pred

        area_pred += (10 ** (ind_reg_models_nopdf_df.loc['Dimension', 'linear'].predict(
            fet_regression.preproc(
                np.array([np.log10(single_Energy_pred), np.log10(single_Cost_pred_ind)]).reshape(1, -1), degree))[0]))

    print('Component parameters: \nQ1:\nVdss: %d, Rds: %f, Qg: %f, Cost: %f, Area: %f', (fets_df.loc[fet_index]['Vdss'], x[0], single_Qg_pred, ))
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
'''
'''


x_predicts.loc[1, 'fet_Qg']
x_predicts.loc[0, 'ind_Rdc']
Si_df.columns
Si_df.loc[:, 'Q_g']
x_predicts['fet_Unit_price']
x_predicts['fet_Qg']
x_predicts.loc[len(x_predicts['fet_Unit_price'].index), 'fet_Unit_price'] = (10 ** fet_reg_models_wpdf_df.loc[
    'RdsCost_product', 'linear'].predict(
    fet_regression.preproc(
        np.array([np.log10(fets_df.loc[fet_index]['Vdss']), np.log10(x[fet_index])]).reshape(1, -1),
        degree))[0]) / x[fet_index]
x_predicts['fet_Qg'].loc[len(x_predicts['fet_Qg'].index), 'fet_Qg'] = \
    (10 ** fet_reg_models_wpdf_df.loc['RdsQg_product', model].predict(
        fet_regression.preproc(
            np.array([np.log10(fets_df.loc[fet_index]['Vdss']), np.log10(x[fet_index]),
                      np.log10(x_predicts.loc[fet_index, 'fet_Unit_price'])]).reshape(1, -1),
            degree))[0]) / x[fet_index]
x_predicts.loc[len(x_predicts['fet_Qg'].index), 'fet_Qg'] = \
    (10 ** fet_reg_models_wpdf_df.loc['RdsQg_product', model].predict(
        fet_regression.preproc(
            np.array([np.log10(fets_df.loc[fet_index]['Vdss']), np.log10(x[fet_index]),
                      np.log10(x_predicts.loc[fet_index, 'fet_Unit_price'])]).reshape(1, -1),
            degree))[0]) / x[fet_index]
x_predicts.loc[i, 'fet_Qg'] = \
    ((10 ** fet_reg_models_wpdf_df.loc['RdsQg_product', model].predict(
        fet_regression.preproc(
            np.array([np.log10(fets_df.loc[fet_index]['Vdss']), np.log10(x[fet_index]),
                      np.log10(x_predicts.loc[fet_index, 'fet_Unit_price'])]).reshape(1, -1),
            degree))[0]) / x[fet_index]).astype(float)
fet_reg_models_wpdf_df.loc['RdsQg_product', model].predict(
    fet_regression.preproc(
        np.array([np.log10(fets_df.loc[fet_index]['Vdss']), np.log10(x[fet_index]),
                  np.log10(x_predicts.loc[fet_index, 'fet_Unit_price'])]).reshape(1, -1),
        degree))[0]
print(fet_reg_models_wpdf_df.loc['RdsQg_product', model].predict(
    fet_regression.preproc(
        np.array([np.log10(fets_df.loc[fet_index]['Vdss']), np.log10(x[fet_index]),
                  np.log10(x_predicts.loc[fet_index, 'fet_Unit_price'])]).reshape(1, -1),
        degree))[0])
10 ** -7
(10 ** fet_reg_models_wpdf_df.loc['RdsQg_product', model].predict(
    fet_regression.preproc(
        np.array([np.log10(fets_df.loc[fet_index]['Vdss']), np.log10(x[fet_index]),
                  np.log10(x_predicts.loc[fet_index, 'fet_Unit_price'])]).reshape(1, -1),
        degree))[0])
print(10 ** fet_reg_models_wpdf_df.loc['RdsQg_product', model].predict(
    fet_regression.preproc(
        np.array([np.log10(fets_df.loc[fet_index]['Vdss']), np.log10(x[fet_index]),
                  np.log10(x_predicts.loc[fet_index, 'fet_Unit_price'])]).reshape(1, -1),
        degree))[0])
(10 ** fet_reg_models_wpdf_df.loc['RdsQg_product', model].predict(
    fet_regression.preproc(
        np.array([np.log10(fets_df.loc[fet_index]['Vdss']), np.log10(x[fet_index]),
                  np.log10(x_predicts.loc[fet_index, 'fet_Unit_price'])]).reshape(1, -1),
        degree))[0]) / x[fet_index]
print((10 ** fet_reg_models_wpdf_df.loc['RdsQg_product', model].predict(
    fet_regression.preproc(
        np.array([np.log10(fets_df.loc[fet_index]['Vdss']), np.log10(x[fet_index]),
                  np.log10(x_predicts.loc[fet_index, 'fet_Unit_price'])]).reshape(1, -1),
        degree))[0]) / x[fet_index])
x_predicts.loc[i, 'fet_Qg'] = \
    (10 ** fet_reg_models_wpdf_df.loc['RdsQg_product', model].predict(
        fet_regression.preproc(
            np.array([np.log10(fets_df.loc[fet_index]['Vdss']), np.log10(x[fet_index]),
                      np.log10(x_predicts.loc[fet_index, 'fet_Unit_price'])]).reshape(1, -1),
            degree))[0]) / x[fet_index]
pd.options.display.float_format = "{:,.9f}".format
x[ind_index + offset]
inds_df.loc[ind_index]['Ind_fsw']
x_predicts.loc[i, 'ind_Rdc'] = 10 ** (ind_reg_models_nopdf_df.loc['DCR_Inductance_product', model].predict(
    fet_regression.preproc(
        np.array([np.log10(ind1), np.log10(1 / inds_df.loc[ind_index]['Current_rating'])]).reshape(1, -1),
        degree))[0]) / ind1
x_predicts.loc[i, 'ind_Rdc']
x_predicts.loc[i, 'ind_Unit_price'] = 10 ** (ind_reg_models_nopdf_df.loc['DCR_Cost_product', model].predict(
    fet_regression.preproc(np.array([np.log10(ind1), np.log10(x_predicts.loc[ind_index, 'ind_Rdc'])]).reshape(1, -1),
                           degree))[0]) / x_predicts.loc[ind_index, 'ind_Rdc']

np.log10(ind1)
np.log10(x_predicts.loc[ind_index, 'ind_Rdc'])
print(I_Q1 ** 2 * x[0] + x[2] * x_predicts.loc[0, 'fet_Qg'] * fets_df.loc[0, 'Vgate'] + \
      I_Q2 ** 2 * x[1] + x[2] * x_predicts.loc[1, 'fet_Qg'] * fets_df.loc[1, 'Vgate'] + x[2] * x_predicts.loc[
          1, 'fet_Qrr'] * fets_df.loc[0, 'Vin'] + \
      inds_df.loc[0, 'Current_rating'] ** 2 * x_predicts.loc[0, 'ind_Rdc'])
fets_df.loc[0, 'Vout'] * fets_df.loc[0, 'Iout']
x_predicts.loc[0, 'fet_Qg']
x_predicts.loc[1, 'fet_Qrr']
c1
total_components
c1 = 5
T1 = fet_regression.preproc(np.array([np.log10(voltage), np.log10(c1)]).reshape(1, -1), degree)
Rdsx = (10 ** fet_reg_RdsCost_model.loc[0, model].predict(np.array(T1))) / c1
c1 = .5
T1 = fet_regression.preproc(np.array([np.log10(voltage), np.log10(c1)]).reshape(1, -1), degree)
Rdsx = (10 ** fet_reg_RdsCost_model.loc[0, model].predict(np.array(T1))) / c1
c1 = .5
T1 = fet_regression.preproc(np.array([np.log10(voltage), np.log10(c1)]).reshape(1, -1), degree)
Rdsx = 10 ** (fet_reg_RdsCost_model.loc[0, model].predict(np.array(T1)) / c1)
c1 = 5
T1 = fet_regression.preproc(np.array([np.log10(voltage), np.log10(c1)]).reshape(1, -1), degree)
Rdsx = 10 ** (fet_reg_RdsCost_model.loc[0, model].predict(np.array(T1)) / c1)
c1 = 5
T1 = fet_regression.preproc(np.array([np.log10(400), np.log10(c1)]).reshape(1, -1), degree)
Rdsx = (10 ** fet_reg_RdsCost_model.loc[0, model].predict(np.array(T1))) / c1
Rdsx
Rds
dc = fets_df.loc[0, 'Vout'] / fets_df.loc[0, 'Vin']
I_Q1 = np.sqrt(dc) * fets_df.loc[0, 'Iout']
I_Q2 = np.sqrt(1 - dc) * fets_df.loc[0, 'Iout']
I_Q1 ** 2 * x[0] + x[2] * x_predicts.loc[0, 'fet_Qg'] * fets_df.loc[0, 'Vgate'] + \
I_Q2 ** 2 * x[1] + x[2] * x_predicts.loc[1, 'fet_Qg'] * fets_df.loc[1, 'Vgate'] + x[2] * x_predicts.loc[1, 'fet_Qrr'] * \
fets_df.loc[0, 'Vin'] + \
inds_df.loc[0, 'Current_rating'] ** 2 * x_predicts.loc[0, 'ind_Rdc']

I_Q1 ** 2 * x[0] + x[2] * x_predicts.loc[0, 'fet_Qg'] * fets_df.loc[0, 'Vgate'] + \
I_Q2 ** 2 * x[1] + x[2] * x_predicts.loc[1, 'fet_Qg'] * fets_df.loc[1, 'Vgate'] + x[2] * x_predicts.loc[1, 'fet_Qrr'] * \
fets_df.loc[0, 'Vin'] + \
inds_df.loc[0, 'Current_rating'] ** 2 * x_predicts.loc[0, 'ind_Rdc']
print(I_Q1 ** 2 * x[0] + x[2] * x_predicts.loc[0, 'fet_Qg'] * fets_df.loc[0, 'Vgate'] + \
      I_Q2 ** 2 * x[1] + x[2] * x_predicts.loc[1, 'fet_Qg'] * fets_df.loc[1, 'Vgate'] + x[2] * x_predicts.loc[
          1, 'fet_Qrr'] * fets_df.loc[0, 'Vin'] + \
      inds_df.loc[0, 'Current_rating'] ** 2 * x_predicts.loc[0, 'ind_Rdc']
      )
fets_df.loc[0, 'Vout'] * fets_df.loc[0, 'Iout']
cost_tot
fet_reg_models_nopdf_df.loc['Pack_case', 'knn']
fet_reg_models_nopdf_df
np.log10(x[fet_index])
x_predicts.loc[fet_index, 'fet_Unit_price']
x_predicts.loc[fet_index, 'fet_Qg']
((ind_reg_models_nopdf_df.loc['Dimension', 'linear'].predict(
    fet_regression.preproc(np.array([np.log10(x_predicts.loc[ind_index, 'ind_Energy']),
                                     np.log10(x_predicts.loc[ind_index, 'ind_Unit_price'])]).reshape(1, -1), degree))[
    0]) / x_predicts.loc[ind_index, 'ind_Rdc'])

print((
        (ind_reg_models_nopdf_df.loc['Dimension', 'linear'].predict(
            fet_regression.preproc(np.array([np.log10(x_predicts.loc[ind_index, 'ind_Energy']),
                                             np.log10(x_predicts.loc[ind_index, 'ind_Unit_price'])]).reshape(1, -1),
                                   degree))[0]) / x_predicts.loc[ind_index, 'ind_Rdc'])
)
ind_reg_models_nopdf_df
output_param
np.log10(x_predicts.loc[ind_index, 'ind_Energy'])
np.log10(x_predicts.loc[ind_index, 'ind_Unit_price'])
(reg_lin.predict(
    preproc(np.array([-3.37, 0.11]).reshape(1, -1), degree))[0])

print((reg_lin.predict(
    preproc(np.array([-3.37, 0.11]).reshape(1, -1), degree))[0]))
10 ** 1.97
for fet_index in fets_df.index:
    area_tot += (10 ** (fet_reg_models_nopdf_df.loc['Pack_case', 'knn'].predict(
        fet_regression.preproc(np.array(
            [np.log10(fets_df.loc[fet_index, 'Vdss']), np.log10(x[fet_index]),
             np.log10(x_predicts.loc[fet_index, 'fet_Unit_price']),
             np.log10(x_predicts.loc[fet_index, 'fet_Qg'])]).reshape(1, -1), degree))[0]))

fets_df
for fet_index in fets_df.index:
    area_tot += (10 ** (fet_reg_models_nopdf_df.loc['Pack_case', 'knn'].predict(
        fet_regression.preproc(np.array(
            [np.log10(fets_df.loc[fet_index, 'Vdss']), np.log10(x[fet_index]),
             np.log10(x_predicts.loc[fet_index, 'fet_Unit_price']),
             np.log10(x_predicts.loc[fet_index, 'fet_Qg'])]).reshape(1, -1), degree))[0]))
    print(area_tot)

fet_reg_models_nopdf_df.loc['Pack_case', 'knn'].predict(
    fet_regression.preproc(np.array(
        [np.log10(fets_df.loc[fet_index, 'Vdss']), np.log10(x[fet_index]),
         np.log10(x_predicts.loc[fet_index, 'fet_Unit_price']),
         np.log10(x_predicts.loc[fet_index, 'fet_Qg'])]).reshape(1, -1), degree))[0])
fet_reg_models_nopdf_df.loc['Pack_case', 'knn'].predict(
fet_regression.preproc(np.array(
    [np.log10(fets_df.loc[fet_index, 'Vdss']), np.log10(x[fet_index]),
     np.log10(x_predicts.loc[fet_index, 'fet_Unit_price']),
     np.log10(x_predicts.loc[fet_index, 'fet_Qg'])]).reshape(1, -1), degree))[0]
print(fet_reg_models_nopdf_df.loc['Pack_case', 'knn'].predict(
    fet_regression.preproc(np.array(
        [np.log10(fets_df.loc[fet_index, 'Vdss']), np.log10(x[fet_index]),
         np.log10(x_predicts.loc[fet_index, 'fet_Unit_price']),
         np.log10(x_predicts.loc[fet_index, 'fet_Qg'])]).reshape(1, -1), degree))[0])
fets_df.loc[fet_index, 'Vdss']
x_predicts.loc[fet_index, 'fet_Unit_price']
x_predicts.loc[fet_index, 'fet_Qg']
reg_knn.loc['Pack_case', 'knn'].predict(
preproc(np.array(
    [np.log10(60), np.log10(0.007), np.log10(0.96),
     np.log10(3.7 * 10 ** -8)]).reshape(1, -1), degree))[0]
reg_knn.predict(
preproc(np.array(
    [np.log10(60), np.log10(0.007), np.log10(0.96),
     np.log10(3.7 * 10 ** -8)]).reshape(1, -1), degree))[0]
reg_knn = model.fit(X, y)
print(reg_knn.predict(
    preproc(np.array(
        [np.log10(60), np.log10(0.007), np.log10(0.96),
         np.log10(3.7 * 10 ** -8)]).reshape(1, -1), degree))[0])
X_over
preproc(np.array(
    [np.log10(60), np.log10(0.007), np.log10(0.96),
     np.log10(3.7 * 10 ** -8)]).reshape(1, -1), degree)
print(preproc(np.array(
    [np.log10(60), np.log10(0.007), np.log10(0.96),
     np.log10(3.7 * 10 ** -8)]).reshape(1, -1), degree))
y_over
area_tot
area_tot = 0
# get the fet areas
for fet_index in fets_df.index:
    area_tot += ((fet_reg_models_nopdf_df.loc['Pack_case', 'knn'].predict(
        fet_regression.preproc(np.array(
            [np.log10(fets_df.loc[fet_index, 'Vdss']), np.log10(x[fet_index]),
             np.log10(x_predicts.loc[fet_index, 'fet_Unit_price']),
             np.log10(x_predicts.loc[fet_index, 'fet_Qg'])]).reshape(1, -1), degree))[0]))
print(area_tot)

# get the inductor areas
for ind_index in inds_df.index:  # inputs for area are ['Energy', 'Unit_price']
    ind1 = inds_df.loc[ind_index, 'Ind_fsw'] / x[ind_index + offset]
# exec("Energy{i} = ind1 * inds_obj_list[i]['Current_rating']")
area_tot += 10 ** ((ind_reg_models_nopdf_df.loc['Dimension', 'linear'].predict(
    fet_regression.preproc(np.array([np.log10(x_predicts.loc[ind_index, 'ind_Energy']),
                                     np.log10(x_predicts.loc[ind_index, 'ind_Unit_price'])]).reshape(1, -1), degree))[
    0]))
print(area_tot)

power_lossx_predicts.loc[1,'fet_Qg']
x_predicts.loc[0,'ind_Rdc']
Si_df.columns
Si_df.loc[:,'Q_g']
x_predicts['fet_Unit_price']
x_predicts['fet_Qg']
x_predicts.loc[len(x_predicts['fet_Unit_price'].index),'fet_Unit_price'] = (10 ** fet_reg_models_wpdf_df.loc['RdsCost_product', 'linear'].predict(
            fet_regression.preproc(
                np.array([np.log10(fets_df.loc[fet_index]['Vdss']), np.log10(x[fet_index])]).reshape(1, -1),
                degree))[0]) / x[fet_index]
x_predicts['fet_Qg'].loc[len(x_predicts['fet_Qg'].index),'fet_Qg'] = \
        (10 ** fet_reg_models_wpdf_df.loc['RdsQg_product', model].predict(
            fet_regression.preproc(
                np.array([np.log10(fets_df.loc[fet_index]['Vdss']), np.log10(x[fet_index]),
                          np.log10(x_predicts.loc[fet_index,'fet_Unit_price'])]).reshape(1, -1),
                degree))[0]) / x[fet_index]
x_predicts.loc[len(x_predicts['fet_Qg'].index),'fet_Qg'] = \
        (10 ** fet_reg_models_wpdf_df.loc['RdsQg_product', model].predict(
            fet_regression.preproc(
                np.array([np.log10(fets_df.loc[fet_index]['Vdss']), np.log10(x[fet_index]),
                          np.log10(x_predicts.loc[fet_index,'fet_Unit_price'])]).reshape(1, -1),
                degree))[0]) / x[fet_index]
x_predicts.loc[i,'fet_Qg'] = \
    ((10 ** fet_reg_models_wpdf_df.loc['RdsQg_product', model].predict(
            fet_regression.preproc(
                np.array([np.log10(fets_df.loc[fet_index]['Vdss']), np.log10(x[fet_index]),
                          np.log10(x_predicts.loc[fet_index,'fet_Unit_price'])]).reshape(1, -1),
                degree))[0]) / x[fet_index]).astype(float)
fet_reg_models_wpdf_df.loc['RdsQg_product', model].predict(
            fet_regression.preproc(
                np.array([np.log10(fets_df.loc[fet_index]['Vdss']), np.log10(x[fet_index]),
                          np.log10(x_predicts.loc[fet_index,'fet_Unit_price'])]).reshape(1, -1),
                degree))[0]
print(fet_reg_models_wpdf_df.loc['RdsQg_product', model].predict(
            fet_regression.preproc(
                np.array([np.log10(fets_df.loc[fet_index]['Vdss']), np.log10(x[fet_index]),
                          np.log10(x_predicts.loc[fet_index,'fet_Unit_price'])]).reshape(1, -1),
                degree))[0])
10**-7
(10 ** fet_reg_models_wpdf_df.loc['RdsQg_product', model].predict(
            fet_regression.preproc(
                np.array([np.log10(fets_df.loc[fet_index]['Vdss']), np.log10(x[fet_index]),
                          np.log10(x_predicts.loc[fet_index,'fet_Unit_price'])]).reshape(1, -1),
                degree))[0])
print(10 ** fet_reg_models_wpdf_df.loc['RdsQg_product', model].predict(
            fet_regression.preproc(
                np.array([np.log10(fets_df.loc[fet_index]['Vdss']), np.log10(x[fet_index]),
                          np.log10(x_predicts.loc[fet_index,'fet_Unit_price'])]).reshape(1, -1),
                degree))[0])
(10 ** fet_reg_models_wpdf_df.loc['RdsQg_product', model].predict(
            fet_regression.preproc(
                np.array([np.log10(fets_df.loc[fet_index]['Vdss']), np.log10(x[fet_index]),
                          np.log10(x_predicts.loc[fet_index,'fet_Unit_price'])]).reshape(1, -1),
                degree))[0]) / x[fet_index]
print((10 ** fet_reg_models_wpdf_df.loc['RdsQg_product', model].predict(
            fet_regression.preproc(
                np.array([np.log10(fets_df.loc[fet_index]['Vdss']), np.log10(x[fet_index]),
                          np.log10(x_predicts.loc[fet_index,'fet_Unit_price'])]).reshape(1, -1),
                degree))[0]) / x[fet_index])
x_predicts.loc[i,'fet_Qg'] = \
        (10 ** fet_reg_models_wpdf_df.loc['RdsQg_product', model].predict(
            fet_regression.preproc(
                np.array([np.log10(fets_df.loc[fet_index]['Vdss']), np.log10(x[fet_index]),
                          np.log10(x_predicts.loc[fet_index,'fet_Unit_price'])]).reshape(1, -1),
                degree))[0]) / x[fet_index]
pd.options.display.float_format = "{:,.9f}".format
x[ind_index + offset]
inds_df.loc[ind_index]['Ind_fsw']
x_predicts.loc[i,'ind_Rdc'] = 10 ** (ind_reg_models_nopdf_df.loc['DCR_Inductance_product', model].predict(
                fet_regression.preproc(
                    np.array([np.log10(ind1), np.log10(1 / inds_df.loc[ind_index]['Current_rating'])]).reshape(1, -1),
                    degree))[0]) / ind1
x_predicts.loc[i,'ind_Rdc']
x_predicts.loc[i,'ind_Unit_price'] = 10 ** (ind_reg_models_nopdf_df.loc['DCR_Cost_product', model].predict(
            fet_regression.preproc(np.array([np.log10(ind1), np.log10(x_predicts.loc[ind_index, 'ind_Rdc'])]).reshape(1, -1),
                                   degree))[0]) / x_predicts.loc[ind_index, 'ind_Rdc']

np.log10(ind1)
np.log10(x_predicts.loc[ind_index, 'ind_Rdc'])
print(I_Q1**2 * x[0] + x[2]*x_predicts.loc[0,'fet_Qg']*fets_df.loc[0,'Vgate'] + \
               I_Q2**2 * x[1] + x[2]*x_predicts.loc[1,'fet_Qg']*fets_df.loc[1,'Vgate'] + x[2]*x_predicts.loc[1,'fet_Qrr']*fets_df.loc[0,'Vin'] + \
               inds_df.loc[0,'Current_rating'] ** 2 * x_predicts.loc[0,'ind_Rdc'])
fets_df.loc[0,'Vout'] * fets_df.loc[0,'Iout']
x_predicts.loc[0,'fet_Qg']
x_predicts.loc[1,'fet_Qrr']
c1
total_components
c1=5
T1 = fet_regression.preproc(np.array([np.log10(voltage), np.log10(c1)]).reshape(1, -1), degree)
Rdsx = (10 ** fet_reg_RdsCost_model.loc[0, model].predict(np.array(T1))) / c1
c1=.5
T1 = fet_regression.preproc(np.array([np.log10(voltage), np.log10(c1)]).reshape(1, -1), degree)
Rdsx = (10 ** fet_reg_RdsCost_model.loc[0, model].predict(np.array(T1))) / c1
c1=.5
T1 = fet_regression.preproc(np.array([np.log10(voltage), np.log10(c1)]).reshape(1, -1), degree)
Rdsx = 10 ** (fet_reg_RdsCost_model.loc[0, model].predict(np.array(T1)) / c1)
c1=5
T1 = fet_regression.preproc(np.array([np.log10(voltage), np.log10(c1)]).reshape(1, -1), degree)
Rdsx = 10 ** (fet_reg_RdsCost_model.loc[0, model].predict(np.array(T1)) / c1)
c1=5
T1 = fet_regression.preproc(np.array([np.log10(400), np.log10(c1)]).reshape(1, -1), degree)
Rdsx = (10 ** fet_reg_RdsCost_model.loc[0, model].predict(np.array(T1))) / c1
Rdsx
Rds
dc = fets_df.loc[0,'Vout'] / fets_df.loc[0,'Vin']
I_Q1 = np.sqrt(dc) * fets_df.loc[0,'Iout']
I_Q2 = np.sqrt(1 - dc) * fets_df.loc[0,'Iout']
I_Q1**2 * x[0] + x[2]*x_predicts.loc[0,'fet_Qg']*fets_df.loc[0,'Vgate'] + \
               I_Q2**2 * x[1] + x[2]*x_predicts.loc[1,'fet_Qg']*fets_df.loc[1,'Vgate'] + x[2]*x_predicts.loc[1,'fet_Qrr']*fets_df.loc[0,'Vin'] + \
               inds_df.loc[0,'Current_rating'] ** 2 * x_predicts.loc[0,'ind_Rdc']

I_Q1**2 * x[0] + x[2]*x_predicts.loc[0,'fet_Qg']*fets_df.loc[0,'Vgate'] + \
               I_Q2**2 * x[1] + x[2]*x_predicts.loc[1,'fet_Qg']*fets_df.loc[1,'Vgate'] + x[2]*x_predicts.loc[1,'fet_Qrr']*fets_df.loc[0,'Vin'] + \
               inds_df.loc[0,'Current_rating'] ** 2 * x_predicts.loc[0,'ind_Rdc']
print(I_Q1**2 * x[0] + x[2]*x_predicts.loc[0,'fet_Qg']*fets_df.loc[0,'Vgate'] + \
               I_Q2**2 * x[1] + x[2]*x_predicts.loc[1,'fet_Qg']*fets_df.loc[1,'Vgate'] + x[2]*x_predicts.loc[1,'fet_Qrr']*fets_df.loc[0,'Vin'] + \
               inds_df.loc[0,'Current_rating'] ** 2 * x_predicts.loc[0,'ind_Rdc']
)
fets_df.loc[0,'Vout'] * fets_df.loc[0,'Iout']
cost_tot
fet_reg_models_nopdf_df.loc['Pack_case', 'knn']
fet_reg_models_nopdf_df
np.log10(x[fet_index])
x_predicts.loc[fet_index,'fet_Unit_price']
x_predicts.loc[fet_index,'fet_Qg']
((ind_reg_models_nopdf_df.loc['Dimension', 'linear'].predict(
        fet_regression.preproc(np.array([np.log10(x_predicts.loc[ind_index, 'ind_Energy']), np.log10(x_predicts.loc[ind_index,'ind_Unit_price'])]).reshape(1, -1), degree))[0]) / x_predicts.loc[ind_index,'ind_Rdc'])

print((
    (ind_reg_models_nopdf_df.loc['Dimension', 'linear'].predict(
        fet_regression.preproc(np.array([np.log10(x_predicts.loc[ind_index, 'ind_Energy']), np.log10(x_predicts.loc[ind_index,'ind_Unit_price'])]).reshape(1, -1), degree))[0]) / x_predicts.loc[ind_index,'ind_Rdc'])
)
ind_reg_models_nopdf_df
output_param
np.log10(x_predicts.loc[ind_index, 'ind_Energy'])
np.log10(x_predicts.loc[ind_index,'ind_Unit_price'])
(reg_lin.predict(
        preproc(np.array([-3.37, 0.11]).reshape(1, -1), degree))[0])

print((reg_lin.predict(
        preproc(np.array([-3.37, 0.11]).reshape(1, -1), degree))[0]))
10**1.97
for fet_index in fets_df.index:
    area_tot += (10 ** (fet_reg_models_nopdf_df.loc['Pack_case', 'knn'].predict(
        fet_regression.preproc(np.array(
            [np.log10(fets_df.loc[fet_index, 'Vdss']), np.log10(x[fet_index]),
             np.log10(x_predicts.loc[fet_index, 'fet_Unit_price']),
             np.log10(x_predicts.loc[fet_index, 'fet_Qg'])]).reshape(1, -1), degree))[0]))

fets_df
for fet_index in fets_df.index:
    area_tot += (10 ** (fet_reg_models_nopdf_df.loc['Pack_case', 'knn'].predict(
        fet_regression.preproc(np.array(
            [np.log10(fets_df.loc[fet_index, 'Vdss']), np.log10(x[fet_index]),
             np.log10(x_predicts.loc[fet_index, 'fet_Unit_price']),
             np.log10(x_predicts.loc[fet_index, 'fet_Qg'])]).reshape(1, -1), degree))[0]))
    print(area_tot)

fet_reg_models_nopdf_df.loc['Pack_case', 'knn'].predict(
        fet_regression.preproc(np.array(
            [np.log10(fets_df.loc[fet_index, 'Vdss']), np.log10(x[fet_index]),
             np.log10(x_predicts.loc[fet_index, 'fet_Unit_price']),
             np.log10(x_predicts.loc[fet_index, 'fet_Qg'])]).reshape(1, -1), degree))[0])
fet_reg_models_nopdf_df.loc['Pack_case', 'knn'].predict(
        fet_regression.preproc(np.array(
            [np.log10(fets_df.loc[fet_index, 'Vdss']), np.log10(x[fet_index]),
             np.log10(x_predicts.loc[fet_index, 'fet_Unit_price']),
             np.log10(x_predicts.loc[fet_index, 'fet_Qg'])]).reshape(1, -1), degree))[0]
print(fet_reg_models_nopdf_df.loc['Pack_case', 'knn'].predict(
        fet_regression.preproc(np.array(
            [np.log10(fets_df.loc[fet_index, 'Vdss']), np.log10(x[fet_index]),
             np.log10(x_predicts.loc[fet_index, 'fet_Unit_price']),
             np.log10(x_predicts.loc[fet_index, 'fet_Qg'])]).reshape(1, -1), degree))[0])
fets_df.loc[fet_index, 'Vdss']
x_predicts.loc[fet_index, 'fet_Unit_price']
x_predicts.loc[fet_index, 'fet_Qg']
reg_knn.loc['Pack_case', 'knn'].predict(
            preproc(np.array(
                [np.log10(60), np.log10(0.007), np.log10(0.96),
                 np.log10(3.7*10**-8)]).reshape(1, -1), degree))[0]
reg_knn.predict(
            preproc(np.array(
                [np.log10(60), np.log10(0.007), np.log10(0.96),
                 np.log10(3.7*10**-8)]).reshape(1, -1), degree))[0]
reg_knn= model.fit(X,y)
print(reg_knn.predict(
            preproc(np.array(
                [np.log10(60), np.log10(0.007), np.log10(0.96),
                 np.log10(3.7*10**-8)]).reshape(1, -1), degree))[0])
X_over
preproc(np.array(
                [np.log10(60), np.log10(0.007), np.log10(0.96),
                 np.log10(3.7*10**-8)]).reshape(1, -1), degree)
print(preproc(np.array(
                [np.log10(60), np.log10(0.007), np.log10(0.96),
                 np.log10(3.7*10**-8)]).reshape(1, -1), degree))
y_over
area_tot
area_tot = 0
# get the fet areas
for fet_index in fets_df.index:
    area_tot += ((fet_reg_models_nopdf_df.loc['Pack_case', 'knn'].predict(
        fet_regression.preproc(np.array(
            [np.log10(fets_df.loc[fet_index, 'Vdss']), np.log10(x[fet_index]), np.log10(x_predicts.loc[fet_index,'fet_Unit_price']),
             np.log10(x_predicts.loc[fet_index,'fet_Qg'])]).reshape(1, -1), degree))[0]))
    print(area_tot)

# get the inductor areas
for ind_index in inds_df.index: # inputs for area are ['Energy', 'Unit_price']
    ind1 = inds_df.loc[ind_index,'Ind_fsw'] / x[ind_index + offset]
    #exec("Energy{i} = ind1 * inds_obj_list[i]['Current_rating']")
    area_tot += 10 ** ((ind_reg_models_nopdf_df.loc['Dimension', 'linear'].predict(
    fet_regression.preproc(np.array([np.log10(x_predicts.loc[ind_index, 'ind_Energy']), np.log10(x_predicts.loc[ind_index,'ind_Unit_price'])]).reshape(1, -1), degree))[0]))
    print(area_tot)

power_loss

'''