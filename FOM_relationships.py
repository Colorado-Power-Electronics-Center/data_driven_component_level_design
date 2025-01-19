'''
    This file contains the scrips to plot the FOMs of all various components for various technologies, and find the
    best-fit line according to the relationship between the FOM and a single input feature of interest. Capture the
    coefficients relating the quantities. Also implement pareto-dominance for plotting these relationships.
    Begin by looking at transistors for Si, SiC, and GaN, with the FOM being Ron*Qg, perhaps color-coded by cost.
'''

from csv import reader

import matplotlib.pyplot as plt
from csv_conversion import csv_to_mosfet
from fet_regression import *
from sklearn.covariance import EllipticEnvelope
import numpy as np
from fet_best_envelope import pareto_optimize
import matplotlib.lines as mlines
import matplotlib.colors as colors
from fet_price_filtering import outlier_removal
from fet_pdf_scraper import parse_pdf_param
from fet_area_filtering import area_filter, area_training, manual_area_filter, area_filter_gd
from sklearn import preprocessing
from sklearn.model_selection import cross_validate
import pandas as pd
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
from scipy.stats import chi2


def all_fom_plotting(component = 'fet'):

    # Plot the capacitor FOMs that we want to capture. First need to make sure the quantities are float values and do quantity
    # computations for what is needed. Then apply data pre-processing techniques: databse filtering, outlier detection.
    # The capacitor data file shown here already has had pareto-dominance applied, this one also has some other features
    # of interest such as mounting type and mfr. part no.
    if component == 'capacitor':
        # load the data we want to use
        file_name = 'csv_files/capacitor_data_extendedOpt.csv'
        cap_df = csv_to_df(file_name)
        data_dims = ['Unit_price', 'Rated_volt', 'Capacitance', 'Size', 'Temp_coef', 'Temp_coef_enc','Mounting_type','Thickness','Mfr_part_no']
        cap_df = cap_df.iloc[:, 1:]
        cap_df.columns = data_dims
        cap_df = cap_df[(cap_df['Capacitance'] >= 10*10**-9) & (cap_df['Rated_volt'] <= 1000)]
        cap_df['Energy'] = .5 * cap_df['Capacitance'] * cap_df['Rated_volt'] ** 2
        cap_df['Volume'] = cap_df['Size'] * cap_df['Thickness']

        var = 'Volume'
        cap_df = outlier_detect(cap_df, var, component='capacitor')
        two_prod_FOM_scatter(cap_df, 'Energy', var, y_var2 = None, component = 'capacitor', option='option2')

    # Plot the inductor FOMs that we want to capture. First need to make sure the quantities are float values and do quantity
    # computations for what is needed. Then apply data pre-processing techniques: databse filtering, outlier detection.
    # The inductor data file shown here already has had pareto-dominance applied, this one also has some other features
    # of interest such as saturating current, which is used to compute some energy-related quantities of interest.
    if component == 'inductor':
        file_name = 'csv_files/inductor_data_wSatCurrent.csv'
        ind_df = csv_to_df(file_name)
        data_dims = ['Unit_price', 'Current_rating', 'DCR', 'Dimension', 'Height', 'Inductance', 'Mfr_part_no', 'Sat_current', 'Energy','Volume']
        ind_df = ind_df.iloc[:, 1:]
        ind_df.columns = data_dims

        ind_df = ind_df[(ind_df['Inductance'] >= 10*10**-9) & (ind_df['DCR'] >= 1)]
        ind_df['Energy'] = .5 * ind_df['Inductance'] * ind_df['Sat_current'] ** 2
        ind_df['Efficiency'] = (ind_df['DCR'] * ind_df['Current_rating'] ** 2) / (.5*
                ind_df['Inductance'] * ind_df['Sat_current']**2)
        # ind_df['Energy_density'] = (ind_df['Energy'] / ind_df['Volume'])
        ind_df['Energy_density'] = .5 * ind_df['Inductance'] * ind_df['Sat_current'] ** 2 / (ind_df['Dimension'] * ind_df['Height'])
        var = 'Volume'
        ind_df = outlier_detect(ind_df, var, component='inductor')
        two_prod_FOM_scatter(ind_df, 'Energy', var, y_var2 = None, component = 'inductor', option='option2')

    # Plot the transistor FOMs that we want to capture. The file used here is after an initial cleaning, gotten from
    # fet_best_envelope.py. Now, separately filter depending on which quantity is being plotted, because we don't want
    # to use feature imputation for plotting, so for each quantity, filter to just include components that have that
    # quantity available. Here, do need to do Pareto-dominance, as well as outlier removal.

    csv_file = 'csv_files/postPreprocessing.csv'

    fet_df = pd.read_csv(csv_file)
    fet_df = fet_df.iloc[:, 1:]
    # var = 'C_oss'
    # fet_df = fet_df.drop(fet_df[(fet_df[var] == 0) | (fet_df[var].isna())].index)
    # df1 = fet_df[fet_df['C_oss']*fet_df['R_ds'] < 10 ** -6]
    # df1['C_oss'] = df1['C_oss']  * 10 ** 12
    # df2 = fet_df[fet_df['C_oss'] *fet_df['R_ds'] > 10 ** -6]
    # fet_df = pd.concat([df1, df2])
    # fet_df = fet_df[fet_df['C_oss']*fet_df['R_ds'] > 7*10**-1]

    # var = 'Q_rr'
    # fet_df = fet_df.drop(fet_df[(fet_df['Q_rr'] == 0) | (fet_df['Q_rr'].isna())].index)
    # df1 = fet_df[fet_df['Q_rr'] < 10 ** -4]
    # df1['Q_rr'] = df1['Q_rr'] * 10 ** 9
    # df2 = fet_df[fet_df['Q_rr'] >= 10 ** -4]
    # fet_df = pd.concat([df1, df2])
    # fet_df[(fet_df['V_dss'] > 100) & (fet_df['R_ds'] * fet_df['Q_rr'] < 1 * 10 ** -1)] = fet_df[
    #                                                                                          (fet_df['V_dss'] > 100) & (
    #                                                                                                      fet_df[
    #                                                                                                          'R_ds'] *
    #                                                                                                      fet_df[
    #                                                                                                          'Q_rr'] < 1 * 10 ** -1)] * 10 ** 3
    # fet_df[(fet_df['V_dss'] > 200) & (fet_df['R_ds'] * fet_df['Q_rr'] < 3 * 10 ** 0)] = fet_df[
    #                                                                                          (fet_df['V_dss'] > 200) & (
    #                                                                                                  fet_df[
    #                                                                                                      'R_ds'] *
    #                                                                                                  fet_df[
    #                                                                                                      'Q_rr'] < 3 * 10 ** 0)] * 10 ** 3

    # filt_df = filt_df.drop(filt_df[(filt_df['C_oss'] == 0) & (filt_df['Q_rr'] == 0)].index).drop_duplicates()
    var = 'Q_g'
    fet_df = fet_df.drop_duplicates('Mfr_part_no')
    fet_df = outlier_detect(fet_df, var)

    # Implement pareto-dominance using all relevant dimensions
    fet_df = pareto_front(fet_df)
    fet_df['Q_g'] = fet_df['Q_g']*10**9
    two_prod_FOM_scatter(fet_df, 'V_dss','R_ds',var,option='option2')

    two_prod_FOM_scatter(fet_df, 'V_dss','R_ds','Q_g',option='option2')

    print('done')



def transistor_fom_plotting():
    # First get the transistor data
    csv_file = 'csv_files/mosfet_data_wmfrPartNo2.csv'
    csv_file = 'csv_files/wManual.csv'


    fet_df = pd.read_csv(csv_file)
    fet_df = fet_df.iloc[:, 1:]
    fet_df.columns = ['Mfr_part_no', 'Unit_price', 'Mfr', 'Series', 'FET_type', 'Technology', 'V_dss', 'I_d', 'V_drive',
                      'R_ds', 'V_thresh', 'Q_g', 'V_gs', 'Input_cap', 'P_diss', 'Op_temp', 'Mount_type', 'Supp_pack',
                      'Pack_case', 'Q_rr','C_oss']
    fet_df = fet_df.reset_index()
    # fet_df['C_oss'] = 0.0
    attr_list = ['Mfr_part_no', 'Unit_price', 'FET_type', 'Technology', 'V_dss', 'R_ds', 'Q_g',
                 'Pack_case'
                 ]

    # fet_df = column_fet_parse(initial_fet_parse(fet_df, attr_list), attr_list)
    # subset_df = fet_df[(fet_df['FET_type'] == 'N')]
    # subset_df = subset_df[(subset_df['Technology'].isin([technology]))]

    # area_df = area_filter(fet_df)
    area_df = fet_df.drop_duplicates('Mfr_part_no')

    # Fill in the datapoints with ones we have Coss and Qrr for, can filter to just look at those later
    # Load the additional scraped qrr and coss data from FET_pdf_tables, observe this data
    csv_file = 'csv_files/FET_pdf_tables.csv'
    add_df = pd.read_csv(csv_file)
    add_df.columns = ['Mfr_part_no', 'Q_rr', 'C_oss']
    filtered_df = pd.DataFrame()
    i=0
    for part in add_df['Mfr_part_no'].unique():
        # print(part)
        # print(add_df[add_df['Mfr_part_no'] == part])
        filt_df = add_df[add_df['Mfr_part_no'] == part]
        filt_df = filt_df.drop(filt_df[filt_df['C_oss'].isna() & filt_df['Q_rr'].isna()].index).drop_duplicates()
        filt_df = filt_df.drop(filt_df[(filt_df['C_oss'] == 0) & (filt_df['Q_rr'] == 0)].index).drop_duplicates()
        # print(filt_df)
        filtered_df = filtered_df.append(filt_df)


    # First add in the manual entries, taken from fet_best_envelope


    # Check if the part number is in the main df
    for part in filtered_df['Mfr_part_no'].unique():
        if len(area_df[area_df['Mfr_part_no'] == part]) > 0:
            coss_index = area_df[area_df['Mfr_part_no'] == part].index
            cossfiltindex = filtered_df[filtered_df['Mfr_part_no'] == part].index
            area_df.loc[coss_index, 'Q_rr'] = filtered_df.loc[cossfiltindex, 'Q_rr'].values[0]*10**-9
            area_df.loc[coss_index, 'C_oss'] = filtered_df.loc[cossfiltindex, 'C_oss'].values[0]*10**-12
            # print(part)
            # print(coss_index[0])
            i+=1

    print('done')

    area_df['R_ds'] = area_df['R_ds'].astype(float)
    area_df['V_dss'] = area_df['V_dss'].astype(float)
    area_df['Unit_price'] = area_df['Unit_price'].astype(float)
    area_df['Q_g'] =area_df['Q_g'].astype(float)
    area_df['Pack_case'] = area_df['Pack_case'].astype(float)
    area_df['Q_rr'] = area_df['Q_rr'].astype(float)
    area_df['C_oss'] = area_df['C_oss'].astype(float)


    # compare and plot all data vs. w/ just outlier vs. w/ outlier and pareto-front
    # compare_preprocessing(area_df, 'V_dss','R_ds','Q_g')

    # Get the residuals of before and after outlier detection, could also train a quick linear model and look at the r^2

    # Implement outlier detection
    area_df = outlier_detect(area_df)

    # Implement pareto-dominance using all relevant dimensions
    area_df = pareto_front(area_df)


    two_prod_FOM_scatter(area_df, 'V_dss','R_ds','Q_g',option='option2')


    RQ_scatter_cost(area_df, 'fet', 'V_dss','R_ds','Q_g')


    print('done')


# Inside this function, determine the Pareto front for each of the materials. Use quantities not including Qrr or Coss,
# initially, bc these had some imputed features. Returns the full datset. To plot the Pareto optimized points for the
# Si case, uncomment the line to filter the cost, and go into the function pareto_optimize() and uncomment lines within
# there to plot the lower Pareto envelope (in red) versus all the datapoints (in blue) on various dimensions, also
# handle the dimensions inside the pareto_optimize() function.
def pareto_front(df):
    data_dims = ['V_dss', 'Unit_price', 'R_ds', 'Q_g', 'FET_type', 'Technology', 'Pack_case', 'Mfr_part_no', 'Q_rr',
                 'C_oss']
    pareto_df = pd.DataFrame()
    new_df = df[(df['Technology'].isin(['MOSFET']))]
    data_dims_paretoFront_dict = {'Power': ['V_dss', 'R_ds'],
                                  'Cost': ['V_dss', 'Unit_price', 'R_ds'], 'Area': ['V_dss', 'Pack_case', 'R_ds'],
                                  'Balanced': ['V_dss', 'Unit_price', 'R_ds', 'Q_g', 'Pack_case', 'Q_rr', 'C_oss'],
                                  'Balanced2': ['V_dss', 'Unit_price', 'R_ds', 'Q_g', 'Pack_case']}

    ################ UNCOMMENT THE FOLLOWING AND UNCOMMENT INSIDE PARETO_OPTIMIZE FUNCTION TO PLOT THE PARETO-FRONT
    ################ IN VARIOUS DIMENSIONS
    # new_df = new_df[(new_df['Unit_price']>.1) & (new_df['Unit_price'] < .5)]
    pareto_opt_data = pareto_optimize(new_df, data_dims, data_dims_paretoFront_dict['Balanced2'],
                                      technology='MOSFET')
    # pareto_opt_data2 = pareto_optimize(new_df, data_dims, data_dims_paretoFront_dict['Power'],
    #                                   technology='MOSFET')
    pareto_opt_df = pd.DataFrame(np.transpose(pareto_opt_data),
                                 columns=data_dims)
    pareto_df = pd.concat([pareto_df, pareto_opt_df])

    new_df = df[(df['Technology'].isin(['SiC','SiCFET']))]
    data_dims_paretoFront_dict = {'Power': ['V_dss', 'R_ds', 'Q_g', 'Q_rr', 'C_oss'],
                                  'Cost': ['V_dss', 'Unit_price', 'R_ds'], 'Area': ['V_dss', 'Pack_case', 'R_ds'],
                                  'Balanced': ['V_dss', 'Unit_price', 'R_ds', 'Q_g', 'Pack_case', 'Q_rr', 'C_oss'],
                                  'Balanced2': ['V_dss', 'Unit_price', 'R_ds', 'Q_g', 'Pack_case']}
    pareto_opt_data = pareto_optimize(new_df, data_dims, data_dims_paretoFront_dict['Balanced2'],
                                      technology='SiCFET')
    pareto_opt_df = pd.DataFrame(np.transpose(pareto_opt_data),
                                 columns=data_dims)
    pareto_df = pd.concat([pareto_df, pareto_opt_df])


    new_df = df[(df['Technology'].isin(['GaNFET']))]
    data_dims_paretoFront_dict = {'Power': ['V_dss', 'R_ds', 'Q_g'], 'Cost': ['V_dss', 'Unit_price', 'R_ds'],
                                  'Area': ['V_dss', 'Pack_case', 'R_ds'],
                                  'Balanced': ['V_dss', 'Unit_price', 'R_ds', 'Q_g', 'Pack_case', 'C_oss'],
                                  'Balanced2': ['V_dss', 'Unit_price', 'R_ds', 'Q_g', 'Pack_case']}
    pareto_opt_data = pareto_optimize(new_df, data_dims, data_dims_paretoFront_dict['Balanced2'],
                                      technology='GaNFET')
    pareto_opt_df = pd.DataFrame(np.transpose(pareto_opt_data),
                                 columns=data_dims)
    pareto_df = pd.concat([pareto_df, pareto_opt_df])




    # here, figure out the best components and add Qrr values for those
    pareto_df['R_ds'] = pareto_df['R_ds'].astype(float)
    pareto_df['V_dss'] = pareto_df['V_dss'].astype(float)
    pareto_df['Unit_price'] = pareto_df['Unit_price'].astype(float)
    pareto_df['Q_g'] = pareto_df['Q_g'].astype(float)
    pareto_df['Pack_case'] = pareto_df['Pack_case'].astype(float)
    pareto_df['Q_rr'] = pareto_df['Q_rr'].astype(float)
    pareto_df['C_oss'] = pareto_df['C_oss'].astype(float)


    return pareto_df


# Uses a multivariate outlier removal technique to get rid of components far outside trend. Takes as inputs the df of
# data, the parameter on the y-axis (assumes some constant x-axis variable for each component type), and the component
# type. Uses the covariance matrix and the distance from the centerpoint, and then a user-defined chi-squared value.
# When needed, can also manually remove some of the components--> some of the component data gotten from different
# sources may have different orders of magnitude if scraped incorrectly.
def outlier_detect(df, param2, component = 'fet', technology = 'MOSFET'):
    if component == 'capacitor':
        ind_df = df.copy()
        ind_df = ind_df[['Energy', param2]]
        ind_df = ind_df.dropna()
        ind_df = ind_df.to_numpy()
        covariance = np.cov(ind_df, rowvar=False)

        # Covariance matrix power of -1
        covariance_pm1 = np.linalg.matrix_power(covariance, -1)

        # Center point
        centerpoint = np.mean(ind_df, axis=0)
        distances = []
        for i, val in enumerate(ind_df):
            p1 = val
            p2 = centerpoint
            distance = (p1 - p2).T.dot(covariance_pm1).dot(p1 - p2)
            distances.append(distance)
        distances = np.array(distances)

        # Cutoff (threshold) value from Chi-Sqaure Distribution for detecting outliers
        cutoff = chi2.ppf(0.9, ind_df.shape[1])

        # Index of outliers
        outlierIndexes = np.where(distances > cutoff)
        ind_df = np.delete(ind_df, outlierIndexes, axis=0)
        ind_df = pd.DataFrame(ind_df, columns=['Energy', 'Efficiency'])
        df = df.drop(df.iloc[outlierIndexes].index)
        # if param2 == 'Efficiency':
        #     df = df[(df['Efficiency'] < 10 ** 6) & (df['Efficiency'] >= 10 ** 1)]
        # if param2 == 'Volume':
        #     # df = df[(df['Efficiency'] < 10 ** -6)]
        #     df1 = df[(df['Energy'] < 10 ** -6) & (df['Volume'] <= 10 ** 2)]
        #     # df1['C'] = df1['C_oss'] * 10 ** 12
        #     df2 = df[df['Energy'] >= 10 ** -6]
        #     df = pd.concat([df1, df2])

        print('done')
        return df


    if component == 'inductor':
        ind_df = df.copy()
        ind_df = ind_df[['Energy', param2]]
        ind_df = ind_df.dropna()
        ind_df = ind_df.to_numpy()
        covariance = np.cov(ind_df, rowvar=False)

        # Covariance matrix power of -1
        covariance_pm1 = np.linalg.matrix_power(covariance, -1)

        # Center point
        centerpoint = np.mean(ind_df, axis=0)
        distances = []
        for i, val in enumerate(ind_df):
            p1 = val
            p2 = centerpoint
            distance = (p1 - p2).T.dot(covariance_pm1).dot(p1 - p2)
            distances.append(distance)
        distances = np.array(distances)

        # Cutoff (threshold) value from Chi-Sqaure Distribution for detecting outliers
        cutoff = chi2.ppf(0.9, ind_df.shape[1])

        # Index of outliers
        outlierIndexes = np.where(distances > cutoff)
        ind_df = np.delete(ind_df, outlierIndexes, axis=0)
        ind_df = pd.DataFrame(ind_df, columns=['Energy', 'Efficiency'])
        df = df.drop(df.iloc[outlierIndexes].index)
        if param2 == 'Efficiency':
            df1 = df[(df['Energy'] >= 5*10 ** -6) & (df['Efficiency'] <= 10 ** 5)]
            # df1['C'] = df1['C_oss'] * 10 ** 12
            df2 = df[df['Energy'] <5* 10 ** -6]
            df = pd.concat([df1, df2])
            df = df[df['Efficiency'] > 10**2]

            # df = df[(df['Efficiency'] < 10 ** 6) & (df['Efficiency'] >= 10**1)]
        if param2 == 'Volume':
            # df = df[(df['Efficiency'] < 10 ** -6)]
            df1 = df[(df['Energy'] < 10 ** -6) & (df['Volume'] <= 10**2)]
            # df1['C'] = df1['C_oss'] * 10 ** 12
            df2 = df[df['Energy'] >= 10 ** -6]
            df = pd.concat([df1, df2])


        print('done')
        return df

    if component == 'fet':
        cutoff_dict = {'MOSFET': 0.999, 'SiCFET': 0.999, 'GaNFET': 0.999}
        area_df = df[(df['Technology'].isin([technology]))]
        # # outdf = area_df[area_df['V_dss'] < 10 ** 3]
        # # indf = area_df[area_df['V_dss'] >= 10 ** 3]
        # ee = EllipticEnvelope(contamination=0.2)
        # # yhat = ee.fit_predict(outdf[['R_ds','Q_g']])
        # yhat = ee.fit_predict(area_df[['R_ds',param2]])
        # mask = yhat != -1
        # # xtrain = pd.concat([outdf[mask], indf])
        # xtrain = area_df[mask]
        ind_df = area_df.copy()
        cap_df = ind_df.copy()
        cap2_df = ind_df.copy()
        cap2_df = cap2_df[cap2_df['V_dss'] >= 10**3]
        # if param2 == 'Q_g':
        #     ind_df['RdsQg'] = ind_df['R_ds'] * ind_df['Q_g']
        #     ind_df = ind_df[['V_dss', 'RdsQg']]
        # else:
        #     ind_df = ind_df[['V_dss', param2]]

        ind_df = ind_df[['V_dss', param2]]
        ind_df['V_dss'] = ind_df['V_dss'].astype(int)
        if param2 in ['Vds_meas', 'I_F']:
            ind_df[param2] = ind_df[param2].astype(int)
        if (technology == 'GaNFET') and (param2 in ['Q_rr', 'I_F']):
            return []

        ind_df = ind_df.dropna()
        ind_df = ind_df.to_numpy()
        # to plot to view relationship
        l1 = [x[0] for x in ind_df]
        l2 = [x[1] for x in ind_df]

        covariance = np.cov(ind_df, rowvar=False)

        # Covariance matrix power of -1
        covariance_pm1 = np.linalg.matrix_power(covariance, -1)

        # Center point
        centerpoint = np.mean(ind_df, axis=0)
        distances = []
        for i, val in enumerate(ind_df):
            p1 = val
            p2 = centerpoint
            distance = (p1 - p2).T.dot(covariance_pm1).dot(p1 - p2)
            distances.append(distance)
        distances = np.array(distances)

        # Cutoff (threshold) value from Chi-Square Distribution for detecting outliers
        cutoff = chi2.ppf(cutoff_dict[technology], ind_df.shape[1])

        # Index of outliers
        outlierIndexes = np.where(distances > cutoff)
        return outlierIndexes[0].tolist()
        # ind_df = np.delete(ind_df, outlierIndexes, axis=0)
        # ind_df = pd.DataFrame(ind_df, columns=['Energy', 'Efficiency'])
        df1 = cap_df.drop(cap_df.iloc[outlierIndexes].index)
        # df1 = pd.concat([df1, cap2_df])
        return df1


    area_df = df[(df['Technology'].isin(['SiC', 'SiCFET']))]
    # ee = EllipticEnvelope(contamination=0.2)
    # yhat = ee.fit_predict(area_df[['R_ds',param2]])
    # mask = yhat != -1
    # xtrain = pd.concat([area_df[mask], xtrain])
    ind_df = area_df.copy()
    cap_df = ind_df.copy()

    # ind_df = ind_df[['V_dss', param2]]
    if param2 == 'Q_g':
        ind_df['RdsQg'] = ind_df['R_ds'] * ind_df['Q_g']
        ind_df = ind_df[['V_dss', 'RdsQg']]
    else:
        ind_df = ind_df[['V_dss', param2]]
    ind_df = ind_df.dropna()
    ind_df = ind_df.to_numpy()
    covariance = np.cov(ind_df, rowvar=False)

    # Covariance matrix power of -1
    covariance_pm1 = np.linalg.matrix_power(covariance, -1)

    # Center point
    centerpoint = np.mean(ind_df, axis=0)
    distances = []
    for i, val in enumerate(ind_df):
        p1 = val
        p2 = centerpoint
        distance = (p1 - p2).T.dot(covariance_pm1).dot(p1 - p2)
        distances.append(distance)
    distances = np.array(distances)

    # Cutoff (threshold) value from Chi-Sqaure Distribution for detecting outliers
    cutoff = chi2.ppf(0.9, ind_df.shape[1])

    # Index of outliers
    outlierIndexes = np.where(distances > cutoff)
    # ind_df = np.delete(ind_df, outlierIndexes, axis=0)
    # ind_df = pd.DataFrame(ind_df, columns=['Energy', 'Efficiency'])
    df2 = cap_df.drop(cap_df.iloc[outlierIndexes].index)

    area_df = df[(df['Technology'].isin(['GaNFET']))]
    # ee = EllipticEnvelope(contamination=0.2)
    # yhat = ee.fit_predict(area_df[['R_ds',param2]])
    # mask = yhat != -1
    # xtrain = pd.concat([area_df[mask], xtrain])
    ind_df = area_df.copy()
    cap_df = ind_df.copy()

    # ind_df = ind_df[['V_dss', param2]]
    if param2 == 'Q_g':
        ind_df['RdsQg'] = ind_df['R_ds'] * ind_df['Q_g']
        ind_df = ind_df[['V_dss', 'RdsQg']]
    else:
        ind_df = ind_df[['V_dss', param2]]
    ind_df = ind_df.dropna()
    ind_df = ind_df.to_numpy()
    covariance = np.cov(ind_df, rowvar=False)

    # Covariance matrix power of -1
    covariance_pm1 = np.linalg.matrix_power(covariance, -1)

    # Center point
    centerpoint = np.mean(ind_df, axis=0)
    distances = []
    for i, val in enumerate(ind_df):
        p1 = val
        p2 = centerpoint
        distance = (p1 - p2).T.dot(covariance_pm1).dot(p1 - p2)
        distances.append(distance)
    distances = np.array(distances)

    # Cutoff (threshold) value from Chi-Sqaure Distribution for detecting outliers
    cutoff = chi2.ppf(0.9, ind_df.shape[1])

    # Index of outliers
    outlierIndexes = np.where(distances > cutoff)
    # ind_df = np.delete(ind_df, outlierIndexes, axis=0)
    # ind_df = pd.DataFrame(ind_df, columns=['Energy', 'Efficiency'])
    df3 = cap_df.drop(cap_df.iloc[outlierIndexes].index)

    return pd.concat([df1,df2,df3])

# This is the old outlier detection method, but wasn't updating the dataframe with the newly filtered datapoints in
# some cases, because was adding back in old datapoints when dropping from a future dataframe.
def outlier_detect_prior(df, param2, component = 'fet'):
    if component == 'capacitor':
        ind_df = df.copy()
        ind_df = ind_df[['Energy', param2]]
        ind_df = ind_df.dropna()
        ind_df = ind_df.to_numpy()
        covariance = np.cov(ind_df, rowvar=False)

        # Covariance matrix power of -1
        covariance_pm1 = np.linalg.matrix_power(covariance, -1)

        # Center point
        centerpoint = np.mean(ind_df, axis=0)
        distances = []
        for i, val in enumerate(ind_df):
            p1 = val
            p2 = centerpoint
            distance = (p1 - p2).T.dot(covariance_pm1).dot(p1 - p2)
            distances.append(distance)
        distances = np.array(distances)

        # Cutoff (threshold) value from Chi-Sqaure Distribution for detecting outliers
        cutoff = chi2.ppf(0.6, ind_df.shape[1])

        # Index of outliers
        outlierIndexes = np.where(distances > cutoff)
        ind_df = np.delete(ind_df, outlierIndexes, axis=0)
        ind_df = pd.DataFrame(ind_df, columns=['Energy', 'Efficiency'])
        df = df.drop(df.iloc[outlierIndexes].index)
        # if param2 == 'Efficiency':
        #     df = df[(df['Efficiency'] < 10 ** 6) & (df['Efficiency'] >= 10 ** 1)]
        # if param2 == 'Volume':
        #     # df = df[(df['Efficiency'] < 10 ** -6)]
        #     df1 = df[(df['Energy'] < 10 ** -6) & (df['Volume'] <= 10 ** 2)]
        #     # df1['C'] = df1['C_oss'] * 10 ** 12
        #     df2 = df[df['Energy'] >= 10 ** -6]
        #     df = pd.concat([df1, df2])

        print('done')
        return df


        # cap1 = Plotting_parameters(df, param2, component)
        # cap1.get_params()
        # cap2 = Plotting_parameters(df,'Energy', component)
        # cap2.get_params()
        # ee = EllipticEnvelope(contamination=0.1)
        # yhat = ee.fit_predict(np.array(cap1.var_label).reshape(-1, 1), np.array(cap2.var_label).reshape(-1, 1))
        # mask = yhat != -1
        # xtrain = df[mask]
        # return xtrain

    if component == 'inductor':
        ind_df = df.copy()
        ind_df = ind_df[['Energy', param2]]
        ind_df = ind_df.dropna()
        ind_df = ind_df.to_numpy()
        covariance = np.cov(ind_df, rowvar=False)

        # Covariance matrix power of -1
        covariance_pm1 = np.linalg.matrix_power(covariance, -1)

        # Center point
        centerpoint = np.mean(ind_df, axis=0)
        distances = []
        for i, val in enumerate(ind_df):
            p1 = val
            p2 = centerpoint
            distance = (p1 - p2).T.dot(covariance_pm1).dot(p1 - p2)
            distances.append(distance)
        distances = np.array(distances)

        # Cutoff (threshold) value from Chi-Sqaure Distribution for detecting outliers
        cutoff = chi2.ppf(0.6, ind_df.shape[1])

        # Index of outliers
        outlierIndexes = np.where(distances > cutoff)
        ind_df = np.delete(ind_df, outlierIndexes, axis=0)
        ind_df = pd.DataFrame(ind_df, columns=['Energy', 'Efficiency'])
        df = df.drop(df.iloc[outlierIndexes].index)
        if param2 == 'Efficiency':
            df1 = df[(df['Energy'] >= 5*10 ** -6) & (df['Efficiency'] <= 10 ** 5)]
            # df1['C'] = df1['C_oss'] * 10 ** 12
            df2 = df[df['Energy'] <5* 10 ** -6]
            df = pd.concat([df1, df2])
            df = df[df['Efficiency'] > 10**2]

            # df = df[(df['Efficiency'] < 10 ** 6) & (df['Efficiency'] >= 10**1)]
        if param2 == 'Volume':
            # df = df[(df['Efficiency'] < 10 ** -6)]
            df1 = df[(df['Energy'] < 10 ** -6) & (df['Volume'] <= 10**2)]
            # df1['C'] = df1['C_oss'] * 10 ** 12
            df2 = df[df['Energy'] >= 10 ** -6]
            df = pd.concat([df1, df2])


        print('done')
        return df

        # ind1 = Plotting_parameters(df, param2,'inductor')
        # ind1.get_params()
        # ind2 = Plotting_parameters(df, 'Energy','inductor')
        # ind2.get_params()
        # ee = EllipticEnvelope(contamination=0.2)
        # # yhat = ee.fit_predict(np.array(ind1.var_label).reshape(-1,1), np.array(ind2.var_label).reshape(-1,1))
        # yhat = ee.fit_predict(np.array(df['Energy']).reshape(-1,1), np.array(df['Efficiency']).reshape(-1,1))
        # mask = yhat != -1
        # xtrain = df[mask]
        # return xtrain


    area_df = df[(df['Technology'].isin(['MOSFET']))]
    # # outdf = area_df[area_df['V_dss'] < 10 ** 3]
    # # indf = area_df[area_df['V_dss'] >= 10 ** 3]
    # ee = EllipticEnvelope(contamination=0.2)
    # # yhat = ee.fit_predict(outdf[['R_ds','Q_g']])
    # yhat = ee.fit_predict(area_df[['R_ds',param2]])
    # mask = yhat != -1
    # # xtrain = pd.concat([outdf[mask], indf])
    # xtrain = area_df[mask]
    ind_df = area_df.copy()
    ind_df = ind_df[['V_dss', param2]]
    ind_df = ind_df.dropna()
    ind_df = ind_df.to_numpy()
    covariance = np.cov(ind_df, rowvar=False)

    # Covariance matrix power of -1
    covariance_pm1 = np.linalg.matrix_power(covariance, -1)

    # Center point
    centerpoint = np.mean(ind_df, axis=0)
    distances = []
    for i, val in enumerate(ind_df):
        p1 = val
        p2 = centerpoint
        distance = (p1 - p2).T.dot(covariance_pm1).dot(p1 - p2)
        distances.append(distance)
    distances = np.array(distances)

    # Cutoff (threshold) value from Chi-Sqaure Distribution for detecting outliers
    cutoff = chi2.ppf(0.6, ind_df.shape[1])

    # Index of outliers
    outlierIndexes = np.where(distances > cutoff)
    ind_df = np.delete(ind_df, outlierIndexes, axis=0)
    # ind_df = pd.DataFrame(ind_df, columns=['Energy', 'Efficiency'])
    df1 = df.drop(df.iloc[outlierIndexes].index)


    area_df = df[(df['Technology'].isin(['SiC', 'SiCFET']))]
    # ee = EllipticEnvelope(contamination=0.2)
    # yhat = ee.fit_predict(area_df[['R_ds',param2]])
    # mask = yhat != -1
    # xtrain = pd.concat([area_df[mask], xtrain])
    ind_df = area_df.copy()
    ind_df = ind_df[['V_dss', param2]]
    ind_df = ind_df.dropna()
    ind_df = ind_df.to_numpy()
    covariance = np.cov(ind_df, rowvar=False)

    # Covariance matrix power of -1
    covariance_pm1 = np.linalg.matrix_power(covariance, -1)

    # Center point
    centerpoint = np.mean(ind_df, axis=0)
    distances = []
    for i, val in enumerate(ind_df):
        p1 = val
        p2 = centerpoint
        distance = (p1 - p2).T.dot(covariance_pm1).dot(p1 - p2)
        distances.append(distance)
    distances = np.array(distances)

    # Cutoff (threshold) value from Chi-Sqaure Distribution for detecting outliers
    cutoff = chi2.ppf(0.6, ind_df.shape[1])

    # Index of outliers
    outlierIndexes = np.where(distances > cutoff)
    ind_df = np.delete(ind_df, outlierIndexes, axis=0)
    # ind_df = pd.DataFrame(ind_df, columns=['Energy', 'Efficiency'])
    df2 = df.drop(df.iloc[outlierIndexes].index)

    area_df = df[(df['Technology'].isin(['GaNFET']))]
    # ee = EllipticEnvelope(contamination=0.2)
    # yhat = ee.fit_predict(area_df[['R_ds',param2]])
    # mask = yhat != -1
    # xtrain = pd.concat([area_df[mask], xtrain])
    ind_df = area_df.copy()
    ind_df = ind_df[['V_dss', param2]]
    ind_df = ind_df.dropna()
    ind_df = ind_df.to_numpy()
    covariance = np.cov(ind_df, rowvar=False)

    # Covariance matrix power of -1
    covariance_pm1 = np.linalg.matrix_power(covariance, -1)

    # Center point
    centerpoint = np.mean(ind_df, axis=0)
    distances = []
    for i, val in enumerate(ind_df):
        p1 = val
        p2 = centerpoint
        distance = (p1 - p2).T.dot(covariance_pm1).dot(p1 - p2)
        distances.append(distance)
    distances = np.array(distances)

    # Cutoff (threshold) value from Chi-Sqaure Distribution for detecting outliers
    cutoff = chi2.ppf(0.6, ind_df.shape[1])

    # Index of outliers
    outlierIndexes = np.where(distances > cutoff)
    ind_df = np.delete(ind_df, outlierIndexes, axis=0)
    # ind_df = pd.DataFrame(ind_df, columns=['Energy', 'Efficiency'])
    df3 = df.drop(df.iloc[outlierIndexes].index)

    return pd.concat([df1,df2,df3])


# Creates the colormap for plotting the datapoint colors according to their cost
def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    # cmap = cmap.reversed()
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap


# This function is explicitly used for generating the three graphics used for comparing data results after various
# preprocessing techniques. This takes the data for all FET materials and first plots with all data, then after just
# outlier detection, then after outlier detection and pareto-dominance. As it is now, need to do some copy-and-pasting
# to get all three materials at each preprocessing step onto the same plot, but should fix this in the future.
def compare_preprocessing(fet_df, x_var, y_var1, y_var2):
    # First for MOSFETs
    ready_df = fet_df[(fet_df['FET_type'] == 'N')]
    ready_df = ready_df[(ready_df['Technology'].isin(['MOSFET']))]
    x = []
    y = []
    X = []
    Y = []
    # print(m)
    x.extend(ready_df.loc[:, x_var])
    y.extend(ready_df.loc[:, y_var1] * ready_df.loc[:, y_var2])
    X.extend(np.log10(ready_df.loc[:, x_var]))
    Y.extend(np.log10(ready_df.loc[:, y_var1] * ready_df.loc[:, y_var2]))
    theta = np.polyfit(X, Y, 2)
    X.sort()
    y_list = [10 ** (theta[0] * g ** 2 + theta[1] * g + theta[2]) for g in X]
    x.sort()
    plt.plot(x, y_list, c='black', linestyle='solid')
    plt.xscale('log')
    plt.yscale('log')

    # then plot w/ after outlier detection
    # outdf = ready_df[ready_df['V_dss'] < 10 ** 3]
    # indf = ready_df[ready_df['V_dss'] >= 10 ** 3]
    ee = EllipticEnvelope(contamination=0.2)
    # yhat = ee.fit_predict(outdf[['R_ds', 'Q_g']])
    yhat = ee.fit_predict(ready_df[['R_ds', 'Q_g']])
    mask = yhat != -1
    # ready_df = pd.concat([outdf[mask], indf])
    ready_df = ready_df[mask]

    # ready_df = outlier_detect(ready_df)
    x = []
    y = []
    X = []
    Y = []
    # print(m)
    x.extend(ready_df.loc[:, x_var])
    y.extend(ready_df.loc[:, y_var1] * ready_df.loc[:, y_var2])
    X.extend(np.log10(ready_df.loc[:, x_var]))
    Y.extend(np.log10(ready_df.loc[:, y_var1] * ready_df.loc[:, y_var2]))
    theta = np.polyfit(X, Y, 2)
    X.sort()
    y_list = [10 ** (theta[0] * g ** 2 + theta[1] * g + theta[2]) for g in X]
    x.sort()
    plt.plot(x, y_list, c='red', linestyle='solid')

    # then plot w/ after outlier detection AND pareto dominance
    data_dims = ['V_dss', 'Unit_price', 'R_ds', 'Q_g', 'FET_type', 'Technology', 'Pack_case', 'Mfr_part_no', 'Q_rr',
                 'C_oss']
    pareto_df = pd.DataFrame()
    new_df = ready_df[(ready_df['Technology'].isin(['MOSFET']))]
    data_dims_paretoFront_dict = {'Power': ['V_dss', 'R_ds', 'Q_g', 'Q_rr', 'C_oss'],
                                  'Cost': ['V_dss', 'Unit_price', 'R_ds'], 'Area': ['V_dss', 'Pack_case', 'R_ds'],
                                  'Balanced': ['V_dss', 'Unit_price', 'R_ds', 'Q_g', 'Pack_case', 'Q_rr', 'C_oss'],
                                  'Balanced2': ['V_dss', 'Unit_price', 'R_ds', 'Q_g', 'Pack_case']}
    area_df = new_df.copy()
    area_df['R_ds'] = area_df['R_ds'].astype(float)
    area_df['V_dss'] = area_df['V_dss'].astype(float)
    area_df['Unit_price'] = area_df['Unit_price'].astype(float)
    area_df['Q_g'] = area_df['Q_g'].astype(float)
    area_df['Pack_case'] = area_df['Pack_case'].astype(float)
    area_df['Q_rr'] = area_df['Q_rr'].astype(float)
    area_df['C_oss'] = area_df['C_oss'].astype(float)
    pareto_opt_data = pareto_optimize(area_df, data_dims, data_dims_paretoFront_dict['Balanced2'],
                                      technology='MOSFET')
    pareto_opt_df = pd.DataFrame(np.transpose(pareto_opt_data),
                                 columns=data_dims)
    ready_df = pd.concat([pareto_df, pareto_opt_df])
    x = []
    y = []
    X = []
    Y = []
    # print(m)
    x.extend(ready_df.loc[:, x_var].astype(float))
    y.extend(ready_df.loc[:, y_var1].astype(float) * ready_df.loc[:, y_var2].astype(float))
    X.extend(np.log10(ready_df.loc[:, x_var].astype(float)))
    Y.extend(np.log10(ready_df.loc[:, y_var1].astype(float) * ready_df.loc[:, y_var2].astype(float)))
    theta = np.polyfit(X, Y, 2)
    X.sort()
    y_list = [10 ** (theta[0] * g ** 2 + theta[1] * g + theta[2]) for g in X]
    x.sort()
    plt.plot(x, y_list, c='blue', linestyle='solid')




    # Then for SiCFETs
    ready_df = fet_df[(fet_df['FET_type'] == 'N')]
    ready_df = ready_df[(ready_df['Technology'].isin(['SiCFET', 'SiC']))]
    x = []
    y = []
    X = []
    Y = []
    # print(m)
    x.extend(ready_df.loc[:, x_var])
    y.extend(ready_df.loc[:, y_var1] * ready_df.loc[:, y_var2])
    X.extend(np.log10(ready_df.loc[:, x_var]))
    Y.extend(np.log10(ready_df.loc[:, y_var1] * ready_df.loc[:, y_var2]))
    theta = np.polyfit(X, Y, 2)
    X.sort()
    y_list = [10 ** (theta[0] * g ** 2 + theta[1] * g + theta[2]) for g in X]
    x.sort()
    plt.plot(x, y_list, c='black', linestyle='dashed')
    plt.xscale('log')
    plt.yscale('log')

    # then plot w/ after outlier detection
    ee = EllipticEnvelope(contamination=0.2)
    yhat = ee.fit_predict(ready_df[['R_ds', 'Q_g']])
    mask = yhat != -1
    ready_df = ready_df[mask]

    # ready_df = outlier_detect(ready_df)
    x = []
    y = []
    X = []
    Y = []
    # print(m)
    x.extend(ready_df.loc[:, x_var])
    y.extend(ready_df.loc[:, y_var1] * ready_df.loc[:, y_var2])
    X.extend(np.log10(ready_df.loc[:, x_var]))
    Y.extend(np.log10(ready_df.loc[:, y_var1] * ready_df.loc[:, y_var2]))
    theta = np.polyfit(X, Y, 2)
    X.sort()
    y_list = [10 ** (theta[0] * g ** 2 + theta[1] * g + theta[2]) for g in X]
    x.sort()
    plt.plot(x, y_list, c='red', linestyle='dashed')

    # then plot w/ after outlier detection AND pareto dominance
    data_dims = ['V_dss', 'Unit_price', 'R_ds', 'Q_g', 'FET_type', 'Technology', 'Pack_case', 'Mfr_part_no', 'Q_rr',
                 'C_oss']
    pareto_df = pd.DataFrame()
    new_df = ready_df[(ready_df['Technology'].isin(['SiCFET','SiC']))]
    data_dims_paretoFront_dict = {'Power': ['V_dss', 'R_ds', 'Q_g', 'Q_rr', 'C_oss'],
                                  'Cost': ['V_dss', 'Unit_price', 'R_ds'], 'Area': ['V_dss', 'Pack_case', 'R_ds'],
                                  'Balanced': ['V_dss', 'Unit_price', 'R_ds', 'Q_g', 'Pack_case', 'Q_rr', 'C_oss'],
                                  'Balanced2': ['V_dss', 'Unit_price', 'R_ds', 'Q_g', 'Pack_case']}
    area_df = new_df.copy()
    area_df['R_ds'] = area_df['R_ds'].astype(float)
    area_df['V_dss'] = area_df['V_dss'].astype(float)
    area_df['Unit_price'] = area_df['Unit_price'].astype(float)
    area_df['Q_g'] = area_df['Q_g'].astype(float)
    area_df['Pack_case'] = area_df['Pack_case'].astype(float)
    area_df['Q_rr'] = area_df['Q_rr'].astype(float)
    area_df['C_oss'] = area_df['C_oss'].astype(float)
    pareto_opt_data = pareto_optimize(area_df, data_dims, data_dims_paretoFront_dict['Balanced2'],
                                      technology='SiCFET')
    pareto_opt_df = pd.DataFrame(np.transpose(pareto_opt_data),
                                 columns=data_dims)
    ready_df = pd.concat([pareto_df, pareto_opt_df])
    x = []
    y = []
    X = []
    Y = []
    # print(m)
    x.extend(ready_df.loc[:, x_var].astype(float))
    y.extend(ready_df.loc[:, y_var1].astype(float) * ready_df.loc[:, y_var2].astype(float))
    X.extend(np.log10(ready_df.loc[:, x_var].astype(float)))
    Y.extend(np.log10(ready_df.loc[:, y_var1].astype(float) * ready_df.loc[:, y_var2].astype(float)))
    theta = np.polyfit(X, Y, 2)
    X.sort()
    y_list = [10 ** (theta[0] * g ** 2 + theta[1] * g + theta[2]) for g in X]
    x.sort()
    plt.plot(x, y_list, c='blue', linestyle='dashed')



    # Finally for GaN

    ready_df = fet_df[(fet_df['FET_type'] == 'N')]
    ready_df = ready_df[(ready_df['Technology'].isin(['GaNFET']))]
    x = []
    y = []
    X = []
    Y = []
    # print(m)
    x.extend(ready_df.loc[:, x_var])
    y.extend(ready_df.loc[:, y_var1] * ready_df.loc[:, y_var2])
    X.extend(np.log10(ready_df.loc[:, x_var]))
    Y.extend(np.log10(ready_df.loc[:, y_var1] * ready_df.loc[:, y_var2]))
    theta = np.polyfit(X, Y, 2)
    X.sort()
    y_list = [10 ** (theta[0] * g ** 2 + theta[1] * g + theta[2]) for g in X]
    x.sort()
    plt.plot(x, y_list, c='black', linestyle='dotted')


    # then plot w/ after outlier detection

    ee = EllipticEnvelope(contamination=0.2)
    yhat = ee.fit_predict(ready_df[['R_ds', 'Q_g']])
    mask = yhat != -1
    ready_df = ready_df[mask]

    # ready_df = outlier_detect(ready_df)
    x = []
    y = []
    X = []
    Y = []
    # print(m)
    x.extend(ready_df.loc[:, x_var])
    y.extend(ready_df.loc[:, y_var1] * ready_df.loc[:, y_var2])
    X.extend(np.log10(ready_df.loc[:, x_var]))
    Y.extend(np.log10(ready_df.loc[:, y_var1] * ready_df.loc[:, y_var2]))
    theta = np.polyfit(X, Y, 2)
    X.sort()
    y_list = [10 ** (theta[0] * g ** 2 + theta[1] * g + theta[2]) for g in X]
    x.sort()
    plt.plot(x, y_list, c='red', linestyle='dotted')

    # then plot w/ after outlier detection AND pareto dominance
    data_dims = ['V_dss', 'Unit_price', 'R_ds', 'Q_g', 'FET_type', 'Technology', 'Pack_case', 'Mfr_part_no', 'Q_rr',
                 'C_oss']
    pareto_df = pd.DataFrame()
    new_df = ready_df[(ready_df['Technology'].isin(['GaNFET']))]
    data_dims_paretoFront_dict = {'Power': ['V_dss', 'R_ds', 'Q_g', 'Q_rr', 'C_oss'],
                                  'Cost': ['V_dss', 'Unit_price', 'R_ds'], 'Area': ['V_dss', 'Pack_case', 'R_ds'],
                                  'Balanced': ['V_dss', 'Unit_price', 'R_ds', 'Q_g', 'Pack_case', 'Q_rr', 'C_oss'],
                                  'Balanced2': ['V_dss', 'Unit_price', 'R_ds', 'Q_g', 'Pack_case']}
    area_df = new_df.copy()
    area_df['R_ds'] = area_df['R_ds'].astype(float)
    area_df['V_dss'] = area_df['V_dss'].astype(float)
    area_df['Unit_price'] = area_df['Unit_price'].astype(float)
    area_df['Q_g'] = area_df['Q_g'].astype(float)
    area_df['Pack_case'] = area_df['Pack_case'].astype(float)
    area_df['Q_rr'] = area_df['Q_rr'].astype(float)
    area_df['C_oss'] = area_df['C_oss'].astype(float)
    pareto_opt_data = pareto_optimize(area_df, data_dims, data_dims_paretoFront_dict['Balanced2'],
                                      technology='GaNFET')
    pareto_opt_df = pd.DataFrame(np.transpose(pareto_opt_data),
                                 columns=data_dims)
    ready_df = pd.concat([pareto_df, pareto_opt_df])
    x = []
    y = []
    X = []
    Y = []
    # print(m)
    x.extend(ready_df.loc[:, x_var].astype(float))
    y.extend(ready_df.loc[:, y_var1].astype(float) * ready_df.loc[:, y_var2].astype(float))
    X.extend(np.log10(ready_df.loc[:, x_var].astype(float)))
    Y.extend(np.log10(ready_df.loc[:, y_var1].astype(float) * ready_df.loc[:, y_var2].astype(float)))
    theta = np.polyfit(X, Y, 2)
    X.sort()
    y_list = [10 ** (theta[0] * g ** 2 + theta[1] * g + theta[2]) for g in X]
    x.sort()
    plt.plot(x, y_list, c='blue', linestyle='dotted')



    print('done')


# Creates a class of visualization parameters for annotating various FOM plots. No need for the plot titles here,
# as the plots we generate don't have titles. Takes as input the other variable in the FOM, for transistors only because
# those are the only components that require plot annotation.
class Visualization_parameters(object):
    def __init__(self, var2):
        self.var2 = var2

    def get_params(self):
        if self.var2 == 'Unit_price':
            self.plot_title = 'Rds Histogram'
            self.ylabel = '$R_{on} * U$ [Ω-\$]'

            self.pt1 = plt.annotate('Si', (2 * 10 ** 2, 7 * 10 ** -3), size=15)
            self.pt2 = plt.annotate('SiC', (1 * 10 ** 3, 2 * 10 ** -1), size=15)
            self.pt3 = plt.annotate('GaN', (10 ** 2, 1 * 10 ** -1), size=15)

        if self.var2 == 'Q_g':
            self.plot_title = 'Rds Histogram'
            self.ylabel = '$R_{on} * Q_g$ [Ω-nC]'
            self.pt1 = plt.annotate('Si', (10 ** 2, 2 * 10 ** 1), size=15)
            self.pt2 = plt.annotate('SiC', (1.5 * 10 ** 3, 9 * 10 ** -1), size=15)
            self.pt3 = plt.annotate('GaN', (10 ** 2, 1 * 10 ** -2), size=15)

        if self.var2 == 'Pack_case':
            self.plot_title = 'Rds Histogram'
            self.ylabel = '$R_{ds} * A$ [Ω-$mm^3$]'
            self.pt1 = plt.annotate('Si', (8 * 10 ** 1, 1 * 10 ** 0), size=15)
            self.pt2 = plt.annotate('SiC', (1.5 * 10 ** 3, 9 * 10 ** -1), size=15)
            self.pt3 = plt.annotate('GaN', (10 ** 2, 1 * 10 ** -3), size=15)

        if self.var2 == 'Q_rr':
            self.plot_title = 'Rds Histogram'
            self.ylabel = '$R_{on} * Q_{rr}$ [Ω-nC]'
            self.pt1 = plt.annotate('Si', (3 * 10 ** 1, 9 * 10 ** -1), size=15)
            self.pt2 = plt.annotate('SiC', (1 * 10 ** 3, 2 * 10 ** 0), size=15)
            self.pt3 = plt.annotate('GaN', (6*10 ** 2, 1 * 10 ** 1), size=15)

        if self.var2 == 'C_oss':
            self.plot_title = 'Rds Histogram'
            self.ylabel = '$R_{on} * C_{oss}$ [Ω-pF]'
            self.pt1 = plt.annotate('Si', (4 * 10 ** 1, 1 * 10 ** 1), size=15)
            self.pt2 = plt.annotate('SiC', (1 * 10 ** 3, 2 * 10 ** 0), size=15)
            self.pt3 = plt.annotate('GaN', (1*10 ** 2, 7 * 10 ** -1), size=15)


# Creates a class of parameters used to create the visual plots. Takes as input the FOM, or for transistors, the other
# variable used in the FOM along w/ Rds, and the df of data which is used to compute the quantity values used in the
# plot for that particular FOM, and the component type.
# Generates the y-axis label. Title of plot typically unused.
class Plotting_parameters(object):
    def __init__(self, df, var2, component):
        self.df = df
        self.var2 = var2
        self.component = component

    def get_params(self):
        if self.component == 'capacitor':
            if self.var2 == 'Energy':
                self.var_label = (0.5 * self.df['Capacitance'] * self.df['Rated_volt']**2)
                self.label = 'Energy [$F-V^2$]'
                self.title = 'Energy'

            if self.var2 == 'Volume':
                self.var_label = self.df['Size'] * self.df['Thickness']
                self.label = 'Volume [$mm^3$]'
                self.title = 'Volume'

            if self.var2 == 'Area':
                self.var_label = self.df['Size']
                self.label = 'Area [$mm^2$]'
                self.title = 'Area'

            if self.var2 == 'Energy density':
                self.var_label = 0.5 * self.df['Capacitance'] * self.df['Rated_volt'] ** 2 / (self.df['Size'] * self.df['Thickness'])
                self.label = 'Energy density [$F-V^2 / mm^3$]'
                self.title = 'Energy density'

            if self.var2 == 'Surface density':
                self.var_label = 0.5 * self.df['Capacitance'] * self.df['Rated_volt'] ** 2 / self.df['Size']
                self.label = 'Surface density [$F-V^2 / mm^2$]'
                self.title = 'Surface density'

            if self.var2 == 'Unit_price':
                self.var_label = self.df['Unit_price']
                self.label = 'Cost [\$]'
                self.title = 'Cost'

            if self.var2 == 'Cost/Energy':
                self.var_label = self.df['Unit_price'] / (0.5 * self.df['Capacitance'] * self.df['Rated_volt'] ** 2)
                self.label = 'Cost/Energy [$\$ / J$]'
                self.title = 'Cost/Energy'

        if self.component == 'inductor':
            if self.var2 == 'PPC':
                self.var_label = self.df['Inductance'] * self.df['Current_rating']
                self.label = 'PPC [H-A]'
                self.title = 'Power processing capability'


            if self.var2 == 'Volume':
                self.var_label = self.df['Dimension'] * self.df['Height']
                self.label = 'Volume [$mm^3$]'
                self.title = 'Volume'

            if self.var2 == 'PPD':
                self.var_label = self.df['Inductance'] * self.df['Current_rating'] / (self.df['Dimension'] * self.df['Height'])
                self.label = 'PPD [$H-A / mm^3$]'
                self.title = 'Power-processing density'

            if self.var2 == 'Energy_density':
                self.var_label =.5*self.df['Inductance'] * self.df['Sat_current']**2 / (self.df['Dimension'] * self.df['Height'])
                self.label = 'Energy density [$H-A^2 / mm^3$]'
                self.title = 'Energy density'

            if self.var2 == 'Power loss':
                self.var_label =self.df['DCR'] * self.df['Current_rating']**2
                self.label = 'Power loss [$Ω-A^2$]'
                self.title = 'Power loss'

            if self.var2 == 'Efficiency':
                self.var_label =(self.df['DCR'] * self.df['Current_rating']**2)/ (self.df['Inductance'] * self.df['Sat_current']**2)
                self.label = 'Loss per Unit Energy [$\dfrac{W}{J}$]'
                self.title = 'Efficiency'
                self.outlier_label = self.df[['DCR','Current_rating','Inductance','Sat_current']]

            if self.var2 == 'Unit_price':
                self.var_label = self.df['Unit_price']
                self.label = 'Cost [\$]'
                self.title = 'Cost'

            if self.var2 == 'Cost/PPC':
                self.var_label = self.df['Unit_price'] / (self.df['Inductance'] * self.df['Current_rating'])
                self.label = 'Cost/PPC [\$/H-A]'
                self.title = 'Cost/PPC'

            if self.var2 == 'Cost/Energy':
                self.var_label = self.df['Unit_price'] / (.5*self.df['Inductance'] * self.df['Current_rating']**2 )
                self.label = 'Cost/Energy [$\$/H-A^2$]'
                self.title = 'Cost/Energy'

            if self.var2 == 'Energy':
                self.var_label = (.5 * self.df['Inductance'] * self.df['Current_sat'] ** 2)
                self.label = 'Energy [$J$]'
                self.title = 'Energy'



# This is the main function used for plotting FOMs. Takes as input the df of data, the x-variable, the two y-variables
# used in the FOM, the component type, and the option.
# Gets the variables used for the plotting visualization from Plotting_parameters class, overlays the data and the
# trendline. Currently, have to manually change between the linear and quadratic plotting, but should update this in the
# future. If want to compare/plot on same plot the data/trends for inductors AND capacitors, this code is included
# in the inductor section and just need to uncomment--should update this in the future.
# At the end, also shows code for getting the residual values of the best-fit line/RSE, which is where we want to see
# how far the trendline is from all the datapoints.
def two_prod_FOM_scatter(fet_df, x_var, y_var1, y_var2, component = 'fet',option='option2'):
    if component == 'capacitor':
        ready_df = fet_df.copy()
        ready_df = ready_df.drop_duplicates(['Mfr_part_no'])
        cmap = plt.get_cmap('hsv')
        cmap = cmap.reversed()
        new_cmap = truncate_colormap(cmap, .72, 1)
        z1 = np.log10(ready_df.loc[:, 'Unit_price'])
        min_, max_ = -2, 2


        x_var_obj = Plotting_parameters(ready_df, x_var, component='capacitor')
        y_var_obj = Plotting_parameters(ready_df, y_var1, component='capacitor')
        x_var_obj.get_params()
        y_var_obj.get_params()

        x = []
        y = []
        X = []
        Y = []
        # print(m)
        x.extend(x_var_obj.var_label)
        y.extend(y_var_obj.var_label)
        X.extend(np.log10(x_var_obj.var_label))
        Y.extend(np.log10(y_var_obj.var_label))
        theta = np.polyfit(X, Y, 2)
        X.sort()
        y_list = [10 ** (theta[0] * g ** 2 + theta[1]*g**1 + theta[2]) for g in X]
        x_list = [10 ** g for g in X]
        plt.plot(x_list, y_list, c='black', linestyle='solid')
        plt.scatter(x,
                    y,
                    marker='.', s=20, edgecolors='grey', linewidths=0.2, c=z1, cmap=new_cmap,
                    alpha=0.8)
        plt.clim(min_, max_)
        plt.xscale('log')
        plt.yscale('log')

        plt.xlabel(x_var_obj.label)
        plt.ylabel(y_var_obj.label)

        plt.title('{:.1f}'.format(theta[0]) + 'x^2' + ' + ' + '{:.1f}'.format(theta[1]) + 'x' + ' + ' + '{:.1f}'.format(
            theta[2]))
        plt.title(y_var_obj.title)

        cbar = plt.colorbar()
        cbar_tick_labels = [round(10 ** float(k.get_text().replace("−", "-")), 1) for k in cbar.ax.get_yticklabels()]
        cbar.ax.set_yticklabels(cbar_tick_labels)  # vertically oriented colorbar
        cbar.ax.set_ylabel('Cost [\$]', rotation=90)
        plt.show()
        print('done')

    if component == 'inductor':
        fet_df = fet_df.drop_duplicates('Mfr_part_no')
        # fet_df['Power_loss'] = fet_df['DCR'] * fet_df['Current_rating'] ** 2
        fet_df['Energy'] = .5 * fet_df['Inductance'] * fet_df['Sat_current'] ** 2
        fet_df['Efficiency'] = (fet_df['DCR'] * fet_df['Current_rating'] ** 2) / (.5*
                    fet_df['Inductance'] * fet_df['Sat_current']**2)
        # fig, ax = plt.subplots()
        ready_df = fet_df.copy()
        cmap = plt.get_cmap('hsv')
        cmap = cmap.reversed()
        new_cmap = truncate_colormap(cmap, .72, 1)
        z1 = np.log10(ready_df.loc[:, 'Unit_price'])
        min_, max_ = -3, 1.398


        x_var_obj = Plotting_parameters(ready_df, x_var,'inductor')
        y_var_obj = Plotting_parameters(ready_df, y_var1,'inductor')
        x_var_obj.get_params()
        y_var_obj.get_params()

        x = []
        y = []
        X = []
        Y = []
        # print(m)
        x.extend(x_var_obj.var_label)
        y.extend(y_var_obj.var_label)
        X.extend(np.log10(x_var_obj.var_label))
        Y.extend(np.log10(y_var_obj.var_label))
        theta = np.polyfit(X, Y, 2)
        X.sort()
        y_list = [10 ** (theta[0] * g ** 2 + theta[1] * g + theta[2]) for g in X]
        x.sort()
        plt.plot(x, y_list, c='black', linestyle='solid')
        plt.scatter(x_var_obj.var_label,
                    y_var_obj.var_label,
                    marker='.', s=25, edgecolors='lightgrey', linewidths=0.05, c=z1, cmap=new_cmap,
                    alpha=0.8)
        plt.clim(min_, max_)
        plt.xscale('log')
        plt.yscale('log')


        plt.xlabel(x_var_obj.label)
        plt.ylabel(y_var_obj.label)


        # plt.title('{:.1f}'.format(theta[0]) + 'x^2' + ' + ' + '{:.1f}'.format(theta[1]) + 'x' + ' + ' + '{:.1f}'.format(
        #     theta[2]))
        # plt.title(y_var_obj.title)

        plt.grid(color='lightgrey', linewidth=1, alpha=0.4)

        cbar = plt.colorbar()
        cbar_tick_labels = [round(10 ** float(k.get_text().replace("−", "-")), 3) for k in cbar.ax.get_yticklabels()]
        cbar_tick_labels[1::2] = ['' for x in cbar_tick_labels[1::2]]
        cbar.ax.set_yticklabels(cbar_tick_labels)  # vertically oriented colorbar
        cbar.ax.set_ylabel('Cost [\$]', rotation=90)
        plt.show()

        # to plot capacitor data on top:
        # file_name = 'csv_files/capacitor_data_extendedOpt.csv'
        # cap_df = csv_to_df(file_name)
        # data_dims = ['Unit_price', 'Rated_volt', 'Capacitance', 'Size', 'Temp_coef', 'Temp_coef_enc', 'Mounting_type',
        #              'Thickness', 'Mfr_part_no']
        # cap_df = cap_df.iloc[:, 1:]
        # cap_df.columns = data_dims
        # cap_df = cap_df[(cap_df['Capacitance'] >= 10 * 10 ** -9) & (cap_df['Rated_volt'] <= 1000)]
        # cap_df['Energy'] = .5 * cap_df['Capacitance'] * cap_df['Rated_volt'] ** 2
        # cap_df['Volume'] = cap_df['Size'] * cap_df['Thickness']
        #
        # var = 'Unit_price'
        # cap_df = outlier_detect(cap_df, var, component='capacitor')

        # ready_df = cap_df.copy()
        # ready_df = ready_df.drop_duplicates(['Mfr_part_no'])
        # cmap = plt.get_cmap('hsv')
        # cmap = cmap.reversed()
        # new_cmap = truncate_colormap(cmap, .72, 1)
        # z1 = np.log10(ready_df.loc[:, 'Unit_price'])
        # min_, max_ = -3, 1.398
        #
        # x_var_obj = Plotting_parameters(ready_df, x_var, component='capacitor')
        # y_var_obj = Plotting_parameters(ready_df, y_var1, component='capacitor')
        # x_var_obj.get_params()
        # y_var_obj.get_params()
        #
        # x = []
        # y = []
        # X = []
        # Y = []
        # # print(m)
        # x.extend(x_var_obj.var_label)
        # y.extend(y_var_obj.var_label)
        # X.extend(np.log10(x_var_obj.var_label))
        # Y.extend(np.log10(y_var_obj.var_label))
        # theta = np.polyfit(X, Y, 2)
        # X.sort()
        # y_list = [10 ** (theta[0] * g ** 2 + theta[1] * g ** 1 + theta[2]) for g in X]
        # x_list = [10 ** g for g in X]
        # plt.plot(x_list, y_list, c='black', linestyle='solid')
        # plt.scatter(x,
        #             y,
        #             marker='X', s=16, edgecolors='lightgrey', linewidths=0.05, c=z1, cmap=new_cmap,
        #             alpha=0.8)
        # plt.clim(min_, max_)

        # si = mlines.Line2D([], [], color='grey', marker='o', ls='', label='Inductors')
        # sic = mlines.Line2D([], [], color='grey', marker='x', ls='', label='Capacitors')
        # legend = plt.legend(handles=[si, sic])

        print('done')


    # get residual values
    ready_df = fet_df[(fet_df['FET_type'] == 'N')]
    ready_df = ready_df[(ready_df['Technology'].isin(['MOSFET']))]
    x = []
    y = []
    X = []
    Y = []
    # print(m)
    x.extend(ready_df.loc[:, x_var])
    y.extend(ready_df.loc[:, y_var1] * ready_df.loc[:, y_var2])
    X.extend(np.log10(ready_df.loc[:, x_var]))
    Y.extend(np.log10(ready_df.loc[:, y_var1] * ready_df.loc[:, y_var2]))
    theta = np.polyfit(X, Y, 2, full=True)
    print('residuals: %s' % theta[1])


    if option == 'option1':
        ready_df = fet_df[(fet_df['FET_type'] == 'N')]
        ready_df = ready_df[(ready_df['Technology'].isin(['MOSFET']))]
        x = []
        y = []
        X = []
        Y = []
        # print(m)
        x.extend(ready_df.loc[:, x_var])
        y.extend(ready_df.loc[:, y_var1] * ready_df.loc[:, y_var2])
        X.extend(np.log10(ready_df.loc[:, x_var]))
        Y.extend(np.log10(ready_df.loc[:, y_var1] * ready_df.loc[:, y_var2]))
        theta = np.polyfit(X, Y, 2)
        X.sort()
        y_list = [10 ** (theta[0]*g**2 + theta[1] * g + theta[2]) for g in X]
        x.sort()
        plt.plot(x, y_list, c='red')
        plt.scatter(ready_df.loc[:, x_var],
                    ready_df.loc[:, y_var1] * ready_df.loc[:, y_var2],
                    marker='.', s=20, c='lightcoral',
                    alpha=0.8)
        plt.xscale('log')
        plt.yscale('log')

        ready_df = fet_df[(fet_df['FET_type'] == 'N')]
        ready_df = ready_df[(ready_df['Technology'].isin(['SiCFET', 'SiC']))]
        x = []
        y = []
        X = []
        Y = []
        # print(m)
        x.extend(ready_df.loc[:, x_var])
        y.extend(ready_df.loc[:, y_var1] * ready_df.loc[:, y_var2])
        X.extend(np.log10(ready_df.loc[:, x_var]))
        Y.extend(np.log10(ready_df.loc[:, y_var1] * ready_df.loc[:, y_var2]))
        theta = np.polyfit(X, Y, 2)
        X.sort()
        y_list = [10 ** (theta[0]*g**2 + theta[1] * g + theta[2]) for g in X]
        x.sort()
        plt.plot(x, y_list, c='blue')
        plt.scatter(ready_df.loc[:, x_var],
                    ready_df.loc[:, y_var1] * ready_df.loc[:, y_var2],
                    marker='x', s=20, c='cornflowerblue',
                    alpha=0.8)
        plt.xscale('log')
        plt.yscale('log')



        ready_df = fet_df[(fet_df['FET_type'] == 'N')]
        ready_df = ready_df[(ready_df['Technology'].isin(['GaNFET']))]
        x = []
        y = []
        X = []
        Y = []
        # print(m)
        x.extend(ready_df.loc[:, x_var])
        y.extend(ready_df.loc[:, y_var1] * ready_df.loc[:, y_var2])
        X.extend(np.log10(ready_df.loc[:, x_var]))
        Y.extend(np.log10(ready_df.loc[:, y_var1] * ready_df.loc[:, y_var2]))
        theta = np.polyfit(X, Y, 2)
        X.sort()
        y_list = [10 ** (theta[0]*g**2 + theta[1] * g + theta[2]) for g in X]
        x.sort()
        plt.plot(x, y_list, c='forestgreen')
        plt.scatter(ready_df.loc[:, x_var],
                    ready_df.loc[:, y_var1] * ready_df.loc[:, y_var2],
                    marker='v', s=30, c='darkseagreen',
                    alpha=0.8)
        plt.xscale('log')
        plt.yscale('log')

    if option=='option2':

        # fig, ax = plt.subplots()
        fet_df = fet_df.drop_duplicates('Mfr_part_no')
        cmap = plt.get_cmap('hsv')
        cmap = cmap.reversed()
        new_cmap = truncate_colormap(cmap, .72, 1)
        ready_df = fet_df[(fet_df['FET_type'] == 'N')]
        z1df = ready_df[(ready_df['Technology'].isin(['MOSFET']))]
        z1 = np.log10(z1df.loc[:, 'Unit_price'])
        z2df = ready_df[(ready_df['Technology'].isin(['SiCFET','SiC']))]
        z2 = np.log10(z2df.loc[:, 'Unit_price'])
        z3df = ready_df[(ready_df['Technology'].isin(['GaNFET']))]
        z3 = np.log10(z3df.loc[:, 'Unit_price'])

        zs = np.concatenate([z1, z2, z3], axis=0)
        min_, max_ = zs.min(), zs.max()

        ready_df = fet_df[(fet_df['FET_type'] == 'N')]
        ready_df = ready_df[(ready_df['Technology'].isin(['MOSFET']))]


        x = []
        y = []
        X = []
        Y = []
        # print(m)
        x.extend(ready_df.loc[:, x_var])
        y.extend(ready_df.loc[:, y_var1] * ready_df.loc[:, y_var2])
        X.extend(np.log10(ready_df.loc[:, x_var]))
        Y.extend(np.log10(ready_df.loc[:, y_var1] * ready_df.loc[:, y_var2]))
        theta = np.polyfit(X, Y, 2)
        X.sort()
        y_list = [10 ** (theta[0] * g ** 2 + theta[1] * g + theta[2]) for g in X]
        x.sort()
        plt.plot(x, y_list, c='black', linestyle='solid')
        plt.scatter(ready_df.loc[:, x_var],
                    ready_df.loc[:, y_var1] * ready_df.loc[:, y_var2],
                    marker='.', s=20, edgecolors='lightgrey', linewidths=0.05, c=z1, cmap=new_cmap,
                    alpha=0.8)
        plt.clim(min_, max_)
        plt.xscale('log')
        plt.yscale('log')

        ready_df = fet_df[(fet_df['FET_type'] == 'N')]
        ready_df = ready_df[(ready_df['Technology'].isin(['SiCFET', 'SiC']))]


        x = []
        y = []
        X = []
        Y = []
        # print(m)
        x.extend(ready_df.loc[:, x_var])
        y.extend(ready_df.loc[:, y_var1] * ready_df.loc[:, y_var2])
        X.extend(np.log10(ready_df.loc[:, x_var]))
        Y.extend(np.log10(ready_df.loc[:, y_var1] * ready_df.loc[:, y_var2]))
        theta = np.polyfit(X, Y, 2)
        X.sort()
        y_list = [10 ** (theta[0] * g ** 2 + theta[1] * g + theta[2]) for g in X]
        x.sort()
        plt.plot(x, y_list, c='black', linestyle='dashed')
        plt.scatter(ready_df.loc[:, x_var],
                    ready_df.loc[:, y_var1] * ready_df.loc[:, y_var2],
                    marker='X', s=20, edgecolors='lightgrey', linewidths=0.05, c=z2, cmap=new_cmap,
                    alpha=0.8)
        plt.clim(min_, max_)

        ready_df = fet_df[(fet_df['FET_type'] == 'N')]
        ready_df = ready_df[(ready_df['Technology'].isin(['GaNFET']))]


        x = []
        y = []
        X = []
        Y = []
        # print(m)
        x.extend(ready_df.loc[:, x_var])
        y.extend(ready_df.loc[:, y_var1] * ready_df.loc[:, y_var2])
        X.extend(np.log10(ready_df.loc[:, x_var]))
        Y.extend(np.log10(ready_df.loc[:, y_var1] * ready_df.loc[:, y_var2]))
        theta = np.polyfit(X, Y, 2)
        X.sort()
        y_list = [10 ** (theta[0] * g ** 2 + theta[1] * g + theta[2]) for g in X]
        x.sort()
        plt.plot(x, y_list, c='black', linestyle='dotted')
        plt.scatter(ready_df.loc[:, x_var],
                    ready_df.loc[:, y_var1] * ready_df.loc[:, y_var2],
                    marker='v', s=20, edgecolors='lightgrey', linewidths=0.05,c=z3, cmap=new_cmap,
                    alpha=0.8)
        plt.clim(min_, max_)

        params = Visualization_parameters(var2 = y_var2)
        params.get_params()

        plt.xlabel('$V_{dss}$ [V]')
        plt.ylabel(params.ylabel)

        si = mlines.Line2D([], [], color='grey', marker='o', ls='', label='Si')
        sic = mlines.Line2D([], [], color='grey', marker='x', ls='', label='SiC')
        gan = mlines.Line2D([], [], color='grey', marker='v', ls='', label='GaN')
        # etc etc
        legend = plt.legend(handles=[si, sic, gan])
        # plt.title('{:.1f}'.format(theta[0]) + 'x^2' + ' + '+'{:.1f}'.format(theta[1]) + 'x' + ' + '+ '{:.1f}'.format(theta[2]))

        pt1 = params.pt1
        pt2 = params.pt2
        pt3 = params.pt3

        plt.grid(color='lightgrey', linewidth=1, alpha=0.4)

        cbar = plt.colorbar()
        cbar_tick_labels = [round(10 ** float(k.get_text().replace("−", "-")), 2) for k in cbar.ax.get_yticklabels()]
        cbar_tick_labels[::2] = ['' for x in cbar_tick_labels[::2]]
        cbar.ax.set_yticklabels(cbar_tick_labels)  # vertically oriented colorbar
        cbar.ax.set_ylabel('Cost [\$]', rotation=90)
        plt.show()


# Create the visual for showing where outliers exist
def outlier_visual():
    file_name = 'csv_files/inductor_data.csv'
    ind_df = csv_to_df(file_name)
    ind_df = ind_df.reset_index()

    attr_list = ['Unit_price', 'Current_rating', 'DCR', 'Dimension', 'Height', 'Inductance', 'Mfr_part_no',
                 'Current_sat']
    ind_df = initial_fet_parse(ind_df, attr_list)

    # Get rid of inductor components without size information
    ind_df = ind_df.dropna(subset=attr_list)
    ind_df = ind_df[ind_df['Inductance'] < 0.1]
    ind_df['Energy'] = (0.5 * ind_df['Inductance'] * (ind_df['Current_sat'] ** 2)).astype(float)
    ind_df['Volume'] = (ind_df['Height'] * (ind_df['Dimension'])).astype(float)
    h = plt.scatter(ind_df['Volume'], ind_df['Energy'], marker='.', s=10, c='blue',edgecolors='grey', linewidths=0.1,
                    alpha=0.8)
    plt.xscale('log')
    plt.yscale('log')
    ind_params_x = Plotting_parameters(ind_df, 'Volume', 'inductor')
    ind_params_x.get_params()
    ind_params_y = Plotting_parameters(ind_df, 'Energy', 'inductor')
    ind_params_y.get_params()
    plt.xlabel(ind_params_x.label)
    plt.ylabel(ind_params_y.label)
    # plt.title(ind_.title)
    circle_rad = 15  # This is the radius, in points
    pntcircle = plt.plot(180, 10 ** -11, 'o',
                         ms=circle_rad * 2, mec='r', mfc='none', mew=2)
    pntlabel = plt.annotate('Outliers', xy=(155, 10 ** -11), xytext=(70, 10 ** -11),
                            textcoords='offset points',
                            color='r', size='large',
                            arrowprops=dict(
                                arrowstyle='simple,tail_width=0.3,head_width=0.8,head_length=0.8',
                                facecolor='r', shrinkB=circle_rad * 1.2)
                            )
    plt.show()

# Plot the data w/ ML models overlaid
def data_and_models():
    # Specifically, start with the RQ FOM. First w/ Vdss as only input, then w/ multiple inputs (Vdss, Cost, Coss).
    # Also get r^2 and MAE for each
    csv_file = 'csv_files/postPreprocessing.csv'

    fet_df = pd.read_csv(csv_file)
    fet_df = fet_df.iloc[:, 1:]
    var = 'Q_g'
    # fet_df = fet_df.drop(fet_df[(fet_df[var] == 0) | (fet_df[var].isna())].index)
    # fet_df = fet_df.drop(fet_df[(fet_df['C_oss'] == 0) | (fet_df['C_oss'].isna())].index)
    # df1 = fet_df[fet_df['C_oss'] * fet_df['R_ds'] < 10 ** -6]
    # df1['C_oss'] = df1['C_oss'] * 10 ** 12
    # df2 = fet_df[fet_df['C_oss'] * fet_df['R_ds'] > 10 ** -6]
    # fet_df = pd.concat([df1, df2])
    # fet_df = fet_df[fet_df['C_oss'] * fet_df['R_ds'] > 7 * 10 ** -1]

    # fet_df = fet_df.drop(fet_df[(fet_df['Q_rr'] == 0) | (fet_df['Q_rr'].isna())].index)
    # df1 = fet_df[fet_df['Q_rr'] < 10 ** -4]
    # df1['Q_rr'] = df1['Q_rr'] * 10 ** 9
    # df2 = fet_df[fet_df['Q_rr'] >= 10 ** -4]
    # fet_df = pd.concat([df1, df2])
    # fet_df[(fet_df['V_dss'] > 100) & (fet_df['R_ds'] * fet_df['Q_rr'] < 1 * 10 ** -1)] = fet_df[
    #                                                                                          (fet_df['V_dss'] > 100) & (
    #                                                                                                      fet_df[
    #                                                                                                          'R_ds'] *
    #                                                                                                      fet_df[
    #                                                                                                          'Q_rr'] < 1 * 10 ** -1)] * 10 ** 3
    # fet_df[(fet_df['V_dss'] > 200) & (fet_df['R_ds'] * fet_df['Q_rr'] < 3 * 10 ** 0)] = fet_df[
    #                                                                                          (fet_df['V_dss'] > 200) & (
    #                                                                                                  fet_df[
    #                                                                                                      'R_ds'] *
    #                                                                                                  fet_df[
    #                                                                                                      'Q_rr'] < 3 * 10 ** 0)] * 10 ** 3

    # filt_df = filt_df.drop(filt_df[(filt_df['C_oss'] == 0) & (filt_df['Q_rr'] == 0)].index).drop_duplicates()
    fet_df['Q_g'] = fet_df['Q_g']*10**9
    fet_df = outlier_detect(fet_df, var)

    # Implement pareto-dominance using all relevant dimensions
    fet_df = pareto_front(fet_df)

    # Train the ML models on this data w/ Vdss as only input, RQ as output
    trained_models = []
    degree = 2

    # rmse_scorer = make_scorer(mean_squared_error, squared=False)
    # Get the X,y for model training
    # fet_df = fet_df[(fet_df['Technology'] == 'SiCFET') | (fet_df['Technology'] == 'SiC')]
    fet_df = fet_df[fet_df['Technology'] == 'MOSFET']
    X = np.array(np.log10(fet_df.loc[:, 'V_dss'])).reshape(-1,1)
    poly = PolynomialFeatures(degree)
    X = poly.fit_transform(X)
    y = np.log10(fet_df.loc[:, 'R_ds'].astype(float) * fet_df.loc[:, 'Q_g'].astype(float))

    scores_df = pd.DataFrame({'scoring_type':['r^2','RMSE','MAE']}).set_index('scoring_type')

    cv = KFold(n_splits=10, random_state=1, shuffle=True)

    model = LinearRegression(fit_intercept=True, normalize=True)
    cv = KFold(n_splits=2, random_state=1, shuffle=True)
    np.mean(cross_val_score(model, X, y, cv=cv, scoring='r2'))

    # new_df = fet_df[(fet_df['Technology'] == 'SiCFET') | (fet_df['Technology'] == 'SiC')]
    # degree = 2
    # X = np.array(np.log10(new_df.loc[:, ['V_dss']]))
    # X = np.repeat(X, 2).reshape(-1, 1)
    # poly = PolynomialFeatures(degree)
    # X = poly.fit_transform(X)
    # y = np.array(np.log10(new_df.loc[:, 'R_ds'].astype(float) * new_df.loc[:, 'Q_rr'].astype(float)))
    # y = np.repeat(y, 2)
    # scores_df = pd.DataFrame({'scoring_type': ['r^2', 'RMSE', 'MAE']}).set_index('scoring_type')
    # cv = KFold(n_splits=10, random_state=1, shuffle=True)
    model = LinearRegression(fit_intercept=True, normalize=True)
    # cv = KFold(n_splits=3, random_state=1, shuffle=True)
    # print(np.mean(cross_val_score(model, X, y, cv=cv, scoring='r2')))
    # print(np.mean(cross_val_score(model, X, y, cv=cv, scoring='neg_mean_absolute_error')))

    model = RandomForestRegressor(min_samples_split = 2, random_state=0)

    # model.fit(X, y)
    # bef_df = before_df[before_df[output_param] != 0]
    # bef_df = bef_df[bef_df[output_param] != np.nan]
    # (X_before, y_before) = X_y_set(bef_df, output_param, attributes, log_x=True, more_inputs=False, pareto=pareto)
    # X_before = preproc(X_before, degree)
    # y_pred = model.predict(X_before)
    # mae_score = mean_absolute_error(y_before, y_pred)

    scores = cross_validate(model, X, y, cv=cv, scoring=['r2', 'neg_mean_absolute_error'], return_train_score=True)
    scores = cross_val_score(model, X, y, cv=cv, scoring='r2')
    scores_df.loc['r^2', 'linear'] = np.mean(scores)
    # scores = cross_val_score(model, X, y, cv=cv, scoring=rmse_scorer)
    # scores_df.loc['RMSE', 'linear'] = np.mean(scores)
    scores = cross_val_score(model, X, y, cv=cv, scoring='neg_mean_absolute_error')
    scores_df.loc['MAE', 'linear'] = np.mean(scores)
    # print('Model: Linear regression')
    # print('Accuracy: %.3f (%.3f)' % (np.mean(scores), np.std(scores)))
    # scores_df.loc[0, 'linear'] = np.mean(scores)
    reg_lin = model.fit(X, y)
    # y_pred = reg_lin.predict(X)
    trained_models.append(reg_lin)


    y_ = []
    T = np.linspace(6,400, 1000)
    for volt in T:
        y_.append(10 ** (reg_lin.predict(poly.fit_transform(np.array(np.log10(volt)).reshape(1, -1)))))

    X = np.log10(fet_df.loc[:, ['V_dss', 'Unit_price']])
    poly = PolynomialFeatures(degree)
    X = poly.fit_transform(X)
    y = np.log10(fet_df.loc[:, 'R_ds'].astype(float) * fet_df.loc[:, 'Q_g'].astype(float))
    scores = cross_val_score(model, X, y, cv=cv, scoring='r2')
    scores_df.loc['r^2', 'linear'] = np.mean(scores)
    # scores = cross_val_score(model, X, y, cv=cv, scoring=rmse_scorer)
    # scores_df.loc['RMSE', 'linear'] = np.mean(scores)
    scores = cross_val_score(model, X, y, cv=cv, scoring='neg_mean_absolute_error')
    scores_df.loc['MAE', 'linear'] = np.mean(scores)
    # print('Model: Linear regression')
    # print('Accuracy: %.3f (%.3f)' % (np.mean(scores), np.std(scores)))
    # scores_df.loc[0, 'linear'] = np.mean(scores)
    reg_lin2 = model.fit(X, y)
    # y_pred = reg_lin.predict(X)
    trained_models.append(reg_lin2)

    # # Now plot the two models on top of the data
    # fet_df = fet_df[fet_df['R_ds'] > 0.005]
    # fet_df = fet_df[fet_df['R_ds'] < 0.05]
    # fet_df = fet_df[fet_df['Unit_price'] > 1]
    # fet_df = fet_df[fet_df['Unit_price'] < 5]



    y_2 = []
    T = np.linspace(6, 400, 1000)
    for volt in T:
        y_2.append(10 ** (reg_lin2.predict(
            poly.fit_transform(np.array([np.log10(volt), np.log10(0.45)]).reshape(1, -1)))))

    fet_df = fet_df[(fet_df['Unit_price'] > 0.2) & (fet_df['Unit_price'] < 0.9)]
    plt.scatter(fet_df.loc[:, 'V_dss'],
                fet_df.loc[:, 'R_ds'] * fet_df.loc[:, 'Q_g'], color='blue', s=0.8)
    plt.plot(T, y_, color='black', linestyle = 'solid',label='single-input', linewidth=.9)
    plt.plot(T, y_2, color='black', linestyle = 'dashed', label='multiple-inputs', linewidth=.9)

    # plt.scatter(X, Y, color='g', s=1.0)
    plt.legend()
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('$V_{dss}$ [V]')
    plt.ylabel('$R_{on} * Q_g$ [Ω-nC]')

    max_volt = 1000
    min_volt = 10.0
    T = np.linspace(min_volt, max_volt, 1000)
    T = np.log10(T)
    model_list = ['RdsQg_product']
    # # model = 'random_forest'
    # fet_reg_models_df = load_models(model_list, file_name)
    # fet_reg_models_df = fet_reg_models_df.reset_index()
    # T_copy = T
    # T = preproc(T.reshape(-1, 1), 1)
    # if (y_var1 == 'Unit_price'):
    #     product_num = 0
    # if (y_var1 == 'Q_g'):
    #     product_num = 1
    # if (y_var1 == 'Q_rr'):
    #     product_num = 2
    RdsCost_y_ = []
    T = np.linspace(min_volt, max_volt, 1000)
    # for volt in T:
    #     # print(volt)
    #     RdsCost_y_.append(10 ** (trained_models[0].predict(
    #         preproc(X).reshape(1, -1), 2))[0])
    # RdsCost_y_ = fet_reg_models_df.loc[product_num, model].predict(np.array([T]))
    y_ = trained_models[0].predict(T)
    attributes = ['V_dss', 'Unit_price']

    # X = df.loc[:, x_var]
    # y = df.loc[:, y_var]
    x_var_params = Visualization_parameters('scatter', 'fet', x_var)
    x_var_params.get_params()
    y_var1_params = Visualization_parameters('scatter', 'fet', y_var1)
    y_var1_params.get_params()
    y_var2_params = Visualization_parameters('scatter', 'fet', y_var2)
    y_var2_params.get_params()

    # xdata = X_test.loc[:, indep_var]
    # ydata = np.array(np.log10(y.loc[X_test.index]))
    plt.scatter(fet_df.loc[:, 'V_dss'],
                fet_df.loc[:, 'R_ds'] * fet_df.loc[:, 'Q_g'], color='g', s=0.8)
    plt.plot(T, y_, color='navy', label=model, linewidth=.9)
    # plt.scatter(X, Y, color='g', s=1.0)
    plt.legend()
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Log[%s]' % x_var_params.label)
    plt.ylabel('Log[%s * %s]' % (y_var1_params.label, y_var2_params.label))
    plt.title('{} * {} vs. Vdss for N-channel FETs, lowest {:0.2f}'.format(y_var1_params.label, y_var2_params.label,
                                                                           percent_keep))
    # plt.legend(['MOSFET','SiCFET','GaNFET'])
    plt.show()
    print('plot made')

    # Logistic regression
    # model_obj = opt_model('logistic')
    # model = LogisticRegression(random_state=0)
    # scores = cross_val_score(model, X, y, cv=cv, scoring='r2')
    # scores_df.loc['r^2', 'rf'] = np.mean(scores)
    # scores = cross_val_score(model, X, y, cv=cv, scoring=rmse_scorer)
    # scores_df.loc['MSE', 'rf'] = np.mean(scores)
    # scores = cross_val_score(model, X, y, cv=cv, scoring='neg_mean_absolute_error')
    # scores_df.loc['MAE', 'rf'] = np.mean(scores)
    # # print('Model: Random forest')
    # # print('Accuracy: %.3f (%.3f)' % (np.mean(scores), np.std(scores)))
    # # scores_df.loc[0, 'rf'] = np.mean(scores)
    # log_rf = model.fit(X, y)
    # trained_models.append(log_rf)

    # # Random forest regression
    # # Create object with the associated model type attributes
    # model_obj = opt_model('random_forest')
    # model = RandomForestRegressor(min_samples_split=model_obj.min_samples_split, random_state=0)
    # scores = cross_val_score(model, X, y, cv=cv, scoring='r2')
    # scores_df.loc['r^2', 'rf'] = np.mean(scores)
    # scores = cross_val_score(model, X, y, cv=cv, scoring=rmse_scorer)
    # scores_df.loc['MSE', 'rf'] = np.mean(scores)
    # scores = cross_val_score(model, X, y, cv=cv, scoring='neg_mean_absolute_error')
    # scores_df.loc['MAE', 'rf'] = np.mean(scores)
    # # print('Model: Random forest')
    # # print('Accuracy: %.3f (%.3f)' % (np.mean(scores), np.std(scores)))
    # # scores_df.loc[0, 'rf'] = np.mean(scores)
    # reg_rf = model.fit(X, y)
    # trained_models.append(reg_rf)
    #
    # # K-nearest-neighbors regression
    # model_obj = opt_model('knn')
    # model = KNeighborsRegressor(n_neighbors=model_obj.n_neighbors, weights=model_obj.weights)
    # scores = cross_val_score(model, X, y, cv=cv, scoring='r2')
    # scores_df.loc['r^2', 'knn'] = np.mean(scores)
    # scores = cross_val_score(model, X, y, cv=cv, scoring='neg_mean_squared_error')
    # scores_df.loc['MSE', 'knn'] = np.mean(scores)
    # scores = cross_val_score(model, X, y, cv=cv, scoring='neg_mean_absolute_error')
    # scores_df.loc['MAE', 'knn'] = np.mean(scores)
    #
    # # print('Model: K-nearest neighbors')
    # # print('Accuracy: %.3f (%.3f)' % (np.mean(scores), np.std(scores)))
    # # scores_df.loc[0, 'knn'] = np.mean(scores)
    # reg_knn = model.fit(X, y)
    # trained_models.append(reg_knn)
    #

    two_prod_FOM_scatter(fet_df, 'V_dss', 'R_ds', var, option='option2')






if __name__ == '__main__':

    # outlier_visual()
    # data_and_models() # Within data and visuals, see instructions for pareto-plotting
    all_fom_plotting(component = 'fet')