'''
    This function looks uses pareto dominance on features we care about to determine the best components used for the regressions.
'''

from csv import reader

# from FOM_relationships import outlier_detect
import matplotlib.pyplot as plt
from csv_conversion import csv_to_mosfet
from fet_regression import *
from fet_price_filtering import outlier_removal
from fet_pdf_scraper import parse_pdf_param
from fet_area_filtering import area_filter, area_training, manual_area_filter, area_filter_gd
from sklearn import preprocessing
from sklearn.model_selection import cross_validate
import pandas as pd
import oapackage
from csv_conversion import df_to_csv, csv_to_df
import seaborn as sn
from fet_visualization import correlation_matrix
from tabulate import tabulate
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import LabelEncoder
from scipy.stats import chi2


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

'''
    Visualize the pareto-dominant datapoints on 2 dimensions
'''
def pareto_visualization(real_vals, optimal_datapoints, var_of_interest, data_dims):
    h = plt.plot(real_vals[0, :], real_vals[var_of_interest, :], '.b', markersize=5, label='Non Pareto-optimal',
                 alpha=0.8)
    hp = plt.plot(optimal_datapoints[0, :], optimal_datapoints[var_of_interest, :], '.r', markersize=7,
                  label='Pareto optimal', alpha=0.6)
    # model_plus_data_vs_Vdss(fet_df, 'V_dss', 'Q_rr', 'R_ds')

    plt.xlabel(data_dims[0], fontsize=16)
    plt.ylabel(data_dims[var_of_interest], fontsize=16)
    plt.title('Pareto-optimized datapoints')

    # plt.xticks([])
    # plt.yticks([])
    _ = plt.legend(loc=1, numpoints=1)
    plt.yscale('log')

'''
    Determine whether or not each value needs to be inverted (depending on whether we want it large or small)
'''
class pareto_var_params(object):
    def __init__(self, component_type, variable):
        self.component_type = component_type
        self.variable = variable

    def get_params(self):
        if self.component_type == 'fet':
            if self.variable == 'V_dss':
                self.inv_val = False
            elif self.variable == 'R_ds':
                self.inv_val = True
            elif self.variable == 'Unit_price':
                self.inv_val = True
            elif self.variable == 'Q_g':
                self.inv_val = True
            elif self.variable == 'Q_rr':
                self.inv_val = True
            elif self.variable == 'FET_type':
                self.inv_val = False
            elif self.variable == 'Technology':
                self.inv_val = False
            elif self.variable == 'Pack_case':
                self.inv_val = True
            elif self.variable == 'C_oss':
                self.inv_val = True
            elif self.variable == 'C_ossp1':
                self.inv_val = True
            elif self.variable == 'Q_rr':
                self.inv_val = True
            elif self.variable == 'I_F':
                self.inv_val = True

        if self.component_type == 'ind':
            if self.variable == 'Unit_price [USD]':
                self.inv_val = True
            elif self.variable == 'Current_Rating [A]':
                self.inv_val = False
            elif self.variable == 'DCR [Ohms]':
                self.inv_val = True
            elif self.variable == 'Dimension':
                self.inv_val = True
            elif self.variable == 'Height':
                self.inv_val = True
            elif self.variable == 'Inductance':
                self.inv_val = False
            elif self.variable == 'Volume [mm^3]':
                self.inv_val = True

        if self.component_type == 'cap':
            if self.variable == 'Unit_price':
                self.inv_val = True
            elif self.variable == 'Rated_volt':
                self.inv_val = False
            elif self.variable == 'Size':
                self.inv_val = True



def pareto_optimize(data_df, data_dims, data_dims_paretoFront, technology, component='fet'):
    if component == 'cap':
        optimal_datapoints = []
        temp_coef_vals = data_df['Temp_coef'].unique()
        for temp_coef_val in temp_coef_vals:
            temp_coef_val_df = data_df[data_df['Temp_coef'] == temp_coef_val]
            cap_vals = temp_coef_val_df['Capacitance'].unique()

            for cap_val in cap_vals:
                cap_val_df = temp_coef_val_df[temp_coef_val_df['Capacitance'] == cap_val]

                noninv_vals = []  # the non-inverted values, used to plot the final values when we want to look at the
                # minimization rather than maximization
                inv_vals = []
                letter_list = list(ascii_lowercase)

                for i in range(len(data_dims)):
                    if data_dims[i] == 'Mfr_part_no' or data_dims[i] == 'Temp_coef' or data_dims[i] == 'Mounting_type':
                        exec('letter_list[i] = cap_val_df[data_dims[i]].to_list()')
                    else:
                        exec('letter_list[i] = cap_val_df[data_dims[i]].astype(float).to_list()')
                    noninv_vals.append(letter_list[i])

                for j in range(len(data_dims_paretoFront)):
                    # create object to know whether to take reciprocal of the variable or not
                    pareto_params = pareto_var_params('cap', data_dims_paretoFront[j])
                    pareto_params.get_params()
                    if pareto_params.inv_val == True:
                        exec('letter_list[j] = np.reciprocal(cap_val_df[data_dims_paretoFront[j]].astype(float).to_list())')
                    else:
                        exec('letter_list[j] = cap_val_df[data_dims_paretoFront[j]].astype(float).to_list()')
                    inv_vals.append(letter_list[j])

                inv_vals = np.array([inv_vals])
                pareto = oapackage.ParetoDoubleLong()
                for ii in range(0, inv_vals.shape[2]):
                    double_vector = []
                    for i in range(len(data_dims_paretoFront)):
                        double_vector.append(inv_vals[0, i, ii])
                    w = oapackage.doubleVector((double_vector))
                    pareto.addvalue(w, ii)

                pareto.show(verbose=1)

                lst = pareto.allindices()
                noninv_vals = np.array(noninv_vals)
                optimal_datapoints.extend(np.transpose(noninv_vals[:, lst]))

        return optimal_datapoints


    if component == 'ind':
        optimal_datapoints = []
        ind_vals = data_df['Inductance [H]'].unique()
        for ind_val in ind_vals:
            ind_val_df = data_df[data_df['Inductance [H]']==ind_val]

            noninv_vals = []  # the non-inverted values, used to plot the final values when we want to look at the
            # minimization rather than maximization
            inv_vals = []
            letter_list = list(ascii_lowercase)

            for i in range(len(data_dims)):
                if data_dims[i] == 'Mfr_part_no':
                    exec('letter_list[i] = ind_val_df[data_dims[i]].to_list()')
                else:
                    exec('letter_list[i] = ind_val_df[data_dims[i]].astype(float).to_list()')
                noninv_vals.append(letter_list[i])

            for j in range(len(data_dims_paretoFront)):
                # create object to know whether to take reciprocal of the variable or not
                pareto_params = pareto_var_params('ind', data_dims_paretoFront[j])
                pareto_params.get_params()
                if pareto_params.inv_val == True:
                    exec('letter_list[j] = np.reciprocal(ind_val_df[data_dims_paretoFront[j]].astype(float).to_list())')
                else:
                    exec('letter_list[j] = ind_val_df[data_dims_paretoFront[j]].astype(float).to_list()')
                inv_vals.append(letter_list[j])

            inv_vals = np.array([inv_vals])
            pareto = oapackage.ParetoDoubleLong()
            for ii in range(0, inv_vals.shape[2]):
                double_vector = []
                for i in range(len(data_dims_paretoFront)):
                    double_vector.append(inv_vals[0, i, ii])
                w = oapackage.doubleVector((double_vector))
                pareto.addvalue(w, ii)

            pareto.show(verbose=1)

            lst = pareto.allindices()
            noninv_vals = np.array(noninv_vals)
            optimal_datapoints.extend(np.transpose(noninv_vals[:, lst]))

        return optimal_datapoints


    if technology == 'MOSFET' or technology == 'GaNFET' or technology == 'SiCFET':
        # determine what the data in data_dims looks like, because that determines how we process it for the pareto
        # front
        noninv_vals = []    # the non-inverted values, used to plot the final values when we want to look at the
                            # minimization rather than maximization
        inv_vals = []
        letter_list = list(ascii_lowercase)

        for col in data_dims_paretoFront:
            if (technology == 'GaNFET') and (col == 'Q_rr'):
                continue
            data_df = data_df[data_df[col].ne(0.0)]

        for i in range(len(data_dims)):
            exec('letter_list[i] = data_df[data_dims[i]].to_list()')
            noninv_vals.append(letter_list[i])

        for j in range(len(data_dims_paretoFront)):
            # create object to know whether to take reciprocal of the variable or not
            if (technology == 'GaNFET') and (data_dims_paretoFront[j] == 'Q_rr'):
                continue
            pareto_params = pareto_var_params('fet',data_dims_paretoFront[j])
            pareto_params.get_params()
            # print(data_dims[j])
            if pareto_params.inv_val == True:
                exec('letter_list[j] = np.reciprocal(data_df[data_dims_paretoFront[j]].to_list())')
            else:
                exec('letter_list[j] = data_df[data_dims_paretoFront[j]].to_list()')
            inv_vals.append(letter_list[j])

        inv_vals = np.array([inv_vals])
        pareto = oapackage.ParetoDoubleLong()
        for ii in range(0, inv_vals.shape[2]):
            double_vector = []
            for i in range(len(data_dims_paretoFront)):
                if (technology == 'GaNFET') and (data_dims_paretoFront[i] == 'Q_rr'):
                    continue
                double_vector.append(inv_vals[0,i,ii])
            w = oapackage.doubleVector((double_vector))
            pareto.addvalue(w, ii)

        pareto.show(verbose=1)

        # code for plotting pareto in 2D. fix data_dims_paretoFront to be 2d and filter the dataframe to have components
        # only within certain price range
        lst = pareto.allindices()
        noninv_vals = np.array(noninv_vals)
        optimal_datapoints = noninv_vals[:, lst]
        var_of_interest = 2

        ###########################
        # h = plt.plot(noninv_vals[0, :].astype(float), noninv_vals[var_of_interest, :].astype(float), '.b', markersize=2,
        #              label='Non Pareto-optimal',
        #              alpha=0.8)
        # hp = plt.plot(optimal_datapoints[0, :].astype(float), optimal_datapoints[var_of_interest, :].astype(float),
        #               '.r', markersize=4,
        #               label='Pareto optimal', alpha=0.6)
        # # model_plus_data_vs_Vdss(fet_df, 'V_dss', 'Q_rr', 'R_ds')
        #
        # plt.xlabel('$V_{dss}$ [V]', fontsize=16)
        # plt.ylabel('$R_{on}$ [Ω]', fontsize=16)
        #
        # plt.xticks(fontsize=12)
        # plt.yticks(fontsize=12)
        # plt.yscale('log')
        # plt.ylim(top=30)
        # _ = plt.legend(loc=4, numpoints=1)
        ###########################

        lst = pareto.allindices()
        noninv_vals = np.array(noninv_vals)
        optimal_datapoints = noninv_vals[:, lst]

        # test with a data_dims length of 2 and plot the results
        var_of_interest = 1
        # pareto_visualization(noninv_vals, optimal_datapoints, var_of_interest, data_dims)

        return optimal_datapoints


'''
    Regenerate pareto-optimized inductor data and/or retrain the desired models on this data
'''
import joblib
def ind_training(data_gen=False, retrain=False, opt_param='Balanced', corr_matrix=False):
    if data_gen:
        file_name = 'csv_files/inductor_data.csv'
        ind_df = csv_to_df(file_name)
        ind_df = ind_df.reset_index()

        attr_list = ['Unit_price', 'Current_rating', 'DCR', 'Dimension', 'Height', 'Inductance','Mfr_part_no', 'Current_sat']
        ind_df = initial_fet_parse(ind_df, attr_list)

        # Get rid of inductor components without size information
        ind_df = ind_df.dropna(subset=attr_list)
        ind_df = ind_df[ind_df['Inductance']<0.1]
        ind_df['Energy'] = (0.5*ind_df['Inductance'] * (ind_df['Current_sat'] ** 2)).astype(float)
        ind_df['Volume'] = (ind_df['Height'] * (ind_df['Dimension'])).astype(float)

        # do pareto optimization on the parameters we care about, as listed in attr_list
        data_dims = ['Unit_price', 'Current_rating', 'DCR', 'Dimension', 'Height', 'Inductance','Mfr_part_no','Current_sat']
        data_dims_paretoFront_dict = {'Power': ['Current_rating', 'DCR'], 'Cost': ['Unit_price', 'Current_rating'], 'Area': ['Dimension', 'Current_rating'], 'Balanced': ['Current_rating', 'DCR', 'Unit_price','Dimension'], 'Balanced2': ['Current_rating', 'DCR', 'Unit_price','Dimension']}
        data_dims_paretoFront = data_dims_paretoFront_dict[opt_param]

        opt_ind_points = pareto_optimize(ind_df, data_dims, data_dims_paretoFront, technology=np.nan, component='ind')
        pareto_opt_df = pd.DataFrame(opt_ind_points,
                                     columns=attr_list)
        pareto_opt_df[['Unit_price', 'Current_rating','DCR','Dimension','Height','Inductance', 'Current_sat']] = pareto_opt_df[['Unit_price', 'Current_rating','DCR','Dimension','Height','Inductance','Current_sat']].apply(pd.to_numeric)
        pareto_opt_df['Energy'] = (pareto_opt_df['Inductance'] * (pareto_opt_df['Current_rating'] ** 2)).astype(float)
        pareto_opt_df['Volume'] = (pareto_opt_df['Height'] * (pareto_opt_df['Dimension'])).astype(float)


        # first plot area vs. energy
        # first filter the df's to include just current ratings around 30
        plot_real_df = ind_df[ind_df['Current_rating'] == 1]
        # plot_real_df = plot_real_df[plot_real_df['Current_rating'] < 35]
        plot_pareto_df = pareto_opt_df[pareto_opt_df['Current_rating'] == 1]
        # plot_pareto_df = plot_pareto_df[plot_pareto_df['Current_rating'] < 35]
        # plot_real_df = ind_df
        # plot_pareto_df = pareto_opt_df

        file_name = '../mosfet_data/joblib_files/inductor_data_plotting_example_vsEnergy'
        # reg_score_and_dump(pareto_opt_df, 'Volume', ['Inductance', 'Current_rating'], file_name, pareto=False, chained=False)
        h = plt.plot((plot_real_df['Energy']), (plot_real_df['Volume']), '.b', markersize=1,
                     label='Non Pareto-optimal',
                     alpha=0.8)
        hp = plt.plot((plot_pareto_df['Energy']), (plot_pareto_df['Volume']), '.r', markersize=1.5,
                      label='Pareto optimal', alpha=0.8)
        file_name = '../mosfet_data/joblib_files/inductor_data_plotting_example_vsEnergy_Volume'
        model = joblib.load(file_name + '.joblib')[0]
        y_ = []
        T = np.linspace(plot_pareto_df['Inductance'].min(), plot_pareto_df['Inductance'].max(), 1000)

        x_=[]
        for value in T:
            x_.append((value * 1 ** 2))
            y_.append(10**(
                model.predict(preproc(np.array([(
                    np.log10((value)),
                    np.log10(1))]).reshape(1, -1), 1))[0]))  # Imax=1

        plt.plot(x_, y_, color='navy', label=model, linewidth=.9)
        plt.xlabel('Energy [J]', fontsize=12)
        plt.ylabel('Volume [$mm^3$]', fontsize=12)
        # plt.title('Pareto-optimized Inductor Datapoints')

        # plt.xticks([])
        # plt.yticks([])
        plt.xscale('log')
        plt.yscale('log')
        _ = plt.legend(loc=2, numpoints=1)
        # model_plus_data_vs_Vdss(fet_df, 'V_dss', 'Q_rr', 'R_ds')

        if corr_matrix:
            attr_list = ['Unit_price', 'Current_rating', 'DCR', 'Dimension', 'Height', 'Inductance', 'Volume',
                         'Volume*DCR','Cost*DCR', 'Energy', 'Inductance*DCR', '1/DCR','L/DCR','L/Cost','L/Volume','L*Imax^2/DCR',
                         'L*Imax^2/Cost', 'L*Imax^2/Volume','L*Imax^2/Area']
            correlation_matrix(pareto_opt_df, attr_list, 'ind')
        file_name = 'csv_files/inductor_data_' + str(opt_param) + 'Opt.csv'
        df_to_csv(pareto_opt_df, 'csv_files/inductor_data_wSatCurrent.csv', 'ind')

        try:
            f = open(file_name, 'r+')
            f.truncate(0)  # need '0' when using r+
            df_to_csv(pareto_opt_df, file_name, 'ind')
            f.close()
        except:
            df_to_csv(pareto_opt_df, file_name, 'ind')

    # load the data we want to use
    file_name = 'csv_files/inductor_data_' + str(opt_param) +'Opt.csv'
    ind_df = csv_to_df(file_name)
    data_dims = ['Unit_price', 'Current_rating', 'DCR', 'Dimension', 'Height', 'Inductance','Mfr_part_no']
    ind_df = ind_df.iloc[:, 1:]
    ind_df.columns = data_dims
    ind_df[['Unit_price', 'Current_rating', 'DCR', 'Dimension', 'Height', 'Inductance']] = ind_df[['Unit_price', 'Current_rating', 'DCR', 'Dimension', 'Height', 'Inductance']].astype(float)
    ind_df['Energy'] = (ind_df['Inductance'] * (ind_df['Current_rating'] ** 2)).astype(float)
    ind_df['Current_rating_inv'] = (1 / (ind_df['Current_rating'])).astype(float)
    ind_df['DCR_Inductance_product'] = (ind_df['DCR'] * ind_df['Inductance']).astype(float)
    ind_df['DCR_Cost_product'] = (ind_df['DCR'] * ind_df['Unit_price']).astype(float)
    ind_df['Dimension_Energy_product'] = (ind_df['Dimension'] * ind_df['Energy']).astype(float)
    ind_df['Dimension_Cost_product'] = (ind_df['Dimension'] * ind_df['Unit_price']).astype(float)
    ind_df['Energy_Cost_product'] = (ind_df['Energy'] * ind_df['Unit_price']).astype(float)
    ind_df['Dimension_Inductance_product'] = (ind_df['Inductance'] * ind_df['Dimension']).astype(float)
    ind_df['Inductance_Cost_product'] = (ind_df['Inductance'] * ind_df['Unit_price']).astype(float)
    ind_df['DCR_Cost_productOfIR'] = (ind_df['DCR'] * ind_df['Unit_price']).astype(float)


    if retrain:
        # train on the pareto-optimized data
        output_param_dict = {'DCR': ['Inductance', 'Current_rating', 'Unit_price', 'Dimension', 'Height'],
                             'Unit_price': ['Inductance', 'Current_rating', 'DCR', 'Dimension', 'Height'],
                             'Dimension': ['Inductance', 'Current_rating', 'DCR', 'Unit_price']}

        output_param_dict = {'DCR_Inductance_product': ['Inductance', 'Current_rating_inv'], 'DCR_Cost_product': [
            'Inductance', 'DCR'], 'Dimension': ['Energy', 'Unit_price'],'Energy':['Unit_price', 'Current_rating'],
                             'Inductance_Cost_product': ['Inductance', 'DCR'], 'Dimension_Energy_product': ['Energy', 'Unit_price'],
                             'Energy_Cost_product': ['Unit_price', 'Current_rating'],'DCR_Cost_productOfIR': ['Current_rating','DCR']}

        # output_param_dict = {'DCR_Inductance_product': ['Inductance', 'Current_rating_inv'], 'Inductance_Cost_product': [
        #     'Inductance', 'DCR'], 'Dimension_Energy_product': ['Energy', 'Unit_price'],'Dimension_Cost_product': ['Energy', 'Unit_price'], 'Dimension': ['Energy', 'Unit_price'], 'Energy': ['Unit_price', 'Current_rating'],
        #                      'Energy_Cost_product': ['Inductance', 'DCR'], 'Dimension_Inductance_product': ['Energy', 'Unit_price']}

        file_name = '../mosfet_data/joblib_files/inductor_models_' + str(opt_param) + 'Opt'
        # reg_score_and_dump(ind_df, 'Dimension', ['Inductance', 'Current_rating', 'DCR', 'Unit_price'], file_name, pareto=False, chained=False)
        # reg_score_and_dump(ind_df, 'Energy', ['Unit_price','Current_rating'], file_name, pareto=False, chained=False)
        # reg_score_and_dump(ind_df, 'Energy_Cost_product', ['Unit_price','Current_rating'], file_name, pareto=False, chained=False)

        output_param_dict = {'DCR': ['Inductance', 'Current_rating', 'Unit_price'],
                             'Unit_price': ['Inductance', 'Current_rating', 'DCR'],
                             'Dimension': ['Inductance', 'Current_rating', 'DCR', 'Unit_price']}

        for output_param, attributes in output_param_dict.items():
            reg_score_and_dump(ind_df, output_param, attributes, file_name, pareto=False, chained=False)
            # reg_score_and_dump_chained(ind_df,  'ind', file_name, component_type = 'ind')
    print('done')

def cap_training(data_gen=True, retrain=False, opt_param='Balanced', corr_matrix=False):
    if data_gen:
        # Send to the reg_score_and_dump_cat() where all chained training occurs (will find the function call farther down,
        # after preparing the data for training).
        # First train the main page data. Have Area, Cap @ 0 Vdc, Vrated as inputs, and cost as output.
        # For these models, train with temp. coefficients in Class II: X5R, X7R, X8R, X7S, and Y5V
        # Also use label encoding on the temp. coefficients, filter based on likely used values, and take pareto front.
        # Put the trained models on the file: str(file_name) + '_' + output_param + '.joblib', file_name = csv_files/capacitor_data.csv,
        # and output param for caps = 'Unit_price'.
        # Following that, train on pdf parameter data. Have Area, Vrated, Cap. calculated, and Vdc as inputs, Cap @ 0Vdc as the output


        extended = False # keep extended = False
        if extended:
            file_name = 'csv_files/capacitor_data_class2_extended.csv'
            cap_df = csv_to_df(file_name)
            cap_df = cap_df.reset_index()
            cap_df = cap_df.iloc[:, 2:]
            cap_df.columns = ['Mfr_part_no', 'Unit_price', 'Stock', 'Supplier', 'Mfr', 'Min_qty', 'Series',
                              'Product_status', 'Capacitance', 'Tolerance','Rated_volt', 'Temp_coef','Op_temp','Features',
                              'Ratings','Applications','Mounting_type','Pack_case',
                              'Size', 'Height','Thickness','Lead_spacing','Lead_style']

            # print(cap_df.columns)
            attr_list = ['Mfr_part_no', 'Unit_price', 'Mfr', 'Series', 'Capacitance', 'Rated_volt', 'Temp_coef', 'Size',
                         'Thickness']
            cap_df = initial_fet_parse(cap_df, attr_list)

            encoder = LabelEncoder()
            encoder.fit(cap_df['Temp_coef'])
            # le_name_mapping = dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))
            # print(le_name_mapping)
            cap_df['Temp_coef_enc'] = encoder.transform(cap_df['Temp_coef'])

            cap_df['Volume'] = (cap_df['Size'] * (cap_df['Thickness'])).astype(float)
            cap1 = cap_df[cap_df['Volume'] < 20]
            cap1 = cap1[cap1['Volume'] > 3]
            cap1['Energy'] = .5 * (cap1['Capacitance'] * ((cap1['Rated_volt'])).astype(float)) ** 2
            cap1 = cap1[cap1['Energy'] < 10 ** -9]

            # Get rid of inductor components without information
            cap_df = cap_df.dropna(subset=attr_list)
            # cap_df = cap_df[cap_df['Inductance']<0.1] # need something like this? check max and min

            if corr_matrix:
                corrMatrix = corr_df[
                    ['Unit_price', 'Capacitance', 'Rated_volt', 'Size', 'Thickness', 'Temp_coef_enc', '1/Capacitance',
                     '1/Rated_volt', '1/Size', '1/Thickness']].corr()
                sn.heatmap(corrMatrix, annot=True)
                sn.set(font_scale=1.5)
                plt.title('Correlation of Capacitor Parameters')
                plt.xticks(rotation=45)
                plt.show()

            # do pareto optimization on the parameters we care about, as listed in attr_list
            data_dims = ['Unit_price', 'Rated_volt', 'Capacitance', 'Size', 'Temp_coef', 'Temp_coef_enc', 'Mounting_type', 'Thickness','Mfr_part_no']
            data_dims_paretoFront_dict = {'Balanced': ['Rated_volt', 'Unit_price'],
                                          'Balanced2': ['Rated_volt', 'Unit_price']}
            data_dims_paretoFront = data_dims_paretoFront_dict[opt_param]

            opt_cap_points = pareto_optimize(cap_df, data_dims, data_dims_paretoFront, technology=np.nan, component='cap')
            pareto_opt_df = pd.DataFrame(opt_cap_points,
                                         columns=data_dims)
            pareto_opt_df[['Rated_volt', 'Capacitance', 'Unit_price', 'Size']] = pareto_opt_df[
                ['Rated_volt', 'Capacitance', 'Unit_price', 'Size']].apply(pd.to_numeric)
            # pareto_opt_df['Energy'] = (pareto_opt_df['Inductance'] * (pareto_opt_df['Current_rating'] ** 2)).astype(float)
            # pareto_opt_df['Volume'] = (pareto_opt_df['Height'] * (pareto_opt_df['Dimension'])).astype(float)

            print('done')





        file_name = 'csv_files/capacitor_data.csv'
        cap_df = csv_to_df(file_name)
        # cap_df = cap_df.reset_index()
        # cap_df = cap_df.iloc[:, 2:]
        # cap_df.columns = ['Mfr_part_no', 'Unit_price', 'Mfr', 'Series', 'Capacitance', 'Rated_volt', 'Temp_coef',
        #                   'Size', 'Thickness']
        # # print(cap_df.columns)
        attr_list = ['Mfr_part_no', 'Unit_price', 'Mfr', 'Series', 'Capacitance', 'Rated_volt', 'Temp_coef', 'Size',
                     'Thickness']
        # cap_df = initial_fet_parse(cap_df, attr_list)

        encoder = LabelEncoder()
        encoder.fit(cap_df['Temp_coef'])
        # le_name_mapping = dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))
        # print(le_name_mapping)
        cap_df['Temp_coef_enc'] = encoder.transform(cap_df['Temp_coef'])

        cap_df['Volume'] = (cap_df['Size'] * (cap_df['Thickness'])).astype(float)
        cap1 = cap_df[cap_df['Volume'] < 20]
        cap1 = cap1[cap1['Volume'] > 3]
        cap1['Energy'] = .5 * (cap1['Capacitance'] * ((cap1['Rated_volt'])).astype(float)) ** 2
        cap1 = cap1[cap1['Energy'] < 10 ** -9]

        # Get rid of inductor components without information
        cap_df = cap_df.dropna(subset=attr_list)
        #cap_df = cap_df[cap_df['Inductance']<0.1] # need something like this? check max and min

        if corr_matrix:
            corrMatrix = corr_df[
                ['Unit_price', 'Capacitance', 'Rated_volt', 'Size', 'Thickness', 'Temp_coef_enc', '1/Capacitance',
                 '1/Rated_volt', '1/Size', '1/Thickness']].corr()
            sn.heatmap(corrMatrix, annot=True)
            sn.set(font_scale=1.5)
            plt.title('Correlation of Capacitor Parameters')
            plt.xticks(rotation=45)
            plt.show()

        # do pareto optimization on the parameters we care about, as listed in attr_list
        data_dims = ['Unit_price', 'Rated_volt', 'Capacitance', 'Size', 'Temp_coef','Temp_coef_enc','Thickness']
        data_dims_paretoFront_dict = {'Balanced': ['Rated_volt', 'Unit_price'], 'Balanced2': ['Rated_volt', 'Unit_price']}
        data_dims_paretoFront = data_dims_paretoFront_dict[opt_param]

        opt_cap_points = pareto_optimize(cap_df, data_dims, data_dims_paretoFront, technology=np.nan, component='cap')
        pareto_opt_df = pd.DataFrame(opt_cap_points,
                                     columns=data_dims)
        pareto_opt_df = pareto_opt_df.replace('nan', np.NaN)
        len(pareto_opt_df.dropna(subset=['Size']))
        pareto_opt_df[['Rated_volt', 'Capacitance', 'Unit_price','Size','Thickness']] = pareto_opt_df[['Rated_volt', 'Capacitance', 'Unit_price','Size','Thickness']].apply(pd.to_numeric)
        # pareto_opt_df['Energy'] = (pareto_opt_df['Inductance'] * (pareto_opt_df['Current_rating'] ** 2)).astype(float)
        # pareto_opt_df['Volume'] = (pareto_opt_df['Height'] * (pareto_opt_df['Dimension'])).astype(float)
        pareto_opt_df['log_Unit_price'] = np.log10(pareto_opt_df['Unit_price'])

        # HERE is where the cap data is sent to be trained. Have chained output but currently only predicting cost as an
        # output for main page, and only cap. @ 0Vdc for pdf params

        reg_score_and_dump_cat(cap_df, training_params='cap_main_page_params', component = 'cap')

        # reg_score_and_dump_cat(cap_df, training_params='cap_area_initialization', component = 'cap')

        # Prepare the cap pdf param data, then send for training
        file_name = 'datasheet_graph_info/capacitor_pdf_data_cleaned.csv'
        cap_df = csv_to_df(file_name)

        reg_score_and_dump_cat(cap_df, training_params='cap_pdf_params', component = 'cap')

        ### Can ignore everything after this.

        dfile_name = 'joblib_files/capacitor_models'
        cap_f = cap_df[cap_df['Capacitance'] >= 10 ** -9]
        reg_score_and_dump_chained(cap_df, 'cap', file_name, component_type='cap')



        plt1 = sn.scatterplot(data=pareto_opt_df, x='Capacitance', y='Rated_volt', hue='Size', legend='brief', s=150)
        plt.yscale('log')
        plt.xscale('log')
        plt.xticks(fontsize=40)
        plt.yticks(fontsize=40)
        plt1.set_xlabel('Capacitance [F]', fontsize=40)
        plt1.set_ylabel('$V_{rated}$ [V]', fontsize=40)
        plt.legend(title='Size [$mm^2$]', fontsize=40, title_fontsize=50)

        pareto_opt_df['log_Size'] = np.log10(pareto_opt_df['Size'])

        plt1 = sn.scatterplot(data=pareto_opt_df, x='Rated_volt', y='Unit_price', hue='log_Size', s=150, alpha=0.3,
                              palette='inferno')
        handles, lables = plt1.get_legend_handles_labels()
        for h in handles:
            h.set_sizes([400])
        plt.yscale('log')
        plt.xscale('log')
        plt.xticks(fontsize=40)
        plt.yticks(fontsize=40)
        plt1.set_xlabel('$V_{rated}$ [V]', fontsize=40)
        plt1.set_ylabel('Cost [$]', fontsize=40)
        plt.legend(handles, lables, title='log10[Size [$mm^2$]]', fontsize=40, title_fontsize=50)


        plt1 = sn.scatterplot(data=pareto_opt_df, x='Capacitance', y='Size', hue='Unit_price', s=150)
        plt.yscale('log')
        plt.xscale('log')
        plt.xticks(fontsize=40)
        plt.yticks(fontsize=40)
        plt1.set_xlabel('Capacitance [F]', fontsize=40)
        plt1.set_ylabel('Size [$mm^2$]', fontsize=40)
        plt.legend(title='Cost [$]', fontsize=40, title_fontsize=50)


        # show a plot of pareto-optimized points
        # first filter the df's to include just voltage ratings around 50, and filter to only show one of the temp_coefs, then plot size vs. capacitance showing the pareto front overlaid with all
        plot_real_df = cap_df[cap_df['Rated_volt'] == 50]
        plot_real_df = plot_real_df[plot_real_df['Temp_coef'] =='X7R']
        # plot_real_df = plot_real_df[plot_real_df['Size'] ==2.5]

        plot_pareto_df = pareto_opt_df[pareto_opt_df['Rated_volt'] == 50]
        plot_pareto_df = plot_pareto_df[plot_pareto_df['Temp_coef'] =='X7R']
        # plot_pareto_df = plot_pareto_df[plot_pareto_df['Size'] ==2.5]

        # plot_real_df = ind_df
        # plot_pareto_df = pareto_opt_df

        file_name = '../mosfet_data/joblib_files/capacitor_data_plotting_example_SizevsCapacitance'
        # reg_score_and_dump(pareto_opt_df, 'Volume', ['Inductance', 'Current_rating'], file_name, pareto=False, chained=False)
        h = plt.plot((plot_real_df['Capacitance']), (plot_real_df['Size']), '.b', markersize=1,
                     label='Non Pareto-optimal',
                     alpha=0.8)
        hp = plt.plot((plot_pareto_df['Capacitance']), (plot_pareto_df['Size']), '.r', markersize=1.5,
                      label='Pareto optimal', alpha=0.8)

        #when have trained models, use this
        # file_name = '../mosfet_data/joblib_files/capacitor_data_plotting_example_SizevsEnergy'
        # model = joblib.load(file_name + '.joblib')[0]
        # y_ = []
        # T = np.linspace(plot_pareto_df['Capacitance'].min(), plot_pareto_df['Capacitance'].max(), 1000)
        #
        # x_=[]
        # for value in T:
        #     x_.append((value * 1 ** 2))
        #     y_.append(10**(
        #         model.predict(preproc(np.array([(
        #             np.log10((value)),
        #             np.log10(1))]).reshape(1, -1), 1))[0]))  # Imax=1
        #
        # plt.plot(x_, y_, color='navy', label=model, linewidth=.9)
        plt.xlabel('Capacitance [F]', fontsize=12)
        plt.ylabel('Size [$mm^2$]', fontsize=12)
        # plt.title('Pareto-optimized Inductor Datapoints')

        # plt.xticks([])
        # plt.yticks([])
        plt.xscale('log')
        plt.yscale('log')
        _ = plt.legend(loc=2, numpoints=1)
        # model_plus_data_vs_Vdss(fet_df, 'V_dss', 'Q_rr', 'R_ds')

        file_name = 'csv_files/capacitor_data_' + str(opt_param) + 'Opt.csv'
        try:
            f = open(file_name, 'r+')
            f.truncate(0)  # need '0' when using r+
            df_to_csv(pareto_opt_df, file_name, 'ind')
            f.close()
        except:
            df_to_csv(pareto_opt_df, file_name, 'ind')

    # load the data we want to use
    file_name = 'csv_files/capacitor_data_' + str(opt_param) +'Opt.csv'
    cap_df = csv_to_df(file_name)
    data_dims = ['Unit_price', 'Rated_volt', 'Capacitance', 'Size', 'Temp_coef','Temp_coef_enc']
    cap_df = cap_df.iloc[:, 1:]
    cap_df.columns = data_dims
    # ind_df[['Unit_price', 'Current_rating', 'DCR', 'Dimension', 'Height', 'Inductance']] = ind_df[['Unit_price', 'Current_rating', 'DCR', 'Dimension', 'Height', 'Inductance']].astype(float)
    # ind_df['Energy'] = (ind_df['Inductance'] * (ind_df['Current_rating'] ** 2)).astype(float)
    # ind_df['Current_rating_inv'] = (1 / (ind_df['Current_rating'])).astype(float)
    # ind_df['DCR_Inductance_product'] = (ind_df['DCR'] * ind_df['Inductance']).astype(float)
    # ind_df['DCR_Cost_product'] = (ind_df['DCR'] * ind_df['Unit_price']).astype(float)
    # ind_df['Dimension_Energy_product'] = (ind_df['Dimension'] * ind_df['Energy']).astype(float)
    # ind_df['Dimension_Cost_product'] = (ind_df['Dimension'] * ind_df['Unit_price']).astype(float)
    # ind_df['Energy_Cost_product'] = (ind_df['Energy'] * ind_df['Unit_price']).astype(float)
    # ind_df['Dimension_Inductance_product'] = (ind_df['Inductance'] * ind_df['Dimension']).astype(float)
    # ind_df['Inductance_Cost_product'] = (ind_df['Inductance'] * ind_df['Unit_price']).astype(float)
    # ind_df['DCR_Cost_productOfIR'] = (ind_df['DCR'] * ind_df['Unit_price']).astype(float)


    if retrain:
        dfile_name = 'joblib_files/capacitor_models'
        cap_f = cap_df[cap_df['Capacitance']>=10**-9]
        reg_score_and_dump_chained(cap_df, 'cap', file_name, component_type='cap')

        # train on the pareto-optimized data
        output_param_dict = {'DCR': ['Inductance', 'Current_rating', 'Unit_price', 'Dimension', 'Height'],
                             'Unit_price': ['Inductance', 'Current_rating', 'DCR', 'Dimension', 'Height'],
                             'Dimension': ['Inductance', 'Current_rating', 'DCR', 'Unit_price']}

        output_param_dict = {'DCR_Inductance_product': ['Inductance', 'Current_rating_inv'], 'DCR_Cost_product': [
            'Inductance', 'DCR'], 'Dimension': ['Energy', 'Unit_price'],'Energy':['Unit_price', 'Current_rating'],
                             'Inductance_Cost_product': ['Inductance', 'DCR'], 'Dimension_Energy_product': ['Energy', 'Unit_price'],
                             'Energy_Cost_product': ['Unit_price', 'Current_rating'],'DCR_Cost_productOfIR': ['Current_rating','DCR']}

        # output_param_dict = {'DCR_Inductance_product': ['Inductance', 'Current_rating_inv'], 'Inductance_Cost_product': [
        #     'Inductance', 'DCR'], 'Dimension_Energy_product': ['Energy', 'Unit_price'],'Dimension_Cost_product': ['Energy', 'Unit_price'], 'Dimension': ['Energy', 'Unit_price'], 'Energy': ['Unit_price', 'Current_rating'],
        #                      'Energy_Cost_product': ['Inductance', 'DCR'], 'Dimension_Inductance_product': ['Energy', 'Unit_price']}

        file_name = '../mosfet_data/joblib_files/inductor_models_' + str(opt_param) + 'Opt'
        # reg_score_and_dump(ind_df, 'Dimension', ['Inductance', 'Current_rating', 'DCR', 'Unit_price'], file_name, pareto=False, chained=False)
        reg_score_and_dump(ind_df, 'Energy', ['Unit_price','Current_rating'], file_name, pareto=False, chained=False)
        reg_score_and_dump(ind_df, 'Energy_Cost_product', ['Unit_price','Current_rating'], file_name, pareto=False, chained=False)

        for output_param, attributes in output_param_dict.items():
            reg_score_and_dump_chained(cap_df,  'ind', file_name, component_type = 'ind')
    print('done')

    # can visualize the models with the data
    # file_name = 'inductor_models_noPDF'
    # model_plus_data_vs_Vdss(ind_df, 'Unit_price', 'Dimension', 'Unity', 'linear', file_name=file_name, component='ind')

from fet_visualization import histogram_df
def gd_training(data_gen=False, retrain=False, opt_param='Balanced', corr_matrix=False):
    if data_gen:
        file_name = 'csv_files/gate_driver_data.csv'
        gd_df = csv_to_df(file_name)
        gd_df = gd_df.reset_index()
        gd_df = gd_df.iloc[:, 2:]
        gd_df.columns = ['Mfr_part_no', 'Unit_price', 'Mfr', 'Driven_config', 'Channel_type', 'Num_drivers',
                         'Gate_type', 'Supply_volt',
                         'Logic_volt', 'Peak_current', 'Input_type', 'High_side_volt', 'R_f_time', 'Op_temp',
                         'Mount_type', 'Pack_case']
        # print(cap_df.columns)
        # encode the driven config, channel type
        attr_list = ['Mfr_part_no', 'Unit_price', 'Mfr',
                          'Peak_current']
        gd_df = initial_fet_parse(gd_df, attr_list)
        gd_df = area_filter_gd(gd_df)
        # gd_df = gd_df[gd_df['Pack_case'].type() == float]
        gd_df = gd_df.dropna(subset=attr_list)
        # histogram_df(gd_df, 'gd', 'Pack_case')
        plot_df = gd_df['Num_drivers'].sort_values()
        plt.scatter(gd_df.loc[:, 'Unit_price'],
                    gd_df.loc[:, 'Pack_case'],
                    s=10,
                    alpha=0.8)

        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('Log[%s]' % 'Area [$mm^2$]')
        plt.ylabel('Log[%s]' % ('Cost [$\$$]'))
        plt.title('Gate driver FOM relationships')
        plt.show()

        histogram_df(gd_df, 'gd', 'Peak_current_source')

        encoder = LabelEncoder()
        encoder.fit(cap_df['Temp_coef'])
        # le_name_mapping = dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))
        # print(le_name_mapping)
        cap_df['Temp_coef_enc'] = encoder.transform(cap_df['Temp_coef'])

        # Get rid of inductor components without information
        cap_df = cap_df.dropna(subset=attr_list)
        #cap_df = cap_df[cap_df['Inductance']<0.1] # need something like this? check max and min

        if corr_matrix:
            corrMatrix = corr_df[
                ['Unit_price', 'Capacitance', 'Rated_volt', 'Size', 'Thickness', 'Temp_coef_enc', '1/Capacitance',
                 '1/Rated_volt', '1/Size', '1/Thickness']].corr()
            sn.heatmap(corrMatrix, annot=True)
            sn.set(font_scale=1.5)
            plt.title('Correlation of Capacitor Parameters')
            plt.xticks(rotation=45)
            plt.show()

        # do pareto optimization on the parameters we care about, as listed in attr_list
        data_dims = ['Unit_price', 'Rated_volt', 'Capacitance', 'Size', 'Temp_coef','Temp_coef_enc']
        data_dims_paretoFront_dict = {'Balanced': ['Rated_volt', 'Unit_price','Size'], 'Balanced2': ['Rated_volt', 'Unit_price','Size']}
        data_dims_paretoFront = data_dims_paretoFront_dict[opt_param]

        opt_cap_points = pareto_optimize(cap_df, data_dims, data_dims_paretoFront, technology=np.nan, component='cap')
        pareto_opt_df = pd.DataFrame(opt_cap_points,
                                     columns=data_dims)
        pareto_opt_df[['Rated_volt', 'Capacitance', 'Unit_price','Size']] = pareto_opt_df[['Rated_volt', 'Capacitance', 'Unit_price','Size']].apply(pd.to_numeric)
        # pareto_opt_df['Energy'] = (pareto_opt_df['Inductance'] * (pareto_opt_df['Current_rating'] ** 2)).astype(float)
        # pareto_opt_df['Volume'] = (pareto_opt_df['Height'] * (pareto_opt_df['Dimension'])).astype(float)


        # show a plot of pareto-optimized points
        # first filter the df's to include just voltage ratings around 50, and filter to only show one of the temp_coefs, then plot size vs. capacitance showing the pareto front overlaid with all
        plot_real_df = cap_df[cap_df['Rated_volt'] == 50]
        plot_real_df = plot_real_df[plot_real_df['Temp_coef'] =='X7R']
        # plot_real_df = plot_real_df[plot_real_df['Size'] ==2.5]

        plot_pareto_df = pareto_opt_df[pareto_opt_df['Rated_volt'] == 50]
        plot_pareto_df = plot_pareto_df[plot_pareto_df['Temp_coef'] =='X7R']
        # plot_pareto_df = plot_pareto_df[plot_pareto_df['Size'] ==2.5]

        # plot_real_df = ind_df
        # plot_pareto_df = pareto_opt_df

        file_name = '../mosfet_data/joblib_files/capacitor_data_plotting_example_SizevsCapacitance'
        # reg_score_and_dump(pareto_opt_df, 'Volume', ['Inductance', 'Current_rating'], file_name, pareto=False, chained=False)
        h = plt.plot((plot_real_df['Capacitance']), (plot_real_df['Size']), '.b', markersize=1,
                     label='Non Pareto-optimal',
                     alpha=0.8)
        hp = plt.plot((plot_pareto_df['Capacitance']), (plot_pareto_df['Size']), '.r', markersize=1.5,
                      label='Pareto optimal', alpha=0.8)

        #when have trained models, use this
        # file_name = '../mosfet_data/joblib_files/capacitor_data_plotting_example_SizevsEnergy'
        # model = joblib.load(file_name + '.joblib')[0]
        # y_ = []
        # T = np.linspace(plot_pareto_df['Capacitance'].min(), plot_pareto_df['Capacitance'].max(), 1000)
        #
        # x_=[]
        # for value in T:
        #     x_.append((value * 1 ** 2))
        #     y_.append(10**(
        #         model.predict(preproc(np.array([(
        #             np.log10((value)),
        #             np.log10(1))]).reshape(1, -1), 1))[0]))  # Imax=1
        #
        # plt.plot(x_, y_, color='navy', label=model, linewidth=.9)
        plt.xlabel('Capacitance [F]', fontsize=12)
        plt.ylabel('Size [$mm^2$]', fontsize=12)
        # plt.title('Pareto-optimized Inductor Datapoints')

        # plt.xticks([])
        # plt.yticks([])
        plt.xscale('log')
        plt.yscale('log')
        _ = plt.legend(loc=2, numpoints=1)
        # model_plus_data_vs_Vdss(fet_df, 'V_dss', 'Q_rr', 'R_ds')

        file_name = 'csv_files/capacitor_data_' + str(opt_param) + 'Opt.csv'
        try:
            f = open(file_name, 'r+')
            f.truncate(0)  # need '0' when using r+
            df_to_csv(pareto_opt_df, file_name, 'ind')
            f.close()
        except:
            df_to_csv(pareto_opt_df, file_name, 'ind')

    # load the data we want to use
    file_name = 'csv_files/capacitor_data_' + str(opt_param) +'Opt.csv'
    cap_df = csv_to_df(file_name)
    data_dims = ['Unit_price', 'Rated_volt', 'Capacitance', 'Size', 'Temp_coef','Temp_coef_enc']
    cap_df = cap_df.iloc[:, 1:]
    cap_df.columns = data_dims
    # ind_df[['Unit_price', 'Current_rating', 'DCR', 'Dimension', 'Height', 'Inductance']] = ind_df[['Unit_price', 'Current_rating', 'DCR', 'Dimension', 'Height', 'Inductance']].astype(float)
    # ind_df['Energy'] = (ind_df['Inductance'] * (ind_df['Current_rating'] ** 2)).astype(float)
    # ind_df['Current_rating_inv'] = (1 / (ind_df['Current_rating'])).astype(float)
    # ind_df['DCR_Inductance_product'] = (ind_df['DCR'] * ind_df['Inductance']).astype(float)
    # ind_df['DCR_Cost_product'] = (ind_df['DCR'] * ind_df['Unit_price']).astype(float)

def pareto_plotting(technology):
    file_name = '../mosfet_data/csv_files/' + str(technology) + '_data_Full.csv'
    df_full = csv_to_df(file_name)
    df_full = df_full.iloc[:, 2:]
    df_full.columns = ['Mfr_part_no', 'Unit_price', 'Mfr', 'Series', 'FET_type', 'Technology', 'V_dss', 'I_d', 'V_drive',
                      'R_ds', 'V_thresh', 'Q_g', 'V_gs', 'Input_cap', 'P_diss', 'Op_temp', 'Mount_type', 'Supp_pack',
                      'Pack_case', 'Q_rr','C_oss']

    file_name = '../mosfet_data/csv_files/' + str(technology) + '_data_Pareto.csv'
    df_pareto = csv_to_df(file_name)
    df_pareto = df_pareto.iloc[:, 1:]
    df_pareto.columns = ['V_dss', 'Unit_price', 'R_ds', 'Q_g', 'FET_type', 'Technology', 'Pack_case', 'Mfr_part_no','Q_rr','C_oss']


    # first plot area vs. energy
    # first filter the df's to include just current ratings around 30
    plot_real_df = df_full[df_full['R_ds']==0.003]
    # plot_real_df = plot_real_df[plot_real_df['Current_rating'] < 35]
    plot_pareto_df = df_pareto[df_pareto['R_ds']==0.003]
    # plot_pareto_df = plot_pareto_df[plot_pareto_df['Current_rating'] < 35]
    # plot_real_df = ind_df
    # plot_pareto_df = pareto_opt_df

    file_name = '../mosfet_data/joblib_files/fet_data_plotting_example_vsVdss'
    # reg_score_and_dump(df_pareto, 'RdsCost_product', ['V_dss', 'R_ds'], file_name, pareto=True, chained=False)
    h = plt.plot((plot_real_df['V_dss']), (plot_real_df['R_ds'] * plot_real_df['Unit_price']), '.b', markersize=1,
                 label='Non Pareto-optimal',
                 alpha=0.8)
    hp = plt.plot((plot_pareto_df['V_dss']), (plot_pareto_df['R_ds'] * plot_pareto_df['Unit_price']), '.r',
                  markersize=1.5,
                  label='Pareto optimal', alpha=0.8)
    file_name = '../mosfet_data/joblib_files/fet_data_plotting_example_vsVdss_RdsCost_product'
    model = joblib.load(file_name + '.joblib')[1]
    y_ = []
    T = np.linspace(plot_pareto_df['V_dss'].min(), plot_pareto_df['V_dss'].max(), 1000)

    x_ = []
    for value in T:
        x_.append((value))
        y_.append(10 ** (
            model.predict(preproc(np.array([(
                np.log10((value)),
                np.log10(.003))]).reshape(1, -1), 1))[0]))  # Rds=1.5 (between the labels we grabbed for plotting)

    plt.plot(x_, y_, color='navy', label='Random Forest regression', linewidth=.9)
    plt.xlabel('$V_{dss}$ [V]', fontsize=12)
    plt.ylabel('$R_{on}$*Cost [$\$*Ω$]', fontsize=12)
    # plt.title('Pareto-optimized Inductor Datapoints')

    # plt.xticks([])
    # plt.yticks([])
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(20,200)
    _ = plt.legend(loc=2, numpoints=1)
    # model_plus_data_vs_Vdss(fet_df, 'V_dss', 'Q_rr', 'R_ds')

import csv

'''
    First takes the capacitor pdf data on capacitor_pdf_data.csv and cleans to get all pdf parameters.
    Then takes pickled scraped main-page data on xl_pdf_sets/pickled_data_capacitors_no_sheets, creates object attributes,
    and writes each object to a new line in new csv file csv_files/capacitor_data_class2_extended.csv.
'''
def capacitor_cleaning():
    # ALSO do the following to get a new .csv with capacitor graph data on delta_C vs. Vdc
    # capacitor areas dictionary [mm]
    cap_area_dict = {1812: 46024.8, 1206: 30632.4, '0805 (2012 Metric)': 2*1.25, '0603 (1608 Metric)': 1.55*0.85, '1206 (3216 Metric)': 3.2*1.6,
                     '0402 (1005 Metric)': 1.02*0.5, '1210 (3225 Metric)': 3.2*2.5, '0201 (0603 Metric)': 0.6*0.3, '2220 (5750 Metric)': 5.7*5,
                     '1812 (4532 Metric)': 4.5*3.2, 'Stacked SMD, 2 J-Lead': 0, '2225 (5763 Metric)': 5.7*6.3, '1808 (4520 Metric)': 4.5*2,
                     'SMD, J-Lead': 0, 'Axial': 0, '01005 (0402 Metric)': 0.4*0.2, 'Radial, Disc': 0, '1825 (4564 Metric)': 4.5*6.4,
                     '0508 (1220 Metric)': 1.2*2, '0306 (0816 Metric)': 0.8*1.6, '3025 (7563 Metric)': 7.5*6.3, '0204 (0510 Metric)': 0.5*1,
                     '2211 (5728 Metric)': 5.7*2.8, '0612 (1632 Metric)': 1.6*3.2}
    cap_pdf_df = pd.read_csv("datasheet_graph_info/capacitor_pdf_data.csv")
    cap_pdf_df.head()
    # Convert Mfr part # to all uppercase
    cap_pdf_df['Mfr part no.'] = cap_pdf_df['Mfr part no.'].str.upper()
    # Adjust capacitor case code to be in mm
    cap_pdf_df['Area [mm]'] = cap_pdf_df['Area [size code]'].map(cap_area_dict)
    # Compute new C at Vdc_meas by multiplying deltaC [%] * Capacitance_at_0Vdc [uF]
    cap_pdf_df['Capacitance_at_Vdc_meas [uF]'] = (1 + 0.01 * cap_pdf_df['deltaC_at_Vdc_meas [%]']) * cap_pdf_df[
        'Capacitance_at_0Vdc [uF]']
    # Write new dataframe to a cleaned .csv to be used in addition to the main page data for pdf data-based training
    cap_pdf_df.to_csv('datasheet_graph_info/capacitor_pdf_data_cleaned.csv', mode='w+')

    # Then clean the capacitor dataset

    repickle = False
    if repickle: # if need to open original .csv and pickle the information
        new_lists = []
        with open("xl_pdf_sets/pickled_data_capacitors_no_sheets", "rb") as f:
            while True:
                try:
                    current_id = pickle.load(f)
                    new_lists.append(current_id)
                except EOFError:
                    print('Pickle ends')
                    break

        for datasheet_set in new_lists:
            # First, create an object with all of the known attributes
            component_obj = DigikeyCap_downloaded(datasheet_set)

            # Then write all objects to a .csv to be used for cleaning (see cap_training for intended format)
            members = [attr for attr in dir(component_obj) if not callable(getattr(component_obj, attr)) and not attr.startswith("__")]
            values = [getattr(component_obj, member) for member in members]
            with open('csv_files/capacitor_data_class2_extended.csv', 'a',newline='') as csvfile:
                wr = csv.writer(csvfile, delimiter=',')
                wr.writerow(values)




    file_name = 'csv_files/capacitor_data_class2_extended.csv'
    cap_df = pd.read_csv(file_name, encoding= 'unicode_escape')
    cap_df = cap_df.reset_index()
    cap_df.columns = ['Index','Applications','Capacitance','Datasheet', 'Features','Height','Lead_spacing','Lead_style','Mfr','Mfr_part_no',
                      'Mounting_type','Op_temp','Pack_case','Rated_volt','Ratings','Series','Size','Stock', 'Temp_coef','Thickness',
                      'Tolerance', 'Unit_price'
                       ]

    # print(cap_df.columns)
    attr_list = ['Mfr_part_no', 'Unit_price', 'Mfr', 'Series', 'Capacitance', 'Rated_volt', 'Temp_coef', 'Size',
                 'Thickness']
    cap_df = initial_fet_parse(cap_df, attr_list)

    # Note: use 'Size' as the series column name, not area
    # cap_df = cap_df[cap_df['Pack_case'].isin(cap_area_dict.keys())]
    # cap_df['Area [mm]'] = cap_df['Pack_case'].map(cap_area_dict)


    # Write contents of parsed csv to be later used by the model training function
    cap_df.to_csv('csv_files\capacitor_data.csv', index=False, mode = 'w+')

    print('done')

from fet_data_parsing import prefix_adj
import re

'''
Cleans all data. 
Note: Inductor dataset already cleaned and in inductor_training_updatedAlgorithm.csv, so this currently only cleans
        transistor data.
Input: none
Output: Data file cleaned_dataset.csv with all relevant columns in trainable form (numerical and categorical).
'''
def data_cleaning_full():

    capacitor_cleaning()


    # new csv, with all the info.
    # steps:
    #   1. clean it
    #   2. drop rows if any of main parameters are missing
    #   2. drop duplicate part numbers

    csv_file = 'csv_files/FET_pdf_complete_database.csv' ### This is a good back-up to go to, but now need to add more columns
    csv_file = 'csv_files/FET_pdf_tables_wt_rr_full.csv'
    fet_df = pd.read_csv(csv_file)

    fet_df = fet_df.iloc[:, 1:]


    fet_df.columns = ['Mfr_part_no', 'Datasheet', 'Unit_price', 'Mfr', 'Series', 'FET_type', 'Technology', 'V_dss', 'I_d', 'V_drive', 'R_ds',
              'V_thresh', 'Q_g', 'V_gs', 'Input_cap', 'P_diss', 'Op_temp', 'Mount_type', 'Supp_pack', 'Pack_case',
              'Q_rr', 't_rr', 'I_S', 'diFdt', 'I_F', 'C_oss', 'Vds_meas']

    # first clean the data, drop rows without necessary info
    attr_list = ['Mfr_part_no', 'Unit_price', 'FET_type', 'Technology', 'V_dss', 'R_ds', 'Q_g', 'Pack_case']
    fet_df = initial_fet_parse(fet_df, attr_list)
    fet_df = area_filter(fet_df)
    fet_df = fet_df.replace(0, np.nan)

    fet_df = fet_df.dropna(subset=attr_list)


    for index, row in fet_df.iterrows():
        # Find the Vds value: filter the Vds values in the V_ds_coss column by finding the first number after VDS
        # For testing, first print the entire entry:
        print(fet_df.loc[index, :])
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

        # Replace the Coss values with np.nan if the Coss value is a string
        try:
            if isinstance(fet_df.loc[index, 'C_oss'], str):
                try:
                    fet_df.loc[index, 'C_oss'] = float(fet_df.loc[index, 'C_oss'])
                except:
                    fet_df.loc[index, 'C_oss'] = np.nan

        except:
            fet_df.loc[index, 'C_oss'] = fet_df.loc[index, 'C_oss']


        # Find IF value
    # for index, row in fet_df.iterrows():

        # Find the I_F value: find instance of 'IF', then find unit letter A, then filter out any other characters
        try:
            text = fet_df.loc[index, 'I_F']
            # print(text)
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
            # first see if can find the IF value in what was captured for diF/dt. If still no, then set IF = np.nan
            try:
                text = fet_df.loc[index, 'diFdt']
                # print(text)
                if_match_list = ['IF', 'I F', 'IS', 'I S', 'ISD', 'I SD']
                for word in if_match_list:
                    if word in text:
                        word = word
                        break
                match = re.compile(word).search(text)
                diFdt_break = [text[:match.start()], text[match.start():]]
                match2 = re.compile("A").search(diFdt_break[1])
                A_break = [diFdt_break[1][:match2.start()], diFdt_break[1][match2.start() + 1:]]

                I_F_value = float(re.sub('[^0-9.-]', '', A_break[0]))
                fet_df.loc[index, 'I_F'] = I_F_value

            except:
                I_F_value = np.nan
                fet_df.loc[index, 'I_F'] = I_F_value
        # print(I_F_value)

    # Also find all other 5 pdf parameters:

    # Find diF/dt value
    # for index, row in fet_df.iterrows():

        # Find the diF/dt value: find instance of 'di/dt', then find unit letter A, then filter out any other characters
        try:
            text = fet_df.loc[index, 'diFdt']
            # print(text)
            diFdt_match_list = ['di/dt']
            for word in diFdt_match_list:
                if word in text:
                    word = word
                    break
            match = re.compile(word).search(text)
            diFdt_break = [text[:match.start()], text[match.start():]]
            match2 = re.compile("A").search(diFdt_break[1])
            A_break = [diFdt_break[1][:match2.start()], diFdt_break[1][match2.start() + 1:]]

            diFdt_value = float(re.sub('[^0-9.-]', '', A_break[0]))
            fet_df.loc[index, 'diFdt'] = diFdt_value

        except:
            diFdt_value = np.nan
            fet_df.loc[index, 'diFdt'] = diFdt_value
        print(diFdt_value)

        # now print all attributes once finished
        print(f'qrr: {fet_df.loc[index, "Q_rr"]}')
        print(f'trr: {fet_df.loc[index, "t_rr"]}')
        print(f'IS: {fet_df.loc[index, "I_S"]}')
        print(f'IF: {fet_df.loc[index, "I_F"]}')
        print(f'diFdt: {fet_df.loc[index, "diFdt"]}')
        print(f'Coss: {fet_df.loc[index, "C_oss"]}')
        print(f'Vds: {fet_df.loc[index, "Vds_meas"]}')






            # Clean the technologies so that SiCFET and SiC are together, call it SiCFET for consistency
    fet_df['Technology'] = fet_df['Technology'].replace(['SiC'], 'SiCFET')
    fet_df = fet_df[fet_df['Technology'] != '']
    # fet_df['Q_rr'] = fet_df['Q_rr'] * 10 ** -9
    # fet_df['t_rr'] = fet_df['t_rr'] * 10 ** -9
    # fet_df['C_oss'] = fet_df['C_oss'] * 10 ** -12
    fet_df = fet_df[fet_df['FET_type'] != '']

    # make a copy of this df to use for the test cases
    open("cleaned_fet_dataset2", "w").close()
    with open('cleaned_fet_dataset2', 'wb') as fet_df_file: # 'cleaned_fet_dataset_noQrr' has the just C_oss and Vds_meas not Qrr, impute Qrr
        # Step 3
        pickle.dump(fet_df, fet_df_file)

'''
Train all parameters. For transistors, separate into parameters obtained from main site and parameters obtained from
datasheets.
Step 1. Remove outliers
Step 2. Take pareto front
Step 3. Train models
'''
# from FOM_relationships import outlier_detect
from sklearn.preprocessing import OneHotEncoder
from sklearn.neighbors import DistanceMetric

def train_all():
    # cap_training()

    cleaned_2 = False

    # Determine if want to train main page params or pdf datasheet-based params
    parameter_training = 'datasheet_params'
    # parameter_training = 'main_page_params'
    # parameter_training = 'inds'

    # Train transistor main site parameters using chained regression.
    with open('cleaned_fet_dataset2', 'rb') as optimizer_obj_file:
        fets_database_df = pickle.load(optimizer_obj_file)
    attr_list = ['Mfr_part_no', 'Unit_price', 'FET_type', 'Technology', 'V_dss', 'R_ds', 'Q_g',
                 'Pack_case']
    fet_df = fets_database_df.drop_duplicates(subset=['Mfr_part_no'])
    # Remove outliers for each variable relationship w/ Vdss, then take Pareto front, distinguishing between
    # FET types and technologies
    pareto_opt_full_df = pd.DataFrame(columns=fet_df.columns)
    for FET_type in fet_df['FET_type'].unique():
        for technology in fet_df['Technology'].unique():
            filtered_df = fet_df[(fet_df['FET_type'] == FET_type) & (fet_df['Technology'] == technology)]
            if len(filtered_df) == 0:
                continue
            # Impute the missing values of Qrr and Coss before doing any pareto-optimizations?

            # Remove outliers -- see FOM_relationships.py:
            # Take w.r.t. Vdss
            pareto_opt_df = pd.DataFrame(columns=filtered_df.columns)
            parameter_list = ['Unit_price', 'R_ds', 'Q_g',
                         'Pack_case']
            outlierIndexes = []
            for parameter in parameter_list:
                print(parameter)
                outlierIndexes.extend(outlier_detect(filtered_df, parameter, 'fet', technology))
                # outlier_removed_df = outlier_detect(filtered_df, parameter, 'fet', technology)
                # pareto_opt_df = pd.concat([pareto_opt_df, outlier_removed_df])
            pareto_opt_df = filtered_df.drop(filtered_df.iloc[outlierIndexes].index)
            #
            # for parameter in parameter_list:
            #     outlier_removed_df = outlier_detect(filtered_df, parameter, 'fet', technology)
            #     pareto_opt_df = pd.concat([pareto_opt_df, outlier_removed_df])

            # Take the Pareto front:
            # dimensions we want to keep
            data_dims_keep = ['V_dss', 'Unit_price', 'R_ds', 'Q_g', 'Pack_case', 'Mfr_part_no', 'Q_rr', 'C_oss', 'I_F',
                              'Vds_meas', 'FET_type', 'Technology']
            # dimensions to consider for the pareto front
            data_dims_pareto = ['V_dss', 'Unit_price', 'R_ds', 'Q_g', 'Pack_case']
            pareto_opt_data = pareto_optimize(pareto_opt_df, data_dims_keep, data_dims_pareto,
                                              technology=technology)
            pareto_opt_df = pd.DataFrame(np.transpose(pareto_opt_data),
                                         columns=data_dims_keep)
            data_dims_floats = ['V_dss', 'Unit_price', 'R_ds', 'Q_g', 'Pack_case', 'Q_rr', 'C_oss', 'I_F', 'Vds_meas']
            for dimension in data_dims_floats:
                pareto_opt_df[dimension] = pareto_opt_df[dimension].astype(float)

            # pareto_opt_df = pareto_opt_df.set_index(['Mfr_part_no'])
            # pareto_opt_df = pareto_opt_df.drop_duplicates()

            # combine pareto front data from all FET types and technologies
            pareto_opt_full_df = pd.concat([pareto_opt_full_df, pareto_opt_df])

        # # write this new data to the csv you will want to use from here on out
        # file_name = '../mosfet_data/csv_files/' + str(technology) + '_data_' + str(opt_param) + 'Opt.csv'
        # # new_df = pareto_opt_df[pareto_opt_df['Q_rr'] != '0.0']
        # try:
        #     f = open(file_name, 'r+')
        #     f.truncate(0)  # need '0' when using r+
        #     pareto_opt_df.to_csv(file_name, mode='a', header=True)
        #     f.close()
        # except:
        #     df_to_csv(pareto_opt_df, file_name, 'fet')

    pareto_opt_full_df = pareto_opt_full_df.drop_duplicates(subset=['Mfr_part_no'])
    pareto_opt_full_df = pareto_opt_full_df.reset_index()
    pareto_opt_full_df = pareto_opt_full_df.drop('index', axis=1)

    if parameter_training == 'main_page_params':
        # Now write this cleaned version to save, for component selection
        with open('cleaned_fet_dataset_main_page_params', 'wb') as fet_df_file:
            # Step 3
            pickle.dump(pareto_opt_full_df, fet_df_file)

        # Make FET type and technology categorical variables using one-hot encoding
        # Converting type of columns to category
        pareto_opt_full_df['Technology'] = pareto_opt_full_df['Technology'].astype('category')
        pareto_opt_full_df['FET_type'] = pareto_opt_full_df['FET_type'].astype('category')

        # Assigning numerical values and storing it in another columns
        pareto_opt_full_df['Technology_new'] = pareto_opt_full_df['Technology'].cat.codes
        pareto_opt_full_df['FET_type_new'] = pareto_opt_full_df['FET_type'].cat.codes
        enc = OneHotEncoder(handle_unknown='ignore')

        # passing bridge-types-cat column (label encoded values of bridge_types)

        enc_df = pd.DataFrame(enc.fit_transform(pareto_opt_full_df[['Technology']]).toarray())
        # merge with main df bridge_df on key values
        enc_df = pareto_opt_full_df.join(enc_df)
        enc_df.drop(['Technology'], axis=1, inplace=True)
        enc_df.rename({0: "GaNFET",
                   1: "MOSFET",
                   2: "SiCFET",
                  },
                      axis="columns", inplace=True)
        enc2_df = pd.DataFrame(enc.fit_transform(enc_df[['FET_type']]).toarray())
        # merge with main df bridge_df on key values
        enc_df = enc2_df.join(enc_df)
        enc_df.drop(['FET_type'], axis=1, inplace=True)
        enc_df.rename({0: "N",
                       1: "P",
                       },
                      axis="columns", inplace=True)

        # Train the ML models on the outlier-removed, Pareto-front, encoded dataset
        # First train the area
        # fet_training(retrain_parameter='area', supplied_df=enc_df)

        # Then train the rest of the parameters
        fet_training(retrain_parameter='FOMs', supplied_df=enc_df, training_params='main_page_params')

        # Then train the initialization models for generating Ron from known voltage and cost
        fet_training(retrain_parameter='initialization', supplied_df=enc_df, training_params='R_ds_initialization')

    elif parameter_training == 'datasheet_params':
        # Train transistor datasheet parameters. Similar approach but include some of these parameters in the outlier
        # detection, pareto front, and as outputs of training (see loss_and_physics.py for more)
        # with open('cleaned_fet_dataset_noQrr', 'rb') as optimizer_obj_file:
        #     fets_database_df = pickle.load(optimizer_obj_file)
        with open('cleaned_fet_dataset2', 'rb') as optimizer_obj_file:
            fets_database_df = pickle.load(optimizer_obj_file)
        attr_list = ['Mfr_part_no', 'Unit_price', 'FET_type', 'Technology', 'V_dss', 'R_ds', 'Q_g',
                     'Pack_case']
        fet_df = fets_database_df.drop_duplicates(subset=['Mfr_part_no'])
        pareto_opt_full_df = pd.DataFrame(columns=fet_df.columns)



        for FET_type in fet_df['FET_type'].unique():
            for technology in fet_df['Technology'].unique():
                filtered_df = fet_df[(fet_df['FET_type'] == FET_type) & (fet_df['Technology'] == technology)]
                # Impute the missing values of Qrr and Coss before doing any pareto-optimizations?

                # Right now, drop the components that don't have pdf datasheet parameters available
                # If GaN, set Q_rr = 0 but don't drop
                if technology == 'GaNFET':
                    fets_database_df = filtered_df
                    fets_database_df.loc[fets_database_df.Mfr_part_no == 'EPC2218', 'Vds_meas'] = \
                        fets_database_df.Vds_meas.replace(np.nan, 50, regex=True)
                    fets_database_df.loc[fets_database_df.Mfr_part_no == 'EPC2218', 'C_oss'] = \
                        fets_database_df.C_oss.replace(np.nan, 562e-12, regex=True)
                    fets_database_df.loc[fets_database_df.Mfr_part_no == 'TP65H035WS', 'Vds_meas'] = \
                        fets_database_df.Vds_meas.replace(np.nan, 400, regex=True)
                    fets_database_df.loc[fets_database_df.Mfr_part_no == 'TP65H035WS', 'C_oss'] = \
                        fets_database_df.C_oss.replace(np.nan, 190e-12, regex=True)
                    fets_database_df.loc[fets_database_df.Mfr_part_no == 'TPH3206PS', 'Vds_meas'] = \
                        fets_database_df.Vds_meas.replace(np.nan, 480, regex=True)
                    fets_database_df.loc[fets_database_df.Mfr_part_no == 'TPH3206PS', 'C_oss'] = \
                        fets_database_df.C_oss.replace(np.nan, 44e-12, regex=True)
                    fets_database_df.loc[fets_database_df.Mfr_part_no == 'TP65H050WS', 'Vds_meas'] = \
                        fets_database_df.Vds_meas.replace(np.nan, 400, regex=True)
                    fets_database_df.loc[fets_database_df.Mfr_part_no == 'TP65H050WS', 'C_oss'] = \
                        fets_database_df.C_oss.replace(np.nan, 130e-12, regex=True)
                    fets_database_df.loc[fets_database_df.Mfr_part_no == 'TP65H050WSQA', 'Vds_meas'] = \
                        fets_database_df.Vds_meas.replace(np.nan, 400, regex=True)
                    fets_database_df.loc[fets_database_df.Mfr_part_no == 'TP65H050WSQA', 'C_oss'] = \
                        fets_database_df.C_oss.replace(np.nan, 130e-12, regex=True)
                    fets_database_df.loc[fets_database_df.Mfr_part_no == 'TP65H050G4WS', 'Vds_meas'] = \
                        fets_database_df.Vds_meas.replace(np.nan, 400, regex=True)
                    fets_database_df.loc[fets_database_df.Mfr_part_no == 'TP65H050G4WS', 'C_oss'] = \
                        fets_database_df.C_oss.replace(np.nan, 110e-12, regex=True)
                    fets_database_df.loc[fets_database_df.Mfr_part_no == 'TP65H480G4JSG-TR', 'Vds_meas'] = \
                        fets_database_df.Vds_meas.replace(np.nan, 400, regex=True)
                    fets_database_df.loc[fets_database_df.Mfr_part_no == 'TP65H480G4JSG-TR', 'C_oss'] = \
                        fets_database_df.C_oss.replace(np.nan, 9e-12, regex=True)
                    fets_database_df.loc[fets_database_df.Mfr_part_no == 'TP90H180PS', 'Vds_meas'] = \
                        fets_database_df.Vds_meas.replace(np.nan, 600, regex=True)
                    fets_database_df.loc[fets_database_df.Mfr_part_no == 'TP90H180PS', 'C_oss'] = \
                        fets_database_df.C_oss.replace(np.nan, 41e-12, regex=True)
                    fets_database_df.loc[fets_database_df.Mfr_part_no == 'TPH3206PD', 'Vds_meas'] = \
                        fets_database_df.Vds_meas.replace(np.nan, 480, regex=True)
                    fets_database_df.loc[fets_database_df.Mfr_part_no == 'TPH3206PD', 'C_oss'] = \
                        fets_database_df.C_oss.replace(np.nan, 44e-12, regex=True)
                    fets_database_df.loc[fets_database_df.Mfr_part_no == 'EPC2034', 'Vds_meas'] = \
                        fets_database_df.Vds_meas.replace(np.nan, 100, regex=True)
                    fets_database_df.loc[fets_database_df.Mfr_part_no == 'EPC2034', 'C_oss'] = \
                        fets_database_df.C_oss.replace(np.nan, 641e-12, regex=True)
                    fets_database_df.loc[fets_database_df.Mfr_part_no == 'EPC2034', 'Q_g'] = \
                        fets_database_df.Q_g.replace(np.nan, 11e-9, regex=True)
                    fets_database_df.loc[fets_database_df.Mfr_part_no == 'EPC2034', 'R_ds'] = \
                        fets_database_df.R_ds.replace(np.nan, 8e-3, regex=True)
                    fets_database_df.loc[fets_database_df.Mfr_part_no == 'EPC2034', 'Unit_price'] = \
                        fets_database_df.Unit_price.replace(np.nan, 9.25, regex=True)
                    fets_database_df.loc[fets_database_df.Mfr_part_no == 'TP65H050G4BS', 'Vds_meas'] = \
                        fets_database_df.Vds_meas.replace(np.nan, 400, regex=True)
                    fets_database_df.loc[fets_database_df.Mfr_part_no == 'TP65H050G4BS', 'C_oss'] = \
                        fets_database_df.C_oss.replace(np.nan, 110e-12, regex=True)
                    fets_database_df.loc[fets_database_df.Mfr_part_no == 'EPC2070', 'Vds_meas'] = \
                        fets_database_df.Vds_meas.replace(np.nan, 50, regex=True)
                    fets_database_df.loc[fets_database_df.Mfr_part_no == 'EPC2214', 'Vds_meas'] = \
                        fets_database_df.Vds_meas.replace(np.nan, 40, regex=True)
                    fets_database_df.loc[fets_database_df.Mfr_part_no == 'EPC2030', 'Vds_meas'] = \
                        fets_database_df.Vds_meas.replace(np.nan, 20, regex=True)
                    fets_database_df.loc[fets_database_df.Mfr_part_no == 'EPC2037', 'Vds_meas'] = \
                        fets_database_df.Vds_meas.replace(np.nan, 50, regex=True)
                    fets_database_df.loc[fets_database_df.Mfr_part_no == 'EPC2036', 'Vds_meas'] = \
                        fets_database_df.Vds_meas.replace(np.nan, 50, regex=True)
                    fets_database_df.loc[fets_database_df.Mfr_part_no == 'EPC2051', 'Vds_meas'] = \
                        fets_database_df.Vds_meas.replace(np.nan, 50, regex=True)
                    fets_database_df.loc[fets_database_df.Mfr_part_no == 'EPC2052', 'Vds_meas'] = \
                        fets_database_df.Vds_meas.replace(np.nan, 50, regex=True)
                    fets_database_df.loc[fets_database_df.Mfr_part_no == 'EPC2039', 'Vds_meas'] = \
                        fets_database_df.Vds_meas.replace(np.nan, 40, regex=True)
                    fets_database_df.loc[fets_database_df.Mfr_part_no == 'EPC2007C', 'Vds_meas'] = \
                        fets_database_df.Vds_meas.replace(np.nan, 50, regex=True)
                    fets_database_df.loc[fets_database_df.Mfr_part_no == 'EPC2012C', 'Vds_meas'] = \
                        fets_database_df.Vds_meas.replace(np.nan, 100, regex=True)
                    fets_database_df.loc[fets_database_df.Mfr_part_no == 'EPC2202', 'Vds_meas'] = \
                        fets_database_df.Vds_meas.replace(np.nan, 50, regex=True)
                    fets_database_df.loc[fets_database_df.Mfr_part_no == 'EPC2212', 'Vds_meas'] = \
                        fets_database_df.Vds_meas.replace(np.nan, 50, regex=True)
                    fets_database_df.loc[fets_database_df.Mfr_part_no == 'EPC2045', 'Vds_meas'] = \
                        fets_database_df.Vds_meas.replace(np.nan, 50, regex=True)
                    fets_database_df.loc[fets_database_df.Mfr_part_no == 'EPC2001C', 'Vds_meas'] = \
                        fets_database_df.Vds_meas.replace(np.nan, 50, regex=True)
                    # can keep adding some more



                    filtered_df = fets_database_df


                    filtered_df = filtered_df.drop(filtered_df[filtered_df['C_oss'].isna() | filtered_df['Vds_meas'].isna()].index)
                    filtered_df[['Q_rr','I_F','diFdt', 't_rr']] = 0
                    # filtered_df = filtered_df.dropna()
                else:
                    fets_database_df = filtered_df

                    added_datasheet_info_dict = {
                                                 'BUK7Y3R5-40E,115': {'Q_rr': 29.6, 'I_F': 20, 'diFdt': 100, 't_rr': 25.4},

                                                 'PSMN1R8-40YLC,115': {'Q_rr': 43, 'I_F': 25, 'diFdt': 100, 't_rr': 37},

                                                 'PSMN1R4-40YLDX': {'Q_rr': 61, 'I_F': 25, 'diFdt': 100, 't_rr': 47},
                                                 'PSMN3R9-100YSFX': {'Q_rr': 44, 'I_F': 25, 'diFdt': 100, 't_rr': 43},
                                                 'PSMN057-200B,118': {'Q_rr': 895, 'I_F': 20, 'diFdt': 100, 't_rr': 133},

                                                 'PSMN1R0-30YLC,115': {'Q_rr': 67, 'I_F': 25, 'diFdt': 100, 't_rr': 45},
                                                 'PSMN0R9-30YLDX': {'Q_rr': 67, 'I_F': 25, 'diFdt': 100, 't_rr': 52},

                                                 'PXN012-60QLJ': {'Q_rr': 13, 'I_F': 10, 'diFdt': 100, 't_rr': 22.1},

                                                 'PMV37ENEAR': {'Q_rr': 6, 'I_F': 1.3, 'diFdt': 100, 't_rr': 13},
                                                 'ZXMN2F34FHTA': {'Q_rr': 1.4, 'I_F': 1.65, 'diFdt': 100, 't_rr': 6.5},
                                                 'BUK6D56-60EX': {'Q_rr': 6, 'I_F': 1.4, 'diFdt': 100, 't_rr': 13},


                                                 }
                    for key1, value1 in added_datasheet_info_dict.items():
                        # print(key1)
                        # print(value1)
                        for key2, value2 in value1.items():
                            # print(key2)
                            # print(value2)
                            fets_database_df.loc[fets_database_df.Mfr_part_no == key1, key2] = value2

                    filtered_df = fets_database_df

                    # filtered_df = filtered_df.replace(np.NaN, 0)
                    # filtered_df = filtered_df.drop(filtered_df[filtered_df['C_oss'].isna() & filtered_df['Q_rr'].isna()].index)
                    filtered_df = filtered_df.drop(filtered_df[filtered_df['C_oss'].isna() | filtered_df['Q_rr'].isna() |
                                                            filtered_df['Vds_meas'].isna() | filtered_df['I_F'].isna() |
                                                            filtered_df['t_rr'].isna() | filtered_df[
                                                                'diFdt'].isna()].index)

                    # filtered_df = filtered_df.dropna()

                if len(filtered_df != 0):
                    # Remove outliers -- see FOM_relationships.py:
                    # Take w.r.t. Vdss
                    pareto_opt_df = pd.DataFrame(columns=filtered_df.columns)
                    parameter_list = ['Unit_price', 'R_ds', 'Q_g']
                    outlierIndexes = []
                    for parameter in parameter_list:
                        print(parameter)
                        outlierIndexes.extend(outlier_detect(filtered_df, parameter, 'fet', technology))
                        # outlier_removed_df = outlier_detect(filtered_df, parameter, 'fet', technology)
                        # pareto_opt_df = pd.concat([pareto_opt_df, outlier_removed_df])
                    pareto_opt_df = filtered_df.drop(filtered_df.iloc[outlierIndexes].index)

                    # Take the Pareto front:
                    # dimensions we want to keep
                    data_dims_keep = ['V_dss', 'Unit_price', 'R_ds', 'Q_g', 'Pack_case', 'Mfr_part_no', 'Q_rr', 't_rr',
                                      'I_F','diFdt', 'C_oss', 'Vds_meas', 'FET_type', 'Technology']
                    # dimensions to consider for the pareto front
                    data_dims_pareto = ['V_dss', 'Unit_price', 'R_ds', 'Q_g', 'Pack_case'] # removed C_oss and Q_rr
                    pareto_opt_data = pareto_optimize(pareto_opt_df, data_dims_keep, data_dims_pareto,
                                                      technology=technology)
                    pareto_opt_df = pd.DataFrame(np.transpose(pareto_opt_data),
                                                 columns=data_dims_keep)
                    data_dims_floats = ['V_dss', 'Unit_price', 'R_ds', 'Q_g', 'Pack_case', 'Q_rr', 't_rr', 'I_F', 'diFdt', 'C_oss', 'Vds_meas']
                    for dimension in data_dims_floats:
                        pareto_opt_df[dimension] = pareto_opt_df[dimension].astype(float)

                    # pareto_opt_df = pareto_opt_df.set_index(['Mfr_part_no'])
                    # pareto_opt_df = pareto_opt_df.drop_duplicates()

                    # combine pareto front data from all FET types and technologies
                    pareto_opt_full_df = pd.concat([pareto_opt_full_df, pareto_opt_df])

        pareto_opt_full_df = pareto_opt_full_df.drop_duplicates(subset=['Mfr_part_no'])
        pareto_opt_full_df = pareto_opt_full_df.reset_index()
        pareto_opt_full_df = pareto_opt_full_df.drop('index', axis=1)

        # fet_df = pareto_opt_full_df.copy(deep=False)
        # Add the Coss,0.1 values to predict these
        gamma_eqn_dict = {'Si_low_volt': [-0.0021, -0.251], 'Si_high_volt': [-0.000569, -0.579],
                          'GaN_low_volt': [-0.00062, -0.355],
                          'GaN_high_volt': [-0.000394, -0.353], 'SiC': [0, -0.4509]}
        pareto_opt_full_df = pareto_opt_full_df.dropna(subset=['C_oss', 'Vds_meas'])
        pareto_opt_full_df = pareto_opt_full_df[pareto_opt_full_df['C_oss'] != 0]
        pareto_opt_full_df = pareto_opt_full_df[pareto_opt_full_df['Vds_meas'] != 0]

        for index, row in pareto_opt_full_df.iterrows():
            if pareto_opt_full_df.loc[index, 'Technology'] == 'MOSFET' and pareto_opt_full_df.loc[index, 'V_dss'] <= 200:
                category = 'Si_low_volt'
            elif pareto_opt_full_df.loc[index, 'Technology'] == 'MOSFET' and pareto_opt_full_df.loc[index, 'V_dss'] > 200:
                category = 'Si_high_volt'
            elif pareto_opt_full_df.loc[index, 'Technology'] == 'GaNFET' and pareto_opt_full_df.loc[index, 'V_dss'] <= 100:
                category = 'GaN_low_volt'
            elif pareto_opt_full_df.loc[index, 'Technology'] == 'GaNFET' and pareto_opt_full_df.loc[index, 'V_dss'] <= 200:
                category = 'GaN_high_volt'
            elif pareto_opt_full_df.loc[index, 'Technology'] == 'SiCFET':
                category = 'SiC'

            gamma = gamma_eqn_dict[category][0] * pareto_opt_full_df.loc[index, 'V_dss'] + gamma_eqn_dict[category][1]

            # now compute Coss,0.1
            pareto_opt_full_df.loc[index, 'C_ossp1'] = pareto_opt_full_df.loc[index, 'C_oss'] / (
                        (0.1 * pareto_opt_full_df.loc[index, 'V_dss'] / pareto_opt_full_df.loc[index, 'Vds_meas']) ** gamma)

            # here is where we will use Q_rr_ds, I_F_ds, and t_rr_ds to compute tau_c and tau_rr for every single datapoint
            # def add(a, b, c):
            #     return a + b + c
            #
            # df['add'] = df.apply(lambda row: add(row['A'],
            #                                      row['B'], row['C']), axis=1)

            # (pareto_opt_full_df.loc[index, 'tau_c'], pareto_opt_full_df.loc[index, 'tau_rr']) = pareto_opt_full_df.apply(lambda row: Qrr_est_new(row['Q_rr'], row['t_rr'], row['I_F']), axis=1)

        pareto_opt_full_df = pareto_opt_full_df.dropna(subset = ['C_ossp1'])
        pareto_opt_full_df = pareto_opt_full_df[pareto_opt_full_df['C_oss'] > 0]
        # pareto_opt_full_df = fet_df.copy(deep=False)

        # Now compute and add the tau_c and tau_rr values to the dataset, need to import the function found at the bottom
        # of fet_optimization_chained_wCaps.py
        from fet_optimization_chained_wCaps import Qrr_est_new
        for index, row in pareto_opt_full_df.iterrows():
            try:
                (tau_c, tau_rr) = Qrr_est_new(pareto_opt_full_df.loc[index, 'Q_rr'],
                                                                                pareto_opt_full_df.loc[index, 't_rr'],
                                                                                pareto_opt_full_df.loc[index, 'I_F'])
                pareto_opt_full_df.loc[index, ['tau_c', 'tau_rr']] = tau_c, tau_rr
            except Exception as e:
                print(e)
                pareto_opt_full_df.loc[index, ['tau_c', 'tau_rr']] = np.nan, np.nan

        # Now write this cleaned version to save, for component selection
        with open('cleaned_fet_dataset_pdf_params3', 'wb') as fet_df_file:
            # Step 3
            pickle.dump(pareto_opt_full_df, fet_df_file)

        # Make FET type and technology categorical variables using one-hot encoding
        pareto_opt_full_df['Technology'] = pareto_opt_full_df['Technology'].astype('category')
        pareto_opt_full_df['FET_type'] = pareto_opt_full_df['FET_type'].astype('category')

        # Assigning numerical values and storing it in another columns
        pareto_opt_full_df['Technology_new'] = pareto_opt_full_df['Technology'].cat.codes
        pareto_opt_full_df['FET_type_new'] = pareto_opt_full_df['FET_type'].cat.codes

        enc = OneHotEncoder(handle_unknown='ignore')
        # passing bridge-types-cat column (label encoded values of bridge_types)

        enc_df = pd.DataFrame(enc.fit_transform(pareto_opt_full_df[['Technology']]).toarray())
        pareto_opt_full_df = pareto_opt_full_df.reset_index()
        # merge with main df bridge_df on key values
        enc_df = pareto_opt_full_df.join(enc_df)
        enc_df.drop(['Technology'], axis=1, inplace=True)
        enc_df.rename({0: "GaNFET",
                       1: "MOSFET",
                       2: "SiCFET",
                       },
                      axis="columns", inplace=True)
        enc2_df = pd.DataFrame(enc.fit_transform(enc_df[['FET_type']]).toarray())
        # merge with main df bridge_df on key values
        enc_df = enc2_df.join(enc_df)
        enc_df.drop(['FET_type'], axis=1, inplace=True)
        enc_df.rename({0: "N",
                       1: "P",
                       },
                      axis="columns", inplace=True)

        # Train the ML models on the outlier-removed, Pareto-front, encoded dataset
        # Train the rest of the parameters: C_oss, Vds_meas, Q_rr, I_F
        # fet_training(retrain_parameter='FOMs', supplied_df=enc_df, training_params='Cossp1_plotting')
        fet_training(retrain_parameter='FOMs', supplied_df=enc_df, training_params='pdf_params')

    # # here include correlation matrix capability
    # corr_df = enc_df[enc_df['N'] == 1.0]
    # corr_df = corr_df[corr_df['MOSFET'] == 1.0]
    # attr_list = ['Unit_price', 'V_dss', 'R_ds', 'Q_g', 'C_oss', 'I_F', 'Q_rr']
    # corr_df2 = corr_df[attr_list]
    # for var in attr_list:
    #     corr_df2['log[' + str(var) + ']'] = np.log10(corr_df2[var].astype('float'))
    #
    # corr_df2['1/log[R_ds]'] = 1 / np.log10(corr_df2['R_ds'].astype('float'))
    # corr_df2['1/R_ds'] = 1 / corr_df2['R_ds']
    # corr_df2['log[1/R_ds]'] = np.log10(1 / corr_df2['R_ds'].astype('float'))
    # attr_list = ['log[Unit_price]', 'log[V_dss]', 'log[R_ds]', 'log[Q_g]',
    #              'log[C_oss]', 'log[I_F]', 'log[Q_rr]', 'log[1/R_ds]']
    #
    # temp_df = corr_df2.reset_index()
    # # attr_list = ['V_dss', 'Unit_price', 'R_ds', 'Q_g', 'Q_rr', '1/Rds', 'RdsQg_product', 'RdsCost_product',
    # #              'RdsQrr_product']
    # temp_df = temp_df.astype('float')
    # corrMatrix = temp_df[attr_list].corr()
    # ax = sn.heatmap(corrMatrix, annot=True, annot_kws={"size": 50}, xticklabels=['log[$U$]','log[$V_{dss}$]','log[$R_{on}$]','log[$Q_g$]','log[$C_{oss,0.1}$]','log[$I_F$]','log[$Q_{rr}$]','log[$1/R_{on}$]'],
    #           yticklabels = ['log[$U$]','log[$V_{dss}$]','log[$R_{on}$]','log[$Q_g$]','log[$C_{oss,0.1}$]','log[$I_F$]','log[$Q_{rr}$]','log[$1/R_{on}$]'])
    # # plt.title(corr_params.title)
    # plt.xticks(rotation=25, fontsize=50)
    # plt.yticks(fontsize=50)
    # # ax = sns.heatmap(corrMatrix)
    # # use matplotlib.colorbar.Colorbar object
    # cbar = ax.collections[0].colorbar
    # # here set the labelsize by 20
    # cbar.ax.tick_params(labelsize=50)
    # plt.show()
    #
    #
    # correlation_matrix(temp_df, attr_list, 'fet')
    #
    # print(corr_df2.astype('float').corr())



    # Retrain the inductor parameters
    csv_file = 'csv_files/inductor_training_updatedAlgorithm.csv'
    ind_df = pd.read_csv(csv_file)

    # first clean the data, drop rows without necessary info
    attr_list = ['Mfr_part_no', 'Unit_price', 'fb', 'b', 'Kfe', 'alpha', 'beta', 'Nturns', 'Ac', 'Core Volume m^3']
    # fet_df = initial_fet_parse(fet_df, attr_list)
    # fet_df = fet_df.replace(0, np.nan)
    #
    # fet_df = fet_df.dropna(subset=attr_list)

    # make a copy of ind_df to use later
    ind_df_copy = ind_df.copy(deep=True)

    # Pareto-optimize data on dimensions of interest
    ind_df['Volume [mm^3]'] = ind_df['Length [mm]'] * ind_df['Width [mm]'] * ind_df['Height [mm]']
    ind_df['Area [mm^2]'] = ind_df['Length [mm]'] * ind_df['Width [mm]']

    data_dims = ['Mfr_part_no', 'Unit_price [USD]', 'Area [mm^2]', 'Current_Rating [A]', 'DCR [Ohms]', 'Volume [mm^3]',
                 'Inductance [H]',
                 'Current_Sat [A]', 'fb [Hz]', 'b', 'Kfe', 'Alpha', 'Beta', 'Nturns', 'Ac [m^2]', 'Core Volume m^3']
    data_dims_paretoFront_dict = {
        'Ind_params': ['Unit_price [USD]', 'Current_Rating [A]', 'DCR [Ohms]', 'Volume [mm^3]']}
    pareto_opt_data = pareto_optimize(ind_df, data_dims, data_dims_paretoFront_dict['Ind_params'], technology='ind',
                                      component='ind')
    pareto_opt_df = pd.DataFrame(pareto_opt_data,
                                 columns=data_dims)
    pareto_opt_df = pareto_opt_df.astype(
        {'Unit_price [USD]': float, 'Current_Rating [A]': float, 'DCR [Ohms]': float, 'Volume [mm^3]': float,
         'Inductance [H]': float,
         'Current_Sat [A]': float, 'fb [Hz]': float, 'b': float, 'Kfe': float, 'Alpha': float, 'Beta': float,
         'Nturns': float, 'Ac [m^2]': float, 'Area [mm^2]': float, 'Core Volume m^3': float})

    # Train with ac inductor parameters as the output:

    pareto_opt_df = pareto_opt_df.drop_duplicates(subset='Mfr_part_no')

    df = pareto_opt_df
    reg_score_and_dump_cat(df, training_params='inductor_params', component = 'ind')
    # Then train the initialization models for generating fsw from known  and current rating
    reg_score_and_dump_cat(df, training_params='fsw_initialization', component = 'ind')

    attr_list = ['Unit_price [USD]', 'Inductance [H]', 'Current_Rating [A]', 'DCR [Ohms]', 'Area [mm^2]']
    corr_df2 = df[attr_list]
    for var in attr_list:
        corr_df2['log[' + str(var) + ']'] = np.log10(corr_df2[var].astype('float'))

    attr_list = ['log[Unit_price [USD]]', 'log[Inductance [H]]', 'log[Current_Rating [A]]', 'log[DCR [Ohms]]',
                 'log[Area [mm^2]]']

    temp_df = corr_df2.reset_index()
    # attr_list = ['V_dss', 'Unit_price', 'R_ds', 'Q_g', 'Q_rr', '1/Rds', 'RdsQg_product', 'RdsCost_product',
    #              'RdsQrr_product']
    temp_df = temp_df.astype('float')
    corrMatrix = temp_df[attr_list].corr()
    ax = sn.heatmap(corrMatrix, annot=True, annot_kws={"size": 40},
                    xticklabels=['log[$U$]', 'log[$L$]', 'log[$I_{rated}$]', 'log[$R_{dc}$]', 'log[$A$]'],
                    yticklabels=['log[$U$]', 'log[$L$]', 'log[$I_{rated}$]', 'log[$R_{dc}$]', 'log[$A$]'])
    # plt.title(corr_params.title)
    plt.xticks(rotation=25, fontsize=40)
    plt.yticks(rotation=0, fontsize=40)
    # ax = sns.heatmap(corrMatrix)
    # use matplotlib.colorbar.Colorbar object
    cbar = ax.collections[0].colorbar
    # here set the labelsize by 20
    cbar.ax.tick_params(labelsize=40)
    plt.show()

    correlation_matrix(df, attr_list, 'ind')
    print('done')


'''
    Regenerate pareto-optimized fet data and/or retrain the desired models on this data. 
    Can specify whether we want classification and/or regression models when training, based on if we want to look
    at area or not, because each uses separate files
'''
def fet_training(data_gen_area = False, retrain_parameter='area', data_gen_FOMs = False, technology = 'MOSFET', opt_param = 'Power', supplied_df = None, training_params = 'main_page_params'):
    # if we want to generate new data, do this. otherwise, go straight to working with the pareto-optimized data
    if data_gen_area:

        # first get the data with no Qrr and train for area classification results, then use the data with Qrr to get
        # the other models
        csv_file = 'csv_files/mosfet_data_csv.csv'
        csv_file = 'csv_files/mosfet_data_wmfrPartNo2.csv'

        fet_df = csv_to_mosfet(csv_file)
        fet_df = fet_df.iloc[:, 1:]
        fet_df.columns = ['Mfr_part_no','Unit_price','Mfr','Series','FET_type','Technology','V_dss','I_d','V_drive','R_ds','V_thresh','Q_g','V_gs','Input_cap','P_diss','Op_temp','Mount_type','Supp_pack','Pack_case','Q_rr']
        fet_df['C_oss'] = 0.0
        attr_list = ['Mfr_part_no', 'Unit_price', 'FET_type', 'Technology', 'V_dss', 'R_ds', 'Q_g',
                     'Pack_case'
                     ]

        fet_df = column_fet_parse(initial_fet_parse(fet_df, attr_list), attr_list)

        if technology == 'SiCFET':
            subset_df = fet_df[(fet_df['FET_type'] == 'N')]
            subset_df = subset_df[(subset_df['Technology'].isin(['SiCFET','SiC']))]

            area_df = area_filter(subset_df)
            # No manual additions, just write directly
            # write this data (in area_df) to a csv
            file_name = '../mosfet_data/csv_files/wManual.csv'
            # new_df = pareto_opt_df[pareto_opt_df['Q_rr'] != '0.0']
            try:
                f = open(file_name, 'r+')
                # f.truncate(0)  # need '0' when using r+
                area_df.to_csv(file_name, mode='a', header=True)
                f.close()
            except:
                df_to_csv(area_df, file_name, 'fet')


        subset_df = fet_df[(fet_df['FET_type'] == 'N')]
        subset_df = subset_df[(subset_df['Technology'].isin([technology]))]

        area_df = area_filter(subset_df)


        # Load the additional scraped qrr and coss data from FET_pdf_tables, observe this data
        # csv_file = 'csv_files/FET_pdf_tables_short.csv'
        # add_df = pd.read_csv(csv_file)
        # add_df.columns = ['Mfr_part_no', 'Q_rr', 'C_oss']
        # filtered_df = pd.DataFrame()
        # for part in add_df['Mfr_part_no'].unique():
        #     # print(part)
        #     # print(add_df[add_df['Mfr_part_no'] == part])
        #     filt_df = add_df[add_df['Mfr_part_no'] == part]
        #     filt_df = filt_df.drop(filt_df[filt_df['C_oss'].isna() & filt_df['Q_rr'].isna()].index).drop_duplicates()
        #     filt_df = filt_df.drop(filt_df[(filt_df['C_oss'] == 0) & (filt_df['Q_rr'] == 0)].index).drop_duplicates()
        #     # print(filt_df)
        #     filtered_df = filtered_df.append(filt_df)

        csv_file = 'csv_files/mosfet_data_csv_manual_additions.csv'
        chunk_sizes = []

        with open(csv_file, 'rt', encoding='utf-8') as f:
            csv_reader = reader(f)
            for line in csv_reader:
                if not line:
                    chunk_sizes.append(csv_reader.line_num)
            f.close()

        if technology == 'GaNFET':
            area_df['Q_rr'] = 0.0
            csv_file = 'csv_files/mosfet_data_csv_manual_additions.csv'
            add_fet_df = pd.read_csv(csv_file, nrows = chunk_sizes[0])
            add_fet_df = pd.DataFrame(add_fet_df)
            # add_fet_df = add_fet_df.iloc[:, 1:]
            add_fet_df.columns = ['Mfr_part_no', 'Unit_price', 'Mfr', 'Series', 'FET_type', 'Technology', 'V_dss',
                                  'I_d',
                                  'V_drive', 'R_ds', 'V_thresh', 'Q_g', 'V_gs', 'Input_cap', 'P_diss', 'Op_temp',
                                  'Mount_type',
                                  'Supp_pack', 'Pack_case', 'Q_rr','C_oss']

            attr_list = ['Mfr_part_no', 'Unit_price', 'FET_type', 'Technology', 'V_dss', 'R_ds', 'Q_g',
                         'Pack_case'
                         ]



            add_fet_df = column_fet_parse(initial_fet_parse(add_fet_df, attr_list), attr_list)
            add_area_df = manual_area_filter(add_fet_df)
            # then check and see how the optimization runs for high voltage and cost
            area_df = pd.concat([area_df, add_area_df], ignore_index=True)

            csv_file = 'gan_data_mfr.csv'
            new_df = pd.read_csv(csv_file).drop_duplicates()
            new_df = new_df.iloc[:, 3:]
            new_df.columns = ['Mfr_part_no', 'Unit_price', 'Mfr', 'Series', 'FET_type', 'Technology', 'V_dss',
                                  'I_d',
                                  'V_drive', 'R_ds', 'V_thresh', 'Q_g', 'V_gs', 'Input_cap', 'P_diss', 'Op_temp',
                                  'Mount_type',
                                  'Supp_pack', 'Pack_case', 'Q_rr']
            new_df['C_oss'] = 0.0
            new_df['Q_rr'] = 0.0
            attr_list = ['Mfr_part_no', 'Unit_price', 'FET_type', 'Technology', 'V_dss', 'R_ds', 'Q_g',
                         'Pack_case'
                         ]
            new_df = column_fet_parse(initial_fet_parse(new_df, attr_list), attr_list)
            new_df = area_filter(new_df)
            area_df = pd.concat([area_df, new_df], ignore_index=True)

            csv_file = 'csv_files/mosfet_data_csv_manual_additions.csv'
            new_df = pd.read_csv(csv_file, skiprows=chunk_sizes[1])
            new_df.columns = ['Mfr_part_no', 'C_oss']
            for index1, row1 in area_df.iterrows():
                for index2, row2 in new_df.iterrows():
                    if row1['Mfr_part_no'] == row2['Mfr_part_no']:
                        area_df.loc[index1, 'C_oss'] = row2['C_oss']

            area_df = area_df.drop_duplicates(['Mfr_part_no'])
            # area_df = area_df.dropna(subset=['C_oss'])
            # area_df = pd.concat([area_df, add_area_df], ignore_index=True)
            area_df['C_oss'] = area_df['C_oss']*10**-12
            area_df['Q_g'] = area_df['Q_g'].apply(lambda x: x * 10 ** -8 if x >= 0.1 else x * 1)
            area_df = area_df.drop_duplicates(['Mfr_part_no'])

            # write this data (in area_df) to a csv
            file_name = '../mosfet_data/csv_files/wManual.csv'
            # new_df = pareto_opt_df[pareto_opt_df['Q_rr'] != '0.0']
            try:
                f = open(file_name, 'r+')
                # f.truncate(0)  # need '0' when using r+
                area_df.to_csv(file_name, mode='a', header=True)
                f.close()
            except:
                df_to_csv(area_df, file_name, 'fet')

            print('done')



        elif technology == 'MOSFET':


            csv_file = 'csv_files/mosfet_data_csv_manual_additions.csv'
            add_fet_df = pd.read_csv(csv_file, skiprows = chunk_sizes[0], nrows = chunk_sizes[1]-chunk_sizes[0])
            # then get the other values from the already existing df
            add_fet_df.columns = ['Mfr_part_no', 'Q_rr', 'C_oss']
            for index, row in add_fet_df.iterrows():
                match_indices = area_df.loc[area_df['Mfr_part_no'] == (row['Mfr_part_no'])].index
                for index in match_indices:
                    area_df.loc[index, ['Q_rr', 'C_oss']] = [row['Q_rr'], row['C_oss']]
            # area_df = area_df.dropna()
            for attr in ['Q_rr', 'C_oss']:
                area_df.loc[:, attr] = area_df.loc[:, attr].apply(lambda x: prefix_adj(str(x)))

            area_df['Q_g'] = area_df['Q_g'].apply(lambda x: x * 10 ** -8 if x >= 0.1 else x * 1)
            area_df = area_df.drop_duplicates(['Mfr_part_no'])

            df = parse_pdf_param(csv_file='csv_files/mosfet_data_wpdf_mfr.csv', pdf_param='Q_rr', component_type='fet',
                                 return_only=False)
            attr_list = ['Mfr_part_no', 'Unit_price', 'FET_type', 'Technology', 'V_dss', 'R_ds', 'Q_g', 'Input_cap',
                         'Pack_case']

            mfr_Qrr_df = column_fet_parse(initial_fet_parse(df, attr_list), attr_list).drop_duplicates(['Mfr_part_no'])
            mfr_Qrr_df['Q_rr'] = mfr_Qrr_df['Q_rr'] * 10 ** -9
            mfr_Qrr_df['C_oss'] = 0.0
            subset_df = mfr_Qrr_df[(mfr_Qrr_df['FET_type'] == 'N')]
            subset_df = subset_df[(subset_df['Technology'].isin([technology]))]
            mfr_Qrr_df = area_filter(subset_df)

            # area_df = area_df.set_index(['Mfr_part_no'])
            for index, row in mfr_Qrr_df.iterrows():
                match_indices = area_df.loc[area_df['Mfr_part_no'] == (row['Mfr_part_no'])].index
                for index in match_indices:
                    area_df.loc[index, 'Q_rr'] = row['Q_rr']
            # area_df = pd.concat([area_df,mfr_Qrr_df], axis=1)

            # write this data (in area_df) to a csv
            file_name = '../mosfet_data/csv_files/wManual.csv'
            # new_df = pareto_opt_df[pareto_opt_df['Q_rr'] != '0.0']
            try:
                f = open(file_name, 'r+')
                # f.truncate(0)  # need '0' when using r+
                area_df.to_csv(file_name, mode='a', header=True)
                f.close()
            except:
                df_to_csv(area_df, file_name, 'fet')

        # impute missing values of Qrr and Coss before doing pareto-optimization
        attr_list = ['V_gs']
            # , 'Input_cap', 'I_d'         ]
        area_df = column_fet_parse(initial_fet_parse(area_df, attr_list), attr_list)

        before_df = area_df.copy()
        if technology != 'GaNFET':
            area_df = impute_features(area_df,'Q_rr')

        area_df = impute_features(area_df,'C_oss')

        attributes = ['V_dss', 'R_ds', 'Unit_price']
        output_var = 'C_oss'
        area_df = area_df[area_df[output_var] != np.nan]
        area_df = area_df[area_df[output_var] != 0.0]

        # reg_score_and_dump(area_df, output_var, attributes, 'impute_scoring_test', pareto=True, chained=False, before_df=before_df)


        # pareto-optimize these points
        area_df = area_df.reset_index()
        data_dims = ['V_dss', 'Unit_price', 'R_ds', 'Q_g', 'FET_type', 'Technology', 'Pack_case', 'Mfr_part_no','Q_rr','C_oss']
        if technology == 'GaNFET':
            data_dims_paretoFront_dict = {'Power': ['V_dss', 'R_ds', 'Q_g'], 'Cost': ['V_dss','Unit_price','R_ds'], 'Area': ['V_dss','Pack_case','R_ds'], 'Balanced': ['V_dss', 'Unit_price', 'R_ds', 'Q_g','Pack_case','C_oss'],'Balanced2':['V_dss', 'Unit_price', 'R_ds', 'Q_g', 'Pack_case']}
        else:
            data_dims_paretoFront_dict = {'Power': ['V_dss', 'R_ds', 'Q_g','Q_rr', 'C_oss'], 'Cost': ['V_dss','Unit_price','R_ds'], 'Area': ['V_dss','Pack_case','R_ds'], 'Balanced': ['V_dss', 'Unit_price', 'R_ds', 'Q_g', 'Pack_case','Q_rr','C_oss'], 'Balanced2':['V_dss', 'Unit_price', 'R_ds', 'Q_g', 'Pack_case']}

        # # visualize the area of the Si FETS
        # cost_list = [0.1, 1.0, 10.0, 100.0]
        # color_list = []
        # for i in range(len(area_df.index)):
        #     # print(parsed_df.iloc[i][' Unit_Price'])
        #     price = area_df.iloc[i]['Unit_price']
        #
        #     # print(split_on_letter(text)[0])
        #
        #     if price < cost_list[0]:
        #         color_list.append('#00da00')
        #     elif cost_list[0] <= price < cost_list[1]:
        #         color_list.append('#ebe700')
        #     elif cost_list[1] <= price < cost_list[2]:
        #         color_list.append('#eba400')
        #     elif cost_list[2] <= price < cost_list[3]:
        #         color_list.append('#f50000')
        #     else:
        #         color_list.append('#800000')
        # plt.scatter(area_df['R_ds'], area_df['Pack_case'], s=0.3, alpha=0.8, color=color_list)
        # plt.xscale('log')
        # plt.yscale('log')
        # plt.xlabel('$R_{on}$ [Ω]')
        # plt.ylabel('Package Area [$mm^2$]')
        # plt.title('Package Area vs. On-Resistance of Silicon FETs')
        # plt.show()

        # write the final full data set to the csv
        file_name = '../mosfet_data/csv_files/' + str(technology) + '_data_imputed.csv'
        # new_df = pareto_opt_df[pareto_opt_df['Q_rr'] != '0.0']
        try:
            f = open(file_name, 'r+')
            f.truncate(0)  # need '0' when using r+
            area_df.to_csv(file_name, mode='a', header=True)
            f.close()
        except:
            df_to_csv(area_df, file_name, 'fet')

        # Impute the missing values of Qrr and Coss before doing any pareto-optimizations
        pareto_opt_data = pareto_optimize(area_df, data_dims, data_dims_paretoFront_dict[opt_param], technology=technology)
        pareto_opt_df = pd.DataFrame(np.transpose(pareto_opt_data),
                                     columns=data_dims)
        # here, figure out the best components and add Qrr values for those
        pareto_opt_df['R_ds'] = pareto_opt_df['R_ds'].astype(float)
        pareto_opt_df['V_dss'] = pareto_opt_df['V_dss'].astype(float)
        pareto_opt_df['Unit_price'] = pareto_opt_df['Unit_price'].astype(float)
        # pareto_opt_df = pareto_opt_df.set_index(['Mfr_part_no'])
        # pareto_opt_df = pareto_opt_df.drop_duplicates()




        # write this new data to the csv you will want to use from here on out
        file_name = '../mosfet_data/csv_files/' + str(technology) + '_data_' + str(opt_param) + 'Opt.csv'
        # new_df = pareto_opt_df[pareto_opt_df['Q_rr'] != '0.0']
        try:
            f = open(file_name, 'r+')
            f.truncate(0)  # need '0' when using r+
            pareto_opt_df.to_csv(file_name, mode='a', header=True)
            f.close()
        except:
            df_to_csv(pareto_opt_df, file_name, 'fet')



    # can visualize the data at any point, e.g. with:
    # FOM_scatter(Si_df, 'V_dss', 'R_ds')


    # train models for what we need in the optimization objective function for area
    if retrain_parameter == 'area':
        # load the data we want to use
        if supplied_df is not None:
            df = supplied_df
        else:
            file_name = '../mosfet_data/csv_files/' + str(technology) + '_data_' + str(opt_param) + 'Opt.csv'
            df = csv_to_df(file_name)

            df = df.iloc[:, 1:]

        # first do the classification on discrete area data
        # use the Si_df, which has all the data, from above
        threshold = 1  # Anything that occurs less than this will be removed.
        allowable_vals = df['Pack_case'].value_counts().loc[lambda x: x >= threshold]
        area_df = df[df['Pack_case'].isin(allowable_vals.index)]

        # using the area data, train models to predict area, write those models to a new file
        file_name = '../mosfet_data/joblib_files/' + str(technology) + '_models_' + str(opt_param) + 'Opt'
        area_training(area_df, file_name)
        print('done')
        return

    # can visualize the models with the data
    # model_plus_data_vs_Vdss(fet_df, 'V_dss', 'Q_rr', 'R_ds')

    # in the optimization, will use these models to make predictions

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


    # if data_gen_FOMs:
    #     # new data that includes Qrr values, for everything except area training
    #     df = parse_pdf_param(csv_file='csv_files/mosfet_data_wpdf3.csv', pdf_param='Q_rr', component_type='fet',
    #                          return_only=False)
    #     # first look at all the datapoints to see which ones to add Qrr for
    #     df = parse_pdf_param(csv_file='csv_files/mosfet_data_wmfrPartNo.csv', pdf_param='Q_rr', component_type='fet',
    #                          return_only=False)
    #     # Q_rr already parsed at this point, don't do it again!
    #     attr_list = ['Mfr_part_no', 'Unit_price', 'FET_type', 'Technology', 'V_dss', 'R_ds', 'Q_g', 'Input_cap',
    #                  'Pack_case']
    #     df = column_fet_parse(initial_fet_parse(df, attr_list), attr_list)
    #     subset_df = df[(df['FET_type'] == 'N')]
    #     subset_df = subset_df[(subset_df['Technology'].isin([technology]))]
    #
    #     # combine values from the manual additions with the current values
    #     csv_file = 'csv_files/mosfet_data_csv_manual_additions.csv'
    #
    #     add_fet_df = csv_to_mosfet(csv_file)
    #     # add_fet_df = add_fet_df.iloc[:, 1:]
    #     add_fet_df.columns = ['Mfr_part_no', 'Unit_price', 'Mfr', 'Series', 'FET_type', 'Technology', 'V_dss',
    #                           'I_d',
    #                           'V_drive', 'R_ds', 'V_thresh', 'Q_g', 'V_gs', 'Input_cap', 'P_diss', 'Op_temp',
    #                           'Mount_type',
    #                           'Supp_pack', 'Pack_case', 'Q_rr']
    #
    #     attr_list = ['Mfr_part_no', 'Unit_price', 'FET_type', 'Technology', 'V_dss', 'R_ds', 'Q_g',
    #                  'Q_rr'
    #                  ]
    #
    #     add_fet_df = column_fet_parse(initial_fet_parse(add_fet_df, attr_list), attr_list)
    #     temp_df = pd.concat([subset_df, add_fet_df], ignore_index=True)
    #
    #     temp_df = temp_df[(temp_df['R_ds'] <= 3.0)]
    #     temp_df = temp_df[(temp_df['Q_g'] <= 0.4 * 10 ** -6)]
    #     # set Q_rr to be consistent
    #     temp_df['Q_rr'] = temp_df['Q_rr'].astype(float)
    #     temp_df = temp_df[temp_df['Q_rr'] < 100000]
    #     temp_df['Q_rr'] = 10 ** -9 * temp_df['Q_rr']
    #
    #     # Pareto-optimize the data
    #     temp_df = temp_df.reset_index()
    #     data_dims = ['Mfr_part_no', 'V_dss', 'Unit_price', 'R_ds', 'Q_g', 'Q_rr','FET_type', 'Technology', 'Pack_case']
    #     data_dims_paretoFront = ['V_dss', 'Unit_price', 'R_ds', 'Q_g','Q_rr']
    #
    #     if technology == 'MOSFET':
    #         data_dims_paretoFront = ['V_dss', 'R_ds', 'Q_g','Q_rr']
    #     elif technology == 'GaNFET':
    #         data_dims_paretoFront = ['V_dss', 'R_ds', 'Q_g', 'Q_rr']
    #     pareto_opt_data = pareto_optimize(temp_df, data_dims, data_dims_paretoFront, technology = technology)
    #     pareto_opt_df = pd.DataFrame(np.transpose(pareto_opt_data),
    #                                  columns=['Mfr_part_no', 'V_dss', 'Unit_price', 'R_ds', 'Q_g', 'Q_rr', 'FET_type', 'Technology',
    #                                           'Pack_case'])
    #     pareto_opt_df['R_ds'] = pareto_opt_df['R_ds'].astype(float)
    #     pareto_opt_df['V_dss'] = pareto_opt_df['V_dss'].astype(float)
    #     pareto_opt_df['Unit_price'] = pareto_opt_df['Unit_price'].astype(float)
    #
    #     # write this new data to the csv you will want to use from here on out
    #     file_name = '../mosfet_data/csv_files/' + str(technology) + '_data_PowerOpt.csv'
    #     try:
    #         f = open(file_name, 'r+')
    #         f.truncate(0)
    #         df_to_csv(pareto_opt_df, file_name, 'fet')
    #         f.close()
    #     except:
    #         df_to_csv(pareto_opt_df, file_name, 'fet')

    # retrain the models for those that have Qrr values for the optimization function on power and cost
    if retrain_parameter == 'FOMs':
        # load the data we want to use
        if supplied_df is not None:
            df = supplied_df

            reg_score_and_dump_cat(df, training_params=training_params)
            return

        else:
            file_name = '../mosfet_data/csv_files/' + str(technology) + '_data_' + str(opt_param) + 'Opt.csv'
            df = csv_to_df(file_name)
            df = df.iloc[:, 1:]
            # df['V_dss_inv'] = (1 / (df['V_dss'])).astype(float)
            # df['R_ds_inv'] = (1 / (df['R_ds'])).astype(float)
            # df['Q_g_inv'] = (1 / (df['Q_g'])).astype(float)


        # Now train the models on the other parameters, not using just the components with associated area values
        file_name = '../mosfet_data/joblib_files/' + str(technology) + '_models_' + str(opt_param) + 'Opt'
        output_param_dict = {
                             'RdsCost_product': ['V_dss', 'R_ds'], 'RdsQg_product': ['V_dss', 'R_ds', 'Unit_price'],
                             'RdsQrr_product': ['V_dss', 'R_ds', 'Unit_price'],
                             'RdsCost_productOfCV': ['V_dss', 'Unit_price'], 'UnitpriceCoss_product': ['V_dss', 'R_ds', 'Unit_price','Q_g']}

        output_param_dict = {'RdsCoss_product': ['V_dss', 'R_ds', 'Unit_price'],
                             'QgCoss_product': ['V_dss', 'R_ds', 'Unit_price','Q_g'],
                             'PriceCoss_product': ['V_dss', 'R_ds', 'Unit_price'],
                             'Coss_product': ['V_dss', 'R_ds', 'Unit_price']}

        output_param_dict = {'Unit_price': ['V_dss', 'R_ds', 'Q_g']}
        # output_param_dict = {'RdsCost_productOfCV': ['V_dss', 'Unit_price']}
        output_param_dict = {'R_ds': ['V_dss', 'Unit_price'],'Q_g': ['V_dss', 'R_ds'],'Unit_price': ['V_dss', 'R_ds', 'Q_g']}

        for output_param, attributes in output_param_dict.items():
            if str(technology) == 'GaNFET':
                if output_param == 'RdsQrr_product':
                    continue
            # reg_score_and_dump_chained(df, technology, file_name, component_type = 'fet')
            reg_score_and_dump(df, output_param, attributes, file_name,pareto=True, chained=False, training_params=training_params)

    if retrain_parameter == 'initialization':
        # load the data we want to use
        if supplied_df is not None:
            df = supplied_df
            reg_score_and_dump_cat(df, training_params=training_params)
            return

    print('done')

    # can visualize the models with the data
    # model_plus_data_vs_Vdss(fet_df, 'V_dss', 'Q_rr', 'R_ds')

    # in the optimization, will use these models to make predictions


    # can visualize the models with the data
    # model_plus_data_vs_Vdss(fet_df, 'V_dss', 'Q_rr', 'R_ds')

    # in the optimization, will use these models to make predictions

import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

'''
    Take the matrix with missing values and use Bayesian regression to impute the remaining values
'''
from sklearn.compose import ColumnTransformer


def FET_type_parse(row):
    entry = (lambda entry: 1 if entry == 'N' else 2)(row['FET_ty'])
    return entry

def Series_parse(entry):
    entry = (lambda entry : 1 if entry == 'MOSFET' else (2 if entry == ('GaNFET' or 'GANFET') else 3))(entry)
    return entry

def impute_features(df, missing_val):
    return_df = df.copy()

    attr='FET_type'
    df[attr] = df[attr].apply(lambda entry: 1 if entry == 'N' or entry == ' N' or entry == 'N ' else 2)
    attr = 'Technology'
    df[attr] = df[attr].apply(
        lambda entry: 1 if entry == 'MOSFET' else 2 if entry == 'GaNFET' or entry == 'GANFET' else 3)

    imp = IterativeImputer(max_iter=10, random_state=0, missing_values=np.nan, sample_posterior=True, skip_complete=True,
                           initial_strategy='median')


    # Take a fraction of the df, include all that has all data, and then a bit of 0.0 rows, to fit, then use the imputer
    # on the entire df
    # take only the columns of the df that are numerical, to be used by the Bayesian regressor.
    # Consider converting N vs. P to be numerical

    attribute_list = ['V_dss','R_ds','Unit_price','Q_g','Pack_case','V_gs','Technology','FET_type'] # 'Input_cap','I_d'
    # filtered_df = df.select_dtypes(include=['int','float64'])
    full_attribute_list = attribute_list + [missing_val]
    filtered_df = df[full_attribute_list]

    # create the dataframe for training
    unfilled_val = filtered_df[filtered_df[missing_val]==0.0].iloc[0:10]
    unfilled_val= np.log10(unfilled_val[attribute_list].astype(float))
    unfilled_val[missing_val] = np.nan
    filled_val = np.log10(filtered_df[filtered_df[missing_val] != 0.0].loc[:, full_attribute_list].astype(float))
    frames = [unfilled_val, filled_val]
    filtered_df_plus_some = pd.concat(frames)
    imp.fit(filtered_df_plus_some)

    # now that the imputer has been fit, transform the entire input matrix
    filtered_df[missing_val] = filtered_df[missing_val].replace(0.0,np.NaN)
    for index, row in filtered_df[~np.isnan(filtered_df[missing_val])].iterrows():
        filtered_df.loc[index,missing_val]=np.log10(row[missing_val])
    filtered_df[attribute_list] = np.log10(filtered_df[attribute_list].astype(float))
    new_list = imp.transform(filtered_df[full_attribute_list])
    new_df = pd.DataFrame(new_list, columns=filtered_df.columns)
    new_df[full_attribute_list] = 10**new_df[full_attribute_list]
    return_df = return_df.drop(missing_val, axis=1)
    new_df.index = return_df.index
    return_df[missing_val] = new_df[missing_val]
    # return_df = pd.concat([df1, df], axis=1, ignore_index=True)

    # train on the Qrr data that we actually have to evaluate its performance
    # X = np.log10(df.loc[:, attributes])
    # y = np.log10(df.loc[:, output_param].astype(float))
    # X = preproc(X, 1)
    cv = KFold(n_splits=10, random_state=1, shuffle=True)
    scores_df = pd.DataFrame({'scoring_type':['r^2','MSE','MAE']}).set_index('scoring_type')

    # X_train, y_train = np.log10(return_df.loc[20000:, ['V_dss', 'Q_g', 'R_ds', 'Unit_price']]), np.log10(
    #     return_df.loc[20000:, [missing_val]])
    # X_test, y_test = filtered_df.dropna(subset=[missing_val], axis=0).loc[0:19999,
    #                  ['V_dss', 'Q_g', 'R_ds', 'Unit_price']], filtered_df.dropna(subset=[missing_val], axis=0).loc[
    #                                                           0:19999,
    #                                                           [missing_val]]
    # Xtest = preproc(X_test, 1)
    # Xtrain = preproc(X_train, 1)
    # rmse_scorer = make_scorer(mean_squared_error, squared=False)
    # model = LinearRegression(fit_intercept=True, normalize=True)
    # model.fit(Xtrain, y_train)
    # y_pred = model.predict(Xtest)
    # print("Mean squared error: %.2f" % mean_squared_error(y_test, y_pred))
    # # The coefficient of determination: 1 is perfect prediction
    # print("Coefficient of determination: %.2f" % r2_score(y_test, y_pred))



    # reg_score_and_dump(return_df, 'Q_rr', ['V_dss','Q_g','R_ds','Unit_price'], 'abc.csv', pareto=True, chained=False)
    return return_df

    # scaler = StandardScaler()
    # newX = scaler.fit_transform(filtered_df_plus_some.values.tolist())
    # imp.fit(newX)
    # newY = scaler.fit_transform(filtered_df.iloc[20:40])
    # imp.transform(newY)
    # new_list = imp.transform(filtered_df)
    #
    # # combine the dataframes from before taking only the numerical columsn, and take the max
    # # to get the new Qrr and Coss, and the other values should be the same
    # new_df = pd.DataFrame(new_list, columns=filtered_df.columns)
    # return_df[['Q_rr','C_oss']] = new_df[['Q_rr','C_oss']]
    # return_df = df.merge(new_df, how='right')
    # print('done')

if __name__ == '__main__':

    train_all()
    # pd.set_option("display.max_rows", None, "display.max_columns", None)
    # pareto_plotting('MOSFET')
    opt_param_list = ['Power', 'Cost', 'Area', 'Balanced', 'Balanced2']
    # gd_training(True, True, opt_param=opt_param_list[3], corr_matrix=False)
    # cap_training(True, True, opt_param=opt_param_list[3], corr_matrix=False)
    ind_training(True, True, opt_param = opt_param_list[3], corr_matrix=False)
    fet_training(data_gen_area=True,  retrain_area=True, data_gen_FOMs=False, retrain_FOMs=True, technology='MOSFET', opt_param = opt_param_list[3]) #typically use Balanced2, only need Balanced for Qrr and Coss
    # fet_training(data_gen_area = False, data_gen_FOMs = False, retrain_area=True, retrain_FOMs = True, technology = 'MOSFET')



    ################## from here below, we are pareto-optimizing all the data without Qrr bc we need more datapoints
                    # to use for classification
    #csv_file = 'mosfet_data_csv_best_final.csv'
    # looking at Qrr values
    csv_file = '../mosfet_data/csv_files/MOSFET_data_wPDF.csv'
    # df_Qrr = parse_pdf_param(csv_file=csv_file, pdf_param='Q_rr', component_type='fet', return_only=True, drop_first=False)
    #
    # # need to first parse the Qrr df
    # attr_list = ['Mfr_part_no', 'Unit_price', 'FET_type', 'Technology', 'V_dss', 'R_ds', 'Q_g', 'Input_cap',
    #              'Pack_case']
    # df_Qrr = column_fet_parse(initial_fet_parse(df_Qrr, attr_list), attr_list)
    #
    # csv_file = '../mosfet_data/csv_files/mosfet_data_csv_wpdf.csv'
    fet_df = csv_to_mosfet(csv_file)
    # df_Qrr = parse_pdf_param(csv_file=csv_file, pdf_param='Q_rr', component_type='fet', return_only=True, drop_first=False)
    #
    # # need to first parse the Qrr df
    # attr_list = ['Mfr_part_no', 'Unit_price', 'FET_type', 'Technology', 'V_dss', 'R_ds', 'Q_g', 'Input_cap',
    #              'Pack_case']
    # df_Qrr = column_fet_parse(initial_fet_parse(df_Qrr, attr_list), attr_list)
    fet_df.columns = ['idx1', 'V_dss', 'Unit_price', 'R_ds', 'Q_g', 'Q_rr', 'FET_type', 'Technology', 'Pack_case']
    fet_df = fet_df.drop('idx1', axis=1)
    subset_df = fet_df[(fet_df['FET_type'] == 'N')]
    subset_df = subset_df[(subset_df['Technology'].isin(['MOSFET']))]
    temp_df = subset_df[(subset_df['R_ds'] <= 3.0)]
    temp_df = temp_df[(temp_df['Q_g'] <= 0.4 * 10 ** -6)]
    # First normalize the data
    temp_df = temp_df.reset_index()
    attr_list = ['V_dss', 'Unit_price', 'R_ds', 'Q_g', 'Q_rr', '1/Rds', 'RdsQg_product', 'RdsCost_product',
                 'RdsQrr_product']
    correlation_matrix(temp_df, attr_list,'fet')
    pareto_opt_data = pareto_optimize(temp_df)
    pareto_opt_df = pd.DataFrame(np.transpose(pareto_opt_data),
                                 columns=['V_dss', 'Unit_price', 'R_ds', 'Q_g','FET_type', 'Technology',
                                          'Pack_case'])
    pareto_opt_df['R_ds'] = pareto_opt_df['R_ds'].astype(float)
    pareto_opt_df['V_dss'] = pareto_opt_df['V_dss'].astype(float)
    pareto_opt_df['Unit_price'] = pareto_opt_df['Unit_price'].astype(float)

    # write this new data to the csv you will want to use from here on out
    file_name = 'pareto_opt_Si_noQrr_final.csv'
    df_to_csv(pareto_opt_df, file_name, 'fet')

    ####################################################################
    pd.set_option("display.max_rows", None, "display.max_columns", None)
    #fet_training(data_gen_area=False,  retrain_area=False, data_gen_FOMs=True, retrain_FOMs=True, technology='MOSFET')
    #fet_training(data_gen_area = False, data_gen_FOMs = False, retrain_area=True, retrain_FOMs = True, technology = 'MOSFET')
    ind_training(True, True)

    df = parse_pdf_param(csv_file='csv_files/mosfet_data_wpdf3.csv', pdf_param='Q_rr', component_type='fet', return_only=False)

    attr_list = ['Mfr_part_no', 'Unit_price', 'FET_type', 'Technology', 'V_dss', 'R_ds', 'Q_g', 'Input_cap',
                 'Pack_case']
    df = column_fet_parse(initial_fet_parse(df, attr_list), attr_list)
    subset_df = df[(df['FET_type'] == 'N')]
    #subset_df = subset_df[(subset_df['Technology'].isin(['MOSFET']))]

    temp_df = subset_df[(subset_df['R_ds'] <= 3.0)]
    temp_df = temp_df[(temp_df['Q_g'] <= 0.4 * 10 ** -6)]
    # set Q_rr to be consistent
    temp_df['Q_rr'] = temp_df['Q_rr'].astype(float)
    temp_df['Q_rr'] = 10 ** -9 * temp_df['Q_rr']


    # First normalize the data
    temp_df = temp_df.reset_index()

    data_dims = ['V_dss', 'Unit_price', 'R_ds', 'Q_g', 'FET_type', 'Technology', 'Pack_case']
    data_dims_paretoFront = ['V_dss', 'Unit_price', 'R_ds', 'Q_g']
    pareto_opt_data = pareto_optimize(temp_df, data_dims, data_dims_paretoFront, technology='MOSFET', component='fet')
    pareto_opt_df = pd.DataFrame(np.transpose(pareto_opt_data), columns=['V_dss', 'Unit_price', 'R_ds', 'Q_g', 'Q_rr','FET_type','Technology','Pack_case'])
    pareto_opt_df['R_ds'] = pareto_opt_df['R_ds'].astype(float)
    pareto_opt_df['V_dss'] = pareto_opt_df['V_dss'].astype(float)
    pareto_opt_df['Unit_price'] = pareto_opt_df['Unit_price'].astype(float)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # load the data with area dimensions available
    file_name = 'csv_files/area_df_Si_wQrr.csv'
    area_df = csv_to_df(file_name)
    area_df.columns =['idx1','V_dss', 'Unit_price', 'R_ds', 'Q_g','Q_rr','FET_type','Technology','Pack_case']
    # Add columns to the dataframe that represent other FOMs
    corr_df = area_df
    corr_df['1/Rds'] = 1 / corr_df['R_ds']
    corr_df['RdsQg_product'] = corr_df['R_ds'] * corr_df['Q_g']
    corr_df['RdsCost_product'] = corr_df['R_ds'] * corr_df['Unit_price']
    corr_df['RdsQrr_product'] = corr_df['R_ds'] * corr_df['Q_rr']
    # corr_df['Energy'] = corr_df['Inductance'] * corr_df['Current_rating'] ** 2
    # corr_df['Inductance*DCR'] = corr_df['Inductance'] * corr_df['DCR']
    corr_df['1/RdsQg_product'] = 1 / (corr_df['RdsQg_product'])
    corr_df['1/RdsCost_product'] = 1 / (corr_df['RdsCost_product'])
    corr_df['1/RdsQrr_product'] = 1 / (corr_df['RdsQrr_product'])

    attr_list = ['V_dss', 'Unit_price', 'R_ds', 'Q_g', 'Q_rr','Pack_case','1/Rds', 'RdsQg_product','RdsCost_product','RdsQrr_product']

    # plot the correlation matrix using seaborn
    correlation_matrix(corr_df, attr_list, 'fet')

    plt.scatter(corr_df['V_dss'], corr_df['Unit_price'])
    plt.show()


    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # write this new data to the csv you will want to use from here on out
    file_name = 'csv_files/pareto_opt_Si_final.csv'
    df_to_csv(pareto_opt_df, file_name, 'fet')
    #train on the pareto-optimized data
    output_param_dict = {'RdsCost_product': ['V_dss', 'R_ds'], 'RdsQg_product': ['V_dss', 'R_ds', 'Unit_price'],'RdsQrr_product': ['V_dss', 'R_ds', 'Unit_price'],
                         'RdsCost_productOfCV':['V_dss','Unit_price']}
    #output_param_dict = {'R_ds': ['V_dss', 'Unit_price'], 'Q_g': ['V_dss', 'R_ds', 'Unit_price'],'Q_rr': ['V_dss', 'R_ds', 'Unit_price']}

    file_name = 'pareto_opt_Si'
    for output_param, attributes in output_param_dict.items():
        reg_score_and_dump(pareto_opt_df, 'V_dss', output_param, attributes, file_name,pareto=True)

    original_df = temp_df.copy()
    scaler = preprocessing.MinMaxScaler()
    names = temp_df.columns
    d = scaler.fit_transform(temp_df[['V_dss', 'Q_g', 'Q_rr', 'Unit_price', 'R_ds']])
    scaled_df = pd.DataFrame(d, columns=['V_dss', 'Q_g', 'Q_rr', 'Unit_price', 'R_ds'])
    temp_df[['V_dss', 'Q_g', 'Q_rr', 'Unit_price', 'R_ds']] = scaled_df[['V_dss', 'Q_g', 'Q_rr', 'Unit_price', 'R_ds']]

    # visualize overall product, remember to do Rds^3
    # RQ_scatter_cost(temp_df, 'V_dss', 'Q_g', 'Q_rr', 'Unit_price', 'R_ds')

'''
    This creates a class for each capacitor based off the downloaded table information, after the tables in the pdf have been 
    opened and scraped and turned into a list of dataframes.
'''


class DigikeyCap_downloaded:
    def __init__(self, datasheet_set):

        try:
            self.mfr_part_no = datasheet_set[0]
            self.datasheet = datasheet_set[1]['Datasheet']
            self.unit_price = datasheet_set[1]['Price']
            self.stock = datasheet_set[1]['Stock']
            self.mfr = datasheet_set[1]['Mfr']
            self.series = datasheet_set[1]['Series']
            self.capacitance = datasheet_set[1]['Capacitance']
            self.tolerance = datasheet_set[1]['Tolerance']
            self.rated_volt = datasheet_set[1]['Voltage - Rated']
            self.temp_coef = datasheet_set[1]['Temperature Coefficient']
            self.op_temp = datasheet_set[1]['Operating Temperature']
            self.features = datasheet_set[1]['Features']
            self.ratings = datasheet_set[1]['Ratings']
            self.applications = datasheet_set[1]['Applications']
            self.mounting_type = datasheet_set[1]['Mounting Type']
            self.pack_case = datasheet_set[1]['Package / Case']
            self.size = datasheet_set[1]['Size / Dimension']
            self.height = datasheet_set[1]['Height - Seated (Max)']
            self.thickness = datasheet_set[1]['Thickness (Max)']
            self.lead_spacing = datasheet_set[1]['Lead Spacing']
            self.lead_style = datasheet_set[1]['Lead Style']



        except:
            print("Component element exception")