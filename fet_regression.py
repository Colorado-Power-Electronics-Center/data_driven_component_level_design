'''
    This file contains the code for creating and training models based on the df, or subsections of the df.
'''

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression, BayesianRidge,Lasso,Ridge
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import mean_squared_error, r2_score, make_scorer
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split, KFold, cross_val_score, cross_validate
import pandas as pd
import math
import mpl_toolkits
import decimal
import numpy.ma as ma
import matplotlib.pyplot as plt
from tabulate import tabulate
import numpy as np
from sklearn.linear_model import LinearRegression, RANSACRegressor, HuberRegressor, TheilSenRegressor, ElasticNet, LogisticRegression
from sklearn.multioutput import RegressorChain
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.kernel_ridge import KernelRidge
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score
import pandas as pd
import math
from sklearn.ensemble import AdaBoostRegressor
from sklearn.neural_network import MLPRegressor
from mpl_toolkits.mplot3d import Axes3D
from joblib import dump, load
from string import ascii_lowercase
import pickle
from fet_data_parsing import column_fet_parse, initial_fet_parse
from fet_visualization import Visualization_parameters





'''
    Takes the desired df, list of input attributes, output parameter, and log_x boolean. Then creates X, the dataframe 
    series of just those specified input attributes with either log or not depending on the value in log_x, 
    and then y, the log value of the desired output parameter. Returns series of X,y to be used for training models.
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

    # Output variables
    if pareto == True:
        if output_param == 'R_ds' or output_param == 'Q_g' or output_param=='Q_rr' or output_param=='C_oss':
            y = np.log10(df.loc[:, output_param].astype(float))
        elif output_param == 'Unit_price':
            y=np.log10(df.loc[:,output_param].astype(float))
        elif output_param == 'RdsQg_product':
            #df.loc[:,'Q_g'] = df.loc[:,'Q_g']*10**8
            y = np.log10(df.loc[:, 'R_ds'].astype(float) * df.loc[:, 'Q_g'].astype(float))
        elif output_param == 'RdsQrr_product':
            df = df[df['Q_rr']!=0]
            X = np.log10(df.loc[:, attributes])
            y = np.log10(df.loc[:, 'R_ds'].astype(float) * df.loc[:, 'Q_rr'].astype(float))
            # entry_list = []
            # for index, row in df.iterrows():
            #     entry = row['Q_rr'] * row['R_ds']
            #     entry_list.append(np.log10(entry))
            # y = pd.Series(entry_list)
            # y = np.log10(df.loc[:, 'R_ds'] * df.loc[:, 'Q_rr'])
        elif output_param == 'RdsCoss_product':
            y = np.log10(df.loc[:, 'R_ds'].astype(float) * df.loc[:, 'C_oss'].astype(float))
        elif output_param == 'QgCoss_product':
            y = np.log10(df.loc[:, 'Q_g'].astype(float) * df.loc[:, 'C_oss'].astype(float))
        elif output_param == 'Coss_product':
            y = np.log10(df.loc[:, 'C_oss'].astype(float))
        elif output_param == 'UnitpriceCoss_product':
            df = df[df['C_oss'] != 0]
            X = np.log10(df.loc[:, attributes])
            y = np.log10(df.loc[:, 'Unit_price'].astype(float) * df.loc[:, 'C_oss'].astype(float))
        elif output_param == 'RdsCost_product':
            y = np.log10(df.loc[:, 'R_ds'].astype(float) * df.loc[:, 'Unit_price'].astype(float))
        elif output_param == 'RdsCost_productOfCV':
            y = np.log10(df.loc[:, 'Unit_price'].astype(float) * df.loc[:, 'R_ds'].astype(float))
        elif output_param == 'C_oss_p1':
            y = np.log10(df.loc[:,'C_oss_p1'].astype(float))
        return X,y
    else:
        y = np.log10(df.loc[:, output_param].astype(float))

    return X,y


'''
    This function takes as input the desired df and a dictionary of parameters (in string form) to filter the df to
    consist of, e.g. keep only N-channel fets, or only Si MOSFETs, and returns the filtered df.
'''


def param_strip(df, params_dict):
    for param,param_value in params_dict.items():
        df = df[(df[param] == param_value)]
    return df


'''
    Evaluates various linear models to find the model characteristics with optimal performance. Takes as input the
    X,y training data set, the optimal preprocessing degree, and a dictionary of the model parameters to be evaluated. 
    In the dictionary, the model parameters should be a list of all the param options. Runs through and for each model 
    parameter, returns a dictionary of just the highest performing value and its score as [value, score]. May consider 
    accessing the model type and its params through a prewritten dict of dicts in the form:
        {model_type1: {param1: [val1, val2, val3], param2: [val1, val2]}, model_type2: {param1: [val1, val2, val3], 
        param2: [val1, val2]}},
    so you can have multiple different known parameters to test for any type of model.
'''
def param_opt(X, y, attributes, model, degree, model_params_dict):
    opt_model_params = {}

    for param, param_value_dict in model_params_dict.items():
        best_score = 0
        # As a separate case, figure out the best degree for linear preprocessing
        if 'degree' in param_value_dict.keys():
            best_degree = 0
            # run through all the possible parameters for that model type that we might want to test
            for param_value in param_value_dict['degree']:

                # Train the models and evaluate performance
                cv = KFold(n_splits=5, random_state=1, shuffle=True)
                X_processed = manual_preproc(X,param_value, attributes)
                model = LinearRegression()
                new_score = np.mean(cross_val_score(model, X_processed, y, cv=cv))
                if new_score > best_score:
                    best_degree = degree
                    best_score = new_score

            # Add the value and score to the opt_model_params dictionary
            opt_model_params[param] = [best_degree, best_score]

        # Otherwise, use the input degree to use for X preprocessing
        else:
            # run through all the possible parameters for that model type that we might want to test
            for param_val in param_value_dict.values:
                best_param_val = 0
                # Train the models and evaluate performance
                cv = KFold(n_splits=5, random_state=1, shuffle=True)
                X_processed = manual_preproc(X, degree)
                new_score = np.mean(cross_val_score(model, X_processed, y, cv=cv))
                if new_score > best_score:
                    best_param_val = param_val
                    best_score = new_score


            # Add the value and score to the opt_model_params dictionary
            opt_model_params[param] = [best_param_val, best_score]

    return opt_model_params


class opt_model():
    def __init__(self, model_type):
        self.model_type = model_type

        if model_type == 'linear':
            self.degree = 0.3
        elif model_type == 'random_forest':
            self.min_samples_split = 2
        elif model_type == 'knn':
            self.weights = 'distance'
            self.n_neighbors = 10
        elif model_type == 'adaboosting':
            self.learning_rate = 1.0
            self.n_estimators = 100
        elif model_type == 'mlp':
            self.max_iter = 500
            self.activation = 'relu'
            self.solver = 'adam'
            self.alpha = 0.001

def rmse_score(y_true, y_pred):
    mean_squared_error(y_true, y_pred, squared=False)
'''
    Takes the df and output_param to train on, and runs through many models. If opt_param_gather boolean = True, calls
    the param_opt() function and gets the optimal parameters for each type of model, and uses those in the training.
    Otherwise, uses the values from the gathered opt_param_dict and uses those in the training models.
'''
def reg_score_and_dump_cat(df, training_params='main_page_params', component = 'fet'):

    param_training_dict = {'main_page_params': {'inputs': ['V_dss', 'R_ds'], 'outputs': ['Q_g','Unit_price','Pack_case'],
                                                'file_name': 'full_dataset_models_chained.joblib', 'order': [0,1,2]},
                           'pdf_params': {'inputs': ['V_dss', 'R_ds', 'Q_g', 'Unit_price'],
                                          'outputs': ['C_ossp1', 'Vds_meas', 'tau_c', 'tau_rr'],
                                          'file_name': 'pdf_dataset_models_chained.joblib', 'order': [0,1,2,3]},
                           'inductor_params': {'inputs': ['Current_Rating [A]', 'Current_Sat [A]', 'Inductance [H]'],
                                               'outputs': ['DCR [Ohms]','Unit_price [USD]', 'Area [mm^2]', 'fb [Hz]', 'b', 'Kfe', 'Alpha', 'Beta', 'Nturns', 'Ac [m^2]', 'Volume [mm^3]', 'Core Volume m^3'],
                                               'file_name': 'inductor_models_chained.joblib', 'order': [0,1,2,3,4,5,6,7,8,9,10,11]},
                           'Cossp1_plotting': {'inputs': ['V_dss', 'R_ds', 'Q_g', 'Unit_price'], 'outputs': ['C_ossp1'], 'file_name': 'Cossp1_plotting_output', 'order': [0]},
                           'R_ds_initialization': {'inputs': ['V_dss', 'Unit_price'], 'outputs': ['R_ds'], 'file_name': 'R_ds_initialization', 'order': [0]},
                           'fsw_initialization': {'inputs': ['Unit_price [USD]', 'Current_Rating [A]'], 'outputs': ['Inductance [H]'], 'file_name': 'fsw_initialization', 'order': [0]},
                           'cap_main_page_params': {'inputs': ['Capacitance','Rated_volt','Size'], 'outputs': ['Unit_price'],
                                                    'file_name': 'cap_main_page_params_models_chained.joblib', 'order': [0]},
                           'cap_pdf_params': {'inputs': ['Vrated [V]','Area [mm]','Vdc_meas [V]','Capacitance_at_Vdc_meas [uF]'],
                                              'outputs': ['Capacitance_at_0Vdc [uF]'], 'file_name': 'cap_pdf_params_models_chained.joblib', 'order': [0]},
                           'cap_area_initialization': {'inputs': ['Capacitance','Rated_volt','Unit_price'], 'outputs': ['Size'],
                                                    'file_name': 'cap_area_initialization_models_chained.joblib', 'order': [0]}
                           }

    # var1 = 'Q_g'
    # param_training_dict['main_page_params']['outputs'] = ['Unit_price','Q_g']
    # param_training_dict['main_page_params']['order'] = [0,1]
    #
    # var1 = 'C_ossp1'
    # param_training_dict['pdf_params']['outputs'] = [var1]
    # param_training_dict['pdf_params']['order'] = [0]

    if training_params == 'pdf_params':
        for index, row in df.iterrows():
            if row['GaNFET'] == 1.0:
                df.loc[index, 'tau_c'] = 0
                df.loc[index, 'tau_rr'] = 0
        df = df.dropna(subset=['tau_c', 'tau_rr'])
        df = df[df['tau_c'] >= df['tau_rr']]

    degree = 1
    rmse_scorer = make_scorer(mean_squared_error, squared=False)
    # Get the X,y for model training
    cv = KFold(n_splits=5, random_state=1, shuffle=True)
    if component == 'fet':
        df = df.dropna(subset=['V_dss', 'R_ds', 'Unit_price', 'Q_g'])
        categorical_attributes = ['N', 'P', 'MOSFET', 'SiCFET', 'GaNFET']
        categorical_attributes = ['N', 'P', 'MOSFET', 'GaNFET'] # Change if also including SiCFETs

        for variable in param_training_dict[training_params]['inputs']:
            new_variable_name = 'log10[' + str(variable) + ']'
            df[new_variable_name] = np.log10(df[variable].astype(float))
            categorical_attributes.append(new_variable_name)
        if training_params != 'R_ds_initialization':
            df = df[df['log10[R_ds]'] != 0]

    elif component == 'ind':
        categorical_attributes = []
        for variable in param_training_dict[training_params]['inputs']:
            new_variable_name = 'log10[' + str(variable) + ']'
            df[new_variable_name] = np.log10(df[variable])
            categorical_attributes.append(new_variable_name)

    elif component == 'cap':
        if training_params != 'cap_pdf_params':
            # Note that any categorical attributes you want to keep, include them in the categorical attributes list, not in the inputs list
            categorical_attributes = ['Temp_coef_enc']
        else:
            categorical_attributes = []
        for variable in param_training_dict[training_params]['inputs']:
            new_variable_name = 'log10[' + str(variable) + ']'
            df[new_variable_name] = np.log10(df[variable].astype(float))
            categorical_attributes.append(new_variable_name)
        # if training_params != 'R_ds_initialization':
        #     df = df[df['log10[R_ds]'] != 0]



    # df = df[df['C_ossp1']>0]
    X = df.loc[:, categorical_attributes]
    X = X.replace(np.NaN, 0)
    X = preproc(X, degree)
    df = df.replace(0, np.NaN)
    output_df = pd.DataFrame()
    for output_var in param_training_dict[training_params]['outputs']:
        output_df[output_var] = np.log10(df[output_var].astype(float))
    output_df = output_df.replace(np.NaN, 0)

    cv = KFold(n_splits=5, random_state=1, shuffle=True)
    scores_df = pd.DataFrame({'scoring_type': ['r^2', 'MSE', 'MAE']}).set_index('scoring_type')
    gamma_final = 0.1
    alpha_final = 0.1
    model = KernelRidge(degree=4, alpha=alpha_final, gamma=gamma_final, kernel='rbf')
    # model = LinearRegression(fit_intercept=True, normalize=True)
    # model_obj = opt_model('random_forest')
    # model = RandomForestRegressor(min_samples_split=model_obj.min_samples_split, random_state=0)
    # model_obj = opt_model('knn')
    # model = KNeighborsRegressor(n_neighbors=model_obj.n_neighbors, weights=model_obj.weights)
    # model_obj = opt_model('mlp')
    # model = MLPRegressor(random_state=1, max_iter=model_obj.max_iter, activation=model_obj.activation,
    #                      solver=model_obj.solver,
    #                      alpha=model_obj.alpha)

    chain = RegressorChain(base_estimator=model, order=param_training_dict[training_params]['order'])
    reg_chain = chain.fit(X, output_df)
    scores = cross_val_score(chain, X, output_df, cv=cv, scoring='neg_mean_absolute_error')

    ### uncomment this when done --> ALSO make sure optimization algorithm sees the correctly trained file! getting rid of nans in pdf params? !!!!!!!
    # dump(reg_chain, 'joblib_files/' + param_training_dict[training_params]['file_name'])

    cv = KFold(n_splits=5, random_state=1, shuffle=True)
    y = output_df
    # y = np.log10(df.loc[:, var1].astype(float)).to_numpy()

    #turn output into log

    # for train_index, test_index in cv.split(X, y):
    #     X_train, X_test = X[train_index], X[test_index]
    #     y_train, y_test = y[train_index], y[test_index]
    #     reg = model.fit(X_train, y_train)
    #
    #     # the coefficients of the regression line
    #     # print('coef: {} \n yint: {}'.format(reg.coef_, reg.intercept_))
    #     y_pred = reg.predict(X_test)
    #     print(reg.score(X_test, y_test))
    #     mse = mean_squared_error(y_test, y_pred)
    #     print('testing root mean squared error: %.2f' % np.sqrt(mse))
    #     cod = r2_score(y_test, y_pred)
    #     print('testing coefficient of determination on testing: %.2f' % cod)
    #     mae = mean_absolute_error(y_test, y_pred)
    #     print('testing mean absolute error: %.2f' % mae)
    #
    #     # see how well the prediction does with the training data itself
    #     reg = model.fit(X_train, y_train)
    #     y_pred = reg.predict(X_train)
    #     mse = mean_squared_error(y_train, y_pred)
    #     print('training root mean squared error: %.2f' % np.sqrt(mse))
    #     cod = r2_score(y_train, y_pred)
    #     print('training coefficient of determination on training: %.2f' % cod)
    #     mae = mean_absolute_error(y_train, y_pred)
    #     print('training mean absolute error: %.2f' % mae)
    print('done')

    ### hyperparameter tuning ###
    # from sklearn.model_selection import RandomizedSearchCV
    # import random
    # param_dist = {"degree": [1, 2, 3, 4, 5],
    #               "alpha": [random.random()],
    #               "gamma": [random.random()],
    #               "kernel": ["linear", "rbf", "laplacian", "polynomial", "sigmoid"]}
    # tree = KernelRidge()
    # tree_cv = RandomizedSearchCV(tree, param_dist, cv=5)
    # tree_cv.fit(X, output_df)
    # print("Tuned Decision Tree Parameters: {}".format(tree_cv.best_params_))
    # print("Best score is {}".format(tree_cv.best_score_))

    # var1 = 'C_ossp1'
    # param_training_dict['pdf_params']['outputs'] = ['C_ossp1']
    # param_training_dict['pdf_params']['order'] = [0]
    #
    # var1 = 'Q_rr'
    # param_training_dict['pdf_params']['outputs'] = ['C_ossp1', 'Q_rr']
    # param_training_dict['pdf_params']['order'] = [0, 1]
    # #
    # var1 = 'I_F'
    # param_training_dict['pdf_params']['outputs'] = ['C_ossp1', 'Q_rr', 'I_F']
    # param_training_dict['pdf_params']['order'] = [0, 1, 2]
    # #
    # var1 = 'Pack_case'
    # param_training_dict['pdf_params']['outputs'] = ['C_ossp1', 'Q_rr', 'I_F', 'Pack_case']
    # param_training_dict['pdf_params']['order'] = [0, 1, 2, 3]

    # var1 = 'DCR [Ohms]'
    # param_training_dict['inductor_params']['outputs'] = ['DCR [Ohms]']
    # param_training_dict['inductor_params']['order'] = [0]
    #
    # var1 = 'Unit_price [USD]'
    # param_training_dict['inductor_params']['outputs'] = ['DCR [Ohms]', 'Unit_price [USD]']
    # param_training_dict['inductor_params']['order'] = [0, 1]
    #
    # var1 = 'Volume [mm^3]'
    # param_training_dict['inductor_params']['outputs'] = ['DCR [Ohms]', var1]
    # param_training_dict['inductor_params']['order'] = [0, 1]
    #
    # # #
    # var1 = 'Dimension [mm^2]'
    # param_training_dict['pdf_params']['outputs'] = ['DCR [Ohms]', 'Volume [mm^3]', var1]
    # param_training_dict['pdf_params']['order'] = [0, 1, 2]
    #
    # var1 = 'Unit_price [USD]'
    # param_training_dict['inductor_params']['outputs'] = ['DCR [Ohms]', 'Volume [mm^3]', var1]
    # param_training_dict['inductor_params']['order'] = [0, 1, 2]
    #
    # var1 = 'fb [Hz]'
    # param_training_dict['inductor_params']['outputs'] = ['DCR [Ohms]', 'Volume [mm^3]', 'Unit_price [USD]', var1]
    # param_training_dict['inductor_params']['order'] = [0, 1, 2, 3]
    #
    # var1 = 'b'
    # param_training_dict['inductor_params']['outputs'] = ['DCR [Ohms]', 'Volume [mm^3]', 'Unit_price [USD]', 'fb [Hz]',
    #                                                      var1]
    # param_training_dict['inductor_params']['order'] = [0, 1, 2, 3, 4]
    #
    # var1 = 'Kfe'
    # param_training_dict['inductor_params']['outputs'] = ['DCR [Ohms]', 'Volume [mm^3]', 'Unit_price [USD]', 'fb [Hz]',
    #                                                      'b', var1]
    # param_training_dict['inductor_params']['order'] = [0, 1, 2, 3, 4, 5]
    #
    # var1 = 'Alpha'
    # param_training_dict['inductor_params']['outputs'] = ['DCR [Ohms]', 'Volume [mm^3]', 'Unit_price [USD]', 'fb [Hz]',
    #                                                      'b', 'Kfe', var1]
    # param_training_dict['inductor_params']['order'] = [0, 1, 2, 3, 4, 5, 6]
    #
    # var1 = 'Beta'
    # param_training_dict['inductor_params']['outputs'] = ['DCR [Ohms]', 'Volume [mm^3]', 'Unit_price [USD]', 'fb [Hz]',
    #                                                      'b', 'Kfe', 'Alpha', var1]
    # param_training_dict['inductor_params']['order'] = [0, 1, 2, 3, 4, 5, 6, 7]
    #
    # var1 = 'Nturns'
    # param_training_dict['inductor_params']['outputs'] = ['DCR [Ohms]', 'Volume [mm^3]', 'Unit_price [USD]', 'fb [Hz]',
    #                                                      'b', 'Kfe', 'Alpha', 'Beta', var1]
    # param_training_dict['inductor_params']['order'] = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    #
    # var1 = 'Ac [m^2]'
    # param_training_dict['inductor_params']['outputs'] = ['DCR [Ohms]', 'Volume [mm^3]', 'Unit_price [USD]', 'fb [Hz]',
    #                                                      'b', 'Kfe', 'Alpha', 'Beta', 'Nturns', var1]
    # param_training_dict['inductor_params']['order'] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    #
    # degree = 1
    # rmse_scorer = make_scorer(mean_squared_error, squared=False)
    # # Get the X,y for model training
    # cv = KFold(n_splits=5, random_state=1, shuffle=True)
    # if component == 'fet':
    #     df = df.dropna(subset=['V_dss', 'R_ds', 'Unit_price', 'Q_g'])
    #     categorical_attributes = ['N', 'P', 'MOSFET', 'SiCFET', 'GaNFET']
    #     for variable in param_training_dict[training_params]['inputs']:
    #         new_variable_name = 'log10[' + str(variable) + ']'
    #         df[new_variable_name] = np.log10(df[variable].astype(float))
    #         categorical_attributes.append(new_variable_name)
    #     if training_params != 'R_ds_initialization':
    #         df = df[df['log10[R_ds]'] != 0]
    #
    # elif component == 'ind':
    #     categorical_attributes = []
    #     for variable in param_training_dict[training_params]['inputs']:
    #         new_variable_name = 'log10[' + str(variable) + ']'
    #         df[new_variable_name] = np.log10(df[variable])
    #         categorical_attributes.append(new_variable_name)
    #
    # df = df[df['C_ossp1'] > 0]
    # X = df.loc[:, categorical_attributes]
    # X = X.replace(np.NaN, 0)
    # X = preproc(X, degree)
    # df = df.replace(0, np.NaN)
    # output_df = pd.DataFrame()
    # for output_var in param_training_dict[training_params]['outputs']:
    #     output_df[output_var] = np.log10(df[output_var].astype(float))
    # output_df = output_df.replace(np.NaN, 0)
    #
    # cv = KFold(n_splits=5, random_state=1, shuffle=True)
    # scores_df = pd.DataFrame({'scoring_type': ['r^2', 'MSE', 'MAE']}).set_index('scoring_type')
    # gamma_final = 0.1
    # alpha_final = 0.1
    # model = KernelRidge(degree=4, alpha=alpha_final, gamma=gamma_final, kernel='rbf')
    # # model = LinearRegression(fit_intercept=True, normalize=True)
    # # model_obj = opt_model('random_forest')
    # # model = RandomForestRegressor(min_samples_split=model_obj.min_samples_split, random_state=0)
    # model_obj = opt_model('knn')
    # model = KNeighborsRegressor(n_neighbors=model_obj.n_neighbors, weights=model_obj.weights)
    # model_obj = opt_model('mlp')
    # model = MLPRegressor(random_state=1, max_iter=model_obj.max_iter, activation=model_obj.activation,
    #                      solver=model_obj.solver,
    #                      alpha=model_obj.alpha)
    #
    # chain = RegressorChain(base_estimator=model, order=param_training_dict[training_params]['order'])
    # reg_chain = chain.fit(X[:800:4], output_df[:800:4])
    # scores = cross_val_score(chain, X, output_df, cv=cv, scoring='neg_root_mean_squared_error')
    # print('testing rmse: %.2f' % np.mean(scores))
    # scores = cross_val_score(chain, X, output_df, cv=cv, scoring='r2')
    # print('testing r2: %.2f' % np.mean(scores))
    # scores = cross_val_score(chain, X, output_df, cv=cv, scoring='neg_mean_absolute_error')
    # print('testing mae: %.2f' % np.mean(scores))
    #
    # y_train = output_df[:800:4]
    # y_pred = reg_chain.predict(X[:800:4])
    # mse = mean_squared_error(y_train, y_pred)
    # print('training root mean squared error: %.2f' % np.sqrt(mse))
    # cod = r2_score(y_train, y_pred)
    # print('training coefficient of determination on training: %.2f' % cod)
    # mae = mean_absolute_error(y_train, y_pred)
    # print('training mean absolute error: %.2f' % mae)

    # # var1 = 'DCR [Ohms]'
    # # param_training_dict['pdf_params']['outputs'] = ['DCR [Ohms]']
    # # param_training_dict['pdf_params']['order'] = [0]
    #
    # var1 = 'Unit_price [USD]'
    # param_training_dict['pdf_params']['outputs'] = ['DCR [Ohms]', 'Unit_price [USD]']
    # param_training_dict['pdf_params']['order'] = [0, 1]
    # # #
    # # var1 = 'I_F'
    # # param_training_dict['pdf_params']['outputs'] = ['C_ossp1', 'Q_rr', 'I_F']
    # # param_training_dict['pdf_params']['order'] = [0, 1, 2]
    # # #
    # # var1 = 'Pack_case'
    # # param_training_dict['pdf_params']['outputs'] = ['C_ossp1', 'Q_rr', 'I_F', 'Pack_case']
    # # param_training_dict['pdf_params']['order'] = [0, 1, 2, 3]
    #
    # degree = 1
    # rmse_scorer = make_scorer(mean_squared_error, squared=False)
    # # Get the X,y for model training
    # cv = KFold(n_splits=5, random_state=1, shuffle=True)
    # if component == 'fet':
    #     df = df.dropna(subset=['V_dss', 'R_ds', 'Unit_price', 'Q_g'])
    #     categorical_attributes = ['N', 'P', 'MOSFET', 'SiCFET', 'GaNFET']
    #     for variable in param_training_dict[training_params]['inputs']:
    #         new_variable_name = 'log10[' + str(variable) + ']'
    #         df[new_variable_name] = np.log10(df[variable].astype(float))
    #         categorical_attributes.append(new_variable_name)
    #     if training_params != 'R_ds_initialization':
    #         df = df[df['log10[R_ds]'] != 0]
    #
    # elif component == 'ind':
    #     categorical_attributes = []
    #     for variable in param_training_dict[training_params]['inputs']:
    #         new_variable_name = 'log10[' + str(variable) + ']'
    #         df[new_variable_name] = np.log10(df[variable])
    #         categorical_attributes.append(new_variable_name)
    #
    # # df = df[df['C_ossp1'] > 0]
    # X = df.loc[:, categorical_attributes]
    # X = X.replace(np.NaN, 0)
    # X = preproc(X, degree)
    # df = df.replace(0, np.NaN)
    # output_df = pd.DataFrame()
    # for output_var in param_training_dict[training_params]['outputs']:
    #     output_df[output_var] = np.log10(df[output_var].astype(float))
    # output_df = output_df.replace(np.NaN, 0)
    #
    # cv = KFold(n_splits=5, random_state=1, shuffle=True)
    # scores_df = pd.DataFrame({'scoring_type': ['r^2', 'MSE', 'MAE']}).set_index('scoring_type')
    # gamma_final = 0.1
    # alpha_final = 0.1
    # model = KernelRidge(degree=4, alpha=alpha_final, gamma=gamma_final, kernel='rbf')
    # # model = LinearRegression(fit_intercept=True, normalize=True)
    # model_obj = opt_model('random_forest')
    # model = RandomForestRegressor(min_samples_split=model_obj.min_samples_split, random_state=0)
    # # model_obj = opt_model('knn')
    # # model = KNeighborsRegressor(n_neighbors=model_obj.n_neighbors, weights=model_obj.weights)
    # # model_obj = opt_model('mlp')
    # # model = MLPRegressor(random_state=1, max_iter=model_obj.max_iter, activation=model_obj.activation,
    # #                      solver=model_obj.solver,
    # #                      alpha=model_obj.alpha)
    #
    # chain = RegressorChain(base_estimator=model, order=param_training_dict[training_params]['order'])
    # reg_chain = chain.fit(X[:800:4], output_df[:800:4])
    # scores = cross_val_score(chain, X, output_df, cv=cv, scoring='neg_root_mean_squared_error')
    # print('testing rmse: %.2f' % np.mean(scores))
    # scores = cross_val_score(chain, X, output_df, cv=cv, scoring='r2')
    # print('testing r2: %.2f' % np.mean(scores))
    # scores = cross_val_score(chain, X, output_df, cv=cv, scoring='neg_mean_absolute_error')
    # print('testing mae: %.2f' % np.mean(scores))
    #
    # y_train = output_df[:800:4]
    # y_pred = reg_chain.predict(X[:800:4])
    # mse = mean_squared_error(y_train, y_pred)
    # print('training root mean squared error: %.2f' % np.sqrt(mse))
    # cod = r2_score(y_train, y_pred)
    # print('training coefficient of determination on training: %.2f' % cod)
    # mae = mean_absolute_error(y_train, y_pred)
    # print('training mean absolute error: %.2f' % mae)


    # # Visualize the model on top of the data
    # max_volt = 1000
    # min_volt = 10.0
    # T = np.linspace(min_volt, max_volt, 1000)
    # T = np.log10(T)
    # model_list = ['RdsCossp1_product']
    # # model = 'random_forest'
    # fet_reg_models_df = load('Cossp1_plotting_output')
    # y_ = []
    # T = np.linspace(min_volt, max_volt, 1000)
    # for volt in T:
    #     # print(volt)
    #     X = preproc(np.array(
    #         [1, 0, 1, 0, 0, np.log10(volt), np.log10(df['R_ds'].mean()),
    #          np.log10(df['Q_g'].mean()), np.log10(df['Unit_price'].mean())]).reshape(1, -1), 1)
    #     y_.append((10 ** (fet_reg_models_df.predict(X))[0]) * 10 ** -15)
    #
    # # filter the dataset we are plotting
    # opt_df = df.copy()
    # opt_df = opt_df[opt_df['Unit_price'] < 10 * df['Unit_price'].mean()]
    # opt_df = opt_df[opt_df['Unit_price'] > 0.1 * df['Unit_price'].mean()]
    # opt_df = opt_df[opt_df['R_ds'] < 10 * df['R_ds'].mean()]
    # opt_df = opt_df[opt_df['R_ds'] > 0.1 * df['R_ds'].mean()]
    # opt_df = opt_df[opt_df['Q_g'] < 10 * df['Q_g'].mean()]
    # opt_df = opt_df[opt_df['Q_g'] > 0.1 * df['Q_g'].mean()]
    #
    # # plt.scatter(opt_df.loc[:, 'V_dss'],
    # #             opt_df.loc[:, 'C_ossp1'] * opt_df.loc[:, 'R_ds'], color='g', s=0.8)
    # plt.scatter(opt_df.loc[:, 'V_dss'],
    #             opt_df.loc[:, 'C_ossp1'] * opt_df.loc[:, 'R_ds'], color='g', s=1.2)
    # plt.plot(T, y_, color='navy', label=model, linewidth=1.5)
    # l = plt.legend()
    # plt.xscale('log')
    # plt.yscale('log')
    # plt.xlabel('$V_{dss}$ [V]', fontsize=16)
    # plt.ylabel('$C_{oss,0.1}R_{on}$ [pF-Ω]', fontsize=16)
    # plt.xlim(10, 600)
    #
    # # plt.xlim((200,1000))
    return

def reg_score_and_dump_chained(df, fet_tech, file_name, component_type):
    trained_models = []
    degree = 1
    rmse_scorer = make_scorer(mean_squared_error, squared=False)

    if component_type == 'fet':
        X_new = np.log10(df.loc[:, ['V_dss', 'R_ds']])
        X_new=preproc(X_new, degree)
        # y_new = df[['Unit_price', 'Q_g', 'Q_rr']].copy()
        y_new = pd.DataFrame()
        new_df = pd.DataFrame()
        new_df['RdsCost_product'] = np.log10(df['Unit_price'].astype(float) * df['R_ds'].astype(float))
        new_df['RdsQg_product'] = np.log10(df['Q_g'].astype(float) * df['R_ds'].astype(float))
        new_df['RdsCoss_product'] = np.log10(df['C_oss'].astype(float) * df['R_ds'].astype(float))
        if fet_tech == 'MOSFET':
            new_df['RdsQrr_product'] = np.log10(df['Q_rr'].astype(float) * df['R_ds'].astype(float))
        elif fet_tech == 'GaNFET':
            new_df['RdsQrr_product'] = 0.0

        y_new['Unit_price'] = np.log10(df['Unit_price'].astype(float))
        y_new['Q_g'] = np.log10(df['Q_g'].astype(float))
        if fet_tech == 'MOSFET':
            y_new['Q_rr'] = np.log10(df['Q_rr'].astype(float))
        elif fet_tech == 'GaNFET':
            y_new['Q_rr'] = 0.0
        y_new['C_oss'] = np.log10(df['C_oss'].astype(float))


        order = [0,1,2,3]

        cv = KFold(n_splits=6, random_state=1, shuffle=True)
        scores_df = pd.DataFrame({'scoring_type': ['r^2', 'RMSE', 'MAE']}).set_index('scoring_type')
        gamma_final = 0.1
        alpha_final = 0.1
        score_init = 0
        if fet_tech == 'MOSFET':
            alpha_final = 0.1
            gamma_final = 0.1
        elif fet_tech == 'GaNFET':
            alpha_final = 0.1
            gamma_final = 0.1
            # for gamma in [0.01, 0.1, 0.5, 1.0, 10]:
            #     for alpha in [0.01, 0.1, 0.5, 1.0, 10]:
            #         model = KernelRidge(degree=4, alpha=alpha, gamma=gamma, kernel='rbf')
            #         chain = RegressorChain(base_estimator=model, order=order)
            #         scores = cross_val_score(chain, X_new, y_new, cv=cv, scoring='neg_mean_absolute_error')
            #         if np.mean(scores) > score_init:
            #             score_init = np.mean(scores)
            #             alpha_final = alpha
            #             gamma_final = gamma

        model = KernelRidge(degree=4, alpha=alpha_final, gamma=gamma_final, kernel='rbf')
        model = LinearRegression(fit_intercept=True, normalize=True)

        chain = RegressorChain(base_estimator=model, order=order)
        scores = cross_val_score(chain, X_new, y_new, cv=cv, scoring='r2')
        scores_df.loc['r^2', 'chain'] = np.mean(scores)
        scores = cross_val_score(chain, X_new, y_new, cv=cv, scoring=rmse_scorer)
        scores_df.loc['MSE', 'chain'] = np.mean(scores)
        scores = cross_val_score(chain, X_new, y_new, cv=cv, scoring='neg_mean_absolute_error')
        scores_df.loc['MAE', 'chain'] = np.mean(scores)
        # reg_chain = model.fit(X_new, y_new)
        # y_pred = reg_lin.predict(X)
        reg_chain = chain.fit(X_new, y_new)
        # print(reg_chain.predict(np.array([1, 3, 0.3]).reshape(1, -1)))

        trained_models.append(reg_chain)

    elif component_type =='ind':
        X_new = np.log10(df.loc[:, ['Inductance','Current_rating']])
        # X_new = np.log10(df.loc[:, ['Energy']])

        X_new = preproc(X_new, degree)
        # y_new = df[['Unit_price', 'Q_g', 'Q_rr']].copy()
        y_new = pd.DataFrame()
        # new_df = pd.DataFrame()
        # new_df['RdsCost_product'] = np.log10(df['Unit_price'].astype(float) * df['R_ds'].astype(float))
        # new_df['RdsQg_product'] = np.log10(df['Q_g'].astype(float) * df['R_ds'].astype(float))
        # new_df['RdsCoss_product'] = np.log10(df['C_oss'].astype(float) * df['R_ds'].astype(float))
        # if fet_tech == 'MOSFET':
        #     new_df['RdsQrr_product'] = np.log10(df['Q_rr'].astype(float) * df['R_ds'].astype(float))
        # elif fet_tech == 'GaNFET':
        #     new_df['RdsQrr_product'] = 0.0

        y_new['Unit_price'] = np.log10(df['Unit_price'].astype(float))
        y_new['DCR'] = np.log10(df['DCR'].astype(float))
        y_new['Dimension'] = np.log10(df['Dimension'].astype(float))
        # y_new['Volume'] = np.log10(df['Dimension'].astype(float) * df['Height'].astype(float))

        order = [0,1,2]

        cv = KFold(n_splits=10, random_state=1, shuffle=True)
        scores_df = pd.DataFrame({'scoring_type': ['r^2', 'RMSE', 'MAE']}).set_index('scoring_type')
        gamma_final = 0.1
        alpha_final = 0.1
        score_init = 0

        # for gamma in [0.01, 0.1, 0.5, 1.0, 10]:
        #     for alpha in [0.01, 0.1, 0.5, 1.0, 10]:
        #         model = KernelRidge(degree=4, alpha=alpha, gamma=gamma, kernel='rbf')
        #         chain = RegressorChain(base_estimator=model, order=order)
        #         scores = cross_val_score(chain, X_new, y_new, cv=cv, scoring='neg_mean_absolute_error')
        #         if np.mean(scores) > score_init:
        #             score_init = np.mean(scores)
        #             alpha_final = alpha
        #             gamma_final = gamma

        model = KernelRidge(degree=4, alpha=alpha_final, gamma=gamma_final, kernel='rbf')
        model = LinearRegression(fit_intercept=True, normalize=True)
        chain = RegressorChain(base_estimator=model, order=order)
        scores = cross_val_score(chain, X_new, y_new, cv=cv, scoring='r2')
        scores_df.loc['r^2', 'chain'] = np.mean(scores)
        scores = cross_val_score(chain, X_new, y_new, cv=cv, scoring=rmse_scorer)
        scores_df.loc['MSE', 'chain'] = np.mean(scores)
        scores = cross_val_score(chain, X_new, y_new, cv=cv, scoring='neg_mean_absolute_error')
        scores_df.loc['MAE', 'chain'] = np.mean(scores)
        # reg_chain = model.fit(X_new, y_new)
        # y_pred = reg_lin.predict(X)
        reg_chain = chain.fit(X_new, y_new)
        # print(reg_chain.predict(np.array([1, 3, 0.3]).reshape(1, -1)))

        trained_models.append(reg_chain)

    elif component_type =='cap':
        degree = 2
        X_new = (df.loc[:, ['Capacitance','Rated_volt','Temp_coef_enc']])
        X_new.loc[:,['Capacitance','Rated_volt']] = np.log10(X_new.loc[:,['Capacitance','Rated_volt']])

        # X_new = np.log10(df.loc[:, ['Energy']])

        X_new = preproc(X_new, degree)
        # y_new = df[['Unit_price', 'Q_g', 'Q_rr']].copy()
        y_new = pd.DataFrame()
        # new_df = pd.DataFrame()
        # new_df['RdsCost_product'] = np.log10(df['Unit_price'].astype(float) * df['R_ds'].astype(float))
        # new_df['RdsQg_product'] = np.log10(df['Q_g'].astype(float) * df['R_ds'].astype(float))
        # new_df['RdsCoss_product'] = np.log10(df['C_oss'].astype(float) * df['R_ds'].astype(float))
        # if fet_tech == 'MOSFET':
        #     new_df['RdsQrr_product'] = np.log10(df['Q_rr'].astype(float) * df['R_ds'].astype(float))
        # elif fet_tech == 'GaNFET':
        #     new_df['RdsQrr_product'] = 0.0

        # y_new['Size'] = np.log10(df['Size'].astype(float))
        y_new['Unit_price'] = np.log10(df['Unit_price'].astype(float))
        # y_new['Volume'] = np.log10(df['Dimension'].astype(float) * df['Height'].astype(float))

        order = [0]

        cv = KFold(n_splits=10, random_state=1, shuffle=True)
        scores_df = pd.DataFrame({'scoring_type': ['r^2', 'RMSE', 'MAE']}).set_index('scoring_type')
        gamma_final = 0.1
        alpha_final = 0.1
        score_init = 0

        # for gamma in [0.01, 0.1, 0.5, 1.0, 10]:
        #     for alpha in [0.01, 0.1, 0.5, 1.0, 10]:
        #         model = KernelRidge(degree=4, alpha=alpha, gamma=gamma, kernel='rbf')
        #         chain = RegressorChain(base_estimator=model, order=order)
        #         scores = cross_val_score(chain, X_new, y_new, cv=cv, scoring='neg_mean_absolute_error')
        #         if np.mean(scores) > score_init:
        #             score_init = np.mean(scores)
        #             alpha_final = alpha
        #             gamma_final = gamma

        model = KernelRidge(degree=4, alpha=alpha_final, gamma=gamma_final, kernel='rbf')
        model = LinearRegression(fit_intercept=True, normalize=True)
        chain = RegressorChain(base_estimator=model, order=order)
        scores = cross_val_score(chain, X_new, y_new, cv=cv, scoring='r2')
        scores_df.loc['r^2', 'chain'] = np.mean(scores)
        scores = cross_val_score(chain, X_new, y_new, cv=cv, scoring=rmse_scorer)
        scores_df.loc['MSE', 'chain'] = np.mean(scores)
        scores = cross_val_score(chain, X_new, y_new, cv=cv, scoring='neg_mean_absolute_error')
        scores_df.loc['MAE', 'chain'] = np.mean(scores)
        # reg_chain = model.fit(X_new, y_new)
        # y_pred = reg_lin.predict(X)
        reg_chain = chain.fit(X_new, y_new)
        # print(reg_chain.predict(np.array([1, 3, 0.3]).reshape(1, -1)))

        trained_models.append(reg_chain)


    if component_type == 'fet':
        output_param = 'Coss'
    elif component_type == 'ind':
        output_param = 'Dimension'
    elif component_type == 'cap':
        output_param = 'Unit_price'
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    pd.set_option('display.colheader_justify', 'center')
    pd.set_option('display.precision', 3)
    fet_column_list = ['scoring_type', 'chain']
    print('Training results: ' + str(output_param))
    print(
        tabulate(scores_df.drop_duplicates(inplace=False), headers=fet_column_list, showindex=True,
                 tablefmt='fancy_grid',
                 floatfmt=".3f"))

    # Write all the models to joblib files to be saved
    dump(trained_models, str(file_name) + '_' + output_param + '.joblib')
    return


def reg_score_and_dump(df, output_param, attributes, file_name,pareto=False, chained=True, before_df=np.nan, training_params='main_page_params'):

    trained_models = []
    degree = 1


    rmse_scorer = make_scorer(mean_squared_error, squared=False)
    # Get the X,y for model training
    cv = KFold(n_splits=5, random_state=1, shuffle=True)
    categorical_attributes = ['N', 'P', 'MOSFET', 'SiCFET', 'GaNFET']
    for variable in attributes:
        new_variable_name = 'log10[' + str(variable) + ']'
        df[new_variable_name] = np.log10(df[variable])
        categorical_attributes.append(new_variable_name)
    # if df['Q_rr'].any() > 0:
    #     X, y = X_y_set(df, 'Pack_case', ['V_dss', 'R_ds', 'Unit_price', 'Q_g','N','P','MOSFET','SiCFET','GaNFET'], True, True, False)
    # else:
    #     X, y = X_y_set(df, 'Pack_case', ['V_dss', 'R_ds', 'Unit_price', 'Q_g', 'N','P','MOSFET','SiCFET','GaNFET'], True, True, False)

    X = df.loc[:, categorical_attributes]
    X = preproc(X, degree)
    y = np.log10(df.loc[:, output_param].astype(float))
    output_var_list = ['Q_g','Unit_price']
    output_df = pd.DataFrame()
    for output_var in output_var_list:
        output_df[output_var] = np.log10(df[output_var].astype(float))

    # (X, y) = X_y_set(df, output_param, attributes, log_x=True, more_inputs=False,pareto=pareto)
    order = [0,1]
    cv = KFold(n_splits=5, random_state=1, shuffle=True)
    scores_df = pd.DataFrame({'scoring_type': ['r^2', 'MSE', 'MAE']}).set_index('scoring_type')
    gamma_final = 0.1
    alpha_final = 0.1
    model = KernelRidge(degree=4, alpha=alpha_final, gamma=gamma_final, kernel='rbf')
    chain = RegressorChain(base_estimator=model, order=order)
    reg_chain = chain.fit(X, output_df)
    scores = cross_val_score(chain, X, output_df, cv=cv, scoring='neg_mean_absolute_error')
    dump(reg_chain, 'full_dataset_models_chained.joblib')
    return

    if chained:
        X_new = np.log10(df.loc[:, ['V_dss', 'R_ds']])
        X_new=preproc(X_new, degree)
        y_new = df[['Unit_price', 'Q_g', 'Q_rr']].copy()
        new_df = pd.DataFrame()
        new_df['RdsCost_product'] = np.log10(df['Unit_price'].astype(float) * df['R_ds'].astype(float))
        new_df['RdsQg_product'] = np.log10(df['Q_g'].astype(float) * df['R_ds'].astype(float))
        new_df['RdsCoss_product'] = np.log10(df['C_oss'].astype(float) * df['R_ds'].astype(float))
        new_df['RdsQrr_product'] = np.log10(df['Q_rr'].astype(float) * df['R_ds'].astype(float))


        order = [0]
        if output_param == 'RdsQg_product' or output_param == 'RdsQrr_product' or output_param == 'RdsCoss_product':
            new_df['RdsQg_product'] = np.log10(df['Q_g'].astype(float) * df['R_ds'].astype(float))
            order = [0,1]
        if output_param == 'RdsQrr_product' or output_param == 'RdsCoss_product':
            new_df['RdsCoss_product'] = np.log10(df['C_oss'].astype(float) * df['R_ds'].astype(float))
            # new_df['RdsQrr_product'] = np.log10(df['Q_rr'].astype(float) * df['R_ds'].astype(float))
            order = [0,1,2]
        # if output_param == 'RdsCoss_product':
        #     new_df['RdsCoss_product'] = np.log10(df['C_oss'].astype(float) * df['R_ds'].astype(float))
        #     order=[0,1,2,3]

        cv = KFold(n_splits=5, random_state=1, shuffle=True)
        scores_df = pd.DataFrame({'scoring_type': ['r^2', 'MSE', 'MAE']}).set_index('scoring_type')
        gamma_final = 0.1
        alpha_final = 0.1
        score_init = 0
        for gamma in [0.01, 0.1, 0.5, 1.0, 10]:
            for alpha in [0.01, 0.1, 0.5, 1.0, 10]:
                model = KernelRidge(degree=4, alpha=alpha, gamma=gamma, kernel='rbf')
                chain = RegressorChain(base_estimator=model, order=order)
                scores = cross_val_score(chain, X_new, new_df, cv=cv, scoring='neg_mean_absolute_error')
                if np.mean(scores) > score_init:
                    score_init = np.mean(scores)
                    alpha_final = alpha
                    gamma_final = gamma
        model = KernelRidge(degree=4, alpha=alpha_final, gamma=gamma_final, kernel='rbf')
        chain = RegressorChain(base_estimator=model, order=order)
        scores = cross_val_score(chain, X_new, new_df, cv=cv, scoring='r2')
        scores_df.loc['r^2', 'chain'] = np.mean(scores)
        scores = cross_val_score(chain, X_new, new_df, cv=cv, scoring=rmse_scorer)
        scores_df.loc['MSE', 'chain'] = np.mean(scores)
        scores = cross_val_score(chain, X_new, new_df, cv=cv, scoring='neg_mean_absolute_error')
        scores_df.loc['MAE', 'chain'] = np.mean(scores)
        # reg_chain = model.fit(X_new, y_new)
        # y_pred = reg_lin.predict(X)
        reg_chain = chain.fit(X_new, new_df)
        print(reg_chain.predict(np.array([1, 3, 0.3]).reshape(1, -1)))
        trained_models.append(reg_chain)

        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 1000)
        pd.set_option('display.colheader_justify', 'center')
        pd.set_option('display.precision', 3)
        fet_column_list = ['scoring_type', 'chain']
        print('Training results: ' + str(output_param))
        print(
            tabulate(scores_df.drop_duplicates(inplace=False), headers=fet_column_list, showindex=True,
                     tablefmt='fancy_grid',
                     floatfmt=".3f"))

        # Write all the models to joblib files to be saved
        dump(trained_models, str(file_name) + '_' + output_param + '.joblib')
        return



        # y = pd.DataFrame[np.log10(df.loc[:, 'Unit_price'].astype(float)), np.log10(df.loc[:, 'Q_g'].astype(float)), np.log10(df.loc[:, 'C_oss'].astype(float))]
        # y_new['Unit_price'] = np.log10(y_new['Unit_price'].astype(float))
        # y_new['Q_g'] = np.log10(y_new['Q_g'].astype(float))
        # y_new['Q_rr'] = np.log10(y_new['Q_rr'].astype(float))
        # y_new['C_oss'] = np.log10(y_new['C_oss'].astype(float))
        #
        # #yhat = chain.predict(np.array([[1, 3, 0.3]]))


        # Creates a dictionary of the optimal model parameters for a model type. Can then use these params to update the
        # object attributes of the chosen model type. Could also figure this out in a separate function call in main loop
        # and manually update the parameters.
        #model_params_dict = {'linear': {'degree': np.linspace(0.1,2.0,num=19)}}
        #opt_model_params = param_opt(X, y, attributes, 'linear', 1.0,model_params_dict)

        # scaler = StandardScaler()
        # X = scaler.fit_transform(X)

    # Preprocess the data to have the bases that works best for linear regression.
    # X = preproc(X, degree)

    # Create object with the associated model type attributes
    # model_obj = opt_model('linear')
    #X = manual_preproc(X, model_obj.degree, attributes)

    # Linear regression
    cv = KFold(n_splits=5, random_state=1, shuffle=True)
    if output_param == 'RdsQrr_product' or output_param == 'RdsCoss_product':
        cv = KFold(n_splits=2, random_state=1, shuffle=True)

    # if df == 'SiC_df':
    #     cv = KFold(n_splits=2, random_state=1, shuffle=True)

    # scores_df = pd.DataFrame({'scoring_type':['r^2','MSE','MAE'],'linear':[], 'rf':[], 'knn':[], 'adaboost':[],'elastic_net':[],'lasso':[],'ridge':[]}).set_index('scoring_type')
    scores_df = pd.DataFrame({'scoring_type':['r^2','RMSE','MAE']}).set_index('scoring_type')

    if chained:
        model = RandomForestRegressor(min_samples_split=2, random_state=1)
        model = BayesianRidge()
        gamma_final=0.1
        alpha_final=0.1
        score_init=0
        for gamma in [0.01, 0.1, 0.5, 1.0, 10]:
            for alpha in [0.01, 0.1, 0.5, 1.0, 10]:
                model = KernelRidge(degree=4, alpha=alpha,gamma=gamma,kernel='rbf')
                chain = RegressorChain(base_estimator=model, order=order)
                scores = cross_val_score(chain, X_new, new_df, cv=cv, scoring='r2')
                if np.mean(scores) > score_init:
                    score_init = np.mean(scores)
                    alpha_final = alpha
                    gamma_final = gamma
        model = KernelRidge(degree=4, alpha=alpha_final, gamma=gamma_final, kernel='rbf')
        chain = RegressorChain(base_estimator=model, order=order)
        scores = cross_val_score(chain, X_new, new_df, cv=cv, scoring='r2')
        scores_df.loc['r^2', 'chain'] = np.mean(scores)
        scores = cross_val_score(chain, X_new, new_df, cv=cv, scoring=rmse_scorer)
        scores_df.loc['MSE', 'chain'] = np.mean(scores)
        scores = cross_val_score(chain, X_new, new_df, cv=cv, scoring='neg_mean_absolute_error')
        scores_df.loc['MAE', 'chain'] = np.mean(scores)
        # reg_chain = model.fit(X_new, y_new)
        # y_pred = reg_lin.predict(X)
        reg_chain = chain.fit(X_new, new_df)
        print(reg_chain.predict(np.array([1, 3, 0.3]).reshape(1, -1)))
        trained_models.append(reg_chain)



    model = LinearRegression(fit_intercept=True, normalize=True)

    # model.fit(X, y)
    # bef_df = before_df[before_df[output_param] != 0]
    # bef_df = bef_df[bef_df[output_param] != np.nan]
    # (X_before, y_before) = X_y_set(bef_df, output_param, attributes, log_x=True, more_inputs=False, pareto=pareto)
    # X_before = preproc(X_before, degree)
    # y_pred = model.predict(X_before)
    # mae_score = mean_absolute_error(y_before, y_pred)

    scores = cross_validate(model, X, y, cv=cv,scoring=['r2','neg_mean_absolute_error'], return_train_score=True)
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
    #y_pred = reg_lin.predict(X)
    trained_models.append(reg_lin)

    # Write all the models to joblib files to be saved
    dump(trained_models, str(file_name) + '_' + output_param + '.joblib')
    return

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

    # Random forest regression
    # Create object with the associated model type attributes
    model = RandomForestRegressor(min_samples_split = model_obj.min_samples_split, random_state=0)
    scores = cross_val_score(model, X, y, cv=cv, scoring='r2')
    scores_df.loc['r^2', 'rf'] = np.mean(scores)
    scores = cross_val_score(model, X, y, cv=cv, scoring=rmse_scorer)
    scores_df.loc['MSE', 'rf'] = np.mean(scores)
    scores = cross_val_score(model, X, y, cv=cv, scoring='neg_mean_absolute_error')
    scores_df.loc['MAE', 'rf'] = np.mean(scores)
    # print('Model: Random forest')
    # print('Accuracy: %.3f (%.3f)' % (np.mean(scores), np.std(scores)))
    # scores_df.loc[0, 'rf'] = np.mean(sco    model_obj = opt_model('random_forest')res)
    reg_rf = model.fit(X, y)
    trained_models.append(reg_rf)

    # K-nearest-neighbors regression
    model_obj = opt_model('knn')
    model = KNeighborsRegressor(n_neighbors=model_obj.n_neighbors, weights=model_obj.weights)
    scores = cross_val_score(model, X, y, cv=cv, scoring='r2')
    scores_df.loc['r^2', 'knn'] = np.mean(scores)
    scores = cross_val_score(model, X, y, cv=cv, scoring='neg_mean_squared_error')
    scores_df.loc['MSE', 'knn'] = np.mean(scores)
    scores = cross_val_score(model, X, y, cv=cv, scoring='neg_mean_absolute_error')
    scores_df.loc['MAE', 'knn'] = np.mean(scores)

    # print('Model: K-nearest neighbors')
    # print('Accuracy: %.3f (%.3f)' % (np.mean(scores), np.std(scores)))
    # scores_df.loc[0, 'knn'] = np.mean(scores)
    reg_knn = model.fit(X, y)
    trained_models.append(reg_knn)

    # Adaboosting regression
    model_obj = opt_model('adaboosting')
    model = AdaBoostRegressor(random_state=0, learning_rate=model_obj.learning_rate, n_estimators=model_obj.n_estimators)
    scores = cross_val_score(model, X, y, cv=cv, scoring='r2')
    scores_df.loc['r^2', 'adaboost'] = np.mean(scores)
    scores = cross_val_score(model, X, y, cv=cv, scoring='neg_mean_squared_error')
    scores_df.loc['MSE', 'adaboost'] = np.mean(scores)
    scores = cross_val_score(model, X, y, cv=cv, scoring='neg_mean_absolute_error')
    scores_df.loc['MAE', 'adaboost'] = np.mean(scores)
    # print('Model: Adaboosting')
    # print('Accuracy: %.3f (%.3f)' % (np.mean(scores), np.std(scores)))
    # scores_df.loc[0, 'adaboost'] = np.mean(scores)
    reg_ada = model.fit(X, y)
    trained_models.append(reg_ada)

    # MLP regression
    model_obj = opt_model('mlp')
    model = MLPRegressor(random_state=1, max_iter=model_obj.max_iter,activation=model_obj.activation,solver=model_obj.solver,
                         alpha=model_obj.alpha)
    # print('Model: MLP')
    # print('Accuracy: %.3f (%.3f)' % (np.mean(scores), np.std(scores)))
    reg_mlp = model.fit(X, y)
    trained_models.append(reg_mlp)

    model_obj = opt_model('elastic_net')
    model = ElasticNet(random_state=1, alpha=0.1)
    # scaler = StandardScaler()
    # scaled_data = scaler.fit_transform(X)
    scores = cross_val_score(model, X, y, cv=cv, scoring='r2')
    scores_df.loc['r^2', 'elastic_net'] = np.mean(scores)
    scores = cross_val_score(model, X, y, cv=cv, scoring='neg_mean_squared_error')
    scores_df.loc['MSE', 'elastic_net'] = np.mean(scores)
    scores = cross_val_score(model, X, y, cv=cv, scoring='neg_mean_absolute_error')
    scores_df.loc['MAE', 'elastic_net'] = np.mean(scores)
    # print('Model: MLP')
    # print('Accuracy: %.3f (%.3f)' % (np.mean(scores), np.std(scores)))
    reg_elastic_net = model.fit(X, y)
    # scores_df.loc[0, 'elastic_net'] = np.mean(scores)
    trained_models.append(reg_elastic_net)

    model = Ridge(alpha=0.1, random_state=1)
    # scaler = StandardScaler()
    # scaled_data = scaler.fit_transform(X)
    scores = cross_val_score(model, X, y, cv=cv, scoring='r2')
    scores_df.loc['r^2', 'ridge'] = np.mean(scores)
    scores = cross_val_score(model, X, y, cv=cv, scoring='neg_mean_squared_error')
    scores_df.loc['MSE', 'ridge'] = np.mean(scores)
    scores = cross_val_score(model, X, y, cv=cv, scoring='neg_mean_absolute_error')
    scores_df.loc['MAE', 'ridge'] = np.mean(scores)
    # print('Model: MLP')
    # print('Accuracy: %.3f (%.3f)' % (np.mean(scores), np.std(scores)))
    reg_ridge = model.fit(X, y)
    # scores_df.loc[0, 'elastic_net'] = np.mean(scores)
    trained_models.append(reg_ridge)

    model = Lasso(random_state=1, alpha=0.1)
    # scaler = StandardScaler()
    # scaled_data = scaler.fit_transform(X)
    scores = cross_val_score(model, X, y, cv=cv, scoring='r2')
    scores_df.loc['r^2', 'lasso'] = np.mean(scores)
    scores = cross_val_score(model, X, y, cv=cv, scoring='neg_mean_squared_error')
    scores_df.loc['MSE', 'lasso'] = np.mean(scores)
    scores = cross_val_score(model, X, y, cv=cv, scoring='neg_mean_absolute_error')
    scores_df.loc['MAE', 'lasso'] = np.mean(scores)
    # print('Model: MLP')
    # print('Accuracy: %.3f (%.3f)' % (np.mean(scores), np.std(scores)))
    reg_lasso = model.fit(X, y)
    # scores_df.loc[0, 'elastic_net'] = np.mean(scores)
    trained_models.append(reg_lasso)

    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    pd.set_option('display.colheader_justify', 'center')
    pd.set_option('display.precision', 3)
    fet_column_list = ['scoring_type','linear', 'rf', 'knn', 'adaboost','mlp','elastic_net','ridge','lasso'
                       ]
    print('Training results: ' + str(output_param))
    print(
        tabulate(scores_df.drop_duplicates(inplace=False), headers=fet_column_list, showindex=True,
                 tablefmt='fancy_grid',
                 floatfmt=".3f"))

    # Write all the models to joblib files to be saved
    # dump(trained_models, str(file_name) + '_' + output_param + '.joblib')


'''
    This function tranforms the basis to have the desired form. Takes as input the highest degree, outputs a new
    bases in the form [1, a^degree, b^degree, a^(degree/2), b^(degree/2), a^-degree, b^-degree], or more, depending on 
    how long the length of input features X is.
'''
def manual_preproc(X, poly_degree, attributes, simple=False,opt=False):
    letter_list = list(ascii_lowercase)
    #var_list = ['V_dss','Unit_price']
    var_list = attributes
    num_vars = 0
    if opt:
        for i in range(len(X)):
            str = letter_list[i]
            locals()[str] = X[i]
            num_vars += 1
        array= [1, locals()[letter_list[0]] ** poly_degree,locals()[letter_list[0]] ** (0.5*poly_degree),locals()[letter_list[0]] ** -poly_degree,locals()[letter_list[1]] ** poly_degree,locals()[letter_list[1]] ** (0.5 * poly_degree),locals()[letter_list[1]] ** -poly_degree]
        return array
        array = np.ones((2,1))

        array = np.concatenate((array[0], [locals()[letter_list[i]] ** poly_degree]))
        if not simple:
            array = np.concatenate((array[0], [locals()[letter_list[i]] ** (0.5*poly_degree)]))
            array = np.concatenate((array[0], [locals()[letter_list[i]] ** -poly_degree]))
        array = np.concatenate((array[1], [locals()[letter_list[i]] ** poly_degree]))
        if not simple:
            array = np.concatenate((array[1], [locals()[letter_list[i]] ** (0.5 * poly_degree)]))
            array = np.concatenate((array[1], [locals()[letter_list[i]] ** -poly_degree]))

        return array

    elif (not opt and type(X) == np.array or type(X) == np.ndarray):
        # create a variable for every attribute in X
        if len(X) == 2:
            str = letter_list[0]
            locals()[str] = X[0]
            str = letter_list[1]
            locals()[str] = X[1]
            array = [1, locals()[letter_list[0]] ** poly_degree, locals()[letter_list[0]] ** (0.5 * poly_degree),
                     locals()[letter_list[0]] ** -poly_degree, locals()[letter_list[1]] ** poly_degree,
                     locals()[letter_list[1]] ** (0.5 * poly_degree), locals()[letter_list[1]] ** -poly_degree]
            return array
        else:
            for i in range(len(X[0])):
                str = letter_list[i]
                locals()[str] = X[:,i]
                num_vars += 1
                #exec("%s = %f" % (str,  X[:, i]))

    else:
        # create a variable for every attribute in X
        for i in range(len(X.columns)):
            str = letter_list[i]
            locals()[str] = X.loc[:,var_list[i]]
            num_vars += 1
            #exec("%s = %f" % (str, X.loc[:,var_list[i]]))

    # Create the array of trainable X
    array = np.ones((len(X), 1))
    for i in range(num_vars):
        array = np.c_[array, locals()[letter_list[i]] ** poly_degree]
        if not simple:
            array = np.c_[array, locals()[letter_list[i]] ** (0.5 * poly_degree)]
            array = np.c_[array, locals()[letter_list[i]] ** -poly_degree]

    return array

'''
    Load the trained models, using the joblib file naming system: str(file_name) + '_' + output_param (then + .joblib).
    Returns a list of trained models, as an array with the output parameter name as the first entry, and then each 
    subsequent model type following, for every entry in output_param_list
'''

def load_models(output_param_list, file_name, component = 'fet', param = 'Rds'):
    if component == 'fet':

        # need two different cases because the area models looks different because of classification
        if param not in ['area']:
            trained_models_dict = {}
            df = pd.DataFrame(columns=['output_param','linear', 'random forest', 'knn', 'adaboosting', 'mlp'])
            for output_param in output_param_list:
                trained_models = load(str(file_name) + '_' + output_param + '.joblib')
                trained_models_dict[output_param] = trained_models
                output_param_lst = [output_param]
                full_list = output_param_lst.extend(trained_models)
                df_series = pd.DataFrame([output_param_lst], columns=['output_param','linear', 'random_forest', 'knn', 'adaboosting', 'mlp'])
                #df_series = pd.DataFrame([output_param, trained_models],columns=['output parameter', 'linear', 'random forest', 'knn', 'adaboosting', 'mlp'])
                df= pd.concat([df, df_series],axis=0)

        else:
            trained_models_dict = {}
            df = pd.DataFrame(columns=['output_param', 'knn', 'random_forest', 'grad_boost', 'decision_tree'])
            for output_param in output_param_list:
                trained_models = load(str(file_name) + '_' + output_param + '.joblib')
                trained_models_dict[output_param] = trained_models
                output_param_lst = [output_param]
                full_list = output_param_lst.extend(trained_models)
                df_series = pd.DataFrame([output_param_lst],
                                         columns=['output_param', 'knn', 'random_forest', 'grad_boost', 'decision_tree'])
                # df_series = pd.DataFrame([output_param, trained_models],columns=['output parameter', 'linear', 'random forest', 'knn', 'adaboosting', 'mlp'])
                df = pd.concat([df, df_series], axis=0)


    elif component == 'ind':
        trained_models_dict = {}
        df = pd.DataFrame(columns=['output_param', 'linear', 'random forest', 'knn', 'adaboosting', 'mlp'])
        for output_param in output_param_list:
            trained_models = load(str(file_name) + '_' + output_param + '.joblib')
            trained_models_dict[output_param] = trained_models
            output_param_lst = [output_param]
            full_list = output_param_lst.extend(trained_models)
            df_series = pd.DataFrame([output_param_lst],
                                     columns=['output_param', 'linear', 'random_forest', 'knn', 'adaboosting', 'mlp'])
            # df_series = pd.DataFrame([output_param, trained_models],columns=['output parameter', 'linear', 'random forest', 'knn', 'adaboosting', 'mlp'])
            df = pd.concat([df, df_series], axis=0)

    return df

'''
    This function makes predictions to fill in sparse data
'''
def bayesion_prediction(df):
    # first train on the old data
    new_val = BayesianRidge()
'''
    This function takes columns with sparse data and uses probabilistic models to add more information.
'''
def add_predicted_vals(df, attributes):
    for attr in attributes:
        non_var_df = df[df[attr]!=0.0]
        # non_var_df[attr] = non_var_d.apply(bayesion_prediction)

'''
    This function performs manual k-fold splitting, in case we want to do things with each split individually, such as
    seeing the individual scores of each split. Takes as input the df, the model name to train with,
    the desired output variable to train the models on, and the polynomial degree to which we set our basis.
    Prints the performance parameters, returns nothing.
'''

def k_fold_splitter(df, output_param, model_name, degree):
    attributes = ['V_dss', 'Unit_price']
    (X,y) =X_y_set(df, output_param)
    X = manual_preproc(X, degree)
    if model_name == 'linear':
        model = LinearRegression()
    elif model_name == 'random forest':
        model = RandomForestRegressor()
    elif model_name == 'knn':
        model = KNeighborsRegressor()
    elif model_name == 'adaboost':
        model = AdaBoostRegressor()
    elif model_name == 'mlp':
        model = MLPRegressor()

    cv = KFold(n_splits=5, random_state=1, shuffle=True)
    for train_index, test_index in cv.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        reg = model.fit(X_train, y_train)

        # the coefficients of the regression line
        print('coef: {} \n yint: {}'.format(reg.coef_, reg.intercept_))
        y_pred = reg.predict(X_test)
        print(reg.score(X_test, y_test))
        mse = mean_squared_error(y_test, y_pred)
        print('mean squared error: %.2f' % mse)
        cod = r2_score(y_test, y_pred)
        print('coefficient of determination on testing: %.2f' % cod)

        # see how well the prediction does with the training data itself
        reg = model.fit(X_train, y_train)
        y_pred = reg.predict(X_test)
        cod = r2_score(y_test, y_pred)
        print('coefficient of determination on training: %.2f' % cod)

'''
    This function takes a single data point (V,C) and predicts the desired output variable. If specified, plots
    that point amidst the data points from the data set with similar (V,C) to visualize how close our prediction is.
    Takes as input the trained_model, the (V,C) data point, the degree of the basis, the output variable to get the
    database data points for, and whether or not to plot everything.
    Outputs the predicted value and the visualized graph.
'''


def point_pred_w_data(trained_model, point, degree, output_var, to_plot):
    (V_dss, cost) = point
    user_df = pd.DataFrame(np.array([[float(V_dss), float(cost)]]),
                           columns=['V_dss', 'Unit_price'])
    output = manual_preproc(user_df, degree)
    power = trained_model.predict(output)
    print('Predicted R_ds with user input voltage and price limit: %f' % 10 ** power)

    if to_plot:
        range_lim = 0.15
        v_rangemin = math.floor(float(V_dss)) - math.floor(range_lim * math.floor(float(V_dss)))
        v_rangemax = math.ceil(float(V_dss)) + math.ceil(range_lim * math.ceil(float(V_dss)))
        c_rangemin = math.floor((float(cost) - range_lim * math.floor(float(cost))))
        c_rangemax = math.ceil((float(cost) + range_lim * math.floor(float(cost))))

        user_df = user_df[(user_df['V_dss'] >= v_rangemin) & (user_df['V_dss'] <= v_rangemax) &
                          (user_df['Unit_price'] >= c_rangemin) & (user_df['Unit_price'] <= c_rangemax)]

        # now plot the results
        fig = plt.figure()
        ax = plt.axes(projection='3d')

        # first plot the database data
        zdata = user_df.loc[:, output_var]
        xdata = user_df.loc[:, 'V_dss']
        ydata = user_df.loc[:, 'Unit_price']
        ax.scatter3D(xdata, ydata, zdata, color='g')

        # then plot the predicted point
        ax.scatter3D(float(V_dss), float(cost), 10 ** power, color='r')


'''
    This function plots the testing and predicted y outputs for easy visual assessment of performance. Takes a list 
    of X testing, y testing, and y prediction data points.
'''


def compare_train_test(X_test, y_test, y_pred):
    plt.scatter(X_test, y_test, color='black')
    plt.plot(X_test, y_pred, color='blue', linewidth=3)
    plt.show()


'''
    This function performs an automatic preprocessing of the basis into the desired power. However, the power must
    be an integer, which is why we separately define our own preprocessing function.
'''


def preproc(X, degree):
    poly = PolynomialFeatures(degree)
    preprocessed_X = poly.fit_transform(X)
    return preprocessed_X



def manual_preproc_opt(X, poly_degree):
    if type(X) == np.array or type(X) == np.ndarray:
        a = X[0]
        b = X[1]
    else:
        a = X.loc[:, 'V_dss']
        b = X.loc[:, 'Unit_price']
    print(a,b)
    array = np.ones((1, 1))
    array = np.c_[array, a ** poly_degree]
    array = np.c_[array, b ** poly_degree]
    array = np.c_[array, a ** (0.5 * poly_degree)]
    array = np.c_[array, b ** (0.5 * poly_degree)]
    array = np.c_[array, a ** -poly_degree]
    array = np.c_[array, b ** -poly_degree]

    return array


'''
    This function takes a list of all the trained models and uses them to predict an exact point. Plots the point for 
    the various models in 3D, then plots the data points from the database in the nearby ranges. Can take any of the 3 
    output_var (R_ds, Q_g, R*Q product) to make a prediction based on the user-supplied V_dss and cost for the part.
    Makes a 3D plot with axis of V_dss, unit price, and the output_var. Outputs the predicted output_var values
    for all the different models, and 3D plots them.
'''


def threeD_models_plus_data(trained_models, fet_df, point, degree, output_var):
    (V_dss, cost) = point
    user_df = pd.DataFrame(np.array([[float(V_dss), float(cost)]]),
                           columns=['V_dss', 'Unit_price'])
    output = manual_preproc(user_df, degree)
    [reg_lin, reg_rf, reg_knn, reg_ada, reg_mlp] = trained_models

    power_lin = reg_lin.predict(output)
    print('Predicted lin_reg %s with user input voltage and price limit: ' % output_var)
    print(10 ** power_lin)

    power_rf = reg_rf.predict(output)
    print('Predicted random_forest %s with user input voltage and price limit: ' % output_var)
    print(10 ** power_rf)

    power_knn = reg_knn.predict(output)
    print('Predicted k-nearest neighbors R_ds with user input voltage and price limit: ' % output_var)
    print(10 ** power_knn)

    power_ada = reg_ada.predict(output)
    print('Predicted adaboost %s with user input voltage and price limit: ' % output_var)
    print(10 ** power_ada)

    power_mlp = reg_mlp.predict(output)
    print('Predicted MLP %s with user input voltage and price limit: ' % output_var)
    print(10 ** power_mlp)

    # Now find the points that are nearby the user input value point, and plot those along with the user's point
    # and prediction to see if the prediction is close to actual data from the database
    range_lim = 0.15
    v_rangemin = math.floor(float(V_dss)) - math.floor(range_lim * math.floor(float(V_dss)))
    v_rangemax = math.ceil(float(V_dss)) + math.ceil(range_lim * math.ceil(float(V_dss)))
    c_rangemin = math.floor((float(cost) - range_lim * math.floor(float(cost))))
    c_rangemax = math.ceil((float(cost) + range_lim * math.floor(float(cost))))

    range_df = fet_df[
        (fet_df['V_dss'] >= v_rangemin) & (fet_df['V_dss'] <= v_rangemax) & (fet_df['Unit price'] >= c_rangemin) & (
                fet_df['Unit_price'] <= c_rangemax)]
    print(range_df.loc[:, output_var])

    # now plot the results
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # first the data around the query point
    zdata = range_df.loc[:, output_var]
    xdata = range_df.loc[:, 'V_dss']
    ydata = range_df.loc[:, 'Unit_price']

    ax.scatter(xdata, ydata, zdata, color='g')

    # then the predicted point for each of the predictions
    ax.scatter3D(float(V_dss), float(cost), 10 ** power_lin, color='r', label='linear')
    ax.scatter3D(float(V_dss), float(cost), 10 ** power_rf, color='b', label='random forest')
    ax.scatter3D(float(V_dss), float(cost), 10 ** power_knn, color='c', label='knn')
    ax.scatter3D(float(V_dss), float(cost), 10 ** power_ada, color='y', label='adaboosting')
    ax.scatter3D(float(V_dss), float(cost), 10 ** power_mlp, color='m', label='mlp')
    ax.set_zlim(0, 0.01)
    plt.legend()
    plt.show()


'''
    This visualization function will take the model and plot it on a linspace. Set either a specific voltage or
    cost, which turns the problem into a 2-D plottable function where we can compare the predicted line with the
    data with one of voltage or cost set. In the creation of the linspace, we need to manually preprocess the array to
    match the basis we have set for our model so that we can get the values at the points across the line.
    Takes as input the trained_model, the dataframe already adjusted for N-channel and MOSFET,
    the degree for preprocessing, whether we want to keep voltage or cost constant, the output variable we trained on.
    We also want to plot at multiple constant values of v or c, including the relevant data points. Add if-type
    statement to handle this.
    Note: want to add this into the visualization scheme.
'''


def model_plus_data(trained_model_list, df, degree, volt_or_cost_const, output_var):
    attributes = ['V_dss', 'Unit_price']
    #X = manual_preproc(new_df.loc[:, attributes], degree)
    #y = np.array(np.log10(new_df.loc[:, output_var]))
    # eventually want to set this to be relative to the opposite parameter, bc the parameter limits will change
    # depending on the opposite parameter values we're looking at.
    max_cost = 10.0
    min_cost = 0.05
    max_volt = 1000
    min_volt = 1.0
    const_v_list = [60.0, 100.0, 600.0]
    const_c_list = [0.50, 1.00, 5.00]
    # If we want to keep voltage constant and plot a variable wrt cost, create a linspace of costs
    if volt_or_cost_const == 'volt':

        for const_volt in const_v_list:
            const_var = 'V_dss'
            indep_var = 'Unit_price'
            # create the linspace from almost 0 to the max cost we want to look at with 100 points in between
            T = np.linspace(min_cost, max_cost, 1000)
            T_copy = T
            # In order to have our V column and our C column to make look like the X data the model coefficients are going
            # to be multiplied by, we need to add the next column to be the constant voltage we want.
            T2 = np.full((1, len(T)), const_volt)
            T = np.append(T2, [T], axis=0)


    # If we want to keep cost constant and plot a variable wrt voltage, create linspace of voltages. For this, will
    # need to eventually add in some sort of ranges around cost.
    elif volt_or_cost_const== 'cost':

        for const_cost in const_c_list:
            const_var = 'Unit_price'
            indep_var = 'V_dss'
            T = np.linspace(min_volt, max_volt, 1000)
            T_copy = T
            T2 = np.full((1, len(T)), const_cost)
            T = np.append([T], T2, axis=0)

    T = np.swapaxes(T, 0, 1)
    T = manual_preproc(T, degree)
    # do for every trained model in the list
    y_ = trained_model_list[1].predict(T)

    # Get the relevant data points
    attributes = ['V_dss', 'Unit_price']
    '''
    X_inter = new_df.loc[:, attributes]
    y = new_df.loc[:, output_var]
    '''
    X = df.loc[:, attributes]
    y = df.loc[:, output_var]
    X_test = X[(X[const_var] == const_volt[0])]
    xdata = X_test.loc[:, indep_var]
    ydata = np.array(np.log10(y.loc[X_test.index]))

    # Now plot the predicted y points based on the model over the linspace, as well as the data points around it.
    # For constant voltage, can plot just the components with that exact set voltage.
    # For constant cost, have to plot within a small range around that cost because very few parts have that exact
    # cost.
    plt.clf()
    plt.ylim(0, 4)
    plt.xlim(0, 2)
    plt.plot(T_copy, 10 ** y_, color='navy', label=trained_model_list[1],linewidth=.6)
    plt.scatter(xdata, 10 ** ydata, color='g', s=1.0)
    plt.legend()
    #plt.xlim(0, 2)
    #plt.ylim(0, 0.4 * 10 ** -8)
    plt.xlabel('Cost [$]')
    plt.ylabel('R_ds [Ohm]')
    plt.title('Preprocessed regression at 60V')

    plt.axis('tight')
    plt.show()


def model_plus_data_vs_Vdss_multInputs(df, x_var, y_var1, y_var2, model, file_name, percent_keep=1.00):
    df = df[df['R_ds']>0.005]
    df = df[df['R_ds']<0.05]
    df = df[df['Unit_price'] > 1]
    df = df[df['Unit_price'] < 5]

    max_volt = 1000
    min_volt = 10.0
    T = np.linspace(min_volt, max_volt, 1000)
    T = np.log10(T)
    model_list = ['RdsCost_product','RdsQg_product','RdsQrr_product']
    # model = 'random_forest'
    fet_reg_models_df= load_models(model_list, file_name)
    fet_reg_models_df = fet_reg_models_df.reset_index()
    T_copy = T
    T = preproc(T.reshape(-1,1), 1)
    if (y_var1 == 'Unit_price'):
        product_num = 0
    if (y_var1 == 'Q_g'):
        product_num = 1
    if (y_var1 == 'Q_rr'):
        product_num = 2
    RdsCost_y_ = []
    T = np.linspace(min_volt, max_volt, 1000)
    for volt in T:
        #print(volt)
        RdsCost_y_.append(10**(fet_reg_models_df.loc[product_num, model].predict(preproc(np.array([np.log10(volt), np.log10(0.01), np.log10(3)]).reshape(1, -1), 2))[0]))
    #RdsCost_y_ = fet_reg_models_df.loc[product_num, model].predict(np.array([T]))
    #y_ = model_df.loc[.predict(T)
    attributes = ['V_dss', 'Unit_price']

    # X = df.loc[:, x_var]
    # y = df.loc[:, y_var]
    x_var_params = Visualization_parameters('scatter', 'fet',x_var)
    x_var_params.get_params()
    y_var1_params = Visualization_parameters('scatter', 'fet',y_var1)
    y_var1_params.get_params()
    y_var2_params = Visualization_parameters('scatter', 'fet',y_var2)
    y_var2_params.get_params()

    # xdata = X_test.loc[:, indep_var]
    # ydata = np.array(np.log10(y.loc[X_test.index]))
    plt.scatter(df.loc[:, x_var],
                df.loc[:, y_var1]*df.loc[:, y_var2], color = 'g', s=0.8)
    plt.plot(T, RdsCost_y_, color='navy', label=model, linewidth=.9)
    #plt.scatter(X, Y, color='g', s=1.0)
    plt.legend()
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Log[%s]' % x_var_params.label)
    plt.ylabel('Log[%s * %s]' % (y_var1_params.label, y_var2_params.label))
    plt.title('{} * {} vs. Vdss for N-channel FETs, lowest {:0.2f}'.format(y_var1_params.label, y_var2_params.label, percent_keep))
    # plt.legend(['MOSFET','SiCFET','GaNFET'])
    plt.show()
    print('plot made')

def model_plus_data_vs_Vdss(df, x_var, y_var1, y_var2, model, file_name, component='fet',degree=1,percent_keep=1.00):
    if component == 'fet':
        max_volt = 1000
        min_volt = 10.0
        T = np.linspace(min_volt, max_volt, 1000)
        T = np.log10(T)
        model_list = ['RdsCost_product','RdsQg_product','RdsQrr_product']
        # model = 'random_forest'
        fet_reg_models_df= load_models(model_list, file_name)
        fet_reg_models_df = fet_reg_models_df.reset_index()
        T_copy = T
        T = preproc(T.reshape(-1,1), 2)
        if (y_var1 == 'Unit_price'):
            product_num = 0
        if (y_var1 == 'Q_g'):
            product_num = 1
        if (y_var1 == 'Q_rr'):
            product_num = 2
        y_ = []
        T = np.linspace(min_volt, max_volt, 1000)
        for volt in T:
            #print(volt)
            y_.append(10**(fet_reg_models_df.loc[product_num, model].predict(preproc(np.array([np.log10(volt)]).reshape(1, -1), degree))[0]))
        #RdsCost_y_ = fet_reg_models_df.loc[product_num, model].predict(np.array([T]))
        #y_ = model_df.loc[.predict(T)
        attributes = ['V_dss', 'Unit_price']

    elif component == 'ind':
        # set a 1/current rating if we are looking at DCR_Inductance, set a DCR if looking at DCR_Cost, set an energy and
        # price range if looking at Dimension
        T = np.linspace(df['Inductance'].min(), df['Inductance'].max(), 1000)
        T = np.log10(T)
        model_list = ['DCR_Inductance_product', 'DCR_Cost_product', 'Dimension']

        fet_reg_models_df = load_models(model_list, file_name, component = 'ind')
        fet_reg_models_df = fet_reg_models_df.reset_index()
        T_copy = T
        T = preproc(T.reshape(-1, 1), 2)
        if (y_var1 == 'Inductance'):
            product_num = 0
        elif (y_var1 == 'Unit_price'):
            product_num = 1
        elif (y_var1 == 'Dimension'):
            product_num = 2

        plot_setpoints = Visualization_parameters('scatter', component, variable=model_list[product_num],refactor=False)
        plot_setpoints.get_params()

        df = df[df[plot_setpoints.set_var] > plot_setpoints.set_var_min]
        df = df[df[plot_setpoints.set_var] < plot_setpoints.set_var_max]

        y_ = []
        if product_num in [0,1]:
            T = np.linspace(df['Inductance'].min(), df['Inductance'].max(), 1000)
            for ind in T:
                # print(volt)
                if product_num == 0:
                    y_.append(10 ** (
                    fet_reg_models_df.loc[product_num, model].predict(preproc(np.array([np.log10(ind), (np.log10(2/(plot_setpoints.set_var_max+plot_setpoints.set_var_min)))]).reshape(1, -1), degree))[
                        0]))
                elif product_num == 1:
                    y_.append(10 ** (
                        fet_reg_models_df.loc[product_num, model].predict(preproc(np.array([np.log10(ind), (
                            np.log10((plot_setpoints.set_var_max + plot_setpoints.set_var_min)/2))]).reshape(1, -1),
                                                                                  degree))[
                            0]))
        elif product_num in [2]:
            T = np.linspace(df['Unit_price'].min(), df['Unit_price'].max(), 1000)
            for cost in T:
                if product_num == 2:
                    y_.append(10 ** (
                        fet_reg_models_df.loc[product_num, model].predict(preproc(np.array([(
                            np.log10((plot_setpoints.set_var_max + plot_setpoints.set_var_min)/2)),np.log10(cost)]).reshape(1, -1),
                                                                                  degree))[
                            0]))
        # RdsCost_y_ = fet_reg_models_df.loc[product_num, model].predict(np.array([T]))
        # y_ = model_df.loc[.predict(T)
        attributes = ['V_dss', 'Unit_price']

    # X = df.loc[:, x_var]
    # y = df.loc[:, y_var]
    x_var_params = Visualization_parameters('scatter', component_type = component, variable = x_var,refactor=False)
    x_var_params.get_params()
    y_var1_params = Visualization_parameters('scatter', component, y_var1,refactor=False)
    y_var1_params.get_params()
    y_var2_params = Visualization_parameters('scatter', component, y_var2, refactor=True)
    y_var2_params.get_params()

    # xdata = X_test.loc[:, indep_var]
    # ydata = np.array(np.log10(y.loc[X_test.index]))
    if y_var2 != 'Unity':
        plt.scatter(df.loc[:, x_var],
                    df.loc[:, y_var1]*df.loc[:, y_var2], color = 'g', s=0.8)
    else:
        plt.scatter(df.loc[:, x_var],
                    df.loc[:, y_var1], color='g', s=0.8)
    plt.plot(T, y_, color='navy', label=model, linewidth=.9)
    #plt.scatter(X, Y, color='g', s=1.0)
    plt.legend()
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Log[%s]' % x_var_params.label)
    plt.ylabel('Log[%s * %s]' % (y_var1_params.label, y_var2_params.label))
    plt.title(plot_setpoints.plot_title)
    # plt.title('{} * {} vs. Vdss for N-channel FETs, lowest {:0.2f}'.format(y_var1_params.label, y_var2_params.label, percent_keep))
    # plt.legend(['MOSFET','SiCFET','GaNFET'])
    plt.show()
    print('plot made')


'''
    This function essentially investigates whether or not we get better out-of-sample performance by splitting up our
    data with respect to another variable, such as cost, and then determining into which bin each testing point falls
    and using that model. Not super useful, because at the end of the day the models we create take into account
    all the different input features we care about, and we want just one regression "line" representing everything.
    Takes as input the df and the desired output parameter to train on. Because this is used more as a way to see
    how this approach works, we assume we are training on the linear regression case. Will tell us how well both types
    perform, and then we can plot if desired.
'''


def lin_bracket_split(fet_df, output_param):
    low_df = fet_df
    high_df = fet_df
    cost_line = 3.0
    # Split the data into the brackets
    for i in fet_df.index:
        try:
            # in this case splitting up based on price
            price = fet_df.loc[i]['Unit_price']
            res = fet_df.loc[i]['R_ds']
            if (price > cost_line):  # or (res > 50):
                low_df = low_df.drop([i])
            if price <= cost_line:
                high_df = high_df.drop([i])
        except:
            print("error in splitting")

    # Now split into X and y. For this case, we won't worry about changing the bases/preprocessing any of our X,
    # so that we can more easily compare just the performance of the splitting tactic.
    attributes = ['V_dss', 'Unit_price']
    X_l = low_df.loc[:, attributes]
    # X_l = preproc(X_l)
    y_l = low_df.loc[:, output_param]
    X_h = high_df.loc[:, attributes]
    y_h = high_df.loc[:, output_param]

    # split the data into training/testing sets. A bit complicated because we need to make sure we keep the same
    # objects for the low and high, training and testing, X and y. Splot up 4/5 training, 1/5 testing.
    if len(attributes) == 1:
        X_train_l = np.array(X_l[:-int(len(X_l) / 5)]).reshape(-1, 1)
        X_test_l = np.array(X_l[-int(len(X_l) / 5):]).reshape(-1, 1)
        X_train_h = np.array(X_h[:-int(len(X_h) / 5)]).reshape(-1, 1)
        X_test_h = np.array(X_h[-int(len(X_h) / 5):]).reshape(-1, 1)
    else:
        X_train_l = np.array(X_l[:-int(len(X_l) / 5)])
        X_test_l = np.array(X_l[-int(len(X_l) / 5):])
        X_train_h = np.array(X_h[:-int(len(X_h) / 5)])
        X_test_h = np.array(X_h[-int(len(X_h) / 5):])

    y_train_l = y_l[:-int(len(X_l) / 5)]
    y_test_l = y_l[-int(len(X_l) / 5):]

    y_train_h = y_h[:-int(len(X_h) / 5)]
    y_test_h = y_h[-int(len(X_h) / 5):]

    # create the linear regression object and train the model
    reg_l = LinearRegression().fit(X_train_l, y_train_l)
    reg_h = LinearRegression().fit(X_train_h, y_train_h)

    # the coefficients of the lines
    print('coef_l: {} \n yint_l: {}'.format(reg_l.coef_, reg_l.intercept_))
    print('coef_h: {} \n yint_h: {}'.format(reg_h.coef_, reg_h.intercept_))

    # Now predict on low and high with our low and high testing data
    y_pred_l = reg_l.predict(X_test_l)
    y_pred_h = reg_h.predict(X_test_h)

    # Need to combine the low and high actual testing y values and predicted y values to look at the overall
    # performance.
    y_test_o = np.concatenate((y_test_l, y_test_h))
    y_pred_o = np.concatenate((y_pred_l, y_pred_h))

    mse = mean_squared_error(y_test_o, y_pred_o)
    print('mean squared error w/ split: %.2f' % mse)
    cod = r2_score(y_test_o, y_pred_o)
    print('coefficient of determination w/ split: %.2f' % cod)

    # Can now plot the testing y and the predicted y, by plotting X_test_o vs y_test_o, X_test_o vs y_pred_o,
    # X_test_l vs y_test_l, X_test_l vs y_pred_l, X_test_h vs y_test_h, X_test_h, vs y_pred_h in order to
    # visualize how well the overall prediction looks compared to the high and low versions. Note, this is all still
    # for the split up version.

    # Now do case with everything combined initially, before training, to compare the performance with the split
    # version.
    X_train = np.concatenate((X_train_l, X_train_h))
    y_train = np.concatenate((y_train_l, y_train_h))
    X_test = np.concatenate((X_test_l, X_test_h))
    y_test = np.concatenate((y_test_l, y_test_h))
    reg = LinearRegression().fit(X_train, y_train)
    y_pred = reg.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    cod = r2_score(y_test, y_pred)
    print('mean squared error w/out split: %.2f' % mse)
    print('coefficient of determination w/out split: %.2f' % cod)

    # Can now add onto the previous plot a line showing X_test vs y_test and X_test vs y_pred to visually compare
    # the split up version and combined version performance

if __name__ == '__main__':
    # If doing something with more attributes, need to make sure all attributes we will be using have been parsed.
    # If looking at Qrr, make sure to do the initial parsing, and then the specifid Qrr parsing function call found in
    # parse_pdf_param in fet_pdf_scraper
    '''
    attr_list = ['Mfr_part_no', 'Unit_price', 'FET_type', 'Technology', 'V_dss', 'R_ds', 'Q_g', 'Input_cap',
                 'Pack_case']
    fet_df = initial_fet_parse(fet_df, attr_list)
    fet_df = column_fet_parse(fet_df, ['V_dss','Unit_price','R_ds','Q_g'])'''