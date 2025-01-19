'''
    This file contains the functions to get rid of components that we wouldn't ever really choose. It looks at various
    ranges on the 2 output variables we have: Rds, Qg, and tries to find what will give us a reasonable trend.
    We choose components that have all 3 variables fall within this range for a given voltage, and then choose the one
    with the lowest cost.
'''

import numpy as np
import time
import multiprocessing
import pandas as pd
from fet_visualization import simple_df_plot2
from csv_conversion import csv_to_mosfet
from fet_regression import X_y_set, opt_model, KFold, RandomForestRegressor, cross_val_score
from fet_data_parsing import column_fet_parse, initial_fet_parse





'''
    This function takes as input the list from multiprocessing as [df, voltage_list, bin_size], where df is the df to be
    filtered, voltage_list is the list of (some of) the unique voltages, and bin_size is a tuple of the fraction of
    standard deviation for each of the components to be filtered.
'''


def component_price_filter(input_list):
    # Grab the appropriate values from the input list
    fet_df = input_list[0].reset_index()
    voltage_list = input_list[1]
    bin_size = input_list[2]
    combined_df = pd.DataFrame(columns = ['Mfr_part_no', 'Unit_price', 'Mfr', 'Series','FET_type', 'Technology', 'V_dss', 'I_d', 'V_drive', 'R_ds', 'V_thresh','Q_g', 'V_gs', 'Input_cap', 'P_diss','Op_temp','Mount_type','Supp_pack','Pack_case'])

    # Run through each unique voltage value from the list, filter the df to just look at that voltage, then filter
    # that voltage to look at each bin for each attribute with the bin size specified in bin_size, and in each of
    # those bins, keep only the element with lowest cost.
    for volt_val in voltage_list:
        temp_df = fet_df[fet_df['V_dss'] == volt_val]

        # get the ranges: final1 is -2 and -9, final2 is -3 and -8, final3 is -1 and -4
        Rds_step=bin_size[0]
        Qg_step=bin_size[1]

        # go through every element, take its Rds and Qg values, get a low and high range on each, and keep track of the
        # indices of all components with cost higher than the component of min cost in that bin. As you keep cycling
        # through the components, skip the ones with indices that fall in the dropped_indices_list in order to speed
        # things up. What remains is just the list of filtered components of lowest cost.

        drop_indices_list = []
        for index,row in temp_df.iterrows():
            if index in np.array(drop_indices_list,dtype=object):
                continue
            else:
                (low_Rds, high_Rds) = (row['R_ds']-Rds_step/2, row['R_ds']+Rds_step/2)
                (low_Qg, high_Qg) = (row['Q_g']-Qg_step/2, row['Q_g']+Qg_step/2)
                temp2_df = temp_df[temp_df['R_ds'] > low_Rds]
                temp2_df = temp2_df[temp2_df['R_ds'] < high_Rds]
                temp2_df = temp2_df[temp2_df['Q_g'] > low_Qg]
                temp2_df = temp2_df[temp2_df['Q_g'] < high_Qg]
                drop_indices = temp2_df[temp2_df['Unit_price'] > (temp2_df['Unit_price'].min())].index
                for drop_i in drop_indices:
                    drop_indices_list.append(drop_i)

        # keep only the filtered elements, and write to the specified csv file line by line using apply()
        temp_df = temp_df.drop(drop_indices_list)

        # If you want to write each entry line by line, include the line below. Otherwise, just write the entire df
        # to a csv at the end.
        # temp_df.apply(df_to_csv(temp_df, 'filtered_fet_data_'+str(Rds_step)+'_'+str(Qg_step),'fet'), axis=1)

        # combine the values from the latest Vdss value with all previous Vdss values
        combined_df = pd.concat([combined_df, temp_df])
        #print('done with voltage')

    #print('done with voltage set')
    return combined_df


'''
    This function takes the dataframe and removes components considered outliers, using method specified by a dictionary
    filter_type, which can take the string 'greater_than' or 'std_dev'. 'greater_than' removes all entries greater 
    than a certain value for a certain attribute, 'std_dev' has a value with the number of standard deviations, and
    removes all components whose value falls outside a certain number of standard deviations for an attribute.
    'greater_than' has a value with the 'greater than' values for each attribute.
'''


def outlier_removal(df, filter_type):
    if 'std_dev' in filter_type.keys():
        # If a fet, take out values outside the standard deviation of Rds and Qg for each voltage value
        unique_vals = np.sort(df['V_dss'].unique())

        # first get a copy of the initial df
        ready_df = df
        # then run through the filtering for each voltage value
        for val in unique_vals:
            temp_df = ready_df[(ready_df['V_dss'] == val)]
            n_std_devs = filter_type['std_dev']

            # A useful line for checking the values and frequencies of an attribute
            #   ready_df['R_ds'].value_counts(sort=True)

            # Don't remove based on cost
            # index = temp_df.index
            # temp_indices = index[(temp_df['Unit_price'] - temp_df['Unit_price'].mean()).abs() > (n_std_devs * temp_df['Unit_price'].std())]
            # ready_df = ready_df.drop(index=temp_indices)

            # Rds:
            index = temp_df.index
            temp_indices1 = index[(temp_df['R_ds'] - temp_df['R_ds'].mean()).abs() > (n_std_devs * temp_df['R_ds'].std())]
            #print(temp_indices)
            ready_df = ready_df.drop(index=temp_indices1)

            # Qg:
            temp_indices2 = index[(temp_df['Q_g'] - temp_df['Q_g'].mean()).abs() > (n_std_devs * temp_df['Q_g'].std())]
            temp_indices2 = [x for x in temp_indices2 if x not in temp_indices1]
            #print(temp_indices)
            ready_df = ready_df.drop(index=temp_indices2)

    if 'max_value' in filter_type.keys():
        ready_df = df[(df['R_ds'] <= filter_type['max_value'][0])]
        ready_df = ready_df[(ready_df['Q_g'] <= filter_type['max_value'][1])]

    return ready_df


''' 
    This function takes as input the parsed df, bin size of Rds and Qg, and number of chunks for multiprocessing, and 
    runs the price filtering function in parallel. Breaks up the Vdss values into chunks for multiprocessing.
'''


def multiprocess_price_filter(fet_df, bin_size, num_chunks):
    # Break up Vdss values into specified number of chunks
    unique_vals = np.sort(fet_df['V_dss'].astype(int).unique())
    Vdss_list = np.array_split(unique_vals, num_chunks)

    # Run the multiprocessing with number of chunks specified, remember pool takes a certain kind of argument
    starttime = time.time()
    pool = multiprocessing.Pool(processes=int(np.ceil(multiprocessing.cpu_count()/2)))
    price_filter_args = [(fet_df, chunk, bin_size) for chunk in Vdss_list]
    results = pool.map(component_price_filter, price_filter_args)

    pool.close()
    pool.join()

    # version using process rather than pool
    # processes = []
    # take in lists of voltages from unique voltages in the fet
    # for voltage_list in Vdss_list:
    #     #p = multiprocessing.Process(target=multithread, args=(link_list,))
    #     bin_size = (1*10**-3,1 * 10 ** -8)
    #     p = multiprocessing.Process(target=component_price_filter2, args=(fet_df,voltage_list,bin_size,))
    #
    #     processes.append(p)
    #     p.start()
    # for process in processes:
    #     process.join()

    # combine the results and return
    results_df = pd.concat(results)
    print('Multiprocessing took {} seconds'.format(time.time() - starttime))
    return results_df

'''
    Return a df of only components whose values were also found within the filtered df
'''
#def Qrr_filter_match(df_Qrr, df_filtered):

def optimize_cost_bin_size():
    csv_file = 'csv_files/mosfet_data_csv.csv'
    fet_df = csv_to_mosfet(csv_file)
    attr_list = ['Mfr_part_no', 'Unit_price', 'FET_type', 'Technology', 'V_dss', 'R_ds', 'Q_g', 'Input_cap',
                 'Pack_case']
    fet_df = column_fet_parse(initial_fet_parse(fet_df, attr_list), attr_list)
    subset_df = fet_df[(fet_df['FET_type'] == 'N')]
    subset_df = subset_df[(subset_df['Technology'].isin(['MOSFET']))]
    n_devs_Rds_list = np.linspace(10 ** -7, 10 ** -5, 10)
    n_devs_Qg_list = np.linspace(10 ** -9, 5 * 10 ** -8, 10)
    n_devs_Rds = 6.8 * 10 ** -5
    n_devs_Qg = 1.16 * 10 ** -7
    for n_Rds in n_devs_Rds_list:
        for n_Qg in n_devs_Qg_list:
            print('Rds val: %s, Qg val: %s' % (n_Rds, n_Qg))
            bin_size = (n_Rds * subset_df['R_ds'].std(), n_Qg * subset_df['Q_g'].std())
            results_df = multiprocess_price_filter(subset_df, bin_size, 3)
            print('length of results_df: %d' % len(results_df))
            # score for performance
            X, y = X_y_set(results_df, 'RQ_product')

            model_obj = opt_model('random_forest')
            model = RandomForestRegressor(min_samples_split=model_obj.min_samples_split, random_state=0)
            cv = KFold(n_splits=5, random_state=1, shuffle=True)
            scores = cross_val_score(model, X, y, cv=cv)
            print('Model: Random forest')
            print('Accuracy: %.3f (%.3f)' % (np.mean(scores), np.std(scores)))

if __name__ == '__main__':
    df = csv_to_mosfet('csv_files/mosfet_data_csv_best_final.csv')
    filtered_df = outlier_removal(df, {'std_dev':2})

    # The following is the function call to component_price_filter(), for given bin sizes of Rds and Qg:

    # Here as a reminder for the component_price_filter() function:
    #   bin_size_list = [(1 * 10 ** -3, 1 * 10 ** -8), (1 * 10 ** -3, 6 * 10 ** -8), (6 * 10 ** -3, 1 * 10 ** -8),
    #                  (6 * 10 ** -4, 6 * 10 ** -9), (1 * 10 ** -2, 1 * 10 ** -7)]
    #   1*10**-3 is x fraction of the std. dev. which = 14.6, x = 6.8*10^-5 for Rds
    #   for Qg, 10**-8 is x fraction of std. dev. = 0.086, x = 1.16e-7
    n_devs_Rds = 6.8 * 10 ** -5
    n_devs_Qg = 1.16 * 10 ** -7
    bin_size = (n_devs_Rds * df['R_ds'].std(), n_devs_Qg * df['Q_g'].std())
    multiprocess_price_filter(df, bin_size, 3)

    # Plotting of the filtered variables at certain voltage values
    volt_set_list = [30, 100, 600]
    for volt_set in volt_set_list:
        ready_df = ready_df[ready_df['V_dss'] == volt_set]
        simple_df_plot2(ready_df, 'Unit_price', 'R_ds')
        simple_df_plot2(ready_df, 'Unit_price', 'Q_g')


