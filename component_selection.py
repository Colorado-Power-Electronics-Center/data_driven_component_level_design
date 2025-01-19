'''
    This file contains the functions for selecting the components after the optimization tool has returned optimized
    parameters. Two methods: 1) brute force using the pdf parameters and improved loss model, and 2) brute force using
    the entire component dataset but the simplistic loss model, thus requiring the designer to go in after and do
    further component selection.
'''

import pickle
import scipy
import pandas as pd
from tabulate import tabulate
import numpy as np
import time
from fet_optimization_chained_wCaps import OptimizerInit, OptimizerFet, OptimizerInductor, OptimizerCapacitor
import matplotlib.pyplot as plt
import inspect
from fet_optimization_chained_wCaps import OptimizerInit
from fet_optimization_chained_wCaps import OptimizerFet
from fet_optimization_chained_wCaps import OptimizerInductor



def resulting_parameter_plotting():
    results = Results()
    results.plot_parameters()

class Results:
    def __init__(self):
        self.fet_tech_list = ['MOSFET', 'GaNFET']
        self.fet_tech_list = ['GaNFET']

    def get_visualization_params(self):
        if self.converter_build_element_param == 'Rds':
            self.legend = ['$R_{on1}$, Si', '$R_{on1}$, GaN', '$R_{on2}$, Si', '$R_{on2}$, GaN']
        else:
            self.legend = ['Si', 'GaN']
        self.fontsize = 15
        visualization_param_dict = {'Rds': ['Total cost [$\$$]','$R_{on}$ [mΩ]',10],
                                    'power_tot': ['Total cost [$\$$]','$P_{loss}/P_{out}$',1/self.Pout],
                                    'fsw': ['Total cost [$\$$]','$f_{sw}$ [kHz]',10**3],
                                    'delta_i': ['Total cost [$\$$]','$\Delta i_L/I_L$',0.1],
                                    'Q1_loss': ['Total cost [$\$$]','$Q1_{loss}/P_{loss, tot}$',1/self.power_tot],
                                    'Q2_loss': ['Total cost [$\$$]','$Q2_{loss}/P_{loss, tot}$',1/self.power_tot],
                                    'wire_loss': ['Total cost [$\$$]','$P_{Lw}/P_{loss, tot}$',1/self.power_tot],
                                    'core_loss': ['Total cost [$\$$]','$P_{Lc}/P_{loss, tot}$',1/self.power_tot]}

        visualization_param_df = pd.DataFrame.from_dict(visualization_param_dict, orient='index',
                                                        columns=['xlabel', 'ylabel', 'param_multiplier'])
        self.xlabel = visualization_param_df.loc[self.converter_build_element_param]['xlabel']
        self.ylabel = visualization_param_df.loc[self.converter_build_element_param]['ylabel']
        self.param_multiplier = visualization_param_df.loc[self.converter_build_element_param]['param_multiplier']


    def plot_parameters(self):
        fet_tech_list = ['MOSFET', 'GaNFET']
        area_constraint = '300'

        # cost_constraint_list = ['10.571428571428571']
        for fet_tech in fet_tech_list:
            parameter_vars = vars()
            i = 0
            if fet_tech == 'MOSFET':
                cost_constraint_list = [2.0, 3.428571428571429, 4.857142857142858, 6.285714285714286, 9.257142857142858,
                                        10.571428571428571]
                cost_constraint_list = [4.0, 10.0]
                cost_constraint_list = [2.0, 3.4, 4.9, 6.3, 8.1, 9.1, 10.6]
                cost_constraint_list = [2.0, 2.7, 3.3, 4.0, 4.2, 4.4, 4.7, 4.9, 5.3, 6.0, 6.3, 6.7, 7.3, 8.0, 8.7, 9.1, 9.3, 10.0, 10.7, 11.3, 12.0]
                cost_constraint_list = [2.0, 2.5, 3.0, 3.4, 3.5, 4.0, 4.5, 4.9, 5.0, 5.5, 6.0]
                # cost_constraint_list = [4.4, 10.0]

            elif fet_tech == 'GaNFET':
                cost_constraint_list = [4.97664, 6.285714285714286, 7.714285714285714, 9.142857142857142,
                                        10.571428571428571]
                cost_constraint_list = [4.0, 10.0]
                cost_constraint_list = [4.5, 4.7, 5.9, 7.1, 8.4, 9.6, 10.8]
                cost_constraint_list = [4.4, 4.7, 4.9, 5.4, 5.9, 6.4, 6.9, 7.1, 7.4, 7.9, 8.4, 8.9, 9.4, 9.6, 10.0, 10.5, 10.8, 11.0, 11.5, 12.0]
                cost_constraint_list = [3.0, 3.4, 3.5, 4.0, 4.5, 4.9, 5.0, 5.5, 6.0]
                # cost_constraint_list = [4.4, 10.0]

            for cost in cost_constraint_list:
                # if fet_tech == 'MOSFET' and cost == 10.0:
                #     with open('optimizer_test_values/optimizer_obj_test_values_' + fet_tech + '_' + str(cost) + '_' + str(area_constraint) + '.p',
                #               'rb') as optimizer_obj_file:
                #         parameter_vars.__setitem__('cost_' + str(i), pickle.load(optimizer_obj_file))
                # else:
                with open('optimizer_test_values/optimizer_obj_test_values_' + fet_tech + '_' + str(cost) + '_' + str(area_constraint),
                          'rb') as optimizer_obj_file:
                    parameter_vars.__setitem__('cost_' + str(i), pickle.load(optimizer_obj_file))
                i += 1
                # Ron1_list.append(myVars['cost_0'].fet1.Rds*10)
                # optimizer_obj = pickle.load(optimizer_obj_file)
                # print("The variables are:")
                # print(myVars)
            # parameter_list = [parameter_vars['cost_' + str(i)].fet1.Rds, parameter_vars['cost_' + str(i)].fet2.Rds,
            #                   parameter_vars['cost_' + str(i)].fsw, parameter_vars['cost_' + str(i)].delta_i]
            converter_build_dict = {'fet1': ['Rds'], 'fet2': ['Rds'], 'converter': ['fsw', 'delta_i']}
            converter_build_dict = {'converter': ['power_tot','fsw', 'delta_i', 'Q1_loss', 'Q2_loss', 'wire_loss', 'core_loss']}

            # fet1_parameter_list = ['Rds']
            # fet2_parameter_list = ['Rds']
            # converter_parameter_list = ['fsw', 'delta_i']
            ready_plot = False
            for converter_build_element in converter_build_dict.keys():
                if converter_build_element == 'fet2':
                    self.linestyle = 'dotted'
                else:
                    self.linestyle = 'solid'
                for self.converter_build_element_param in converter_build_dict[converter_build_element]:
                    for fet_tech in fet_tech_list:
                        parameter_vars = vars()
                        i = 0
                        if fet_tech == 'MOSFET':
                            self.plot_color = 'b'
                            cost_constraint_list = [2.0, 3.428571428571429, 4.857142857142858, 6.285714285714286,
                                                    9.257142857142858,
                                                    10.571428571428571]
                            cost_constraint_list = [4.0, 10.0]
                            cost_constraint_list = [2.0, 3.4, 4.9, 6.3, 8.1, 9.1, 10.6]
                            cost_constraint_list = [2.0, 2.7, 3.3,4.0, 4.2, 4.4, 4.7, 4.9, 5.3, 6.0, 6.3, 6.7,
                                                    7.3, 8.0, 8.7, 9.1, 9.3, 10.0, 10.7, 11.3, 12.0]
                            cost_constraint_list = [2.0, 2.5, 3.0, 3.4, 3.5, 4.0, 4.5, 4.9, 5.0, 5.5, 6.0]

                            # cost_constraint_list = [4.4, 10.0]

                        elif fet_tech == 'GaNFET':
                            self.plot_color = 'g'
                            cost_constraint_list = [4.97664, 6.285714285714286, 7.714285714285714, 9.142857142857142,
                                                    10.571428571428571]
                            cost_constraint_list = [4.0, 10.0]
                            cost_constraint_list = [4.5, 4.7, 5.9, 7.1, 8.4, 9.6, 10.8]
                            cost_constraint_list = [4.4, 4.9, 5.4, 5.9, 6.4, 6.9, 7.1, 7.4, 7.9, 8.4, 8.9,
                                                    9.4, 9.6, 10.0, 10.5, 10.8, 11.0, 11.5, 12.0]
                            cost_constraint_list = [3.0, 3.4, 3.5, 4.0, 4.5, 4.9, 5.0, 5.5, 6.0]

                            # cost_constraint_list = [4.4, 10.0]

                        for cost in cost_constraint_list:
                            # if fet_tech == 'MOSFET' and cost == 10.0:
                            #     with open('optimizer_test_values/optimizer_obj_test_values_' + fet_tech + '_' + str(cost) + '_' + str(
                            #             area_constraint) + '.p',
                            #               'rb') as optimizer_obj_file:
                            #         parameter_vars.__setitem__('cost_' + str(i), pickle.load(optimizer_obj_file))
                            # else:
                            with open('optimizer_test_values/optimizer_obj_test_values_' + fet_tech + '_' + str(cost) + '_' + str(
                                    area_constraint),
                                      'rb') as optimizer_obj_file:
                                parameter_vars.__setitem__('cost_' + str(i), pickle.load(optimizer_obj_file))
                            # with open('optimizer_obj_test_values_' + fet_tech + '_' + str(cost) + '_' + str(area_constraint),
                            #           'rb') as optimizer_obj_file:
                            #     parameter_vars.__setitem__('cost_' + str(i), pickle.load(optimizer_obj_file))
                            i += 1


                        opt_var_list = []
                        for i in range(len(cost_constraint_list)):
                            self.Pout = parameter_vars['cost_' + str(i)].Pout
                            self.power_tot = parameter_vars['cost_' + str(i)].power_tot
                            self.get_visualization_params()

                            if self.converter_build_element_param == 'wire_loss':
                                opt_var_list.append((parameter_vars['cost_' + str(i)].converter_df.loc[0, 'Iout'] ** 2 * parameter_vars['cost_' + str(i)].ind1.R_dc \
                                + parameter_vars['cost_' + str(i)].ind1.Rac_total) * self.param_multiplier)

                            elif self.converter_build_element_param == 'core_loss':
                                opt_var_list.append((parameter_vars['cost_' + str(i)].ind1.Core_volume * parameter_vars['cost_' + str(i)].ind1.IGSE_total) * self.param_multiplier)

                            else:
                                for component_element in inspect.getmembers(parameter_vars['cost_' + str(i)]):
                                    print(component_element)
                                    # covering the individual components
                                    if component_element[0] == converter_build_element:
                                        for j in inspect.getmembers(component_element[1]):
                                            # print(j[0])
                                            if j[0] == self.converter_build_element_param:
                                                print(j)
                                                opt_var_list.append(j[1] * self.param_multiplier)

                                    # covering the entire converter build
                                    else:
                                        for j in inspect.getmembers(parameter_vars['cost_' + str(i)]):
                                            # print(j[0])
                                            if j[0] == self.converter_build_element_param:
                                                print(j)
                                                opt_var_list.append(j[1] * self.param_multiplier)
                                                break
                                        if converter_build_element == 'converter':
                                            break


                        if len(opt_var_list) == len(cost_constraint_list):
                            plt.plot(cost_constraint_list[0:len(opt_var_list)], opt_var_list, c=self.plot_color, linewidth=4, linestyle=self.linestyle)
                            ready_plot = True

                    if ready_plot:
                        legend = plt.legend(self.legend, fontsize=self.fontsize)
                        plt.xlabel(self.xlabel, fontsize=self.fontsize)
                        plt.ylabel(self.ylabel, fontsize=self.fontsize)
                        plt.xticks(fontsize=self.fontsize)
                        plt.yticks(fontsize=self.fontsize)
                        # plt.ylim(0, .15)
                        # plt.scatter(3, 0.0272, s=100)
                        # pt1 = plt.annotate('Selected design', (3.3, .026), size=15)
                        plt.grid(color='lightgrey', linewidth=1, alpha=0.4)
                        plt.show()
                        ready_plot = False

                            # for j in inspect.getmembers(parameter_vars['cost_' + str(i)].fet1):
                            #     # print(j[0])
                            #     if j[0] == fet1_param:
                            #         print(j)
                            #         opt_var_list.append(j[1] * 10)



def predict_components():

    fet_tech = 'MOSFET'
    cost_constraint = 5.0 # cost constraint must be a float
    area_constraint = 800 # area constraint, could generalize this
    num_comps = 10
    topology = 'buck'

    fet_tech = 'MOSFET'
    cost_constraint = 19.67  # cost constraint must be a float
    area_constraint = 1000  # area constraint, could generalize this
    num_comps = 10
    topology = 'microinverter_combined'

    # fet_tech = 'MOSFET'
    # cost_constraint = 5.0  # cost constraint must be a float
    # area_constraint = 800  # area constraint, could generalize this
    # num_comps = 5
    # topology = 'boost'


    with open('optimizer_test_values/optimizer_obj_' + topology + '_' + fet_tech + '_' + str(cost_constraint) + '_' + str(area_constraint), 'rb') as optimizer_obj_file:
        # Step 3
        optimizer_obj = pickle.load(optimizer_obj_file)
        # optimizer_obj = OptimizerInit(**optimizer_obj_file.__dict__)

    # optimizer_obj.fsw = 0.175
    # optimizer_obj.fsw = 0.6
    # print(f'{o.power_tot},{o.Cost_tot},{o.Area_tot}')
    # print(f'{o.Q1_loss},{o.Q2_loss},{o.L1_loss}')
    # print(f'{o.fet1.Area},{o.fet2.Area},{o.ind1.Area},{o.cap1.Area},{o.cap2.Area}')
    # print(f'{o.fet1.Cost},{o.fet2.Cost},{o.ind1.Cost},{o.cap1.Cost},{o.cap2.Cost}')
    start_opt = time.time()
    optimization_case = make_component_predictions_bf(optimizer_obj, 'pdf_params', cost_constraint, area_constraint, num_comps)

    optimization_case.filter_components()
    end_opt = time.time()
    print(f'until optimize combinations: {(end_opt-start_opt)}')

    # with open('optimizer_obj_test_values', 'rb') as optimizer_obj_file:
    #     # Step 3
    #     optimizer_obj = pickle.load(optimizer_obj_file)
    # fet1_case.fets_list, fet2_case.fets_list, ind1_case.inds_list, cap1_case.caps_list, cap2_case.caps_list

    # with open('fet1_case.fets_list', 'rb') as optimizer_obj_file:
    #     # Step 3
    #     optimization_case.fet1_case_fet_list = pickle.load(optimizer_obj_file)
    # with open('fet2_case.fets_list', 'rb') as optimizer_obj_file:
    #     # Step 3
    #     optimization_case.fet2_case_fet_list = pickle.load(optimizer_obj_file)
    # with open('ind1_case.inds_list', 'rb') as optimizer_obj_file:
    #     # Step 3
    #     optimization_case.ind1_case_ind_list = pickle.load(optimizer_obj_file)
    # with open('cap1_case.caps_list', 'rb') as optimizer_obj_file:
    #     # Step 3
    #     optimization_case.cap1_case_cap_list = pickle.load(optimizer_obj_file)
    # with open('cap2_case.caps_list', 'rb') as optimizer_obj_file:
    #     # Step 3
    #     optimization_case.cap2_case_cap_list = pickle.load(optimizer_obj_file)

    optimization_case.optimize_combinations()

def compare_combinations():
    set_combo_dict = {'0': {'fet_tech': 'MOSFET', 'cost_constraint': 5.0, 'area_constraint': 800, 'num_comps': 10, 'topology': 'boost', 'Mfr_part_nos_dict': {'fet_mfr_part_nos': ['PXN012-60QLJ', 'PXN012-60QLJ'],
                                           'ind_mfr_part_nos': ['SRP1265C-8R2M'],
                                           'cap_mfr_part_nos': ['GRM31CC72A475ME11L', 'GRM32EC72A106ME05L']}},
                      '1': {'fet_tech': 'MOSFET', 'cost_constraint': 5.0, 'area_constraint': 800, 'num_comps': 10, 'topology': 'buck', 'Mfr_part_nos_dict': {'fet_mfr_part_nos': ['PXN012-60QLJ', 'PXN012-60QLJ'],
                                           # 'ind_mfr_part_nos': ['SRP1265C-8R2M'],
                                            'ind_mfr_part_nos': ['SRP1770TA-390M'],
                                           # 'ind_mfr_part_nos': ['ASPIAIG-Q1513-330M-T'],
                                           'cap_mfr_part_nos': ['GRM31CC72A475ME11L', 'CL21A226MAYNNNE']}},
                      '2': {'fet_tech': 'GaNFET', 'cost_constraint': 5.0, 'area_constraint': 800, 'num_comps': 5, 'topology': 'buck',
                            'Mfr_part_nos_dict': {'fet_mfr_part_nos': ['EPC2052', 'EPC2052'],
                                                  'ind_mfr_part_nos': ['MMD-06CZ-R68M-V1-RU'],
                                                  'cap_mfr_part_nos': ['GRM31CC72A475ME11L', 'CL21A226MAYNNNE']}},
                      '3': {'fet_tech': 'GaNFET', 'cost_constraint': 5.0, 'area_constraint': 800, 'num_comps': 10, 'topology': 'buck',
                            'Mfr_part_nos_dict': {'fet_mfr_part_nos': ['EPC2052', 'EPC2052'],
                                                  'ind_mfr_part_nos': ['MMD-06CZ-R68M-V1-RU'],
                                                  'cap_mfr_part_nos': ['GRM31CC72A475ME11L', 'CL21A226MAYNNNE']}},
                      '4': {'fet_tech': 'MOSFET', 'cost_constraint': 11.0, 'area_constraint': 800, 'num_comps': 10, 'topology': 'buck',
                            'Mfr_part_nos_dict': {'fet_mfr_part_nos': ['SIS176LDN-T1-GE3', 'PXN012-60QLJ'],
                                                  'ind_mfr_part_nos': ['AMDLA2213Q-470MT'],
                                                  'cap_mfr_part_nos': ['GRM32EC72A106ME05L', 'C3216X5R1E476M160AC']}},

                      '5': {'fet_tech': 'GaNFET', 'cost_constraint': 11.0, 'area_constraint': 800, 'num_comps': 10, 'topology': 'buck',
                            'Mfr_part_nos_dict': {'fet_mfr_part_nos': ['EPC2204', 'EPC2204'],
                                                  # 'ind_mfr_part_nos': ['ASPIAIG-Q1010-6R8M-T'],
                                                  'ind_mfr_part_nos': ['SRP1513CA-100M'],
                                                  'cap_mfr_part_nos': ['GRM32EC72A106ME05L', 'C3216X5R1E336M160AC']}},
                      '6': {'fet_tech': 'MOSFET', 'cost_constraint': 19.67, 'area_constraint': 1000, 'num_comps': 10, 'topology': 'microinverter_combined',
                            'Mfr_part_nos_dict': {'fet_mfr_part_nos': ['SIR104LDP-T1-RE3', 'SIDR668DP-T1-GE3','SIR104LDP-T1-RE3'],
                                                  'ind_mfr_part_nos': ['MPX1D2213L6R8'],
                                                  'cap_mfr_part_nos': []}},
                      '7': {'fet_tech': 'MOSFET', 'cost_constraint': 5.0, 'area_constraint': 800, 'num_comps': 10,
                            'topology': 'buck',
                            'Mfr_part_nos_dict': {'fet_mfr_part_nos': ['PXN012-60QLJ', 'PXN012-60QLJ'],
                                                  'ind_mfr_part_nos': ['SRP1770TA-390M'],
                                                  'cap_mfr_part_nos': ['GRM31CC72A475ME11L', 'CL21A226MAYNNNE']}},
                      '8': {'fet_tech': 'GaNFET', 'cost_constraint': 11.0, 'area_constraint': 800, 'num_comps': 10,
                            'topology': 'buck',
                            'Mfr_part_nos_dict': {'fet_mfr_part_nos': ['EPC2204', 'EPC2204'],
                                                  'ind_mfr_part_nos': ['SRP1513CA-100M'],
                                                  'cap_mfr_part_nos': ['GRM32EC72A106ME05L', 'C3216X5R1E336M160AC']}},
                      }
    dict_index = '1'
    fet_tech = set_combo_dict[dict_index]['fet_tech']
    cost_constraint = set_combo_dict[dict_index]['cost_constraint']
    area_constraint = set_combo_dict[dict_index]['area_constraint']
    num_comps = set_combo_dict[dict_index]['num_comps']
    topology = set_combo_dict[dict_index]['topology']


    with open(
            'optimizer_test_values/optimizer_obj_' + topology + '_' + fet_tech + '_' + str(cost_constraint) + '_' + str(
                area_constraint), 'rb') as optimizer_obj_file:
        # Step 3
        optimizer_obj = pickle.load(optimizer_obj_file)
        # optimizer_obj = OptimizerInit(**optimizer_obj_file.__dict__)

    # for microinverter
    # TODO: when predicting losses at a different power level/switching frequency, make changes here
    # If you want to set your own frequency
    # Pout = 430
    # Vpv = 50
    # Vbus = optimizer_obj.converter_df.loc[0,'Vbus']
    # optimizer_obj.converter_df.loc[0, 'Vin'] = Vpv
    # optimizer_obj.fsw = (1-Vpv/Vbus)*(Vpv**2) / (2*(7.5e-6)*Pout) / optimizer_obj.ind_normalizer
    # optimizer_obj.converter_df.loc[0, 'Iin'] = Pout / Vpv
    # optimizer_obj.converter_df.loc[0, 'Iout'] = Pout / 40
    # optimizer_obj.IL = optimizer_obj.converter_df.loc[0,'Iin']
    # optimizer_obj.dc = 1 - Vpv / Vbus
    # TODO: check if IL and Duty Cycle get changed when these things are altered
    # For buck
    # optimizer_obj.fsw = 10
    optimizer_obj.fsw = 0.05
    optimizer_obj.converter_df.loc[0, 'Iout'] = 5
    optimizer_obj.IL = optimizer_obj.converter_df.loc[0, 'Iout']
    optimizer_obj.Pout = optimizer_obj.converter_df.loc[0, 'Iout'] * optimizer_obj.converter_df.loc[0, 'Vout']
    # # self.database_df['fb [Hz]'] = 1000

    start_opt = time.time()
    optimization_case = make_component_predictions_bf(optimizer_obj, 'pdf_params', cost_constraint, area_constraint,
                                                      num_comps)
    optimization_case.opt_obj.bounds_dict['comp_selection_delta_i_max'] = 50
    optimization_case.opt_obj.cost_constraint = 13

    optimization_case.practical = True
    optimization_case.theoretical = True

    optimization_case.Mfr_part_nos_dict = set_combo_dict[dict_index]['Mfr_part_nos_dict']

    # Here, put in code for optimizing fsw. Then, set self.opt_obj.fsw = the optimized switching frequency, and print the loss
    # parameters with that optimized switching frequency.
    # Actually, optimize fsw inside print_all_parameters? or do outside of it in case you just want to do that? I feel
    # like we should do it inside, right after optimizing the combinations with the set/given components
    # or maybe we want it outside, so that it doesn't mess with the optimizing the combinations. yeah, probably want it outside.
    # optimization_case.optimize_fsw()
    # for now, putting optimize_fsw() inside print_all_parameters()
    optimization_case.print_all_parameters()
    pass


def normalize_x(df, vars_list, obj):
    # Q1_df[var_name + '_normalized'] = (x_pnt - np.min(df[var_name]) / (np.max(df[var_name]) - np.min(df[var_name])))
    df_to_obj_dict = {'Unit_price': 'Cost', 'R_ds': 'Rds_normalized', 'Q_g': 'Qg', 'Pack_case': 'Area', 'C_ossp1': 'Cossp1',
                      'tau_c':'tau_c','tau_rr':'tau_rr',
                      'DCR [Ohms]': 'R_dc',
                      'Area [mm^2]': 'Area', 'Unit_price [USD]': 'Cost', 'Inductance [H]': 'L',
                      'Size': 'Area', 'Capacitance': 'Capacitance', 'R_dc': 'R_dc', 'L': 'L', 'Area':'Area', 'Cost': 'Cost',
                      'Rac_total':'Rac_total', 'IGSE_total':'IGSE_total', 'Core_volume':'Core_volume', 'IGSE_loss': 'IGSE_loss'}
    for var_name in vars_list:
        if var_name == 'R_ds':
            obj.Rds_normalized = obj.Rds * obj.fet_normalizer
        # df['Normalized_score'] += np.abs((df[var_name] - getattr(obj, df_to_obj_dict[var_name]))) / np.nanmax(np.append(np.array(df[var_name]),getattr(obj, df_to_obj_dict[var_name])))
        df['Normalized_score'] += (df[var_name] - getattr(obj, df_to_obj_dict[var_name])) / np.nanmax(np.append(np.array(df[var_name]),getattr(obj, df_to_obj_dict[var_name])))

        # Note that each of the above values is between 0 and 1
        # df['Normalized_score'] += (df[var_name] - np.min(df[var_name])) / (np.max(df[var_name]) - np.min(df[var_name]))
    # Then find avg. score value
    df = df.applymap(lambda x: np.real(x))
    df['Normalized_score'] = df['Normalized_score'] / len(vars_list)
    df['Normalized_score'] = df['Normalized_score'].astype(float)
    return df

class fet_predictor:
    def __init__(self, selection_method, opt_obj, fet_obj):
        self.opt_obj = opt_obj
        self.selection_method = selection_method
        if self.selection_method.database == 'pdf_params':
            self.recommended_params_df = pd.DataFrame.from_dict(
            {'Vdss': [fet_obj.Vdss], 'Rds': [fet_obj.Rds],
             'Qg': [fet_obj.Qg],
             'Cossp1': [fet_obj.Cossp1],
             'Tau_c': [fet_obj.tau_c],
             'Tau_rr': [fet_obj.tau_rr],
             'Cost': [fet_obj.Cost],
             'Area': [
                 fet_obj.Area]})

        elif self.selection_method.database == 'main_page_params':
            self.recommended_params_df = pd.DataFrame.from_dict(
                {'Vdss': [fet_obj.Vdss], 'Rds': [fet_obj.Rds],
                 'Qg': [fet_obj.Qg],
                 'Cost': [fet_obj.Cost],
                 'Area': [
                     fet_obj.Area]})


    def database_df_filter(self, opt_obj, fet_obj, num_comps, fet_Mfr_part_no = None):
        from component_selection_case_statements import determine_norm_scored_vars
        # low_range = 5
        # high_range = 10

        if self.selection_method.database == 'pdf_params':
            if opt_obj.fet_tech == 'GaNFET':
                # First, open the dataset and add on the additional GaN components
                # Check where values are NaN with: gan_df[gan_df['Vds_meas'].isna()]
                with open('cleaned_fet_dataset_pdf_params3', 'rb') as optimizer_obj_file:
                    # with open('cleaned_fet_dataset', 'rb') as optimizer_obj_file:
                    fets_database_df = pickle.load(optimizer_obj_file)

            else:
                with open('cleaned_fet_dataset_pdf_params3', 'rb') as optimizer_obj_file:
                    # with open('cleaned_fet_dataset_pdf_params_noQrr', 'rb') as optimizer_obj_file:
                    fets_database_df = pickle.load(optimizer_obj_file)
                # Drop if tau_c or tau_rr == np.nan
                #     fets_database_df[['tau_c', 'tau_rr']] = 0
                    # Here, first impute Qrr and trr
                    # then compute tau_c and tau_rr--check matches what we would expect to get

                    # impute Qrr and trr

                    fets_database_df = fets_database_df.dropna(subset=['tau_c', 'tau_rr']) # include this if using
                                                # 'cleaned_fet_dataset_pdf_params2' for Si as well, currently just
                                                # using for GaN bc includes other added GaN devices



        elif self.selection_method.database == 'main_page_params':
            with open('cleaned_fet_dataset_main_page_params', 'rb') as optimizer_obj_file:
                fets_database_df = pickle.load(optimizer_obj_file)

        if num_comps == 1:
            fets_database_df = fets_database_df[fets_database_df['Mfr_part_no'] == fet_Mfr_part_no]

        bmin = 0.8
        bmax = 1.2

        # First take care of all required filtering where range doesn't matter
        Q1_df = fets_database_df.loc[(fets_database_df['V_dss'] >= fet_obj.Vdss)]
        # Q1_df = Q1_df.loc[(Q1_df['Unit_price'] <= 1.5*fet_obj.Cost)]
        Q1_df = Q1_df.loc[(Q1_df['Pack_case'] <= 1.5*fet_obj.Area)]

        Q1_df = Q1_df.drop_duplicates(subset=['Mfr_part_no'])
        Q1_df = Q1_df.loc[Q1_df['Technology'] == opt_obj.fet_tech]
        Q1_df = Q1_df.loc[Q1_df['FET_type'] == 'N']

        # for un_fet in self.unavailable_fets:
        #     Q1_df.drop(Q1_df[Q1_df['Mfr_part_no'] == un_fet].index, inplace=True)

        # Now score for closeness using normalization formula

        # should be something like 'for fet_obj', but already inside fet_obj, so check fet_obj.fet_index, and have another
        # function call in component_selection_case_statements

        # if fet_obj.fet_index == 0:
        #     vars_list = ['Unit_price', 'R_ds', 'Q_g', 'Pack_case', 'C_ossp1'] # only tau_c, tau_rr, and Cossp1 for Q2
        # elif fet_obj.fet_index == 1:
        #     vars_list = ['Unit_price', 'R_ds', 'Q_g', 'Pack_case', 'tau_c', 'tau_rr', 'C_ossp1'] # only tau_c, tau_rr, and Cossp1 for Q2
        # elif fet_obj.fet_index == 2:
        #     vars_list = ['Unit_price', 'R_ds', 'Q_g', 'Pack_case', 'tau_c', 'tau_rr', 'C_ossp1']

        # Q1_df = Q1_df.loc[(Q1_df['Unit_price'] <= 2*fet_obj.Cost)]
        # Q1_df = Q1_df.loc[(Q1_df['R_ds'] <= 2**fet_obj.Rds * opt_obj.fet_normalizer)]
        # Q1_df = Q1_df.loc[(Q1_df['Q_g'] <= 2*fet_obj.Qg)]
        # Q1_df = Q1_df.loc[(Q1_df['Pack_case'] <= fet_obj.Area)]
        vars_list = determine_norm_scored_vars(opt_obj, fet_obj)
        Q1_df['Normalized_score'] = 0
        Q1_df = normalize_x(Q1_df, vars_list, fet_obj)
        Q1_df['Normalized_score'] = Q1_df['Normalized_score'].astype(float)
        self.database_df = Q1_df.nsmallest(num_comps, 'Normalized_score')
        print('len fet database_df: ' + str(len(self.database_df)))
        return



    def print_parameters(self, fet_obj, n):
        fet_column_list = ['V_dss [V]', 'R_ds [mΩ]', 'Q_g [nC]', 'C_ossp1 [pF]', 'tau_c', 'tau_rr', 'Unit_price [$]',
                           'Area [mm^2]']


        print('FET ' + str(n) + ' recommended parameters:')
        print(tabulate(self.recommended_params_df.drop_duplicates(inplace=False), headers=fet_column_list, showindex=False,
                       tablefmt='fancy_grid',
                       floatfmt=".13f"))

    def compute_loss_equations(self, opt_obj, fet_obj=None):
        # self.database_df['category'] = self.database_df.apply(lambda x: self.compute_category_bf(x.V_dss), axis=1)
        # self.database_df['kT'] = self.database_df.apply(lambda x: self.compute_kT_bf(x.category, x.V_dss), axis=1)
        # self.database_df['Power_loss'] = self.I_Q1 ** 2 * self.database_df['kT'] * self.database_df['R_ds'] + \
        #                       self.x_result[2] * self.database_df['Q_g'] * self.fets_df.loc[0, 'Vgate']

        if fet_obj == None:
            fets_objects = self.database_df.to_records()
        else:
            fets_objects = [fet_obj]
        # fet = fets[0]
        fets_list = []

        for fet in fets_objects:
            # create an OptimizerFet object for each row of the dataframe
            # param_dict = {'opt_var': 'power', 'plotting_var': 'cost', 'set_constraint': 'area',
            #               'set_constraint_val': 600,
            #               'example_num': 1, 'tech_list': ['MOSFET'], 'num_points': 2,
            #               'plotting_range': [4, 10], 'predict_components': False}

            if fet_obj == None:
                fetX = OptimizerFet(opt_obj.param_dict, 0)
                fetX.Mfr_part_no, fetX.Vdss, fetX.Rds, fetX.Qg, fetX.Cost, fetX.Area, fetX.tau_c, fetX.tau_rr, fetX.Cossp1, fetX.I_F, fetX.Vds_meas = \
                    fet.Mfr_part_no, fet.V_dss, fet.R_ds, fet.Q_g, fet.Unit_price, fet.Pack_case, fet.tau_c, fet.tau_rr, fet.C_ossp1, fet.I_F, fet.Vds_meas

            else:
                fetX = fet
                fetX.Mfr_part_no, fetX.Vdss, fetX.Rds, fetX.Qg, fetX.Cost, fetX.Area, fetX.tau_c, fetX.tau_rr, fetX.Cossp1, fetX.I_F, fetX.Vds_meas = \
                    fet.Mfr_part_no, fet.Vdss, fet.Rds.values[0], fet.Qg.values[0], fet.Cost.values[0], fet.Area.values[0], fet.tau_c, fet.tau_rr, fet.Cossp1.values[0], fet.I_F, fet.Vds_meas

            if opt_obj.fet_tech == 'MOSFET':
                fetX.p = 3
                fetX.t_rise = 20 * 10 ** -9  # for low-voltage Si grouping
                fetX.t_off = 10 * 10 ** -9  # for Si low-voltage grouping

                # for low-voltage Si grouping
            elif opt_obj.fet_tech == 'GaNFET':
                fetX.p = 1.5
                fetX.t_rise = 10 * 10 ** -9  # for low-voltage Si grouping
                fetX.t_off = 5 * 10 ** -9  # for Si low-voltage grouping

            fets_list.append(fetX)

        self.fets_list = fets_list



class ind_predictor:
    def __init__(self, opt_obj, ind_obj):
        self.opt_obj = opt_obj
        self.recommended_params_df = pd.DataFrame(
        {'Inductance': [ind_obj.L],
         'I_rated': [ind_obj.I_rated],
         'I_sat': [ind_obj.I_sat],
         'Rdc': [ind_obj.R_dc],
         'Cost': [ind_obj.Cost], 'Area': [ind_obj.Area], 'f_b': [ind_obj.f_b], 'b': [ind_obj.b],
         'N_turns': [ind_obj.N_turns], 'A_c': [ind_obj.A_c], 'Alpha': [ind_obj.Alpha],
         'Beta': [ind_obj.Beta], 'K_fe': [ind_obj.K_fe], 'Core_volume': [ind_obj.Core_volume]})

    def database_df_filter(self, ind_obj, num_comps):
        # don't need the following line bc right now have the database already, need to make sure it's a dataframe, then can turn that
        # into inds_database_df and go from there
        # inds_database_df = pd.read_csv('csv_files/inductor_training_updatedAlgorithm.csv')
        inds_database_df = pd.DataFrame([o.__dict__ for o in self.inds_list])
        # After this, will need to change some of the filtering names for x_normalized bc now it matches what the objects are.
        #   --> actually just need to update what's in var_list, otherwise will be fine inside normalize_x
        # After filtering, will need to use compute_loss_equations() again to get into a list that can be used for
        # checking combinations, but this time will have object attributes for the delta_i, Rac_total, and IGSE_total.

        # import sys
        # np.set_printoptions(threshold=sys.maxsize)

        L1_df = inds_database_df.loc[(ind_obj.I_rated <= inds_database_df['I_rated'])]
        L1_df = L1_df.loc[(5 * ind_obj.Cost > L1_df['Cost'])]

        unavailable_inds = ['ASPIAIG-Q1513-220M-T', 'ASPIAIG-Q1513-330M-T', 'ASPIAIG-Q1513-8R2M-T',
                            'EXL1V1010-5R6-R',
                            'ASPIAIG-Q1010-5R6M-T', 'AMDLA1707S-6R8MT', 'ASPIAIG-Q1513-150M-T', 'AMPLA1306S-5R6MT',
                            'AMPLA7030S-R75MT', 'AMPLA7050Q-1R2MT', 'EXL1V1010-150-R', 'B82482M1822M000',
                            'B82482M1682M000',
                            'ETQ-P3MR68KVN', '0925-221H', 'AMDLA2213Q-470MT', 'AMDLA2213Q-330MT','AMPLA7050Q-R60MT','EXL1V1010-8R2-R',
                            'ASPIAIG-Q1513-220M-T', 'ASPIAIG-Q1513-330M-T', 'ASPIAIG-Q1513-8R2M-T', 'EXL1V1010-5R6-R',
                            'ASPIAIG-Q1010-5R6M-T', 'AMDLA1707S-6R8MT', 'ASPIAIG-Q1513-150M-T', '0925-221H',
                            'AMPLA7030S-R60MT', 'FPT1006-340-R',
                            'CTX50-7-52M-R', 'AMDLA2213Q-470MT', 'AMDLA2213Q-330MT'
                            ]
        unavailable_inds = []
        for un_ind in unavailable_inds:
            L1_df.drop(L1_df[L1_df['Mfr_part_no'] == un_ind].index, inplace=True)
        L1_df = L1_df[L1_df['Mfr'] != 'EPCOS - TDK Electronics']
        L1_df = L1_df.drop_duplicates(subset=['Mfr_part_no'])
        # L1_df = L1_df.loc[(L1_df['DCR [Ohms]'] <= ind_obj.R_dc)]
        # L1_df = L1_df.loc[(L1_df['L'] <= 2*ind_obj.L) & (L1_df['L'] >= 0.5*ind_obj.L)]

        ### May want to add back in the following lines in some capacity
        # L1_df = L1_df.loc[
        #     (L1_df['Area'] <= 1.5 * ind_obj.Area)]
        # L1_df = L1_df.loc[(L1_df['Cost'] <= 1.5 * ind_obj.Cost)]
        L1_df['IGSE_loss'] = L1_df['IGSE_total'] * L1_df['Core_volume']
        vars_list = ['R_dc', 'L', 'Cost', 'Area', 'Rac_total', 'IGSE_loss']
        L1_df['Normalized_score'] = 0
        L1_df = normalize_x(L1_df, vars_list, ind_obj)
        self.database_df = L1_df.nsmallest(num_comps, 'Normalized_score')
        print('len ind database_df: ' + str(len(self.database_df)))
        return



    def print_parameters(self, fet_obj, n):
        ind_column_list = ['Inductance [µH]', 'Current_rating [A]', 'R_dc [mΩ]', 'Unit_price [$]', 'Area [mm^2]'
                           ]

        print('Inductor ' + str(n) + ' recommended parameters:')
        self.rec_params_df = self.recommended_params_df[['Inductance', 'I_rated', 'Rdc', 'Cost', 'Area']]
        print(tabulate(self.rec_params_df.drop_duplicates(inplace=False), headers=ind_column_list, showindex=False,
                       tablefmt='fancy_grid',
                       floatfmt=".13f"))

    def compute_loss_equations_final(self, opt_obj):
        inds_objects = self.database_df.to_records()
        inds_list = []
        for ind in inds_objects:
            indX = OptimizerInductor(self.opt_obj.param_dict, 0)
            indX.Mfr_part_no, indX.I_rated, indX.I_sat, indX.Cost, indX.R_dc, indX.L, indX.Area, indX.f_b, indX.b, indX.N_turns, indX.A_c, \
            indX.Alpha, indX.Beta, indX.K_fe, indX.Core_volume, indX.Mfr, indX.delta_i, indX.Rac_total, indX.IGSE_total, indX.delta_B = \
                ind.Mfr_part_no, ind.I_rated, ind.I_sat, ind.Cost, ind.R_dc, ind.L, ind.Area, ind.f_b, ind.b, ind.N_turns, ind.A_c, \
                ind.Alpha, ind.Beta, ind.K_fe, ind.Core_volume, ind.Mfr, ind.delta_i, ind.Rac_total, ind.IGSE_total, ind.delta_B  # this is what .to_records() does, can index either way
            inds_list.append(indX)

        self.inds_list = inds_list

    def compute_loss_equations(self, opt_obj, ind_obj=None):
        from component_selection_case_statements import set_inductor_attributes
        # self.database_df['category'] = self.database_df.apply(lambda x: self.compute_category_bf(x.V_dss), axis=1)
        # self.database_df['kT'] = self.database_df.apply(lambda x: self.compute_kT_bf(x.category, x.V_dss), axis=1)
        # self.database_df['Power_loss'] = self.I_Q1 ** 2 * self.database_df['kT'] * self.database_df['R_ds'] + \
        #                       self.x_result[2] * self.database_df['Q_g'] * self.fets_df.loc[0, 'Vgate']
        if ind_obj == None:
            inds_objects = self.database_df.to_records()
        else:
            inds_objects = [ind_obj]
        # inds_objects = self.database_df.to_records()
        # fet = fets[0]
        inds_list = []

        for ind in inds_objects:
            # create an OptimizerFet object for each row of the dataframe
            # param_dict = {'opt_var': 'power', 'plotting_var': 'cost', 'set_constraint': 'area',
            #               'set_constraint_val': 600,
            #               'example_num': 1, 'tech_list': ['MOSFET'], 'num_points': 2,
            #               'plotting_range': [4, 10], 'predict_components': False}
            if ind_obj == None:
                indX = OptimizerInductor(self.opt_obj.param_dict, 0)
                indX.Mfr_part_no, indX.I_rated, indX.I_sat, indX.Cost, indX.R_dc, indX.L, indX.Area, indX.f_b, indX.b, indX.N_turns, indX.A_c,\
                    indX.Alpha, indX.Beta, indX.K_fe, indX.Core_volume, indX.Mfr = \
                    ind['Mfr_part_no'], ind['Current_Rating [A]'], ind['Current_Sat [A]'], ind['Unit_price [USD]'], ind['DCR [Ohms]'], ind['Inductance [H]'], ind['Length [mm]']*ind['Width [mm]'], ind['fb [Hz]'], ind.b, ind.Nturns, ind['Ac [m^2]'],\
                    ind.Alpha, ind.Beta, ind.Kfe, ind['Core Volume m^3'], ind.Mfr # this is what .to_records() does, can index either way

                # if indX.L > 60*10**-6:
                #     print(indX.L)
                # opt_obj.ind1 = indX
                # opt_obj.set_optimization_variables(opt_obj.x_result)

                # Still need to re-compute delta_i here, set this as another object attribute?
                indX = set_inductor_attributes(indX, self.opt_obj)

                if (indX.delta_i < self.opt_obj.bounds_dict['comp_selection_delta_i_min']) or (indX.delta_i > self.opt_obj.bounds_dict['comp_selection_delta_i_max']):  # Translates to delta_i [%] = 1% or 70% as limits

                    continue # set to be very high so won't ever use, as associated losses will be very high

                # print('for initial computing of Rac and IGSE:')
                # print(indX.delta_i)
                self.opt_obj.delta_i = indX.delta_i
                start = time.time()
                indX.compute_Rac(self.opt_obj)

                end = time.time()
                # print(f'Rac comp time: {(end-start)}')
                # print(indX.Rac_total)
                # indX.delta_B = 1
                start = time.time()
                indX.compute_IGSE(self.opt_obj)
                end = time.time()
                # print(f'IGSE comp time: {(end-start)}')
                indX.IGSE_loss = indX.IGSE_total * indX.Core_volume
                # print(indX.IGSE_total)

            else:
                indX = ind
                indX.IGSE_loss = indX.IGSE_total * indX.Core_volume


            inds_list.append(indX)

        self.inds_list = inds_list





class cap_predictor:
    def __init__(self, opt_obj, cap_obj):
        self.opt_obj = opt_obj
        self.recommended_params_df = pd.DataFrame.from_dict(
        {'Vrated': [cap_obj.V_rated], 'Capacitance': [cap_obj.Cap_0Vdc],
         'Cost': [cap_obj.Cost],
         'Area': [
             cap_obj.Area]})
        self.cap_index = cap_obj.cap_index


    def database_df_filter(self, opt_obj, fet_obj, num_comps, cap_Mfr_part_no):
        # low_range = 2
        # high_range = 6
        caps_database_df = pd.read_csv('csv_files/capacitor_data.csv')
        if num_comps == 1:
            caps_database_df = caps_database_df[caps_database_df['Mfr_part_no'] == cap_Mfr_part_no]

        # Q1_df = caps_database_df.loc[(caps_database_df['Rated_volt'] >= fet_obj.V_rated) & (caps_database_df['Rated_volt'] <= 2*fet_obj.V_rated)]
        Q1_df = caps_database_df.loc[(caps_database_df['Rated_volt'] >= fet_obj.V_rated)]

        Q1_df = Q1_df.drop_duplicates(subset=['Mfr_part_no'])
        unavailable_caps = ['C3225X7R2A155K200AB', 'C1210C155K1RAC7800', 'NTTFS6H880NLTAG', 'NTTFS6H860NLTAG',
                            'NVTFS6H854NTAG', 'NVTFS6H854NWFTAG', 'NTMFS6H848NT1G', 'NTMFS6H836NT1G', 'NVMFS6H848NLT1G',
                            'NVTFS8D1N08HTAG','CTX50-7-52M-R']
        for un_fet in unavailable_caps:
            Q1_df.drop(Q1_df[Q1_df['Mfr_part_no'] == un_fet].index, inplace=True)
        Q1_df = Q1_df[Q1_df['Series'] != 'SV2220']
        Q1_df = Q1_df[Q1_df['Series'] != 'NTD']
        Q1_df = Q1_df.loc[(Q1_df['Unit_price'] <= 5*fet_obj.Cost)]
        # Q1_df = Q1_df.loc[(Q1_df['Size'] <= 2 * fet_obj.Area)]
        # Q1_df = Q1_df.loc[(Q1_df['Capacitance'] >= 0.95 * fet_obj.Capacitance)]
        if num_comps != 1:
            if fet_obj.cap_index == 1:
                if (len(Q1_df.loc[(Q1_df['Capacitance'] >= 5 * fet_obj.Capacitance)]) > 0):
                    Q1_df = Q1_df.loc[(Q1_df['Capacitance'] >= 5 * fet_obj.Capacitance)]
                else:
                    Q1_df = Q1_df.loc[(Q1_df['Capacitance'] >= fet_obj.Capacitance)]
            else:
                Q1_df = Q1_df.loc[(Q1_df['Capacitance'] >= fet_obj.Capacitance)]

        vars_list = ['Unit_price', 'Size']
        Q1_df['Normalized_score'] = 0
        Q1_df = normalize_x(Q1_df, vars_list, fet_obj)
        self.database_df = Q1_df.nsmallest(num_comps, 'Normalized_score')
        print('len cap database_df: ' + str(len(self.database_df)))
        return



    def compute_loss_equations(self, opt_obj, cap_obj=None):
        # self.database_df['category'] = self.database_df.apply(lambda x: self.compute_category_bf(x.V_dss), axis=1)
        # self.database_df['kT'] = self.database_df.apply(lambda x: self.compute_kT_bf(x.category, x.V_dss), axis=1)
        # self.database_df['Power_loss'] = self.I_Q1 ** 2 * self.database_df['kT'] * self.database_df['R_ds'] + \
        #                       self.x_result[2] * self.database_df['Q_g'] * self.fets_df.loc[0, 'Vgate']

        if cap_obj == None:
            caps_objects = self.database_df.to_records()
        else:
            caps_objects = [cap_obj]
        # fet = fets[0]
        caps_list = []

        for cap in caps_objects:

            if cap_obj == None:
                capX = OptimizerCapacitor(opt_obj.param_dict, 0)
                capX.V_rated, capX.Cap, capX.Cost, capX.Area, capX.Mfr_part_no = \
                    cap.Rated_volt, cap.Capacitance, cap.Unit_price, cap.Size, cap.Mfr_part_no

            else:
                capX = cap
                capX.V_rated, capX.Cap, capX.Cost, capX.Area, capX.Mfr_part_no = \
                    cap.Rated_volt, cap.Capacitance, cap.Unit_price, cap.Size, cap.Mfr_part_no

            caps_list.append(capX)

        self.caps_list = caps_list

        # self.recommended_comps_df = pd.DataFrame.from_records(vars(s) for s in caps_list).loc[0:30]



    def print_parameters(self, fet_obj, n):
        fet_column_list = ['V_rated [V]','Capacitance [F]','Cost [$]','Area [mm]']


        print('Cap ' + str(n) + ' recommended parameters:')
        # Note, the following uses recommended_params_df, not recommended_comps_df, and 'headers' just gives titles to the
        # columns already in the order of recommended_params_df
        print(tabulate(self.recommended_params_df.drop_duplicates(inplace=False), headers=fet_column_list, showindex=False,
                       tablefmt='fancy_grid',
                       floatfmt=".13f"))

    ### Also need to add something as seen at end of compute_power loss eqns in fets, where adding for vars onto a list, to be used
    # for brute-force search


class converter_predictor:
    def __init__(self, opt_obj):
        self.recommended_params_df = pd.DataFrame.from_dict(
            {'frequency': [opt_obj.fsw], 'delta_i': [opt_obj.delta_i], 'Total cost': [opt_obj.Cost_tot], 'Total area': [opt_obj.Area_tot],
             'Total Ploss/Pout': [opt_obj.power_tot / opt_obj.Pout]})



class make_component_predictions_bf:
    def __init__(self, opt_obj, database, cost_constraint, area_constraint, num_comps):
        self.opt_obj = opt_obj
        self.database = database
        self.cost_constraint = cost_constraint
        self.area_constraint = area_constraint
        self.opt_obj.param_dict['set_constraint_val'] = area_constraint
        self.opt_obj.param_dict['plotting_range'] = [cost_constraint, cost_constraint + 1]
        self.num_comps = num_comps
        self.time_start = time.time()
        self.time_end = 0
        # keep 5 and 15


    def filter_components(self, Mfr_part_no_dict = None):
        if Mfr_part_no_dict != None:
            self.Mfr_part_no_dict = Mfr_part_no_dict
            self.num_comps = 1
        pd.set_option("display.max_rows", None, "display.max_columns", None)

        print('Total cost: %f, Total area: %f, Total power loss/Pout: %f\n' % (
            self.opt_obj.Cost_tot, self.opt_obj.Area_tot, self.opt_obj.power_tot / self.opt_obj.Pout
        ))
        # self.fet1.Rds = self.fet1.Ron * self.fet_normalizer
        # self.fet2.Rds = self.fet2.Ron * self.fet_normalizer
        self.opt_obj.delta_i = self.opt_obj.delta_i / self.opt_obj.delta_i_normalizer
        self.opt_obj.fsw = self.opt_obj.fsw * self.opt_obj.ind_normalizer

        print('delta i: %f, fsw: %f\n' % (
            self.opt_obj.delta_i, self.opt_obj.fsw
        ))

        self.opt_obj.delta_i = self.opt_obj.delta_i * self.opt_obj.delta_i_normalizer
        self.opt_obj.fsw = self.opt_obj.fsw / self.opt_obj.ind_normalizer
        # self.opt_obj.fet1.fet_tech = self.opt_obj.fet_tech
        # self.opt_obj.fet2.fet_tech = self.opt_obj.fet_tech

        # Create fet cases of the database for each fet in the design
        self.fets_cases = []
        fet_index = 0
        for fet in self.opt_obj.fet_list:
            if not fet.make_new_opt_var:
                self.fet_case = fet_predictor(self, self.opt_obj, fet)
                self.fets_cases.append(self.fet_case)
                fet_index += 1
                continue
            self.fet_case = fet_predictor(self, self.opt_obj, fet)
            fet.fet_index = fet_index
            self.fet_case.database_df_filter(self.opt_obj, fet, self.num_comps, Mfr_part_no_dict['fet_mfr_part_nos'][fet_index] if self.num_comps == 1 else None)  # after this, have what we need on self.database_df --> need to turn this back into list
            self.fet_case.compute_loss_equations(self.opt_obj)
            self.fets_cases.append(self.fet_case)
            fet_index += 1


        self.inds_cases = []
        ind_index = 0
        for ind in self.opt_obj.ind_list:
            self.ind_case = ind_predictor(self.opt_obj, ind)
            ind.ind_index = ind_index
            self.ind_case.database_df = pd.read_csv('csv_files/inductor_training_updatedAlgorithm.csv')
            if self.num_comps == 1:
                self.ind_case.database_df = self.ind_case.database_df[self.ind_case.database_df['Mfr_part_no'] == Mfr_part_no_dict['ind_mfr_part_nos'][ind_index]]
                self.ind_case.compute_loss_equations(self.opt_obj)

            else:
                self.ind_case.compute_loss_equations(self.opt_obj)

            self.ind_case.database_df_filter(ind,
                                              self.num_comps)  # what if we computed these loss equations prior to filtering? we already have the opt_obj anyways
            self.ind_case.compute_loss_equations_final(self.opt_obj)
            self.inds_cases.append(self.ind_case)
            ind_index += 1



        self.caps_cases = []
        cap_index = 0
        for cap in self.opt_obj.cap_list:
            self.cap_case = cap_predictor(self.opt_obj, cap)
            cap.cap_index = cap_index
            self.cap_case.database_df_filter(self.opt_obj, cap, self.num_comps, Mfr_part_no_dict['cap_mfr_part_nos'][
                cap_index] if self.num_comps == 1 else None)  # after this, have what we need on self.database_df --> need to turn this back into list
            self.cap_case.compute_loss_equations(self.opt_obj)
            self.caps_cases.append(self.cap_case)
            cap_index += 1


        print('continue')



    def optimize_combinations(self):
        '''
        Compute the optimal combination of recommended components
        :return: None, but set an attribute with optimal components
        '''

        # now search the database for parts with similar parameters

        # drop ones w/ NaN or 0
        attr_list = ['Mfr_part_no', 'Unit_price', 'FET_type', 'Technology', 'V_dss', 'R_ds', 'Q_g', 'C_oss', 'Vds_meas',
                     'Q_rr', 'I_F']
        import itertools
        def flatten_list(nested_list):
            return (list(itertools.chain(*nested_list)))


        grid_list = []
        for fet_case in self.fets_cases:
            try:
                grid_list.append(fet_case.fets_list)
            except:
                # grid_list.append(self.fets_cases[2].fets_list)
                continue
        for ind_case in self.inds_cases:
            grid_list.append(ind_case.inds_list)
        for cap_case in self.caps_cases:
            grid_list.append(cap_case.caps_list)

        grid_list_mult = 1
        for el in grid_list:
            grid_list_mult *= len(el)

        new_grid = np.array(np.meshgrid(*grid_list)).T

        while len(new_grid) != grid_list_mult:
            new_grid = flatten_list(new_grid)
        print(len(new_grid))

        if len(new_grid) == 1:
            while (any(isinstance(sub, list) for sub in new_grid) or any(
                    isinstance(sub, np.ndarray) for sub in new_grid)):
                new_grid = list(itertools.chain(*new_grid))
            new_grid = [new_grid]



        # Create an object for each combination
        start = time.time()
        self.combo_list = []
        for combo in new_grid:
            # print(new_grid[combo])
            # print(self.component_combo(new_grid[combo],self.opt_obj, self.database, self.cost_constraint, self.area_constraint))
            ###
            # combo_list.append(component_combo(new_grid[combo],self.opt_obj, self.database, self.cost_constraint, self.area_constraint))
            new_combo = component_combo2(combo, self.database, self.cost_constraint, self.area_constraint, self.opt_obj)
            # make a function that sets parameters of additional components equal to those that are set
            for fet1 in new_combo.opt_obj.fet_list:
                if not fet1.make_new_opt_var:
                    for index, fet2 in enumerate(new_combo.opt_obj.fet_list):
                        if fet1.Rds == fet2.Rds:
                            # get index of fet2, that is where it is in new_combo[x], add another to new_combo that is that same component.
                            # this won't mess up the meshgrid bc we've already done that. The user already set the Rds' equal in
                            # optimization_tool_case_statements --> set_optimization_variables()
                            new_combo.combo = np.append(new_combo.combo, new_combo.combo[index])
                            break
                # could do this same thing w/ inductors and/or capacitors if also have ones that are set equal
            self.combo_list.append(new_combo)

        end = time.time()
        print(f'create combo time: {end-start}')

        from component_selection_case_statements import make_original_arrays, make_new_arrays, vectorize_equations, compute_total_power, make_valid_arrays, set_combo_variables



        # use vectorization to perform object functions on all combination objects in combo_list
        vec_f = np.vectorize(component_combo2.compute_total_cost)
        vec_f(self.combo_list)
        vec_f = np.vectorize(component_combo2.compute_total_area)
        vec_f(self.combo_list)
        vectorize_equations(self)


        # import numexpr as ne
        start = time.time()

        compute_total_power(self)
        # vec_f = np.vectorize(component_combo2.compute_total_power)
        # vec_f(self.combo_list)
        end = time.time()
        # print('method1 = %f' % (end - start))


        # Then create 3 separate arrays: One w/ cost less than constraint, one w/ area less than constraint, and one
        # w/ power loss to later take the minimum of
        Cost_arr = np.array([item.Cost_tot for item in self.combo_list])
        Area_arr = np.array([item.Area_tot for item in self.combo_list])
        Power_arr = np.array([item.power_tot for item in self.combo_list])
        make_original_arrays(self)



        # keeps the same size of the original arrays, has true and false in each of the indices
        new_area_arr = Area_arr < 1 * self.opt_obj.area_constraint
        new_cost_arr = Cost_arr < 1 * self.opt_obj.cost_constraint
        make_new_arrays(self)
        # if (any(new_area_arr) == True) or (any(new_cost_arr) == True): # then have a true value

        # Could use something where checks if there are no viable combinations, goes back and swaps out with new components and re-run
        self.valid_arr = np.logical_and(new_area_arr, new_cost_arr)
        make_valid_arrays(self)
        # valid_arr = np.logical_and(valid_arr, new_C2_arr)

        # map the true/false values onto the power array
        start = time.time()
        valid_power_arr = np.multiply(Power_arr, self.valid_arr)
        valid_power_arr[valid_power_arr == 0] = np.nan

        # Turn multi-dimensional list of valid_power_arr into numpy array and then determine best 5 indices/component combinations
        valid_power_arr = np.asarray(valid_power_arr)
        best_idx_list = []
        unavailable_fets = ['NVTFS6H854NTAG', 'NVTFS6H854NWFTAG', 'NTMFS6H848NT1G']
        unavailable_inds = ['ASPIAIG-Q1513-220M-T', 'ASPIAIG-Q1513-330M-T', 'ASPIAIG-Q1513-8R2M-T', 'EXL1V1010-5R6-R',
                            'ASPIAIG-Q1010-5R6M-T', 'AMDLA1707S-6R8MT', 'ASPIAIG-Q1513-150M-T','0925-221H','AMPLA7030S-R60MT']
        unavailable_fets = []
        unavailable_inds = []
        iter = 0
        iter_total = 5
        if self.num_comps == 1:
            iter_total = 1
            self.test_combo = self.combo_list[0]
        while iter < iter_total:
            # note that the different combination objects are in combo_list, so index through that, and then each combo
            # has objects of each component: .fet1, .fet2, etc.
            # .fet1_case, .fet2_case, etc. are the predictor objects, made using opt_obj.fet1, opt_obj.fet2, etc.,
            # and contain information about the predicted values

            best_idx = np.where(valid_power_arr == np.nanmin(valid_power_arr))

            # The following line is not yet generalized, can write the following however you want, given that the
            # best power loss components are found in best_idx[0][0], and all are appended to best_idx_list
            if self.combo_list[best_idx[0][0]].fet1.Mfr_part_no in unavailable_fets or \
                    self.combo_list[best_idx[0][0]].fet2.Mfr_part_no in unavailable_fets or \
                    self.combo_list[best_idx[0][0]].ind1.Mfr_part_no in unavailable_inds:
                valid_power_arr[best_idx[0][0]] = 10e4
                continue
            best_idx_list.append((best_idx[0][0],
                                  valid_power_arr[
                                      best_idx[0][0]],Cost_arr[
                                      best_idx[0][0]], Area_arr[
                                      best_idx[0][0]]))
            valid_power_arr[best_idx[0][0]] = 10e4
            iter += 1

        for best_idx in best_idx_list:
            # print(best_idx)
            # print("Ploss/Pout: {}, total cost: {}, total area: {}, fet1: {}, fet2: {}, ind1: {}, cap1: {}, cap2: {}".format(
            #     (best_idx[1] / self.opt_obj.Pout), best_idx[2], best_idx[3],
            #     self.combo_list[best_idx[0]].fet1.Mfr_part_no,
            #     self.combo_list[best_idx[0]].fet2.Mfr_part_no,
            #     self.combo_list[best_idx[0]].ind1.Mfr_part_no, self.combo_list[best_idx[0]].cap1.Mfr_part_no, self.combo_list[best_idx[0]].cap2.Mfr_part_no))
            print("Ploss/Pout: {}, total cost: {}, total area: {}".format(
                (best_idx[1] / self.opt_obj.Pout), best_idx[2], best_idx[3] ))
            print(f"Components: \n {self.combo_list[best_idx[0]].combo}")
            print("Manufacturer part numbers: ")
            for obj in self.combo_list[best_idx[0]].combo:
                print(obj.Mfr_part_no)

        self.Ploss_div_Pout = best_idx_list[0][1] / self.opt_obj.Pout

        self.time_end = time.time()
        print('full run = %f' % (self.time_end - self.time_start))

        print('done')


    ### unused function, consider removing ###
    # This function takes the optimized parameters and prints them, then prints the parameters of the selected components.
    # Also prints the losses broken down by contribution, as well as each component's cost and area contributions
    def compare_parameters(self):

        if self.practical:
            # First, print all parameters of selected components

            Mfr_part_nos_dict = self.Mfr_part_nos_dict
            self.filter_components(Mfr_part_nos_dict)

            # After this, have all the computed quantities for each component, can now skip to the printing part
            Mfr_part_nos_list = []
            for list_val in Mfr_part_nos_dict.values():
                for entry in list_val:
                    Mfr_part_nos_list.append(entry)

            # print(
            #     f'Printing parameters of selected components: Q1: {Mfr_part_nos_list[0]}, Q2: {Mfr_part_nos_list[1]}, L1: {Mfr_part_nos_list[2]}, '
            #     f'C1: {Mfr_part_nos_list[3]}, C2: {Mfr_part_nos_list[4]}')
            #
            # fet1_Mfr_part_no = Mfr_part_nos_list[0]
            # fet1_lst = [x for x in self.fet1_case.fets_list if
            #             x.Mfr_part_no == fet1_Mfr_part_no]  # list of all elements with .n==30
            # fet1_ds = fet1_lst[0]
            # print(f'Vdss: {fet1_ds.Vdss}, Ron: {fet1_ds.Rds}, Qg: {fet1_ds.Qg}, cost: {fet1_ds.Cost}, area: {fet1_ds.Area}')
            #
            # fet2_Mfr_part_no = Mfr_part_nos_list[1]
            # fet2_lst = [x for x in self.fet2_case.fets_list if
            #             x.Mfr_part_no == fet2_Mfr_part_no]  # list of all elements with .n==30
            # fet2_ds = fet2_lst[0]
            # print(
            #     f'Vdss: {fet2_ds.Vdss}, Ron: {fet2_ds.Rds}, Qg: {fet2_ds.Qg}, tau_c: {fet2_ds.tau_c}, tau_rr: {fet2_ds.tau_rr}, Cossp1: {fet2_ds.Cossp1}, cost: {fet2_ds.Cost}, area: {fet2_ds.Area}')
            #
            # ind1_Mfr_part_no = Mfr_part_nos_list[2]
            # ind1_lst = [x for x in self.ind1_case.inds_list if
            #             x.Mfr_part_no == ind1_Mfr_part_no]  # list of all elements with .n==30
            # ind1_ds = ind1_lst[0]
            # print(
            #     f'L: {ind1_ds.L}, Irated: {ind1_ds.I_rated}, Rdc: {ind1_ds.R_dc}, cost: {ind1_ds.Cost}, area: {ind1_ds.Area}, '
            #     f'Rac_total: {ind1_ds.Rac_total}, IGSE_total: {ind1_ds.IGSE_total}, core volume: {ind1_ds.Core_volume}')
            #
            # cap1_Mfr_part_no = Mfr_part_nos_list[3]
            # cap1_lst = [x for x in self.cap1_case.caps_list if
            #             x.Mfr_part_no == cap1_Mfr_part_no]  # list of all elements with .n==30
            # cap1_ds = cap1_lst[0]
            # print(f'Capacitance: {cap1_ds.Cap}, Vrated: {cap1_ds.V_rated}, cost: {cap1_ds.Cost}, area: {cap1_ds.Area}')
            #
            # cap2_Mfr_part_no = Mfr_part_nos_list[4]
            # cap2_lst = [x for x in self.cap2_case.caps_list if
            #             x.Mfr_part_no == cap2_Mfr_part_no]  # list of all elements with .n==30
            # cap2_ds = cap2_lst[0]
            # print(f'Capacitance: {cap2_ds.Cap}, Vrated: {cap2_ds.V_rated}, cost: {cap2_ds.Cost}, area: {cap2_ds.Area}')

            # Now get all the related loss/cost/area contribution break-downs: call optimize_combinations() with the single
            # component on each list, then should have all the necessary parameters on that object

            # All attributes of combination are put on test_combo.Cost_tot
            self.optimize_combinations()

            return

    def print_all_parameters(self):
        from component_selection_case_statements import print_practical, print_theoretical
        if self.practical:
            Mfr_part_nos_dict = self.Mfr_part_nos_dict
            # TODO: why are we filtering components when we already have part list commented out filter_components and optimize_combinations
            # TODO: no fets cases without this
            self.filter_components(Mfr_part_nos_dict)

            # After this, have all the computed quantities for each component, can now skip to the printing part
            Mfr_part_nos_list = []
            for list_val in Mfr_part_nos_dict.values():
                for entry in list_val:
                    Mfr_part_nos_list.append(entry)

            # No test combo without this
            self.optimize_combinations()
            # TODO: this should at least be a boolean variable, won't always be wanted
            # self.optimize_fsw()
            print_practical(self)

        if self.theoretical:
            with open('optimizer_test_values/optimizer_obj_' + self.opt_obj.topology + '_' + self.opt_obj.fet_tech + '_' + str(
                    self.cost_constraint) + '_' + str(self.area_constraint), 'rb') as optimizer_obj_file:
                # Step 3
                optimizer_obj = pickle.load(optimizer_obj_file)
            optimization_case = make_component_predictions_bf(optimizer_obj, 'pdf_params', self.cost_constraint,
                                                              self.area_constraint, self.num_comps)

            print_theoretical(optimization_case)

        # fet1_objects = [fet_object(**kwargs) for kwargs in self.fet1_case.recommended_comps_df[
        #     ['Area', 'Cost', 'Mfr_part_no', 'Vdss', 'Rds', 'Qg', 'tau_c', 'tau_rr', 'Cossp1', 'I_F',
        #      'Vds_meas']].to_dict(orient='records')]







    def optimize_fsw(self):
        from component_selection_case_statements import bounds_check
        # given the parameters of the component, and the power loss model used, determine the switching frequency
        # leading to lowest power loss

        # want to look at self.test_combo.combo, and re-compute the power loss. see what is useful from the optimize_combinations
        # code, can maybe reuse some of that here
        from component_selection_case_statements import compute_total_power, set_inductor_attributes
        self.combo_list = []
        self.combo_list.append(self.test_combo)
        fsw_loss_list = []
        fsw_list = np.arange(10000, 3000000, 10000).tolist()
        for fsw in fsw_list:

            self.opt_obj.fsw = fsw/self.opt_obj.ind_normalizer
            # nope, really need to recompute delta_I for real, bc we have a new fsw !!!
            # from component_selection_case_statements import set_inductor_attributes
            # try:
            # TODO: iterate through all the inductors in the design, also figure out what to do if we want to keep delta_i set
            for obj in self.combo_list[0].combo:
                try:
                    set_inductor_attributes(obj, self.combo_list[0].opt_obj)
                except:
                    continue

                # set_inductor_attributes(self.combo_list[0].ind1, self.combo_list[0].opt_obj)
                # self.opt_obj.delta_i = self.combo_list[0].ind1.delta_i
            # except:
                # self.opt_obj.delta_i = self.combo_list[0].opt_obj.delta_i
            # self.opt_obj.delta_i = self.combo_list[0].opt_obj.delta_i / self.combo_list[0].opt_obj.delta_i_normalizer
            compute_total_power(self)
            # TODO: clean up the code around this
            print(f'fsw: {fsw}, delta_i: {self.combo_list[0].opt_obj.delta_i}, Ploss/Pout: {self.combo_list[0].power_tot / self.opt_obj.Pout}')
            fsw_loss_list.append([fsw, self.combo_list[0].power_tot / self.opt_obj.Pout, self.combo_list[0].opt_obj.delta_i])

        # TODO: add another function that does additional checks, e.g. for microinverter_combined check that delta_i is in the right range
        fsw_loss_list = bounds_check(fsw_loss_list, self.opt_obj)
        min_idx = [el[1] for el in fsw_loss_list].index(min([el[1] for el in fsw_loss_list]))
        min_fsw, min_loss, min_delta_i = fsw_loss_list[min_idx]
        print(f'min fsw: {min_fsw}, min loss: {min_loss}, min delta_i: {min_delta_i}')

        self.opt_obj.fsw = min_fsw/self.opt_obj.ind_normalizer
        for obj in self.combo_list[0].combo:
            try:
                set_inductor_attributes(obj, self.combo_list[0].opt_obj)
            except:
                continue
        compute_total_power(self)
        print(self.opt_obj.power_tot/self.opt_obj.Pout)
        # First select the component combination used

    #     # self.fet1_case = fet_predictor(self, self.opt_obj, self.opt_obj.fet1)
    #     # self.fet1_case.database_df_filter(self.opt_obj, self.opt_obj.fet1)
    #     fet1 = next((x for x in self.fet1_case.fets_list if x.Mfr_part_no == self.Mfr_part_nos_list[0]), None)
    #     fet2 = next((x for x in self.fet2_case.fets_list if x.Mfr_part_no == self.Mfr_part_nos_list[1]), None)
    #     ind1 = next((x for x in self.ind1_case.inds_list if x.Mfr_part_no == self.Mfr_part_nos_list[2]), None)
    #     cap1 = next((x for x in self.cap1_case.caps_list if x.Mfr_part_no == self.Mfr_part_nos_list[3]), None)
    #     cap2 = next((x for x in self.cap2_case.caps_list if x.Mfr_part_no == self.Mfr_part_nos_list[4]), None)
    #
    #     # fet1 = self.fet1_case.database_df.loc[self.fet1_case.database_df['Mfr_part_no'] == 'SIS176LDN-T1-GE3']
    #     # fet2 = self.fet2_case.database_df.loc[self.fet2_case.database_df['Mfr_part_no'] == 'SIS178LDN-T1-GE3']
    #     # ind1 = self.ind1_case.database_df.loc[self.ind1_case.database_df['Mfr_part_no'] == 'MWLA1707S-6R8MT']
    #
    #     # fet1 = self.fet1_case.database_df.loc[self.fet1_case.database_df['Mfr_part_no'] == 'AON6284']
    #     # fet2 = self.fet2_case.database_df.loc[self.fet2_case.database_df['Mfr_part_no'] == 'NTMFS6D1N08HT1G']
    #     # ind1 = self.ind1_case.database_df.loc[self.ind1_case.database_df['Mfr_part_no'] == '74439370082']
    #
    #     # fet1 = self.fet1_case.database_df.loc[self.fet1_case.database_df['Mfr_part_no'] == 'EPC2052']
    #     # fet2 = self.fet2_case.database_df.loc[self.fet2_case.database_df['Mfr_part_no'] == 'EPC2052']
    #     # ind1 = self.ind1_case.database_df.loc[self.ind1_case.database_df['Mfr_part_no'] == 'LMLP07C7M1R0DTAS']
    #
    #     # fet1 = self.fet1_case.database_df.loc[self.fet1_case.database_df['Mfr_part_no'] == 'EPC2052']
    #     # fet2 = self.fet2_case.database_df.loc[self.fet2_case.database_df['Mfr_part_no'] == 'EPC2204']
    #     # ind1 = self.ind1_case.database_df.loc[self.ind1_case.database_df['Mfr_part_no'] == 'SRP1513CA-100M']
    #
    #     power_arr = []
    #     fsw_list = [1*10e3, 1*10e4, 2*10e4, 3*10e4, 4*10e4, 5*10e4, 7*10e4, 10e5, 2*10e5, 3*10e5, 4*10e5, 5*10e5]
    #     fsw_list = np.arange(10000, 300000, 10000).tolist()
    #     # fsw_list = [2190496.885057]
    #     # fsw_list = [174901.667170]
    #
    #     # fsw_list = [1*10e4,1.4*10e4, 1.5*10e4, 1.6*10e4, 2*10e4, 3*10e4, 4*10e4]
    #     # fsw_list = [2000000]
    #     for fsw in fsw_list:
    #
    #         self.opt_obj.fsw = fsw/self.opt_obj.ind_normalizer
    #         # may want to adjust the following so directly takes the object combo, but for now allow for it to be done
    #         # manually for testing purposes
    #         combo = [fet1, fet2, ind1, cap1, cap2]
    #         optimization_case = component_combo2(combo,self.opt_obj, self.database, self.cost_constraint, self.area_constraint)
    #         optimization_case.ind1.delta_i = optimization_case.opt_obj.delta_i_normalizer * (
    #                 optimization_case.opt_obj.converter_df.loc[0, 'Vin'] - optimization_case.opt_obj.converter_df.loc[
    #             0, 'Vout']) * optimization_case.opt_obj.dc * (
    #                                1 / (optimization_case.opt_obj.fsw * optimization_case.opt_obj.ind_normalizer)) / (
    #                                2 * optimization_case.ind1.L * optimization_case.opt_obj.converter_df.loc[0, 'Iout'])
    #         if (optimization_case.ind1.delta_i < 0.1) or (optimization_case.ind1.delta_i > 7):  # Translates to delta_i [%] = 1% or 70% as limits
    #             print(f'fsw = {fsw} not allowed, delta_i = {optimization_case.ind1.delta_i}')
    #             # continue  # set to be very high so won't ever use, as associated losses will be very high
    #
    #         self.opt_obj.delta_i = optimization_case.ind1.delta_i
    #
    #         # print('computing for optimized fsw: Rac and IGSE')
    #         optimization_case.ind1.compute_Rac(self.opt_obj)
    #         optimization_case.ind1.compute_IGSE(self.opt_obj)
    #
    #         optimization_case.compute_total_power()
    #
    #         # print(
    #         #     f'Total Ploss/Pout: {optimization_case.Power_tot / optimization_case.opt_obj.Pout}, Total cost: {optimization_case.Cost_tot}, total area: {optimization_case.Area_tot}')
    #         # print(f'fet1 loss breakdown:')
    #         # print(f'Rds losses: {optimization_case.fet1.Rds_loss}, Qg losses: {optimization_case.fet1.Qg_loss}')
    #         # print(f'fet2 loss breakdown:')
    #         # print(f'Rds losses: {optimization_case.fet2.Rds_loss}, Qg losses: {optimization_case.fet2.Qg_loss},'
    #         #       f'Cdsq losses: {optimization_case.fet2.Cdsq_loss}, toff losses: {optimization_case.fet2.toff_loss}')
    #         # if optimization_case.opt_obj.fet_tech == 'MOSFET':
    #         #     print(f'Qrr losses: {optimization_case.fet2.Qrr_loss}')
    #         # print(f'ind1 loss breakdown:')
    #         # print(f'Rdc losses: {optimization_case.ind1.Rdc_loss}, Rac losses: {optimization_case.ind1.Rac_loss},'
    #         #       f'IGSE losses: {optimization_case.ind1.IGSE_loss}')
    #         # print(f'Loss contribution by components:')
    #         # print(
    #         #     f'Q1: {optimization_case.fet1.Loss}, Q2: {optimization_case.fet2.Loss}, L1: {optimization_case.ind1.Loss}, total sum: {optimization_case.fet1.Loss + optimization_case.fet2.Loss + optimization_case.ind1.Loss}')
    #         #
    #         print(f'fsw: {fsw}, Ploss/Pout: {optimization_case.Power_tot/self.opt_obj.Pout}')
    #         print('\n')
    #
    # #     pass

class component_combo2(OptimizerInit):
    def __init__(self, comp_list,database, cost_constraint, area_constraint, opt_obj):

        start = time.time()
        ## PUT THIS BACK?
        # super().__init__(param_dict=opt_obj.param_dict)


        end = time.time()
        # print(f'super call:{end-start}')

        param_dict = opt_obj.param_dict

        self.opt_var = param_dict['opt_var']
        self.plotting_var = param_dict['plotting_var']
        self.set_constraint = param_dict['set_constraint']
        self.example_num = param_dict['example_num']
        self.plotting_range = param_dict['plotting_range']
        self.tech_list = param_dict['tech_list']
        self.num_points = param_dict['num_points']
        self.predict_components = param_dict['predict_components']
        self.fet_tech = param_dict['test_fet_tech']
        self.topology = param_dict['topology']
        self.status = 0
        self.plotting_val = 0


        # def set_design_dfs(self):
        self.get_params()
        self.create_plot_vars()

        self.converter_df = pd.DataFrame(data=self.converter_data_dict)
        self.fets_df = pd.DataFrame(data=self.fet_data_dict)
        self.inds_df = pd.DataFrame(data=self.ind_data_dict)
        self.caps_df = pd.DataFrame(data=self.cap_data_dict)
        # self.constraints_df = pd.DataFrame(data=self.constraints_data_dict)
        self.Pout = self.converter_df.loc[0, 'Vout'] * self.converter_df.loc[0, 'Iout']
        self.total_components = len(self.fets_df) + len(self.inds_df) + len(self.caps_df)


        start = time.time()

        self.combo = comp_list
        self.Cost_tot = 0
        self.Area_tot = 0
        self.Power_tot = 0
        self.database = database
        self.cost_constraint = cost_constraint
        self.area_constraint = area_constraint
        self.opt_obj = opt_obj
        end = time.time()
        # print(f'everything else call:{end-start}')


    def compute_total_cost(self):
        self.Cost_tot = 0
        for obj in self.combo:
            self.Cost_tot += obj.Cost
        # print('cost computed')
        # self.Cost_tot += 3 * self.combo[2].Cost

    def compute_total_area(self):
        self.Area_tot = 0
        for obj in self.combo:
            self.Area_tot += obj.Area
        # print('area computed')
        # self.Area_tot += 3 * self.combo[2].Area




if __name__ == '__main__':
    # risky change
    pd.set_option("display.max_rows", 100, "display.max_columns", 100)

    # data_cleaning()
    # ML_model_training()
    # resulting_parameter_plotting()

    predict_components()
    compare_combinations()
    print('done')