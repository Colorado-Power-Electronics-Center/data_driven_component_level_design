'''
    This file contains the functions to find the optimal distribution of resources to minimize some objective function,
    weighted for minimizing power loss, cost, and/or area, subject to power loss, cost, and/or area constraint.
'''

import numpy as np
import scipy
from scipy.optimize import minimize, basinhopping, fsolve
import matplotlib.pyplot as plt
import fet_regression
import pandas as pd
from fet_database_partsearch import user_COTS_choice
from tabulate import tabulate
import joblib
from scipy.optimize import Bounds
import time
from coreloss import coreloss
import pickle
import warnings
from fet_area_filtering import area_filter
import weakref
import csv

class OptimizerInit:

    def __init__(self, param_dict):

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
        self.prior_opt_var = 10000
        self.prior_variable_constraint = 10000
        self.slsqp_points = []
        self.freq_points = []
        # self.Q1_loss_points = []
        # self.Q2_loss_points = []
        # self.L1_loss_points = []
        self.total_cost_points = []
        self.total_area_points = []
        self.MOSFET_overall_points = []
        self.GaNFET_overall_points = []

        self.MOSFET_freq_points = []
        self.GaNFET_freq_points = []
        self.MOSFET_delta_i_points = []
        self.GaNFET_delta_i_points = []
        # self.MOSFET_Q1_loss_points = []
        # self.MOSFET_Q2_loss_points = []
        # self.MOSFET_L1_loss_points = []
        # self.GaNFET_Q1_loss_points = []
        # self.GaNFET_Q2_loss_points = []
        # self.GaNFET_L1_loss_points = []
        self.MOSFET_total_cost_points = []
        self.MOSFET_total_area_points = []
        self.GaNFET_total_cost_points = []
        self.GaNFET_total_area_points = []
        self.MOSFET_total_power_points = []
        self.GaNFET_total_power_points = []

        self.x_result = 0
        self.opt_var_reset = 10000
        self.constraint_reset = 0

        # self.optimization_table = {"power": self.power_pred_tot, "cost": self.cost_pred_tot, "area": self.area_pred_tot}
        self.bounds_table = {"power": self.power_bounds_fcn1, "power2": self.power_bounds_fcn2,
                             "cost": self.cost_bounds_fcn, "area": self.area_bounds_fcn}
        self.previous_opt_var = self.prior_opt_var
        self.previous_variable_constraint = self.prior_variable_constraint
        self.degree = 1
        self.model = 'chain'

        self.fold = 10e-10
        self.fnew = self.fold - 1
        self.x0 = []

        # self.dc = self.converter_df.loc[0, 'Vout'] / self.converter_df.loc[0, 'Vin']

        self.fet_vals = []
        self.ind_vals = []
        self.power_tot = 0
        self.area_tot = 0
        self.cost_tot = 0
        # TODO: could change this to 10e3 to try to speed up convergence
        self.ind_normalizer = 10 ** 4 # we changed to 10 **4
        self.fet_normalizer = 10 ** -3 # we changed to 10**-3
        self.delta_i_normalizer = 10 ** 1
        self.cap_normalizer = 10 ** 0
        self.power_normalizer = 10 ** 1
        self.Pold = 1
        self.Pnew = 0
        self.I_Q1 = 0
        self.I_Q2 = 0
        self.power_divider = 0
        self.Q1_loss = 0
        self.Q2_loss = 0
        self.L1_loss = 0
        self.power_tot_prev = 0
        self.fun = 0
        self.previous_variable_constraint = 0

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

        self.offset = len(self.fets_df)


        if self.set_constraint == 'area':
            self.area_constraint = param_dict['set_constraint_val']
        elif self.set_constraint == 'power':
            self.power_constraint = param_dict['set_constraint_val']
        elif self.set_constraint == 'cost':
            self.cost_constraint = param_dict['set_constraint_val']

    def get_visualization_params(self):
        self.legend = ['Si', 'GaN']
        self.fontsize = 15
        if self.opt_var == 'power' and self.plotting_var == 'cost':
            self.xlabel = 'Total cost [$\$$]'
            self.ylabel = '$P_{loss}/P_{out}$'
            self.title = 'Ploss/Pout of converter vs. cost constraint, Ex. %d, area constraint = %d' % (
                self.example_num, self.area_constraint)

        elif self.opt_var == 'power' and self.plotting_var == 'area':
            self.xlabel = 'Total area [$mm^2$]'
            self.ylabel = '$P_{loss}/P_{out}$'
            self.title = plt.title(
                'Ploss/Pout of converter vs. area constraint, Ex. %d, cost constraint = $\$${:.2f}'.format(
                    self.cost_constraint) % (
                    self.example_num))

        elif self.opt_var == 'cost' and self.plotting_var == 'power':
            self.xlabel = '$P_{loss}/P_{out}$'
            self.ylabel = 'Total cost [$\$$]'
            self.title = 'Total cost of converter vs. power loss constraint, Ex. %d, area constraint = %d' % (
                self.example_num, self.area_constraint)

        elif self.opt_var == 'cost' and self.plotting_var == 'area':
            self.xlabel = 'Total area [$mm^2$]'
            self.ylabel = 'Total cost [$\$$]'
            self.title = 'Total cost of converter vs. area constraint, Ex. %d, power constraint = {:.2f}'.format(
                self.power_constraint) % (
                             self.example_num)

        elif self.opt_var == 'area' and self.plotting_var == 'power':
            self.xlabel = '$P_{loss}/P_{out}$'
            self.ylabel = 'Total area [$mm^2$]'
            self.title = 'Total area of converter vs. power loss constraint, Ex. %d, cost constraint = $\$${:.2f}'.format(
                self.cost_constraint) % (
                             self.example_num)

        elif self.opt_var == 'area' and self.plotting_var == 'cost':
            self.xlabel = 'Total cost [$\$$]'
            self.ylabel = 'Total area [$mm^2$]'
            self.title = 'Total area of converter vs. cost constraint, Ex. %d, power constraint = {:.2f}'.format(
                self.power_constraint) % (
                             self.example_num)

    def make_component_predictions_bf(self):
        pd.set_option("display.max_rows", None, "display.max_columns", None)

        print('Total cost: %f, Total area: %f, Total power loss/Pout: %f\n' % (
            self.cost_tot, self.area_tot, self.power_tot / self.Pout
        ))
        self.x_result[0] = self.x_result[0] * self.fet_normalizer
        self.x_result[1] = self.x_result[1] * self.fet_normalizer
        self.x_result[2] = self.x_result[2] * self.ind_normalizer
        # self.power_tot = self.power_tot*self.power_normalizer

        Q1_rec_df = pd.DataFrame.from_dict(
            {'Vdss': [self.fets_df.loc[0, 'Vdss']], 'Rds': [self.x_result[0]],
             'Qg': [self.fet_vals[0]['Qg'][0]],
             'Coss': [self.fet_vals[0]['Coss'][0]],
             'Qrr': [self.fet_vals[1]['Qrr'][0]],
             'Cost': [self.fet_vals[0]['Cost'][0]],
             'Area': [
                 self.fet_vals[0]['Area'][0]]})  # , columns=pd.MultiIndex.from_product([['Q1']], names=['Component:']))

        Q1_rec_df.style.set_caption("Q1 Recommended Parameters")
        Q2_rec_df = pd.DataFrame({'Vdss': [self.fets_df.loc[1, 'Vdss']], 'Rds': [self.x_result[1]],
                                  'Qg': [self.fet_vals[1]['Qg'][0]],
                                  'Coss': [self.fet_vals[1]['Coss'][0]],
                                  'Qrr': [self.fet_vals[1]['Qrr'][0]],
                                  'Cost': [self.fet_vals[1]['Cost'][0]],
                                  'Area': [self.fet_vals[1]['Area'][0]]})
        Q2_rec_df.style.set_caption("Q2 Recommended Parameters")
        L1_rec_df = pd.DataFrame(
            {'Inductance': [self.inds_df.loc[0, 'Ind_fsw'] / (self.x_result[2])],
             'Current_rating': [self.inds_df.loc[0, 'Current_rating']],
             'Rdc': [self.ind_vals[0]['DCR'][0] * 10 ** 3],
             'Cost': [self.ind_vals[0]['Cost'][0]], 'Area': [self.ind_vals[0]['Area'][0]]})
        L1_rec_df.style.set_caption("L1 Recommended Parameters")
        converter_rec_df = pd.DataFrame.from_dict(
            {'frequency': [self.x_result[2]], 'Total cost': [self.cost_tot], 'Total area': [self.area_tot],
             'Total Ploss/Pout': [self.power_tot / self.Pout]})

        # set the variables so they are in the units we want
        Q1_rec_df['Rds'] = Q1_rec_df['Rds']
        Q2_rec_df['Rds'] = Q2_rec_df['Rds']
        Q1_rec_df['Qg'] = Q1_rec_df['Qg'] * 10 ** 9
        Q2_rec_df['Qg'] = Q2_rec_df['Qg'] * 10 ** 9
        Q1_rec_df['Qrr'] = Q1_rec_df['Qrr'] * 10 ** 9
        Q2_rec_df['Qrr'] = Q2_rec_df['Qrr'] * 10 ** 9
        Q1_rec_df['Coss'] = Q1_rec_df['Coss'] * 10 ** 12
        Q2_rec_df['Coss'] = Q2_rec_df['Coss'] * 10 ** 12
        L1_rec_df['Rdc'] = L1_rec_df['Rdc']
        L1_rec_df['Inductance'] = L1_rec_df['Inductance'] * self.ind_normalizer

        fet_column_list = ['V_dss [V]', 'R_ds [mΩ]', 'Q_g [nC]', 'C_oss [pF]', 'Q_rr [nC]', 'Unit_price [$]',
                           'Area [mm^2]']
        ind_column_list = ['Inductance [µH]', 'Current_rating [A]', 'R_dc [mΩ]', 'Unit_price [$]', 'Area [mm^2]'
                           ]
        performance_column_list = ['frequency [Hz]', 'Total cost [$]', 'Total area [mm^2]', 'Total Ploss/Pout']
        print('Q1 recommended parameters:')
        print(tabulate(Q1_rec_df.drop_duplicates(inplace=False), headers=fet_column_list, showindex=False,
                       tablefmt='fancy_grid',
                       floatfmt=".1f"))
        print('Q2 recommended parameters:')
        print(tabulate(Q2_rec_df.drop_duplicates(inplace=False), headers=fet_column_list, showindex=False,
                       tablefmt='fancy_grid',
                       floatfmt=".1f"))
        print('L1 recommended parameters:')
        print(tabulate(L1_rec_df.drop_duplicates(inplace=False), headers=ind_column_list, showindex=False,
                       tablefmt='fancy_grid',
                       floatfmt=".1f"))
        print('Converter expected performance using recommended parameters:')
        print(tabulate(converter_rec_df, headers=performance_column_list, showindex=False,
                       tablefmt='fancy_grid',
                       floatfmt=".3f"))

        # now search the database for parts with similar parameters

        with open('fet_df_test_values', 'rb') as optimizer_obj_file:
            # Step 3
            fets_database_df = pickle.load(optimizer_obj_file)

        # drop ones w/ NaN or 0
        attr_list = ['Mfr_part_no', 'Unit_price', 'FET_type', 'Technology', 'V_dss', 'R_ds', 'Q_g', 'C_oss', 'Vds_meas',
                     'Q_rr', 'I_F']
        fets_database_df = fets_database_df.replace(0, np.nan)
        # fets_database_df = fets_database_df.dropna(subset=attr_list)
        fets_database_df = fets_database_df[fets_database_df['Technology'] == self.fet_tech]

        xlsx_file = 'csv_files/inductor_training_updatedAlgorithm.csv'
        inds_database_df = pd.read_csv(xlsx_file)

        # filter the fet df to find a component with the following parameters we care about (within an x% range):
        #   Vdss, cost, Rds, Qg
        # for Q1:
        # Q1_df = fets_database_df.iloc[
        #     (fets_database_df['V_dss'] - self.fets_df.loc[0]['Vdss']).abs().argsort()[:(n - 0)]]
        # Q1_df = Q1_df.iloc[(Q1_df['Unit_price'] - self.fet_vals[0]['Cost'][0]).abs().argsort()[:(n - 5)]]
        # Q1_df = Q1_df.iloc[(Q1_df['R_ds'] - self.x_result[0]).abs().argsort()[:(n - 10)]]
        # Q1_df = Q1_df.iloc[(Q1_df['Q_g'] - self.fet_vals[0]['Qg'][0]).abs().argsort()[:(n - 15)]]
        # Q1_df = Q1_df.iloc[(Q1_df['Pack_case'] - self.fet_vals[0]['Area'][0]).abs().argsort()[:(n - 20)]]
        b = 5
        Q1_df = fets_database_df.loc[fets_database_df['V_dss'] > self.fets_df.loc[0]['Vdss']]
        Q1_df = Q1_df.loc[(Q1_df['Unit_price'] > 0.5 * self.fet_vals[0]['Cost'][0]) & (Q1_df['Unit_price'] < b *
                                                                                       self.fet_vals[0]['Cost'][0])]
        Q1_df = Q1_df.loc[(Q1_df['R_ds'] > 0.001 * self.x_result[0]) & (Q1_df['R_ds'] < b * self.x_result[0])]
        Q1_df = Q1_df.loc[
            (Q1_df['Q_g'] > 0.001 * self.fet_vals[0]['Qg'][0]) & (Q1_df['Q_g'] < b * self.fet_vals[0]['Qg'][0])]
        Q1_df = Q1_df.loc[(Q1_df['Pack_case'] > 0.001 * self.fet_vals[0]['Area'][0]) & (Q1_df['Pack_case'] < b *
                                                                                        self.fet_vals[0]['Area'][0])]
        # Q1_df = Q1_df.loc[(Q1_df['C_oss'] > 0.5*self.fet_vals[0]['Coss'][0]) & (Q1_df['C_oss'] < 1.2*self.fet_vals[0]['Coss'][0])]
        # Q1_df = Q1_df.loc[(Q1_df['Q_rr'] > 0.5*self.fet_vals[0]['Qrr'][0]) & (Q1_df['Q_rr'] < 1.2*self.fet_vals[0]['Qrr'][0])]

        # Maybe divide by the max. to normalize values, and then sum, and then choose the top 20 w/ closest?

        # Q2_df = fets_database_df.iloc[(fets_database_df['V_dss'] - self.fets_df.loc[1]['Vdss']).abs().argsort()[:(n-0)]]
        # Q2_df = Q2_df.iloc[(Q2_df['R_ds'] - self.x_result[1]).abs().argsort()[:(n-5)]]
        # Q2_df = Q2_df.iloc[(Q2_df['Q_g'] - self.fet_vals[1]['Qg'][0]).abs().argsort()[:(n-10)]]
        # Q2_df = Q2_df.iloc[(Q2_df['Unit_price'] - self.fet_vals[1]['Cost'][0]).abs().argsort()[:(n-15)]]
        # Q2_df = Q2_df.iloc[(Q2_df['Pack_case'] - self.fet_vals[1]['Area'][0]).abs().argsort()[:(n-20)]]
        Q2_df = fets_database_df.loc[fets_database_df['V_dss'] > self.fets_df.loc[1]['Vdss']]
        Q2_df = Q2_df.loc[(Q2_df['Unit_price'] > 0.5 * self.fet_vals[1]['Cost'][0]) & (Q2_df['Unit_price'] < b *
                                                                                       self.fet_vals[1]['Cost'][0])]
        Q2_df = Q2_df.loc[(Q2_df['R_ds'] > 0.001 * self.x_result[1]) & (Q2_df['R_ds'] < b * self.x_result[1])]
        Q2_df = Q2_df.loc[
            (Q2_df['Q_g'] > 0.001 * self.fet_vals[1]['Qg'][0]) & (Q2_df['Q_g'] < b * self.fet_vals[1]['Qg'][0])]
        Q2_df = Q2_df.loc[(Q2_df['Pack_case'] > 0.001 * self.fet_vals[1]['Area'][0]) & (Q2_df['Pack_case'] < b *
                                                                                        self.fet_vals[1]['Area'][0])]
        # Q2_df = Q2_df.iloc[(Q2_df['C_oss'] > 0.5 * self.fet_vals[1]['Coss'][0]) & (
        #             Q2_df['C_oss'] < 1.2 * self.fet_vals[1]['Coss'][0])]
        # Q2_df = Q2_df.iloc[
        #     (Q2_df['Q_rr'] > 0.5 * self.fet_vals[1]['Qrr'][0]) & (Q2_df['Q_rr'] < 1.2 * self.fet_vals[1]['Qrr'][0])]

        # L1_df = inds_database_df[(self.inds_df.loc[0]['Current_rating'] - inds_database_df['Current_Rating [A]']) <= 0]
        # L1_df = L1_df.iloc[
        #     (self.inds_df.loc[0]['Current_rating'] - L1_df['Current_Rating [A]']).argsort()[25:]]
        # L1_df = L1_df.iloc[(L1_df['DCR [Ohms]'] - self.ind_vals[0]['DCR'][0]).abs().argsort()[:20]]
        # L1_df = L1_df.iloc[
        #     (L1_df['Inductance [H]'] - (self.inds_df.loc[0, 'Ind_fsw'] / (self.x_result[2]))).abs().argsort()[:15]]
        # L1_df = L1_df.iloc[
        #     (L1_df['Length [mm]'] * L1_df['Width [mm]'] - self.ind_vals[0]['Area'][0]).abs().argsort()[:10]]
        # L1_df = L1_df.iloc[(L1_df['Unit_price [USD]'] - self.ind_vals[0]['Cost'][0]).abs().argsort()[:5]]
        L1_df = inds_database_df.loc[(self.inds_df.loc[0]['Current_rating'] < inds_database_df['Current_Rating [A]'])]
        L1_df = L1_df.loc[(L1_df['DCR [Ohms]'] < 1.2 * self.ind_vals[0]['DCR'][0]) & (L1_df['DCR [Ohms]'] > 0.5 *
                                                                                      self.ind_vals[0]['DCR'][0])]
        L1_df = L1_df.loc[
            (L1_df['Inductance [H]'] < 1.2 * (self.inds_df.loc[0, 'Ind_fsw'] / (self.x_result[2]))) & (L1_df[
                                                                                                           'Inductance [H]'] > 0.5 * (
                                                                                                                   self.inds_df.loc[
                                                                                                                       0, 'Ind_fsw'] / (
                                                                                                                   self.x_result[
                                                                                                                       2])))]
        L1_df = L1_df.loc[
            (L1_df['Length [mm]'] * L1_df['Width [mm]'] < 1.2 * self.ind_vals[0]['Area'][0]) & (
                    L1_df['Length [mm]'] * L1_df['Width [mm]'] > 0.5 * self.ind_vals[0]['Area'][0])]
        L1_df = L1_df.loc[(L1_df['Unit_price [USD]'] < 1.2 * self.ind_vals[0]['Cost'][0]) & (
                L1_df['Unit_price [USD]'] > 0.5 * self.ind_vals[0]['Cost'][0])]

        # n = 35
        # Q1_df = fets_database_df.iloc[(fets_database_df['V_dss'] - self.fets_df.loc[0]['Vdss']).abs().argsort()[:(n-0)]]
        # Q1_df = Q1_df.iloc[(Q1_df['Unit_price'] - self.fet_vals[0]['Cost'][0]).abs().argsort()[:(n-5)]]
        # Q1_df = Q1_df.iloc[(Q1_df['R_ds'] - self.x_result[0]).abs().argsort()[:(n-10)]]
        # Q1_df = Q1_df.iloc[(Q1_df['Q_g'] - self.fet_vals[0]['Qg'][0]).abs().argsort()[:(n-15)]]
        # Q1_df = Q1_df.iloc[(Q1_df['C_oss'] - self.fet_vals[0]['Coss'][0]).abs().argsort()[:(n-20)]]
        # Q1_df = Q1_df.iloc[(Q1_df['Q_rr'] - self.fet_vals[0]['Qrr'][0]).abs().argsort()[:(n-25)]]
        # Q1_df = Q1_df.iloc[(Q1_df['Pack_case'] - self.fet_vals[0]['Area'][0]).abs().argsort()[:(n-30)]]
        #
        # # Maybe divide by the max. to normalize values, and then sum, and then choose the top 20 w/ closest?
        #
        # Q2_df = fets_database_df.iloc[(fets_database_df['V_dss'] - self.fets_df.loc[1]['Vdss']).abs().argsort()[:35]]
        # Q2_df = Q2_df.iloc[(Q2_df['R_ds'] - self.x_result[1]).abs().argsort()[:30]]
        # Q2_df = Q2_df.iloc[(Q2_df['Q_g'] - self.fet_vals[1]['Qg'][0]).abs().argsort()[:25]]
        # Q2_df = Q2_df.iloc[(Q2_df['C_oss'] - self.fet_vals[1]['Coss'][0]).abs().argsort()[:20]]
        # Q2_df = Q2_df.iloc[(Q2_df['Q_rr'] - self.fet_vals[1]['Qrr'][0]).abs().argsort()[:15]]
        # Q2_df = Q2_df.iloc[(Q2_df['Unit_price'] - self.fet_vals[1]['Cost'][0]).abs().argsort()[:10]]
        # Q2_df = Q2_df.iloc[(Q2_df['Pack_case'] - self.fet_vals[1]['Area'][0]).abs().argsort()[:5]]
        #
        # L1_df = inds_database_df[(self.inds_df.loc[0]['Current_rating'] - inds_database_df['Current_Rating [A]']) <= 0]
        # L1_df = L1_df.iloc[
        #     (self.inds_df.loc[0]['Current_rating'] - L1_df['Current_Rating [A]']).argsort()[25:]]
        # L1_df = L1_df.iloc[(L1_df['DCR [Ohms]'] - self.ind_vals[0]['DCR'][0]).abs().argsort()[:20]]
        # L1_df = L1_df.iloc[
        #     (L1_df['Inductance [H]'] - (self.inds_df.loc[0, 'Ind_fsw'] / (self.x_result[2]))).abs().argsort()[:15]]
        # L1_df = L1_df.iloc[
        #     (L1_df['Length [mm]'] * L1_df['Width [mm]'] - self.ind_vals[0]['Area'][0]).abs().argsort()[:10]]
        # L1_df = L1_df.iloc[(L1_df['Unit_price [USD]'] - self.ind_vals[0]['Cost'][0]).abs().argsort()[:5]]

        # Here is where we would brute-force combinations, keep the top ~20 of each closest parameter component, then search over all these
        # by computing the power loss and find the minimum of that

        # compute the power loss of each component, then later sum to find the loss of each combination

        # Using the methods already written out, compute the power loss with the more complex loss model for each
        # component in the subset of the database. This is the approach using all the dataset that includes ALL
        # parameters we care about

        # First, determine the category of each component--
        Q1_df['category'] = Q1_df.apply(lambda x: self.compute_category_bf(x.V_dss), axis=1)
        Q1_df['kT'] = Q1_df.apply(lambda x: self.compute_kT_bf(x.category, x.V_dss), axis=1)
        Q1_df['Power_loss'] = self.I_Q1 ** 2 * Q1_df['kT'] * Q1_df['R_ds'] + \
                              self.x_result[2] * Q1_df['Q_g'] * self.fets_df.loc[0, 'Vgate']

        Q2_df['category'] = Q2_df.apply(lambda x: self.compute_category_bf(x.V_dss), axis=1)
        Q2_df['kT'] = Q2_df.apply(lambda x: self.compute_kT_bf(x.category, x.V_dss), axis=1)
        Q2_df['Cdsq'] = Q2_df.apply(lambda x: self.compute_Cdsq_bf(x.category, x.V_dss, x.C_oss, x.Vds_meas), axis=1)
        Q2_df['Qrr_est'] = Q2_df.apply(lambda x: self.compute_Qrr_est_bf(x.Q_rr, x.I_F), axis=1)
        Q2_df['Power_loss'] = self.I_Q2 ** 2 * Q2_df['kT'] * Q2_df['R_ds'] + \
                              self.x_result[2] * Q2_df['Q_g'] * self.fets_df.loc[1, 'Vgate'] + \
 \
                              self.x_result[2] * Q2_df['Cdsq'] * self.converter_df.loc[
                                  0, 'Vin'] ** 2 \
                              + self.I_off_Q1 ** 2 * self.t_off ** 2 * self.x_result[2] / (
                                      48 * 10 ** -12 * Q2_df['C_ossp1'])

        if self.fet_tech != 'GaNFET':
            Q2_df['Power_loss'] += (self.I_d2_off * self.t_rise / 2) * self.converter_df.loc[0, 'Vin'] * self.x_result[
                2] \
                                   + Q2_df['Qrr_est'] * 10 ** -9 * self.converter_df.loc[0, 'Vin'] * self.x_result[2]

        L1_df['Rac_total'] = L1_df.apply(lambda x: self.compute_Rac_bf(x['DCR [Ohms]'], x['fb [Hz]'], x.b), axis=1)
        L1_df['IGSE_total'] = L1_df.apply(
            lambda x: self.compute_IGSE_bf(x['Inductance [H]'], x['Nturns'], x['Ac [m^2]'], x['Alpha'], x['Beta'],
                                           x['Kfe']), axis=1)
        L1_df['Power_loss'] = self.converter_df.loc[0, 'Iout'] ** 2 * L1_df['DCR [Ohms]'] \
                              + L1_df['Rac_total'] + 10 ** -9 * L1_df['Length [mm]'] * L1_df['Height [mm]'] * L1_df[
                                  'Width [mm]'] * \
                              L1_df['IGSE_total']

        # compute the power loss of each combination by implementing the brute force method

        start = time.time()
        x1, y1, z1 = np.meshgrid(Q1_df['Unit_price'], Q2_df['Unit_price'], L1_df['Unit_price [USD]'],
                                 indexing='ij',
                                 sparse=True)
        cost_arr = x1 + y1 + z1
        x1, y1, z1 = np.meshgrid(Q1_df['Pack_case'], Q2_df['Pack_case'], L1_df['Length [mm]'] * L1_df['Width [mm]'],
                                 indexing='ij', sparse=True)
        area_arr = x1 + y1 + z1
        x1, y1, z1 = np.meshgrid(Q1_df['Power_loss'], Q2_df['Power_loss'], L1_df['Power_loss'], indexing='ij',
                                 sparse=True)
        power_arr = x1 + y1 + z1
        end = time.time()
        # print('method1 = %f' % (end - start))

        # check combination meets constraints
        start = time.time()
        self.set_constraint_val = 200
        if self.opt_var == 'power':
            if self.plotting_var == 'cost':
                self.cost_constraint = self.plotting_val.astype('float')
                self.area_constraint = self.set_constraint_val

        new_area_arr = area_arr < 1.3 * self.area_constraint
        new_cost_arr = cost_arr < 1.3 * self.cost_constraint
        valid_arr = np.logical_and(new_area_arr, new_cost_arr)
        end = time.time()
        # print('method2 = %f' % (end - start))

        # map the true/false values onto the power array
        start = time.time()
        valid_power_arr = np.multiply(power_arr, valid_arr)
        valid_power_arr[valid_power_arr == 0] = np.nan
        try:
            best_idx = np.where(valid_power_arr == np.nanmin(valid_power_arr))
            best_idx = [best_idx[0][0], best_idx[1][0], best_idx[2][0]]
        except:
            best_idx = np.nan
        end = time.time()
        # print('method3 = %f' % (end - start))

        # self.result = [valid_power_arr[best_idx[0]][best_idx[1]][best_idx[2]], (
        # self.df1_chunk['Index'].iloc[best_idx[0]], self.df2['Index'].iloc[best_idx[1]],
        # df3_filt['Index'].iloc[best_idx[2]])]

        # Now write the script for manual selection, using components not scraped from pdfs and selecting the closest based on those pararmeters
        # Use the mosfet_data_wmfrPartNo2 database for this component selection
        from csv_conversion import csv_to_mosfet
        from fet_data_parsing import column_fet_parse, initial_fet_parse

        csv_file = 'csv_files/mosfet_data_wmfrPartNo2.csv'

        fet_df = csv_to_mosfet(csv_file)
        fet_df = fet_df.iloc[:, 1:]
        fet_df.columns = ['Mfr_part_no', 'Unit_price', 'Mfr', 'Series', 'FET_type', 'Technology', 'V_dss', 'I_d',
                          'V_drive', 'R_ds', 'V_thresh', 'Q_g', 'V_gs', 'Input_cap', 'P_diss', 'Op_temp', 'Mount_type',
                          'Supp_pack', 'Pack_case', 'Q_rr']
        fet_df['C_oss'] = 0.0
        attr_list = ['Mfr_part_no', 'Unit_price', 'FET_type', 'Technology', 'V_dss', 'R_ds', 'Q_g',
                     'Pack_case'
                     ]

        fet_df = column_fet_parse(initial_fet_parse(fet_df, attr_list), attr_list)
        fet_df = area_filter(fet_df)
        fet_df = fet_df[fet_df['Technology'] == self.fet_tech]
        fets_database_df = fet_df.drop_duplicates(subset='Mfr_part_no')
        # Now use brute force to determine based off a limited number of parameters those with closest values

        n = 25
        # Q1_df = fets_database_df.iloc[
        #     (fets_database_df['V_dss'] - self.fets_df.loc[0]['Vdss']).abs().argsort()[:(n - 0)]]
        # Q1_df = Q1_df.iloc[(Q1_df['Unit_price'] - self.fet_vals[0]['Cost'][0]).abs().argsort()[:(n - 5)]]
        # Q1_df = Q1_df.iloc[(Q1_df['R_ds'] - self.x_result[0]).abs().argsort()[:(n - 10)]]
        # Q1_df = Q1_df.iloc[(Q1_df['Q_g'] - self.fet_vals[0]['Qg'][0]).abs().argsort()[:(n - 15)]]
        # Q1_df = Q1_df.iloc[(Q1_df['Pack_case'] - self.fet_vals[0]['Area'][0]).abs().argsort()[:(n - 20)]]
        b = 1.2
        b = 5
        Q1_df = fets_database_df[fets_database_df['V_dss'] > self.fets_df.loc[0]['Vdss']]
        Q1_df = Q1_df.loc[(Q1_df['Unit_price'] > 0.5 * self.fet_vals[0]['Cost'][0]) & (Q1_df['Unit_price'] < b *
                                                                                       self.fet_vals[0]['Cost'][0])]
        Q1_df = Q1_df.loc[(Q1_df['R_ds'] > 0.001 * self.x_result[0]) & (Q1_df['R_ds'] < b * self.x_result[0])]
        Q1_df = Q1_df.loc[
            (Q1_df['Q_g'] > 0.001 * self.fet_vals[0]['Qg'][0]) & (Q1_df['Q_g'] < b * self.fet_vals[0]['Qg'][0])]
        Q1_df = Q1_df.loc[(Q1_df['Pack_case'] > 0.001 * self.fet_vals[0]['Area'][0]) & (Q1_df['Pack_case'] < b *
                                                                                        self.fet_vals[0]['Area'][0])]

        # Maybe divide by the max. to normalize values, and then sum, and then choose the top 20 w/ closest?

        # Q2_df = fets_database_df.iloc[(fets_database_df['V_dss'] - self.fets_df.loc[1]['Vdss']).abs().argsort()[:(n-0)]]
        # Q2_df = Q2_df.iloc[(Q2_df['R_ds'] - self.x_result[1]).abs().argsort()[:(n-5)]]
        # Q2_df = Q2_df.iloc[(Q2_df['Q_g'] - self.fet_vals[1]['Qg'][0]).abs().argsort()[:(n-10)]]
        # Q2_df = Q2_df.iloc[(Q2_df['Unit_price'] - self.fet_vals[1]['Cost'][0]).abs().argsort()[:(n-15)]]
        # Q2_df = Q2_df.iloc[(Q2_df['Pack_case'] - self.fet_vals[1]['Area'][0]).abs().argsort()[:(n-20)]]
        Q2_df = fets_database_df[fets_database_df['V_dss'] > self.fets_df.loc[1]['Vdss']]
        Q2_df = Q2_df.loc[(Q2_df['Unit_price'] > 0.5 * self.fet_vals[1]['Cost'][0]) & (
                    Q2_df['Unit_price'] < b * self.fet_vals[1]['Cost'][0])]
        Q2_df = Q2_df.loc[(Q2_df['R_ds'] > 0.001 * self.x_result[1]) & (Q2_df['R_ds'] < b * self.x_result[1])]
        Q2_df = Q2_df.loc[
            (Q2_df['Q_g'] > 0.001 * self.fet_vals[1]['Qg'][0]) & (Q2_df['Q_g'] < b * self.fet_vals[1]['Qg'][0])]
        Q2_df = Q2_df.loc[(Q2_df['Pack_case'] > 0.001 * self.fet_vals[1]['Area'][0]) & (
                    Q2_df['Pack_case'] < b * self.fet_vals[1]['Area'][0])]

        # L1_df = inds_database_df[(self.inds_df.loc[0]['Current_rating'] - inds_database_df['Current_Rating [A]']) <= 0]
        # L1_df = L1_df.iloc[
        #     (self.inds_df.loc[0]['Current_rating'] - L1_df['Current_Rating [A]']).argsort()[25:]]
        # L1_df = L1_df.iloc[(L1_df['DCR [Ohms]'] - self.ind_vals[0]['DCR'][0]).abs().argsort()[:20]]
        # L1_df = L1_df.iloc[
        #     (L1_df['Inductance [H]'] - (self.inds_df.loc[0, 'Ind_fsw'] / (self.x_result[2]))).abs().argsort()[:15]]
        # L1_df = L1_df.iloc[
        #     (L1_df['Length [mm]'] * L1_df['Width [mm]'] - self.ind_vals[0]['Area'][0]).abs().argsort()[:10]]
        # L1_df = L1_df.iloc[(L1_df['Unit_price [USD]'] - self.ind_vals[0]['Cost'][0]).abs().argsort()[:5]]
        L1_df = inds_database_df[(self.inds_df.loc[0]['Current_rating'] < inds_database_df['Current_Rating [A]'])]
        L1_df = L1_df.loc[(L1_df['DCR [Ohms]'] < 1.2 * self.ind_vals[0]['DCR'][0]) & (
                    L1_df['DCR [Ohms]'] > 0.5 * self.ind_vals[0]['DCR'][0])]
        L1_df = L1_df.loc[
            (L1_df['Inductance [H]'] < 1.2 * (self.inds_df.loc[0, 'Ind_fsw'] / (self.x_result[2]))) & (
                        L1_df['Inductance [H]'] > 0.5 * (self.inds_df.loc[0, 'Ind_fsw'] / (self.x_result[2])))]
        L1_df = L1_df.loc[
            (L1_df['Length [mm]'] * L1_df['Width [mm]'] < 1.2 * self.ind_vals[0]['Area'][0]) & (
                        L1_df['Length [mm]'] * L1_df['Width [mm]'] > 0.5 * self.ind_vals[0]['Area'][0])]
        L1_df = L1_df.loc[(L1_df['Unit_price [USD]'] < 1.2 * self.ind_vals[0]['Cost'][0]) & (
                    L1_df['Unit_price [USD]'] > 0.5 * self.ind_vals[0]['Cost'][0])]

        # Here is where we would brute-force combinations, keep the top ~20 of each closest parameter component, then search over all these
        # by computing the power loss and find the minimum of that

        # compute the power loss of each component, then later sum to find the loss of each combination

        # Using the methods already written out, compute the power loss with the more complex loss model for each
        # component in the subset of the database. This is the approach using all the dataset that includes ALL
        # parameters we care about

        # First, determine the category of each component--
        Q1_df['category'] = Q1_df.apply(lambda x: self.compute_category_bf(x.V_dss), axis=1)
        Q1_df['kT'] = Q1_df.apply(lambda x: self.compute_kT_bf(x.category, x.V_dss), axis=1)
        Q1_df['Power_loss'] = self.I_Q1 ** 2 * Q1_df['kT'] * Q1_df['R_ds'] + \
                              self.x_result[2] * Q1_df['Q_g'] * self.fets_df.loc[0, 'Vgate']

        Q2_df['category'] = Q2_df.apply(lambda x: self.compute_category_bf(x.V_dss), axis=1)
        Q2_df['kT'] = Q2_df.apply(lambda x: self.compute_kT_bf(x.category, x.V_dss), axis=1)
        # Q2_df['Cdsq'] = Q2_df.apply(lambda x: self.compute_Cdsq_bf(x.category, x.V_dss, x.C_oss, x.Vds_meas), axis=1)
        # Q2_df['Qrr_est'] = Q2_df.apply(lambda x: self.compute_Qrr_est_bf(x.Q_rr, x.I_F), axis=1)
        Q2_df['Power_loss'] = self.I_Q2 ** 2 * Q2_df['kT'] * Q2_df['R_ds'] + \
                              self.x_result[2] * Q2_df['Q_g'] * self.fets_df.loc[1, 'Vgate']

        # self.x_result[2] * Q2_df['Cdsq'] * self.converter_df.loc[
        #     0, 'Vin'] ** 2 \
        # + self.I_off_Q1 ** 2 * self.t_off ** 2 * self.x_result[2] / (
        #         48 * 10 ** -12 * Q2_df['C_ossp1'])

        if self.fet_tech != 'GaNFET':
            Q2_df['Power_loss'] += (self.I_d2_off * self.t_rise / 2) * self.converter_df.loc[0, 'Vin'] * self.x_result[
                2] \
                # + Q2_df['Qrr_est'] * 10 ** -9 * self.converter_df.loc[0, 'Vin'] * self.x_result[2]

        L1_df['Rac_total'] = L1_df.apply(lambda x: self.compute_Rac_bf(x['DCR [Ohms]'], x['fb [Hz]'], x.b), axis=1)
        L1_df['IGSE_total'] = L1_df.apply(
            lambda x: self.compute_IGSE_bf(x['Inductance [H]'], x['Nturns'], x['Ac [m^2]'], x['Alpha'], x['Beta'],
                                           x['Kfe']), axis=1)
        L1_df['Power_loss'] = self.converter_df.loc[0, 'Iout'] ** 2 * L1_df['DCR [Ohms]'] \
                              + L1_df['Rac_total'] + 10 ** -9 * L1_df['Length [mm]'] * L1_df['Height [mm]'] * L1_df[
                                  'Width [mm]'] * \
                              L1_df['IGSE_total']

        # compute the power loss of each combination by implementing the brute force method

        start = time.time()
        x1, y1, z1 = np.meshgrid(Q1_df['Unit_price'], Q2_df['Unit_price'], L1_df['Unit_price [USD]'],
                                 indexing='ij',
                                 sparse=True)
        cost_arr = x1 + y1 + z1
        x1, y1, z1 = np.meshgrid(Q1_df['Pack_case'], Q2_df['Pack_case'], L1_df['Length [mm]'] * L1_df['Width [mm]'],
                                 indexing='ij', sparse=True)
        area_arr = x1 + y1 + z1
        x1, y1, z1 = np.meshgrid(Q1_df['Power_loss'], Q2_df['Power_loss'], L1_df['Power_loss'], indexing='ij',
                                 sparse=True)
        power_arr = x1 + y1 + z1
        end = time.time()
        # print('method1 = %f' % (end - start))

        # check combination meets constraints
        start = time.time()
        self.set_constraint_val = 200
        if self.opt_var == 'power':
            if self.plotting_var == 'cost':
                self.cost_constraint = self.plotting_val.astype('float')
                self.area_constraint = self.set_constraint_val

        new_area_arr = area_arr < 1.3 * self.area_constraint
        new_cost_arr = cost_arr < 1.3 * self.cost_constraint
        valid_arr = np.logical_and(new_area_arr, new_cost_arr)
        end = time.time()
        # print('method2 = %f' % (end - start))

        # map the true/false values onto the power array
        start = time.time()
        valid_power_arr = np.multiply(power_arr, valid_arr)
        valid_power_arr[valid_power_arr == 0] = np.nan
        try:
            best_idx = np.where(valid_power_arr == np.nanmin(valid_power_arr))
            best_idx = [best_idx[0][0], best_idx[1][0], best_idx[2][0]]
        except:
            best_idx = np.nan
        end = time.time()

        power = 0

        self.I_Q1 = np.sqrt(self.dc) * self.converter_df.loc[0, 'Iout']
        self.I_Q2 = np.sqrt(1 - self.dc) * self.converter_df.loc[0, 'Iout']

        # Now implement the brute force method
        combo_df = pd.DataFrame()
        combo_df['Power_tot'] = Q1_df['Power_loss'] + Q2_df['Power_loss'] + L1_df['Power_loss']

        # print(x)
        # print(self.power_tot)
        if self.fet_tech == 'MOSFET':
            self.power_divider = 10 # used to be 10
        if self.fet_tech == 'GaNFET':
            self.power_divider = 100

        if self.power_tot < self.power_tot_prev + .0001 and self.power_tot > self.power_tot_prev - .0001:
            self.cont_counter += 1
        else:
            self.cont_counter = 0
        self.power_tot_prev = self.power_tot
        var = self.power_tot / self.Pout
        # print('power_tot: %f' % var)
        return self.power_tot / self.power_divider

        # reset the units for better display purposes
        Q1_dup = Q1_df.copy(deep=True)
        Q2_dup = Q2_df.copy(deep=True)
        L1_dup = L1_df.copy(deep=True)

        Q1_dup['R_ds'] = Q1_dup['R_ds'] * 10 ** 3
        Q2_dup['R_ds'] = Q2_dup['R_ds'] * 10 ** 3
        Q1_dup['Q_g'] = Q1_dup['Q_g'] * 10 ** 9
        Q2_dup['Q_g'] = Q2_dup['Q_g'] * 10 ** 9
        Q1_dup['Q_rr'] = Q1_dup['Q_rr'] * 10 ** 9
        Q2_dup['Q_rr'] = Q2_dup['Q_rr'] * 10 ** 9
        Q1_dup['C_oss'] = Q1_dup['C_oss'] * 10 ** 12
        Q2_dup['C_oss'] = Q2_dup['C_oss'] * 10 ** 12
        L1_dup['R_dc'] = L1_dup['R_dc'] * 10 ** 3
        L1_dup['Inductance'] = L1_dup['Inductance'] * 10 ** 6

        # if fet_technology == 'GaNFET':
        #     Q1_dup = area_filter(Q1_dup)
        #     Q2_dup = area_filter(Q2_dup)
        # then the user chooses which components they want to go with, and this tells them what the expected loss is given
        # these choices. For Qrr, if not present, give the expected Qrr in the sum term.
        # also, find some way to put all this information onto a nice info-graphic from within the program

        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 1000)
        pd.set_option('display.colheader_justify', 'center')
        pd.set_option('display.precision', 3)
        fet_column_list = ['V_dss [V]', 'Unit_price [$]', 'R_ds [mΩ]', 'Q_g [nC]', 'FET_type', 'Technology',
                           'Area [mm^2]', 'Mfr_part_no', 'Qrr [nC]', 'Coss [pF]']
        ind_column_list = ['Unit_price [$]', 'Current_rating [A]', 'R_dc [mΩ]', 'Area [mm^2]', 'Height [mm]',
                           'Inductance [µH]', 'Mfr_part_no']
        Q1_dup = Q1_dup.drop_duplicates(inplace=False)
        Q2_dup = Q2_dup.drop_duplicates(inplace=False)
        L1_dup = L1_dup.drop_duplicates(inplace=False)

        print('Available Q1 components:')
        print(tabulate(Q1_dup, headers=fet_column_list, showindex=False,
                       tablefmt='fancy_grid', floatfmt=".2f"))
        print('Available Q2 components:')
        print(tabulate(Q2_dup, headers=fet_column_list, showindex=False,
                       tablefmt='fancy_grid', floatfmt=".2f"))
        print('Available L1 components:')
        print(tabulate(L1_dup, headers=ind_column_list, showindex=False,
                       tablefmt='fancy_grid', floatfmt=".2f"))

        [user_Q1_index, user_Q2_index, user_L1_index] = user_COTS_choice(Q1_df, Q2_df, L1_df)
        user_prefs = {'Q1': int(user_Q1_index), 'Q2': int(user_Q2_index), 'L1': int(user_L1_index)}
        COTS_total_cost = Q1_df.iloc[user_prefs['Q1']]['Unit_price'] + Q2_df.iloc[user_prefs['Q2']]['Unit_price'] + \
                          L1_df.iloc[user_prefs['L1']]['Unit_price']
        # to get area of FETs need to convert the package sizes to actual areas using the dictionary
        # if fet_technology == 'GaNFET':
        #     Q1_df = area_filter(Q1_df)
        #     Q2_df = area_filter(Q2_df)
        COTS_total_area = Q1_df.iloc[user_prefs['Q1']]['Pack_case'] + Q2_df.iloc[user_prefs['Q2']]['Pack_case'] + \
                          L1_df.iloc[user_prefs['L1']]['Dimension']
        COTS_total_powerfrac = (self.I_Q1 ** 2 * Q1_df.iloc[user_prefs['Q1']]['R_ds'] + self.x_result[2] *
                                Q1_df.iloc[user_prefs['Q1']][
                                    'Q_g'] * self.fets_df.loc[0, 'Vgate'] + \
                                self.I_Q2 ** 2 * Q2_df.iloc[user_prefs['Q2']]['R_ds'] + self.x_result[2] *
                                Q2_df.iloc[user_prefs['Q2']][
                                    'Q_g'] * self.fets_df.loc[1, 'Vgate'] + self.x_result[2] * self.fet_vals[1]['Qrr'][
                                    0] * self.converter_df.loc[0, 'Vin'] +
                                (1 / 2) * self.x_result[2] * self.fet_vals[0]['Coss'][0] *
                                self.converter_df.loc[0, 'Vin'] ** 2 +
                                (1 / 2) * self.x_result[2] * self.fet_vals[1]['Coss'][0] *
                                self.converter_df.loc[0, 'Vin'] ** 2 +
                                self.converter_df.loc[0, 'Iout'] ** 2 * L1_df.iloc[user_prefs['L1']]['R_dc']) / (
                                       self.converter_df.loc[0, 'Vout'] * self.converter_df.loc[0, 'Iout'])
        # print('Using the parameters of the chosen COTS components:\nTotal cost = %f, Total area = %f, Total power loss fraction = %f\n' % (COTS_total_cost, COTS_total_area, COTS_total_powerfrac))
        COTS_performance_df = pd.DataFrame.from_dict({'Total cost': [COTS_total_cost], 'Total area': [COTS_total_area],
                                                      'Total Ploss/Pout': [COTS_total_powerfrac]})
        performance_column_list = ['Total cost [$]', 'Total area [mm^2]', 'Total Ploss/Pout']
        print('Expected performance given these component choices:')
        print(
            tabulate(COTS_performance_df.drop_duplicates(inplace=False), headers=performance_column_list,
                     showindex=False, tablefmt='fancy_grid',
                     floatfmt=".3f"))

        print('done')

    def make_component_predictions(self):
        pd.set_option("display.max_rows", None, "display.max_columns", None)

        print('Total cost: %f, Total area: %f, Total power loss/Pout: %f\n' % (
            self.Cost_tot, self.Area_tot, self.fun / self.Pout
        ))
        self.x_result[0] = self.x_result[0] * self.fet_normalizer * 10 ** 3
        self.x_result[1] = self.x_result[1] * self.fet_normalizer * 10 ** 3
        self.x_result[2] = self.x_result[2] * self.ind_normalizer
        # self.power_tot = self.power_tot*self.power_normalizer

        Q1_rec_df = pd.DataFrame.from_dict(
            {'Vdss': [self.fets_df.loc[0, 'Vdss']], 'Rds': [self.x_result[0]],
             'Qg': [self.fet_vals[0]['Qg'][0]],
             'Coss': [self.fet_vals[0]['Coss'][0]],
             'Qrr': [self.fet_vals[1]['Qrr'][0]],
             'Cost': [self.fet_vals[0]['Cost'][0]],
             'Area': [
                 self.fet_vals[0]['Area'][0]]})  # , columns=pd.MultiIndex.from_product([['Q1']], names=['Component:']))

        Q1_rec_df.style.set_caption("Q1 Recommended Parameters")
        Q2_rec_df = pd.DataFrame({'Vdss': [self.fets_df.loc[1, 'Vdss']], 'Rds': [self.x_result[1]],
                                  'Qg': [self.fet_vals[1]['Qg'][0]],
                                  'Coss': [self.fet_vals[1]['Coss'][0]],
                                  'Qrr': [self.fet_vals[1]['Qrr'][0]],
                                  'Cost': [self.fet_vals[1]['Cost'][0]],
                                  'Area': [self.fet_vals[1]['Area'][0]]})
        Q2_rec_df.style.set_caption("Q2 Recommended Parameters")
        L1_rec_df = pd.DataFrame(
            {'Inductance': [self.inds_df.loc[0, 'Ind_fsw'] / (self.x_result[2])],
             'Current_rating': [self.inds_df.loc[0, 'Current_rating']],
             'Rdc': [self.ind_vals[0]['DCR'][0] * 10 ** 3],
             'Cost': [self.ind_vals[0]['Cost'][0]], 'Area': [self.ind_vals[0]['Area'][0]]})
        L1_rec_df.style.set_caption("L1 Recommended Parameters")
        converter_rec_df = pd.DataFrame.from_dict(
            {'frequency': [self.x_result[2]], 'Total cost': [self.cost_tot], 'Total area': [self.area_tot],
             'Total Ploss/Pout': [self.power_tot / self.Pout]})

        # set the variables so they are in the units we want
        Q1_rec_df['Rds'] = Q1_rec_df['Rds']
        Q2_rec_df['Rds'] = Q2_rec_df['Rds']
        Q1_rec_df['Qg'] = Q1_rec_df['Qg'] * 10 ** 9
        Q2_rec_df['Qg'] = Q2_rec_df['Qg'] * 10 ** 9
        Q1_rec_df['Qrr'] = Q1_rec_df['Qrr'] * 10 ** 9
        Q2_rec_df['Qrr'] = Q2_rec_df['Qrr'] * 10 ** 9
        Q1_rec_df['Coss'] = Q1_rec_df['Coss'] * 10 ** 12
        Q2_rec_df['Coss'] = Q2_rec_df['Coss'] * 10 ** 12
        L1_rec_df['Rdc'] = L1_rec_df['Rdc']
        L1_rec_df['Inductance'] = L1_rec_df['Inductance'] * self.ind_normalizer

        fet_column_list = ['V_dss [V]', 'R_ds [mΩ]', 'Q_g [nC]', 'C_oss [pF]', 'Q_rr [nC]', 'Unit_price [$]',
                           'Area [mm^2]']
        ind_column_list = ['Inductance [µH]', 'Current_rating [A]', 'R_dc [mΩ]', 'Unit_price [$]', 'Area [mm^2]'
                           ]
        performance_column_list = ['frequency [Hz]', 'Total cost [$]', 'Total area [mm^2]', 'Total Ploss/Pout']
        print('Q1 recommended parameters:')
        print(tabulate(Q1_rec_df.drop_duplicates(inplace=False), headers=fet_column_list, showindex=False,
                       tablefmt='fancy_grid',
                       floatfmt=".1f"))
        print('Q2 recommended parameters:')
        print(tabulate(Q2_rec_df.drop_duplicates(inplace=False), headers=fet_column_list, showindex=False,
                       tablefmt='fancy_grid',
                       floatfmt=".1f"))
        print('L1 recommended parameters:')
        print(tabulate(L1_rec_df.drop_duplicates(inplace=False), headers=ind_column_list, showindex=False,
                       tablefmt='fancy_grid',
                       floatfmt=".1f"))
        print('Converter expected performance using recommended parameters:')
        print(tabulate(converter_rec_df, headers=performance_column_list, showindex=False,
                       tablefmt='fancy_grid',
                       floatfmt=".3f"))

        # now search the database for parts with similar parameters

        # csv_file = '../mosfet_data/csv_files/' + str(self.fet_tech) + '_data_BalancedOpt.csv'
        csv_file = '../mosfet_data/csv_files/' + str(self.fet_tech) + '_data_BalancedOpt.csv'
        # csv_file = '../mosfet_data/csv_files/FET_pdf_tables_wVds_full2.csv'
        #
        #
        # # fets_database_df = pd.read_csv(csv_file)
        # # fets_database_df = fets_database_df.reset_index()
        # # data_dims = ['V_dss', 'Unit_price', 'R_ds', 'Q_g', 'FET_type', 'Technology', 'Pack_case', 'Mfr_part_no', 'Q_rr',
        # #              'C_oss']
        # # fets_database_df = fets_database_df.iloc[:, 2:]
        # # fets_database_df.columns = data_dims
        #
        # fets_database_df = pd.read_csv(csv_file)
        #
        # fets_database_df = fets_database_df.iloc[:, 1:]
        # fets_database_df.columns = ['Mfr_part_no', 'Unit_price', 'Mfr', 'Series', 'FET_type', 'Technology', 'V_dss', 'I_d',
        #                   'V_drive',
        #                   'R_ds', 'V_thresh', 'Q_g', 'V_gs', 'Input_cap', 'P_diss', 'Op_temp', 'Mount_type',
        #                   'Supp_pack',
        #                   'Pack_case', 'Q_rr', 'C_oss', 'I_F', 'Vds_meas']
        with open('fet_df_test_values', 'rb') as optimizer_obj_file:
            # Step 3
            fets_database_df = pickle.load(optimizer_obj_file)

        # drop ones w/ NaN or 0
        attr_list = ['Mfr_part_no', 'Unit_price', 'FET_type', 'Technology', 'V_dss', 'R_ds', 'Q_g', 'C_oss', 'Vds_meas',
                     'Q_rr', 'I_F']
        fets_database_df = fets_database_df.replace(0, np.nan)
        fets_database_df = fets_database_df.dropna(subset=attr_list)

        # csv_file = 'csv_files/inductor_data_BalancedOpt.csv'
        # inds_database_df = pd.read_csv(csv_file)
        # inds_database_df = inds_database_df.reset_index()
        # inds_database_df = inds_database_df.iloc[:, 2:]
        # attr_list = ['Unit_price', 'Current_rating', 'R_dc', 'Dimension', 'Height', 'Inductance', 'Mfr_part_no', 'Sat_current', 'Energy', 'Volume']
        # inds_database_df.columns = attr_list
        xlsx_file = 'csv_files/inductor_training_updatedAlgorithm.csv'
        inds_database_df = pd.read_csv(xlsx_file)
        # data_dims = ['Mfr_part_no', 'Unit_price [USD]', 'Current_Rating [A]', 'DCR [Ohms]', 'Volume [mm^3]',
        #              'Inductance [H]',
        #              'Current_Sat [A]', 'fb [Hz]', 'b', 'Kfe', 'Alpha', 'Beta', 'Nturns', 'Ac [m^2]']
        # inds_database_df.columns = data_dims

        # filter the fet df to find a component with the following parameters we care about (within an x% range):
        #   Vdss, cost, Rds, Qg
        # for Q1:
        Q1_df = fets_database_df.iloc[(fets_database_df['V_dss'] - self.fets_df.loc[0]['Vdss']).abs().argsort()[:35]]
        Q1_df = Q1_df.iloc[(Q1_df['R_ds'] - self.x_result[0]).abs().argsort()[:30]]
        Q1_df = Q1_df.iloc[(Q1_df['Q_g'] - self.fet_vals[0]['Qg'][0]).abs().argsort()[:25]]
        Q1_df = Q1_df.iloc[(Q1_df['C_oss'] - self.fet_vals[0]['Coss'][0]).abs().argsort()[:20]]
        Q1_df = Q1_df.iloc[(Q1_df['Q_rr'] - self.fet_vals[0]['Qrr'][0]).abs().argsort()[:15]]
        Q1_df = Q1_df.iloc[(Q1_df['Unit_price'] - self.fet_vals[0]['Cost'][0]).abs().argsort()[:10]]
        Q1_df = Q1_df.iloc[(Q1_df['Pack_case'] - self.fet_vals[0]['Area'][0]).abs().argsort()[:5]]

        # Maybe divide by the max. to normalize values, and then sum, and then choose the top 20 w/ closest?

        Q2_df = fets_database_df.iloc[(fets_database_df['V_dss'] - self.fets_df.loc[1]['Vdss']).abs().argsort()[:35]]
        Q2_df = Q2_df.iloc[(Q2_df['R_ds'] - self.x_result[1]).abs().argsort()[:30]]
        Q2_df = Q2_df.iloc[(Q2_df['Q_g'] - self.fet_vals[1]['Qg'][0]).abs().argsort()[:25]]
        Q2_df = Q2_df.iloc[(Q2_df['C_oss'] - self.fet_vals[1]['Coss'][0]).abs().argsort()[:20]]
        Q2_df = Q2_df.iloc[(Q2_df['Q_rr'] - self.fet_vals[1]['Qrr'][0]).abs().argsort()[:15]]
        Q2_df = Q2_df.iloc[(Q2_df['Unit_price'] - self.fet_vals[1]['Cost'][0]).abs().argsort()[:10]]
        Q2_df = Q2_df.iloc[(Q2_df['Pack_case'] - self.fet_vals[1]['Area'][0]).abs().argsort()[:5]]

        L1_df = inds_database_df[(self.inds_df.loc[0]['Current_rating'] - inds_database_df['Current_Rating [A]']) <= 0]
        L1_df = L1_df.iloc[
            (self.inds_df.loc[0]['Current_rating'] - L1_df['Current_Rating [A]']).argsort()[25:]]
        L1_df = L1_df.iloc[(L1_df['DCR [Ohms]'] - self.ind_vals[0]['DCR'][0]).abs().argsort()[:20]]
        L1_df = L1_df.iloc[
            (L1_df['Inductance [H]'] - (self.inds_df.loc[0, 'Ind_fsw'] / (self.x_result[2]))).abs().argsort()[:15]]
        L1_df = L1_df.iloc[
            (L1_df['Length [mm]'] * L1_df['Width [mm]'] - self.ind_vals[0]['Area'][0]).abs().argsort()[:10]]
        L1_df = L1_df.iloc[(L1_df['Unit_price [USD]'] - self.ind_vals[0]['Cost'][0]).abs().argsort()[:5]]

        # Here is where we would brute-force combinations, keep the top ~20 of each closest parameter component, then search over all these
        # by computing the power loss and find the minimum of that

        # compute the power loss of each component, then later sum to find the loss of each combination

        # Using the methods already written out, compute the power loss with the more complex loss model for each
        # component in the subset of the database. This is the approach using all the dataset that includes ALL
        # parameters we care about

        # First, determine the category of each component--
        Q1_df['category'] = Q1_df.apply(lambda x: self.compute_category_bf(x.V_dss), axis=1)
        Q1_df['kT'] = Q1_df.apply(lambda x: self.compute_kT_bf(x.category, x.V_dss), axis=1)
        Q1_df['Power_loss'] = self.I_Q1 ** 2 * Q1_df['kT'] * Q1_df['R_ds'] + \
                              self.x_result[2] * Q1_df['Q_g'] * self.fets_df.loc[0, 'Vgate']

        Q2_df['category'] = Q2_df.apply(lambda x: self.compute_category_bf(x.V_dss), axis=1)
        Q2_df['kT'] = Q2_df.apply(lambda x: self.compute_kT_bf(x.category, x.V_dss), axis=1)
        Q2_df['Cdsq'] = Q2_df.apply(lambda x: self.compute_Cdsq_bf(x.category, x.V_dss, x.C_oss, x.Vds_meas), axis=1)
        Q2_df['Qrr_est'] = Q2_df.apply(lambda x: self.compute_Qrr_est_bf(x.Q_rr, x.I_F), axis=1)
        Q2_df['Power_loss'] = self.I_Q2 ** 2 * Q2_df['kT'] * Q2_df['R_ds'] + \
                              self.x_result[2] * Q2_df['Q_g'] * self.fets_df.loc[1, 'Vgate'] + \
 \
                              self.x_result[2] * Q2_df['Cdsq'] * self.converter_df.loc[
                                  0, 'Vin'] ** 2 \
                              + self.I_off_Q1 ** 2 * self.t_off ** 2 * self.x_result[2] / (
                                          48 * 10 ** -12 * Q2_df['C_ossp1'])

        if self.fet_tech != 'GaNFET':
            Q2_df['Power_loss'] += (self.I_d2_off * self.t_rise / 2) * self.converter_df.loc[0, 'Vin'] * self.x_result[
                2] \
                                   + Q2_df['Qrr_est'] * 10 ** -9 * self.converter_df.loc[0, 'Vin'] * self.x_result[2]

        L1_df['Rac_total'] = L1_df.apply(lambda x: self.compute_Rac_bf(x['DCR [Ohms]'], x['fb [Hz]'], x.b), axis=1)
        L1_df['IGSE_total'] = L1_df.apply(
            lambda x: self.compute_IGSE_bf(x['Inductance [H]'], x['Nturns'], x['Ac [m^2]'], x['Alpha'], x['Beta'],
                                           x['Kfe']), axis=1)
        L1_df['Power_loss'] = self.converter_df.loc[0, 'Iout'] ** 2 * L1_df['DCR [Ohms]'] \
                              + L1_df['Rac_total'] + 10 ** -3 * L1_df['Length [mm]'] * 10 ** -3 * L1_df['Width [mm]'] * \
                              L1_df['IGSE_total']

        # compute the power loss of each combination

        power = 0

        self.I_Q1 = np.sqrt(self.dc) * self.converter_df.loc[0, 'Iout']
        self.I_Q2 = np.sqrt(1 - self.dc) * self.converter_df.loc[0, 'Iout']

        ###### OLD LOSS MODEL: ###########
        # self.Q1_loss = self.I_Q1 ** 2 * self.fet_normalizer * self.x_result[0] + \
        #                self.ind_normalizer * self.x_result[2] * self.fet_vals[0]['Qg'][0] * self.fets_df.loc[0, 'Vgate'] \
        #                + .5 * self.ind_normalizer * self.x_result[2] * self.fet_vals[0]['Coss'][0] * self.converter_df.loc[
        #                    0, 'Vin'] ** 2
        # self.Q2_loss = self.I_Q2 ** 2 * self.fet_normalizer * self.x_result[1] + \
        #                self.ind_normalizer * self.x_result[2] * self.fet_vals[1]['Qg'][0] * self.fets_df.loc[1, 'Vgate'] \
        #                + self.ind_normalizer * self.x_result[2] * self.fet_vals[1]['Qrr'][0] * self.converter_df.loc[0, 'Vin'] + \
        #                .5 * self.ind_normalizer * self.x_result[2] * self.fet_vals[1]['Coss'][0] * self.converter_df.loc[
        #                    0, 'Vin'] ** 2
        # self.L1_loss = self.converter_df.loc[0, 'Iout'] ** 2 * self.ind_vals[0]['DCR'][0]

        ####### NEW LOSS MODEL: ###########
        self.Q1_loss = self.I_Q1 ** 2 * Q1_df['kT'] * self.fet_normalizer * self.x_result[0] + \
                       self.ind_normalizer * self.x_result[2] * self.fet_vals[0]['Qg'][0] * self.fets_df.loc[0, 'Vgate']
        self.Q2_loss = self.I_Q2 ** 2 * self.fet_vals[1]['kT'][0] * self.fet_normalizer * self.x_result[1] + \
                       self.ind_normalizer * self.x_result[2] * self.fet_vals[1]['Qg'][0] * self.fets_df.loc[
                           1, 'Vgate'] + \
 \
                       self.ind_normalizer * self.x_result[2] * self.fet_vals[1]['Cdsq'][0] * self.converter_df.loc[
                           0, 'Vin'] ** 2 \
                       + self.I_off_Q1 ** 2 * self.t_off ** 2 * self.ind_normalizer * self.x_result[2] / (
                                   48 * self.C_ossp1)
        if self.fet_tech != 'GaNFET':
            self.Q2_loss += (self.I_d2_off * self.t_rise / 2) * self.converter_df.loc[0, 'Vin'] * self.ind_normalizer * \
                            self.x_result[2] \
                            + self.Qrr_est * self.converter_df.loc[0, 'Vin'] * self.ind_normalizer * self.x_result[2]

        self.L1_loss = self.converter_df.loc[0, 'Iout'] ** 2 * self.ind_vals[0]['DCR'][0] \
                       + self.Rac_total + self.Volume * self.IGSE_total

        self.power_tot = self.Q1_loss + self.Q2_loss + self.L1_loss

        # print(x)
        # print(self.power_tot)
        if self.fet_tech == 'MOSFET':
            self.power_divider = 10
        if self.fet_tech == 'GaNFET':
            self.power_divider = 100

        if self.power_tot < self.power_tot_prev + .0001 and self.power_tot > self.power_tot_prev - .0001:
            self.cont_counter += 1
        else:
            self.cont_counter = 0
        self.power_tot_prev = self.power_tot
        var = self.power_tot / self.Pout
        # print('power_tot: %f' % var)
        return self.power_tot / self.power_divider

        # reset the units for better display purposes
        Q1_dup = Q1_df.copy(deep=True)
        Q2_dup = Q2_df.copy(deep=True)
        L1_dup = L1_df.copy(deep=True)

        Q1_dup['R_ds'] = Q1_dup['R_ds'] * 10 ** 3
        Q2_dup['R_ds'] = Q2_dup['R_ds'] * 10 ** 3
        Q1_dup['Q_g'] = Q1_dup['Q_g'] * 10 ** 9
        Q2_dup['Q_g'] = Q2_dup['Q_g'] * 10 ** 9
        Q1_dup['Q_rr'] = Q1_dup['Q_rr'] * 10 ** 9
        Q2_dup['Q_rr'] = Q2_dup['Q_rr'] * 10 ** 9
        Q1_dup['C_oss'] = Q1_dup['C_oss'] * 10 ** 12
        Q2_dup['C_oss'] = Q2_dup['C_oss'] * 10 ** 12
        L1_dup['R_dc'] = L1_dup['R_dc'] * 10 ** 3
        L1_dup['Inductance'] = L1_dup['Inductance'] * 10 ** 6

        # if fet_technology == 'GaNFET':
        #     Q1_dup = area_filter(Q1_dup)
        #     Q2_dup = area_filter(Q2_dup)
        # then the user chooses which components they want to go with, and this tells them what the expected loss is given
        # these choices. For Qrr, if not present, give the expected Qrr in the sum term.
        # also, find some way to put all this information onto a nice info-graphic from within the program

        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 1000)
        pd.set_option('display.colheader_justify', 'center')
        pd.set_option('display.precision', 3)
        fet_column_list = ['V_dss [V]', 'Unit_price [$]', 'R_ds [mΩ]', 'Q_g [nC]', 'FET_type', 'Technology',
                           'Area [mm^2]', 'Mfr_part_no', 'Qrr [nC]', 'Coss [pF]']
        ind_column_list = ['Unit_price [$]', 'Current_rating [A]', 'R_dc [mΩ]', 'Area [mm^2]', 'Height [mm]',
                           'Inductance [µH]', 'Mfr_part_no']
        Q1_dup = Q1_dup.drop_duplicates(inplace=False)
        Q2_dup = Q2_dup.drop_duplicates(inplace=False)
        L1_dup = L1_dup.drop_duplicates(inplace=False)

        print('Available Q1 components:')
        print(tabulate(Q1_dup, headers=fet_column_list, showindex=False,
                       tablefmt='fancy_grid', floatfmt=".2f"))
        print('Available Q2 components:')
        print(tabulate(Q2_dup, headers=fet_column_list, showindex=False,
                       tablefmt='fancy_grid', floatfmt=".2f"))
        print('Available L1 components:')
        print(tabulate(L1_dup, headers=ind_column_list, showindex=False,
                       tablefmt='fancy_grid', floatfmt=".2f"))

        [user_Q1_index, user_Q2_index, user_L1_index] = user_COTS_choice(Q1_df, Q2_df, L1_df)
        user_prefs = {'Q1': int(user_Q1_index), 'Q2': int(user_Q2_index), 'L1': int(user_L1_index)}
        COTS_total_cost = Q1_df.iloc[user_prefs['Q1']]['Unit_price'] + Q2_df.iloc[user_prefs['Q2']]['Unit_price'] + \
                          L1_df.iloc[user_prefs['L1']]['Unit_price']
        # to get area of FETs need to convert the package sizes to actual areas using the dictionary
        # if fet_technology == 'GaNFET':
        #     Q1_df = area_filter(Q1_df)
        #     Q2_df = area_filter(Q2_df)
        COTS_total_area = Q1_df.iloc[user_prefs['Q1']]['Pack_case'] + Q2_df.iloc[user_prefs['Q2']]['Pack_case'] + \
                          L1_df.iloc[user_prefs['L1']]['Dimension']
        COTS_total_powerfrac = (self.I_Q1 ** 2 * Q1_df.iloc[user_prefs['Q1']]['R_ds'] + self.x_result[2] *
                                Q1_df.iloc[user_prefs['Q1']][
                                    'Q_g'] * self.fets_df.loc[0, 'Vgate'] + \
                                self.I_Q2 ** 2 * Q2_df.iloc[user_prefs['Q2']]['R_ds'] + self.x_result[2] *
                                Q2_df.iloc[user_prefs['Q2']][
                                    'Q_g'] * self.fets_df.loc[1, 'Vgate'] + self.x_result[2] * self.fet_vals[1]['Qrr'][
                                    0] * self.converter_df.loc[0, 'Vin'] +
                                (1 / 2) * self.x_result[2] * self.fet_vals[0]['Coss'][0] *
                                self.converter_df.loc[0, 'Vin'] ** 2 +
                                (1 / 2) * self.x_result[2] * self.fet_vals[1]['Coss'][0] *
                                self.converter_df.loc[0, 'Vin'] ** 2 +
                                self.converter_df.loc[0, 'Iout'] ** 2 * L1_df.iloc[user_prefs['L1']]['R_dc']) / (
                                       self.converter_df.loc[0, 'Vout'] * self.converter_df.loc[0, 'Iout'])
        # print('Using the parameters of the chosen COTS components:\nTotal cost = %f, Total area = %f, Total power loss fraction = %f\n' % (COTS_total_cost, COTS_total_area, COTS_total_powerfrac))
        COTS_performance_df = pd.DataFrame.from_dict({'Total cost': [COTS_total_cost], 'Total area': [COTS_total_area],
                                                      'Total Ploss/Pout': [COTS_total_powerfrac]})
        performance_column_list = ['Total cost [$]', 'Total area [mm^2]', 'Total Ploss/Pout']
        print('Expected performance given these component choices:')
        print(
            tabulate(COTS_performance_df.drop_duplicates(inplace=False), headers=performance_column_list,
                     showindex=False, tablefmt='fancy_grid',
                     floatfmt=".3f"))

        print('done')

    def compute_kT_bf(self, category, V_dss):
        kT_eqn_dict = {'Si_low_volt': [0.233, 1.15], 'Si_high_volt': [0.353, 0.827],
                       'GaN_low_volt': [0.066, 1.34],
                       'GaN_high_volt': [0.148, 1.145], 'SiC': [0.572, -0.323]}

        kT = kT_eqn_dict[category][0] * np.log10(V_dss) + kT_eqn_dict[category][1]
        return kT

    def compute_category_bf(self, V_dss):
        if self.fet_tech == 'MOSFET' and V_dss <= 200:
            category = 'Si_low_volt'
        elif self.fet_tech == 'MOSFET' and V_dss > 200:
            category = 'Si_high_volt'
        elif self.fet_tech == 'GaNFET' and V_dss <= 100:
            category = 'GaN_low_volt'
        elif self.fet_tech == 'GaNFET' and V_dss <= 200:
            category = 'GaN_high_volt'
        elif self.fet_tech == 'SiCFET':
            category = 'SiC'
        return category

    def compute_Cdsq_bf(self, category, V_dss, C_oss, Vds_meas):
        gamma_eqn_dict = {'Si_low_volt': [-0.0021, -0.251], 'Si_high_volt': [-0.000569, -0.579],
                          'GaN_low_volt': [-0.00062, -0.355],
                          'GaN_high_volt': [-0.000394, -0.353], 'SiC': [0, -0.4509]}
        self.gamma = gamma_eqn_dict[category][0] * V_dss + gamma_eqn_dict[category][1]

        # now compute Coss,0.1
        Cossp1 = C_oss / (
                (0.1 * V_dss / Vds_meas) ** self.gamma)

        Cdsq = scipy.integrate.quad(lambda Vds: (1 / self.converter_data_dict['Vin'][0]) * Cossp1 * (
                0.1 * V_dss / Vds) ** self.gamma, 0, self.converter_data_dict['Vin'][0])

        return Cdsq[0] * 10 ** -12

    def compute_Qrr_est_bf(self, Q_rr, I_F):
        # compute Qrr_est based on paper equation (1)
        p = 3  # for low-voltage Si grouping
        # self.delta_i = 0.1 # for now, soon will be an optimization variable
        self.I_d2_off = self.converter_df.loc[0, 'Iout'] - self.delta_i
        Qrr_est = Q_rr * (self.I_d2_off / I_F) ** (1 / p)
        # Qrr_est = Qrr_est[0]
        self.t_rise = 20 * 10 ** -9  # for low-voltage Si grouping

        return Qrr_est

    def compute_Rac_bf(self, DCR, f_b, b):
        from scipy.fft import fft
        Rac_total = 0
        Rac_total += DCR * self.converter_df.loc[0, 'Iout'] ** 2
        for k in np.linspace(2, 11, num=10, dtype=int):
            f = k * self.x_result[2]
            Rac = DCR * max(1, ((k - 1) * f / f_b) ** b)

            # compute the waveshape to use with the fft
            running_time = np.linspace(0, 1, num=1000)
            current = running_time.copy()
            for i in range(len(running_time)):
                if running_time[i] < self.dc:
                    current[i] = self.converter_df.loc[0, 'Iout'] - self.delta_i + (
                                2 * self.delta_i * running_time[i]) / self.dc
                else:
                    current[i] = self.converter_df.loc[0, 'Iout'] + self.delta_i - (
                                2 * self.delta_i * (running_time[i] - self.dc)) / (1 - self.dc)

            # get the fast-fourier transform (fft) of the current signal I_rms
            current_harmonics = 2 * abs(fft(current) / len(current))
            Rac_k = Rac * current_harmonics[k - 1] ** 2 / 2
            Rac_total += Rac_k

        return Rac_total

        # print('done')

    def compute_IGSE_bf(self, Inductance, N_turns, A_c, Alpha, Beta, K_fe):

        # now set the parameters to pass to Bailey's script
        delta_B = Inductance * self.delta_i / (N_turns * A_c)
        B_vector = [-delta_B, delta_B, -delta_B]
        Ts = 1 / (self.x_result[2])
        self.t_vector = [0, self.dc * Ts, Ts]
        IGSE_total = 0
        IGSE_total = coreloss(self.t_vector, B_vector, Alpha, Beta, K_fe)
        return IGSE_total

    def plotting_init(self):
        self.status = 0
        self.degree = 1
        self.model = 'chain'

        self.x0 = []

        self.fet_vals = []
        self.ind_vals = []
        self.cap_vals = []
        self.x_result = 0
        self.power_tot = 0
        self.area_tot = 0
        self.cost_tot = 0

        self.Q1_loss = 0
        self.Q2_loss = 0
        self.L1_loss = 0
        self.fun = 0


    def create_plot_vars(self):
        self.plot_range_list = np.linspace(self.plotting_range[0],
                                           self.plotting_range[1], self.num_points)

    # def init_cap(self, cap_index):
    #     # need capacitance and rated_volt as the inputs
    #     # use the voltage ripple or rms current calculation?
    #     # rated voltage is a known design quantity, given by the starting example
    #     # capacitance will be gotten either from voltage ripple requirement of rms current calculation
    #     if self.opt_var == 'power':
    #         # don't predict the cost, just use an initial value to start. still use chained to get the area, but this
    #         # will rely on other parameters previously predicted, like switching frequency. if this is entirely a
    #         # function of other variables, maybe we don't need to have another variable to optimize over? caps will
    #         # just be part of the equation? -->so then wouldn't need to initialize them at all
    #         c1 = self.variable_constraint / (2 * self.total_components)
    #     elif self.opt_var == 'cost':
    #         self.x0.append(1000000 / self.ind_normalizer)
    #         return
    #         # how to get initial fsw guess from this? could choose a generic one and see how that goes to start
    #     elif self.opt_var == 'area':
    #         self.x0.append(100000 / self.ind_normalizer)
    #         return
    #
    #     Rated_volt1 = self.caps_df.loc[cap_index]['Rated_volt']
    #     Cap_rip_prod = self.caps_df.loc[cap_index]['Cap_rip']
    #
    #     file_name = '../mosfet_data/joblib_files/capacitor_models_' + 'Balanced' + 'Opt' + '_Unit_price.joblib'
    #     # fet_reg_Energy_model = fet_regression.load_models(['Energy_Cost_product'], file_name)
    #     # fet_reg_Energy_model = fet_reg_Energy_model.reset_index()
    #     cap_reg_Unit_price_model = joblib.load(file_name)[0]
    #     T1 = fet_regression.preproc(np.array([np.log10(), np.log10(Rated_volt1)]).reshape(1, -1), self.degree)
    #     Cap1 = (10 ** cap_reg_Unit_price_model.predict(np.array(T1)))
    #     fsw1 = Cap_rip_prod / Cap1
    #     # fsw1 = [500000]
    #     # if fsw1[0] > 10**6:
    #     #     fsw1[0] = 5*10**5
    #
    #     self.x0.append(fsw1[0] / self.ind_normalizer)

    def func(self, co, x):
        return self.dIL * (1 / (8 * self.ind_normalizer * x[0 + self.offset] * co[0]) + 10 ** -0.54 * co[0] ** (
                    1 - 0.57)) - self.dVo

    from scipy.optimize import fsolve
    # def predict_cap(self, x, cap_index):
    #     file_name = '../mosfet_data/joblib_files/capacitor_models' + '_Unit_price.joblib'
    #     cap_model = joblib.load(file_name)[0]
    #     cap1 = fsolve(self.func, 10 ** -8, x)[0]
    #     # cap1 = 10**12*-self.dIL*(1+2**(3+0)*5**0*(self.ind_normalizer * x[0 + self.offset]))/(8*(self.ind_normalizer * x[0 + self.offset])*(10**1*self.dIL-self.dVo)) #can replace the 0 before self.offset w/ ind_index
    #     # cap1 = self.caps_df.loc[cap_index]['Cap_rip'] / (self.cap_normalizer * x[cap_index + self.offset])
    #     X = fet_regression.preproc(
    #         np.array([np.log10(cap1), np.log10(self.caps_df.loc[cap_index]['Rated_volt']), 1]).reshape(1, -1),
    #         2)[0]
    #     cap_model_df = pd.DataFrame(10 ** (cap_model.predict(X.reshape(1, -1))), columns=['Area', 'Cost'])
    #     # self.ind_vals[ind_index] = ind_models_df.loc['Area', self.model].predict(X)
    #     self.cap_vals.append(cap_model_df.to_dict())
    #     # print(self.cap_vals[0]['Area'])
    #     # print(self.cap_vals[0]['Cost'])

    def cost_pred_tot(self, x):
        for fet_obj in self.fet_list:
            fet_obj.predict_fet(self, x)
        for ind_obj in self.ind_list:
            ind_obj.predict_ind(self, x)
        for cap_obj in self.cap_list:
            cap_obj.predict_cap(self, x)

        self.Cost_tot = 0
        for fet in self.fet_list:
            self.Cost_tot += fet.Cost
        for ind in self.ind_list:
            self.Cost_tot += ind.Cost
        for cap in self.cap_list:
            self.Cost_tot += cap.Cost

        # TODO: This is not a good permanent solution but it does work
        if self.topology == "microinverter_combined":
            self.Cost_tot += 3 * self.fet_list[2].Cost

        # self.cost_tot = cost
        # # print('cost: %f' % cost)
        return self.Cost_tot

    def cost_bounds_fcn(self, x):
        if self.plotting_var == 'cost':
            self.cost_constraint = self.plotting_val
        return (self.cost_constraint - self.cost_pred_tot(x))

    def cost_bounds_fcn1(self):
        if self.plotting_var == 'cost':
            self.cost_constraint = self.plotting_val
        return (.2 + self.cost_constraint - self.cost_pred_tot(x))

    def cost_bounds_fcn2(self):
        if self.plotting_var == 'cost':
            self.cost_constraint = self.plotting_val
        return -(-.2 + self.cost_constraint - self.cost_pred_tot(x))

    def area_pred_tot(self, x):
        for fet_obj in self.fet_list:
            fet_obj.predict_fet(self, x)
        for ind_obj in self.ind_list:
            ind_obj.predict_ind(self, x)
        for cap_obj in self.cap_list:
            cap_obj.predict_cap(self, x)

        self.Area_tot = 0
        for fet in self.fet_list:
            self.Area_tot += fet.Area
        for ind in self.ind_list:
            self.Area_tot += ind.Area
        for cap in self.cap_list:
            self.Area_tot += cap.Area

        # TODO: again This is not a good permanent solution
        if self.topology == "microinverter_combined":
            self.Area_tot += 3 * self.fet_list[2].Area
        #
        # self.area_tot = area
        # # print('area: %f' % area)
        return self.Area_tot

    def area_bounds_fcn(self, x):
        if self.plotting_var == 'area':
            self.area_constraint = self.plotting_val

        return (self.area_constraint - self.area_pred_tot(x)) / 10

    def get_params(self):
        if self.example_num == 1 and self.topology == 'buck':
            self.converter_data_dict = {'Vin': [48], 'Vout': [12], 'Iout': [5]}
            if self.fet_tech == 'MOSFET' or self.fet_tech == 'SiCFET':
                self.fet_data_dict = {'Vdss': [60, 60], 'Vgate': [10, 10]}
            elif self.fet_tech == 'GaNFET':
                self.fet_data_dict = {'Vdss': [60, 60], 'Vgate': [5, 5]}
            self.ind_data_dict = {'Current_rating': [10], 'Ind_fsw': [4.6], 'Current_sat': [1.2 * 10]}
            self.cap_data_dict = {'Voltage_rating': [1.4*self.converter_data_dict['Vin'][0], 1.4*self.converter_data_dict['Vout'][0]],
                                  'Vdc': [self.converter_data_dict['Vin'][0], self.converter_data_dict['Vout'][0]]}

        if self.example_num == 1 and self.topology == 'boost':
            self.converter_data_dict = {'Vin': [12], 'Vout': [48], 'Iout': [2]}
            if self.fet_tech == 'MOSFET' or self.fet_tech == 'SiCFET':
                self.fet_data_dict = {'Vdss': [60, 60], 'Vgate': [10, 10]}
            elif self.fet_tech == 'GaNFET':
                self.fet_data_dict = {'Vdss': [60, 60], 'Vgate': [5, 5]}
            self.ind_data_dict = {'Current_rating': [30], 'Ind_fsw': [4.6], 'Current_sat': [1.2 * 10]}
            self.cap_data_dict = {'Voltage_rating': [1.4*self.converter_data_dict['Vin'][0], 1.4*self.converter_data_dict['Vout'][0]],
                                  'Vdc': [self.converter_data_dict['Vin'][0], self.converter_data_dict['Vout'][0]]}

        if self.example_num == 4:
            self.converter_data_dict = {'Vin': [12], 'Vout': [5], 'Iout': [5]}
            if self.fet_tech == 'MOSFET' or self.fet_tech == 'SiCFET':
                self.fet_data_dict = {'Vdss': [20, 20], 'Vgate': [10, 10]}
            elif self.fet.tech == 'GaNFET':
                self.fet_data_dict = {'Vdss': [20, 20], 'Vgate': [5, 5]}
            self.ind_data_dict = {'Current_rating': [25], 'Ind_fsw': [2.87]}
            self.constraints_data_dict = {'area_constraint': [300], 'power_constraint': [np.nan],
                                          'example_cost_range': [(2, 20)], 'bounds_list_min': [[0.001, 0.001, .01]],
                                          'bounds_list_max': [[300, 300, 100]], 'weights': [[0, 1, 0]]}
            self.dIL = 0.1 * self.converter_data_dict['Vout'][0] * self.converter_data_dict['Iout'][0] / \
                       self.converter_data_dict['Vin'][0]
            self.dVo = 0.1 * self.converter_data_dict['Vout'][0]
            self.cap_data_dict = {'Rated_volt': [20]}

        # This is the example for QZVS Boost used in the QCells microinverter
        if self.example_num == 1 and self.topology == 'QZVS_boost':
            self.converter_data_dict = {'Vin': [32.94], 'Vout': [60], 'Iout': [7.18]}
            if self.fet_tech == 'MOSFET' or self.fet_tech == 'SiCFET':
                self.fet_data_dict = {'Vdss': [100, 100], 'Vgate': [10, 10]}
            elif self.fet_tech == 'GaNFET':
                self.fet_data_dict = {'Vdss': [100, 100], 'Vgate': [5, 5]}
            self.ind_data_dict = {'Current_rating': [30], 'Ind_fsw': [4.6], 'Current_sat': [1.2 * 30]}
            self.cap_data_dict = {'Voltage_rating': [1.4*self.converter_data_dict['Vin'][0]],
                                  'Vdc': [self.converter_data_dict['Vin'][0]]}

        if self.example_num == 1 and self.topology == '2L_inverter':
            # DC bus voltage, RMS output voltage  and RMS output current
            self.converter_data_dict = {'Vin': [60], 'Vout': [40],'Iout': [15.27/np.sqrt(2)]}
            if self.fet_tech == 'MOSFET' or self.fet_tech == 'SiCFET':
                self.fet_data_dict = {'Vdss': [100], 'Vgate': [10]}
            elif self.fet_tech == 'GaNFET':
                self.fet_data_dict = {'Vdss': [100], 'Vgate': [5]}
            self.ind_data_dict = {}
            self.cap_data_dict = {}
            # Could add back input cap (or not, seems like we will size energy storage cap separately)
            # self.cap_data_dict = {'Voltage_rating': [1.4 * self.converter_data_dict['Vin'][0]],
            #                       'Vdc': [self.converter_data_dict['Vin'][0]]}

        # Note: energy storage cap between boost and inverter is sized separately
        if self.example_num == 1 and self.topology == 'microinverter_combined':
            # DC bus voltage, RMS output voltage  and RMS output current
            self.converter_data_dict = {'Vin': [32.94], 'Iin': [13.05],'Vbus': [60], 'Vout': [40],'Iout': [15.27/np.sqrt(2)]}
            if self.fet_tech == 'MOSFET' or self.fet_tech == 'SiCFET':
                self.fet_data_dict = {'Vdss': [100, 100, 100], 'Vgate': [10, 10, 10]}
            elif self.fet_tech == 'GaNFET':
                self.fet_data_dict = {'Vdss': [100, 100, 100], 'Vgate': [5, 5, 5]}
            self.ind_data_dict = {'Current_rating': [30], 'Ind_fsw': [4.6], 'Current_sat': [1.2 * 30]}
            # self.cap_data_dict = {'Voltage_rating': [1.4*self.converter_data_dict['Vin'][0]],
            #                       'Vdc': [self.converter_data_dict['Vin'][0]]}
            self.cap_data_dict = {}

    def set_optimization_variables(self, x):
        # Make sure none of the optimization variables are <= 0 or will hit convergence issues
        for index in range(len(x)):
            if x[index] < 0.000000001:
                x[index] = 0.0000000001

        if self.topology == 'buck':
            self.fet1.Rds = x[0]
            self.fet2.Rds = x[1]
            self.fsw = x[2]
            self.delta_i = x[3]
            if len(x) > 5:
                self.cap1.Area = x[4]
                self.cap2.Area = x[5]
            delta_V = 0.05 * self.converter_df.loc[0, 'Vin']
            self.cap1.Capacitance = self.converter_df.loc[0, 'Iout'] * self.dc * (1 - self.dc) / (
                    2 * (x[2] * self.ind_normalizer) * delta_V)
            # delta_V spec is: delta_V/Vout <= 1%, call it = 1%.
            delta_V = 0.01 * self.converter_df.loc[0, 'Vout']
            self.cap2.Capacitance = self.delta_i * (1 / (x[2] * self.ind_normalizer)) / (8 * delta_V)
            self.ind1.L = (self.converter_df.loc[0, 'Vin'] * self.dc) * (
                    1 / (self.fsw * self.ind_normalizer)) / (
                                  2 * (self.delta_i / self.delta_i_normalizer) * self.converter_df.loc[0, 'Iout'])
            self.ind1.delta_B = self.ind1.L * self.converter_df.loc[0, 'Iout'] * (
                        self.delta_i / self.delta_i_normalizer) / (
                                           self.ind1.N_turns * self.ind1.A_c)

        elif self.topology == 'boost':
            self.fet1.Rds = x[0]
            self.fet2.Rds = x[1]
            self.fsw = x[2]
            self.delta_i = x[3]
            if len(x) > 5:
                self.cap1.Area = x[4]
                self.cap2.Area = x[5]

            delta_V = 0.05 * self.converter_df.loc[0, 'Vin']
            self.cap1.Capacitance = self.converter_df.loc[0, 'Iout'] * self.dc * (1 - self.dc) / (
                    2 * (x[2] * self.ind_normalizer) * delta_V)
            # delta_V spec is: delta_V/Vout <= 1%, call it = 1%.
            delta_V = 0.01 * self.converter_df.loc[0, 'Vout']
            self.cap2.Capacitance = self.converter_df.loc[0, 'Iout'] * self.dc * (
                    1 / (x[2] * self.ind_normalizer)) / (2 * delta_V)
            self.converter_df.loc[0, 'Iin'] = self.converter_df.loc[0, 'Iout'] / (1 - self.dc)
            self.ind1.L = (self.converter_df.loc[0, 'Vin'] * self.dc) * (
                    1 / (self.fsw * self.ind_normalizer)) / (
                                  2 * (self.delta_i / self.delta_i_normalizer) * self.converter_df.loc[0, 'Iin'])
            self.ind1.delta_B = self.ind1.L * self.converter_df.loc[0, 'Iin']  * (
                    self.delta_i / self.delta_i_normalizer) / (
                                        self.ind1.N_turns * self.ind1.A_c)

        elif self.topology == 'QZVS_boost':
            self.fet1.Rds = x[0]
            self.fet2.Rds = x[1]
            self.fsw = x[2]
            # self.delta_i = x[3]
            self.delta_i = 10
            if len(x) > 5:
                self.cap1.Area = x[4]
                # TODO: this comment out was done for boost not QZVS so cap area results are weird, worth rechecking if time
                # self.cap2.Area = x[5]

            delta_V = 0.05 * self.converter_df.loc[0, 'Vin']
            self.cap1.Capacitance = self.converter_df.loc[0, 'Iout'] * self.dc * (1 - self.dc) / (
                    2 * (x[2] * self.ind_normalizer) * delta_V)
            # delta_V spec is: delta_V/Vout <= 1%, call it = 1%.

            self.converter_df.loc[0, 'Iin'] = self.converter_df.loc[0, 'Iout'] / (1 - self.dc)
            self.ind1.L = (self.converter_df.loc[0, 'Vin'] * self.dc) * (
                    1 / (self.fsw * self.ind_normalizer)) / (
                                  2 * (self.delta_i / self.delta_i_normalizer) * self.converter_df.loc[0, 'Iin'])
            self.ind1.delta_B = self.ind1.L * self.converter_df.loc[0, 'Iin'] * (
                    self.delta_i / self.delta_i_normalizer) / (
                                        self.ind1.N_turns * self.ind1.A_c)

        elif self.topology == '2L_inverter':
            self.fet1.Rds = x[0]

            self.fsw = 20e3 / self.ind_normalizer

        elif self.topology == 'microinverter_combined':
            self.fet1.Rds = x[0]
            self.fet2.Rds = x[1]
            self.fet3.Rds = x[2]
            # TODO: to make 4 switches, just add components but set equal to fet3.Rds
            self.fsw = x[3]
            self.delta_i = x[4]
            # if len(x) > 5:
            #     self.cap1.Area = x[5]
            #
            # delta_V = 0.05 * self.converter_df.loc[0, 'Vin']
            # self.cap1.Capacitance = (self.converter_df.loc[0, 'Iin'] * self.dc) / (
            #             2 * (x[3] * self.ind_normalizer) * delta_V)


            self.ind1.L = (self.converter_df.loc[0, 'Vin'] * self.dc) * (
                    1 / (self.fsw * self.ind_normalizer)) / (
                                  2 * (self.delta_i / self.delta_i_normalizer) * self.converter_df.loc[0, 'Iin'])
            self.ind1.delta_B = self.ind1.L * self.converter_df.loc[0, 'Iin'] * (
                    self.delta_i / self.delta_i_normalizer) / (
                                        self.ind1.N_turns * self.ind1.A_c)

    def create_component_lists(self, param_dict):
        if param_dict['topology'] == 'buck':
            self.fet1 = OptimizerFet(param_dict, 0)
            self.fet2 = OptimizerFet(param_dict, 1)
            self.ind1 = OptimizerInductor(param_dict, 0)
            self.cap1 = OptimizerCapacitor(param_dict, 0)
            self.cap2 = OptimizerCapacitor(param_dict, 1)
            self.fet_list.extend([self.fet1, self.fet2])
            self.ind_list.extend([self.ind1])
            self.cap_list.extend([self.cap1, self.cap2])

            self.dc = self.converter_df.loc[0, 'Vout'] / self.converter_df.loc[0, 'Vin']


        if param_dict['topology'] == 'boost':
            self.fet1 = OptimizerFet(param_dict, 0)
            self.fet2 = OptimizerFet(param_dict, 1)
            self.ind1 = OptimizerInductor(param_dict, 0)
            self.cap1 = OptimizerCapacitor(param_dict, 0)
            self.cap2 = OptimizerCapacitor(param_dict, 1)

            self.fet_list.extend([self.fet1, self.fet2])
            self.ind_list.extend([self.ind1])
            self.cap_list.extend([self.cap1, self.cap2])

            self.dc = 1 - (self.converter_df.loc[0, 'Vin'] / self.converter_df.loc[0, 'Vout'])

        if param_dict['topology'] == 'QZVS_boost':
            self.fet1 = OptimizerFet(param_dict, 0)
            self.fet2 = OptimizerFet(param_dict, 1)
            self.ind1 = OptimizerInductor(param_dict, 0)
            self.cap1 = OptimizerCapacitor(param_dict, 0)

            self.fet_list.extend([self.fet1, self.fet2])
            self.ind_list.extend([self.ind1])
            self.cap_list.extend([self.cap1])

            self.dc = 1 - (self.converter_df.loc[0, 'Vin'] / self.converter_df.loc[0, 'Vout'])

        if param_dict['topology'] == '2L_inverter':
            self.fet1 = OptimizerFet(param_dict, 0)

            self.fet_list.extend([self.fet1])

        if param_dict['topology'] == 'microinverter_combined':
            self.fet1 = OptimizerFet(param_dict, 0)
            self.fet2 = OptimizerFet(param_dict, 1)
            self.fet3 = OptimizerFet(param_dict, 2)
            self.fet_list.extend([self.fet1, self.fet2, self.fet3])

            self.ind1 = OptimizerInductor(param_dict, 0)
            self.ind_list.extend([self.ind1])

            # self.cap1 = OptimizerCapacitor(param_dict, 0)
            # self.cap_list.extend([self.cap1])

            self.dc = 1 - (self.converter_df.loc[0, 'Vin'] / self.converter_df.loc[0, 'Vbus'])
            pass


    def power_pred_tot(self, x):

        if not isinstance(x, np.ndarray):
            self.fet_normalizer = 1

        if isinstance(x, np.ndarray):
            # Make sure none of the optimization variables are <= 0 or will hit convergence issues
            for index in range(len(x)):
                if x[index] < 0.000000001:
                    x[index] = 0.0000000001

            # 1. It is up to the designer to set the optimization variables based on the components in their design.
            self.set_optimization_variables(x)

            # Make predictions for all components based on the optimization variables.
            # These functions are independent of the loss models, and only use the ML models.
            for fet_obj in self.fet_list:
                fet_obj.predict_fet(self, x)
            for ind_obj in self.ind_list:
                ind_obj.predict_ind(self, x)
            for cap_obj in self.cap_list:
                cap_obj.predict_cap(self, x)

        # 2. It is up to the designer to write the specific loss model here, depending on the specific components used
        # in their design. Object attributes can be created for any given component in the design by writing:
        # self.<component>.<attribute> = <value>. If the designer wants to change the physics-based loss models, they
        # may do so here, but will have to write another object function and call that.

        if self.topology == 'buck':

            self.fet2.I_d2_off = self.converter_df.loc[0, 'Iout'] - (self.delta_i / self.delta_i_normalizer) * \
                                 self.converter_df.loc[
                                     0, 'Iout']

            # predict Cossp1 for the fet
            self.fet2.compute_Cdsq(self)
            self.fet2.Cdsq = \
                scipy.integrate.quad(
                    lambda Vds: (1 / self.converter_data_dict['Vin'][0]) * self.fet2.Cossp1 * 10 ** -12 * (
                            0.1 * self.fet2.Vdss / Vds) ** self.fet2.gamma, 0, self.converter_data_dict['Vin'][0])[0]

            # fet_model_df['Cdsq'] = Cdsq[0]

            # Add the info to the df about kT, the factor to multiply Ron by based on Vdss
            self.fet1.compute_kT(self)
            self.fet2.compute_kT(self)
            # fet_model_df['kT'] = kT

            # Compute Qrr,est based on I_F prediction and Qrr prediction
            if self.fet_tech != 'GaNFET':
                self.fet2.compute_Qrr_est(self)

            # compute values related to P_Qoff loss
            self.fet1.I_off_Q1 = self.converter_df.loc[0, 'Iout'] + (self.delta_i / self.delta_i_normalizer) * \
                                 self.converter_df.loc[0, 'Iout']

            # self.fet_vals.append(fet_model_df.to_dict())

            power = 0

            self.I_Q1 = np.sqrt(self.dc) * self.converter_df.loc[0, 'Iout']
            self.I_Q2 = np.sqrt(1 - self.dc) * self.converter_df.loc[0, 'Iout']

            # compute the IGSE including Vc
            self.ind1.compute_IGSE(self)

            # compute Rac values
            self.ind1.compute_Rac(self)

        # 3. It is up to the designer to write the loss model for their topology. Separate losses by specific loss
        #    contribution and by component.
        if self.topology == 'buck':
            ####### NEW LOSS MODEL: ###########
            # self.compute_loss() --> use this generalized method to compute the loss, so that can pull from the same
            # equation in the component selection method, see compute_kT for how
            self.fet1.Rds_loss = self.I_Q1 ** 2 * self.fet1.kT * self.fet_normalizer * self.fet1.Rds
            self.fet1.Qg_loss = self.ind_normalizer * self.fsw * self.fet1.Qg * self.fet1.Vgate
            self.Q1_loss = self.fet1.Rds_loss + self.fet1.Qg_loss

            self.fet2.Rds_loss = self.I_Q2 ** 2 * self.fet2.kT * self.fet_normalizer * self.fet2.Rds
            self.fet2.Qg_loss = self.ind_normalizer * self.fsw * self.fet2.Qg * self.fet2.Vgate
            # have already predicted Coss,0.1, use this to take the integral along with gamma to compute Cds,q

            self.fet2.Cdsq_loss = self.ind_normalizer * self.fsw * self.fet2.Cdsq * self.converter_df.loc[
                0, 'Vin'] ** 2
            self.fet2.toff_loss = self.fet1.I_off_Q1 ** 2 * self.fet2.t_off ** 2 * self.ind_normalizer * self.fsw / (
                    48 * (10 ** -12 * self.fet2.Cossp1))
            self.Q2_loss = self.fet2.Rds_loss + self.fet2.Qg_loss + self.fet2.Cdsq_loss + self.fet2.toff_loss

            if self.fet_tech != 'GaNFET':
                self.fet2.Qrr_loss = self.converter_df.loc[0, 'Vin'] * (
                        self.fet2.Q_rr + (self.fet2.t_rr * (self.converter_df.loc[0, 'Iout'] -
                                                            (self.converter_df.loc[
                                                                 0, 'Iout'] * self.delta_i / self.delta_i_normalizer)))) \
                                     * self.ind_normalizer * self.fsw
                self.Q2_loss += self.fet2.Qrr_loss

            self.ind1.Rdc_loss = self.converter_df.loc[0, 'Iout'] ** 2 * self.ind1.R_dc
            self.ind1.Rac_loss = self.ind1.Rac_total
            self.ind1.IGSE_loss = self.ind1.Core_volume * self.ind1.IGSE_total
            self.L1_loss = self.ind1.Rdc_loss + self.ind1.Rac_loss + self.ind1.IGSE_loss

            self.power_tot = self.Q1_loss + self.Q2_loss + self.L1_loss

        if self.topology == 'boost':
            self.fet2.I_d2_off = self.converter_df.loc[0, 'Iout'] - (self.delta_i / self.delta_i_normalizer) * \
                                 self.converter_df.loc[
                                     0, 'Iout']

            # TODO: I'm not sure this should be calculating for fet2, redo math and check if should be fet1
            # predict Cossp1 for the fet
            self.fet2.compute_Cdsq(self)
            self.fet2.Cdsq = \
                scipy.integrate.quad(
                    lambda Vds: (1 / self.converter_data_dict['Vout'][0]) * self.fet2.Cossp1 * 10 ** -12 * (
                            0.1 * self.fet2.Vdss / Vds) ** self.fet2.gamma, 0, self.converter_data_dict['Vout'][0])[0]

            # fet_model_df['Cdsq'] = Cdsq[0]

            # Add the info to the df about kT, the factor to multiply Ron by based on Vdss
            self.fet1.compute_kT(self)
            self.fet2.compute_kT(self)
            # fet_model_df['kT'] = kT

            # Compute Qrr,est based on I_F prediction and Qrr prediction
            if self.fet_tech != 'GaNFET':
                self.fet2.compute_Qrr_est(self)

            # compute values related to P_Qoff loss
            self.fet1.I_off_Q1 = self.converter_df.loc[0, 'Iout'] + (self.delta_i / self.delta_i_normalizer) * \
                                 self.converter_df.loc[0, 'Iout']

            # self.fet_vals.append(fet_model_df.to_dict())

            # TODO: this is wrong? should be sqrt(D)*IL? (or better RMS)
            self.I_Q1 = np.sqrt(self.dc) * self.converter_df.loc[0, 'Iout']
            self.I_Q2 = np.sqrt(1 - self.dc) * self.converter_df.loc[0, 'Iout']

            # compute the IGSE including Vc
            self.ind1.compute_IGSE(self)

            # compute Rac values
            self.ind1.compute_Rac_boost(self)

        if self.topology == 'boost':
            ####### NEW LOSS MODEL: ###########
            # self.compute_loss() --> use this generalized method to compute the loss, so that can pull from the same
            # equation in the component selection method, see compute_kT for how
            self.fet1.Rds_loss = self.I_Q1 ** 2 * self.fet1.kT * self.fet_normalizer * self.fet1.Rds
            self.fet1.Qg_loss = self.ind_normalizer * self.fsw * self.fet1.Qg * self.fet1.Vgate
            self.Q1_loss = self.fet1.Rds_loss + self.fet1.Qg_loss

            self.fet2.Rds_loss = self.I_Q2 ** 2 * self.fet2.kT * self.fet_normalizer * self.fet2.Rds
            self.fet2.Qg_loss = self.ind_normalizer * self.fsw * self.fet2.Qg * self.fet2.Vgate
            self.fet2.Cdsq_loss = self.ind_normalizer * self.fsw * self.fet2.Cdsq * self.converter_df.loc[
                0, 'Vout'] ** 2
            self.fet2.toff_loss = self.fet1.I_off_Q1 ** 2 * self.fet2.t_off ** 2 * self.ind_normalizer * self.fsw / (
                    48 * (10 ** -12 * self.fet2.Cossp1))
            self.Q2_loss = self.fet2.Rds_loss + self.fet2.Qg_loss + self.fet2.Cdsq_loss + self.fet2.toff_loss

            if self.fet_tech != 'GaNFET':
                self.fet2.Qrr_loss = self.converter_df.loc[0, 'Vout'] * (
                        self.fet2.Q_rr + (self.fet2.t_rr * (self.converter_df.loc[0, 'Iout'] -
                                                            ((self.converter_df.loc[0, 'Iout']) * self.delta_i / self.delta_i_normalizer)))) \
                                     * self.ind_normalizer * self.fsw
                self.Q2_loss += self.fet2.Qrr_loss

            self.ind1.Rdc_loss = (self.converter_df.loc[0, 'Iout'] / (1 - self.dc)) ** 2 * self.ind1.R_dc
            self.ind1.Rac_loss = self.ind1.Rac_total
            self.ind1.IGSE_loss = self.ind1.Core_volume * self.ind1.IGSE_total
            self.L1_loss = self.ind1.Rdc_loss + self.ind1.Rac_loss + self.ind1.IGSE_loss

            self.power_tot = self.Q1_loss + self.Q2_loss + self.L1_loss

        if self.topology == 'QZVS_boost':

            # Add the info to the df about kT, the factor to multiply Ron by based on Vdss
            self.fet1.compute_kT(self)
            self.fet2.compute_kT(self)

            # compute values related to P_Qoff loss - assumes 100% ripple so
            self.fet1.I_off_Q1 = self.converter_df.loc[0, 'Iout'] * (1/(1-self.dc)) * 2

            power = 0

            self.I_Q1 = np.sqrt(self.dc) * self.converter_df.loc[0, 'Iout']
            self.I_Q2 = np.sqrt(1 - self.dc) * self.converter_df.loc[0, 'Iout']

            # compute the IGSE including Vc
            self.ind1.compute_IGSE(self)

            # compute Rac values
            self.ind1.compute_Rac_boost(self)

        if self.topology == 'QZVS_boost':

            # equation in the component selection method, see compute_kT for how
            self.fet1.Rds_loss = self.fet_normalizer * self.fet1.Rds * self.fet1.kT * ( self.I_Q1 * np.sqrt(self.dc) *
                                 np.sqrt(1+(1/3)*(self.delta_i / self.delta_i_normalizer / self.I_Q1)**2) )**2
            self.fet1.Qg_loss = self.ind_normalizer * self.fsw * self.fet1.Qg * self.fet1.Vgate
            self.fet1.toff_loss = self.fet1.I_off_Q1 ** 2 * self.fet1.t_off ** 2 * self.ind_normalizer * self.fsw / (
                    48 * (10 ** -12 * self.fet2.Cossp1))
            self.Q1_loss = self.fet1.Rds_loss + self.fet1.Qg_loss + self.fet1.toff_loss

            self.fet2.Rds_loss = self.fet_normalizer * self.fet2.Rds * self.fet1.kT * ( self.I_Q2 * np.sqrt(1-self.dc) *
                                 np.sqrt(1+(1/3)*(self.delta_i / self.delta_i_normalizer / self.I_Q2)**2) )**2
            self.fet2.Qg_loss = self.ind_normalizer * self.fsw * self.fet2.Qg * self.fet2.Vgate
            self.Q2_loss = self.fet2.Rds_loss + self.fet2.Qg_loss

            self.ind1.Rdc_loss = (self.converter_df.loc[0, 'Iout'] / (1 - self.dc)) ** 2 * self.ind1.R_dc
            self.ind1.Rac_loss = self.ind1.Rac_total
            self.ind1.IGSE_loss = self.ind1.Core_volume * self.ind1.IGSE_total
            self.L1_loss = self.ind1.Rdc_loss + self.ind1.Rac_loss + self.ind1.IGSE_loss

            self.power_tot = self.Q1_loss + self.Q2_loss + self.L1_loss

        if self.topology == '2L_inverter':

            self.fet1.compute_kT(self)

            # Line frequency
            F = 60
            Ns = int(self.fsw * self.ind_normalizer / F)

            total_E_Qrr = 0
            total_E_Off = 0
            total_E_On = 0

            for i in range(0, int(Ns/2)):
                Iac = np.sqrt(2) * self.converter_df.loc[0, 'Iout'] * np.sin(2*np.pi*(i/Ns))
                if self.fet_tech != 'GaNFET':
                    self.fet1.I_d2_off = Iac
                    self.fet1.compute_Qrr_est(self, Iac)
                    E_Qrr = self.converter_df.loc[0, 'Vin'] * (self.fet1.Q_rr + self.fet1.t_rr * Iac)
                    total_E_Qrr += 2 * E_Qrr

                E_off = Iac**2 * self.fet1.t_off**2 / (48 * 10**-12 * self.fet1.Cossp1)
                total_E_Off += 2 * E_off

                self.fet1.compute_Cdsq(self)
                self.fet1.Cdsq = scipy.integrate.quad(
                        lambda Vds: (1 / self.converter_data_dict['Vin'][0]) * self.fet1.Cossp1 * 10 ** -12 * (0.1 * self.fet1.Vdss / Vds) ** self.fet1.gamma, 0, self.converter_data_dict['Vin'][0])[0]
                E_on = self.fet1.Cdsq * self.converter_data_dict['Vin'][0]
                total_E_On += 2 * E_on

            self.fet1.toff_loss = 2 * F * total_E_Off
            self.fet1.Qrr_loss = 2 * F * total_E_Qrr
            self.fet1.Cdsq_loss = 2 * F * total_E_On
            self.fet1.Qg_loss = 4 * self.ind_normalizer * self.fsw * self.fet1.Qg * self.fet1.Vgate
            # TODO: this has to be Rds not Ron (why are there both anyway??)
            self.fet1.Rds_loss = 2 * self.converter_df.loc[0, 'Iout']**2 * self.fet1.Rds * self.fet_normalizer
            self.Q1_loss = self.fet1.toff_loss + self.fet1.Qrr_loss + self.fet1.Cdsq_loss + self.fet1.Qg_loss + self.fet1.Rds_loss

            self.power_tot = self.Q1_loss

        if self.topology == 'microinverter_combined':

            # Add the info to the df about kT, the factor to multiply Ron by based on Vdss
            self.fet1.compute_kT(self)
            self.fet2.compute_kT(self)

            # compute values related to P_Qoff loss - assumes 100% ripple so
            self.fet1.I_off_Q1 = self.converter_df.loc[0, 'Iin'] * 2

            power = 0

            # TODO: RMS calculations, check with Skye that this is right?
            self.I_Q1 = self.converter_df.loc[0, 'Iin'] * np.sqrt(self.dc) * np.sqrt(1+(1/3)*(self.delta_i / self.delta_i_normalizer / self.converter_df.loc[0, 'Iin'])**2)
            self.I_Q2 = self.converter_df.loc[0, 'Iin'] * np.sqrt(1 - self.dc) * np.sqrt(1+(1/3)*(self.delta_i / self.delta_i_normalizer / self.converter_df.loc[0, 'Iin'])**2)

            # compute the IGSE including Vc
            self.ind1.compute_IGSE(self)

            # compute Rac values
            self.ind1.compute_Rac_boost(self)

            # equation in the component selection method, see compute_kT for how
            self.fet1.Rds_loss = self.fet_normalizer * self.fet1.Rds * self.fet1.kT * ( self.I_Q1 )**2
            self.fet1.Qg_loss = self.ind_normalizer * self.fsw * self.fet1.Qg * self.fet1.Vgate
            self.fet1.toff_loss = self.fet1.I_off_Q1 ** 2 * self.fet1.t_off ** 2 * self.ind_normalizer * self.fsw / (
                    48 * (10 ** -12 * self.fet2.Cossp1))
            self.Q1_loss = self.fet1.Rds_loss + self.fet1.Qg_loss + self.fet1.toff_loss

            self.fet2.Rds_loss = self.fet_normalizer * self.fet2.Rds * self.fet1.kT * ( self.I_Q2 )**2
            self.fet2.Qg_loss = self.ind_normalizer * self.fsw * self.fet2.Qg * self.fet2.Vgate
            self.Q2_loss = self.fet2.Rds_loss + self.fet2.Qg_loss

            # Using RMS value of current
            # TODO: DC loss shouldn't be RMS current, right?
            self.ind1.Rdc_loss = self.converter_df.loc[0, 'Iin']**2 * self.ind1.R_dc
            self.ind1.Rac_loss = self.ind1.Rac_total
            self.ind1.IGSE_loss = self.ind1.Core_volume * self.ind1.IGSE_total
            self.L1_loss = self.ind1.Rdc_loss + self.ind1.Rac_loss + self.ind1.IGSE_loss

            self.fet3.compute_kT(self)
            # Line frequency
            F = 60
            # TODO: microinverter hardcoded at 20 kHz
            inv_fsw = 20e3
            Ns = int(inv_fsw / F)

            total_E_Qrr = 0
            total_E_Off = 0
            total_E_On = 0

            for i in range(0, int(Ns / 2)):
                Iac = np.sqrt(2) * self.converter_df.loc[0, 'Iout'] * np.sin(2 * np.pi * (i / Ns))
                if self.fet_tech != 'GaNFET':
                    self.fet3.I_d2_off = Iac
                    self.fet3.compute_Qrr_est(self, Iac)
                    E_Qrr = self.converter_df.loc[0, 'Vbus'] * (self.fet3.Q_rr + self.fet3.t_rr * Iac)
                    total_E_Qrr += 2 * E_Qrr

                E_off = Iac ** 2 * self.fet3.t_off ** 2 / (48 * 10 ** -12 * self.fet3.Cossp1)
                total_E_Off += 2 * E_off

                self.fet3.compute_Cdsq(self)
                self.fet3.Cdsq = scipy.integrate.quad(
                    lambda Vds: (1 / self.converter_data_dict['Vbus'][0]) * self.fet3.Cossp1 * 10 ** -12 * (
                                0.1 * self.fet3.Vdss / Vds) ** self.fet3.gamma, 0, self.converter_data_dict['Vbus'][0])[
                    0]
                E_on = self.fet3.Cdsq * self.converter_data_dict['Vbus'][0]
                total_E_On += 2 * E_on

            self.fet3.toff_loss = 2 * F * total_E_Off
            self.fet3.Qrr_loss = 2 * F * total_E_Qrr
            self.fet3.Cdsq_loss = 2 * F * total_E_On
            self.fet3.Qg_loss = 4 * inv_fsw * self.fet3.Qg * self.fet3.Vgate
            self.fet3.Rds_loss = 2 * self.converter_df.loc[0, 'Iout'] ** 2 * self.fet3.Rds * self.fet_normalizer
            self.Q3_loss = self.fet3.toff_loss + self.fet3.Qrr_loss + self.fet3.Cdsq_loss + self.fet3.Qg_loss + self.fet3.Rds_loss
            self.power_tot = self.Q1_loss + self.Q2_loss + self.L1_loss + self.Q3_loss

        if self.fet_tech == 'MOSFET':
            self.power_divider = 10
        if self.fet_tech == 'GaNFET':
            self.power_divider = 100

        return self.power_tot / self.power_divider

    def power_pred_tot_boost(self, x):
        print(self.fsw)
        print(x)

    def power_pred_tot_hold(self, x):
        print(self.fsw)
        # print(x)
        for index in range(len(x)):
            if x[index] < 0.000000001:
                x[index] = 0.0000000001
        self.fet_vals = []
        self.ind_vals = []
        self.cap_vals = []
        # self.delta_i = x[3]  # for now, if don't want to be an optimization variable replace w/ 0.1

        # Set the optimization variables
        self.fet1.Rds = x[0]
        self.fet2.Rds = x[1]
        self.fsw = x[2]
        self.delta_i = x[3]
        self.cap1.Area = x[4]
        self.cap2.Area = x[5]

        self.fet1.predict_fet(self)
        self.fet2.predict_fet(self)
        self.ind1.predict_ind(self)
        self.cap1.predict_cap(self)
        self.cap2.predict_cap(self)

        # Now have all the quantities we need for the specific power loss model
        # Add the info to the df about Cds,q
        self.fet2.I_d2_off = self.converter_df.loc[0, 'Iout'] - (self.delta_i / self.delta_i_normalizer) * \
                             self.converter_df.loc[
                                 0, 'Iout']

        # predict Cossp1 for the fet
        self.fet2.compute_Cdsq(self)
        # fet_model_df['Cdsq'] = Cdsq[0]

        # Add the info to the df about kT, the factor to multiply Ron by based on Vdss
        self.fet1.compute_kT(self)
        self.fet2.compute_kT(self)
        # fet_model_df['kT'] = kT

        # Compute Qrr,est based on I_F prediction and Qrr prediction
        if self.fet_tech != 'GaNFET':
            self.fet2.compute_Qrr_est(self)

        # compute values related to P_Qoff loss
        self.fet1.I_off_Q1 = self.converter_df.loc[0, 'Iout'] + (self.delta_i / self.delta_i_normalizer) * \
                             self.converter_df.loc[0, 'Iout']

        # self.fet_vals.append(fet_model_df.to_dict())

        power = 0

        self.I_Q1 = np.sqrt(self.dc) * self.converter_df.loc[0, 'Iout']
        self.I_Q2 = np.sqrt(1 - self.dc) * self.converter_df.loc[0, 'Iout']

        # compute the IGSE including Vc
        self.ind1.compute_IGSE(self)

        # compute Rac values
        self.ind1.compute_Rac(self)

        # print('start')
        # print(x)
        # print(self.fet_vals[0])
        # print(self.fet_vals[1])
        # print(self.ind_vals[0])
        # print(power)
        # print('end')

        # Making some adjustments to the way this is computed--adding in new Qrr quantities, losses due to Cdsq, and
        # adjusting Ron loss contributions to scale with junction temperature

        ###### OLD LOSS MODEL: ###########
        # self.Q1_loss = self.I_Q1 ** 2 * self.fet_normalizer * x[0] + \
        #                self.ind_normalizer * x[2] * self.fet_vals[0]['Qg'][0] * self.fets_df.loc[0, 'Vgate'] \
        #                + .5 * self.ind_normalizer * x[2] * self.fet_vals[0]['Coss'][0] * self.converter_df.loc[
        #                    0, 'Vin'] ** 2
        # self.Q2_loss = self.I_Q2 ** 2 * self.fet_normalizer * x[1] + \
        #                self.ind_normalizer * x[2] * self.fet_vals[1]['Qg'][0] * self.fets_df.loc[1, 'Vgate'] \
        #                + self.ind_normalizer * x[2] * self.fet_vals[1]['Qrr'][0] * self.converter_df.loc[0, 'Vin'] + \
        #                .5 * self.ind_normalizer * x[2] * self.fet_vals[1]['Coss'][0] * self.converter_df.loc[
        #                    0, 'Vin'] ** 2
        # self.L1_loss = self.converter_df.loc[0, 'Iout'] ** 2 * self.ind_vals[0]['DCR'][0]

        ####### NEW LOSS MODEL: ###########
        # self.compute_loss() --> use this generalized method to compute the loss, so that can pull from the same
        # equation in the component selection method, see compute_kT for how
        self.fet1.Rds_loss = self.I_Q1 ** 2 * self.fet1.kT * self.fet_normalizer * self.fet1.Rds
        self.fet1.Qg_loss = self.ind_normalizer * self.fsw * self.fet1.Qg * self.fet1.Vgate
        self.Q1_loss = self.fet1.Rds_loss + self.fet1.Qg_loss

        # self.Q1_loss = self.I_Q1 ** 2 * self.fet1.kT * self.fet_normalizer * self.fet1.Rds + \
        #                self.ind_normalizer * self.fsw * self.fet1.Qg * self.fet1.Vgate

        self.fet2.Rds_loss = self.I_Q2 ** 2 * self.fet2.kT * self.fet_normalizer * self.fet2.Rds
        self.fet2.Qg_loss = self.ind_normalizer * self.fsw * self.fet2.Qg * self.fet2.Vgate
        self.fet2.Cdsq_loss = self.ind_normalizer * self.fsw * self.fet2.Cdsq * self.converter_df.loc[
                           0, 'Vin'] ** 2
        self.fet2.toff_loss = self.fet1.I_off_Q1 ** 2 * self.fet2.t_off ** 2 * self.ind_normalizer * self.fsw / (
                                   48 * (10**-12 * self.fet2.Cossp1))
        self.Q2_loss = self.fet2.Rds_loss + self.fet2.Qg_loss + self.fet2.Cdsq_loss + self.fet2.toff_loss

 #        self.Q2_loss = self.I_Q2 ** 2 * self.fet2.kT * self.fet_normalizer * self.fet2.Rds + \
 #                       self.ind_normalizer * self.fsw * self.fet2.Qg * self.fet2.Vgate + \
 # \
 #                       self.ind_normalizer * self.fsw * self.fet2.Cdsq * self.converter_df.loc[
 #                           0, 'Vin'] ** 2 \
 #                       + self.fet1.I_off_Q1 ** 2 * self.fet2.t_off ** 2 * self.ind_normalizer * self.fsw / (
 #                                   48 * (10**-12 * self.fet2.Cossp1))

        if self.fet_tech != 'GaNFET':
            self.fet2.Qrr_loss = self.converter_df.loc[0, 'Vin'] * (self.fet2.Q_rr + (self.fet2.t_rr * (self.converter_df.loc[0, 'Iout'] -
                                                                                (self.converter_df.loc[0, 'Iout'] * self.delta_i/self.delta_i_normalizer)) ))\
                                                               * self.ind_normalizer * self.fsw
            self.Q2_loss += self.fet2.Qrr_loss

            # self.Q2_loss += self.converter_df.loc[0, 'Vin'] * (self.fet2.Q_rr + (self.fet2.t_rr * (self.converter_df.loc[0, 'Iout'] -
            #                                                                     (self.converter_df.loc[0, 'Iout'] * self.delta_i/self.delta_i_normalizer)) ))\
            #                                                    * self.ind_normalizer * self.fsw
                # (self.fet2.I_d2_off * self.t_rise / 2) * self.converter_df.loc[
                # 0, 'Vin'] * self.ind_normalizer * self.fsw \
                #             + self.fet2.Qrr_est * self.converter_df.loc[0, 'Vin'] * self.ind_normalizer * self.fsw

        self.ind1.Rdc_loss = self.converter_df.loc[0, 'Iout'] ** 2 * self.ind1.R_dc
        self.ind1.Rac_loss = self.ind1.Rac_total
        self.ind1.IGSE_loss = self.ind1.Core_volume * self.ind1.IGSE_total
        self.L1_loss = self.ind1.Rdc_loss + self.ind1.Rac_loss + self.ind1.IGSE_loss

        # self.L1_loss = self.converter_df.loc[0, 'Iout'] ** 2 * self.ind1.R_dc \
        #                + self.ind1.Rac_total + self.ind1.Core_volume * self.ind1.IGSE_total

        self.power_tot = self.Q1_loss + self.Q2_loss + self.L1_loss

        # print(x)
        # print(self.power_tot)
        if self.fet_tech == 'MOSFET':
            self.power_divider = 10
        if self.fet_tech == 'GaNFET':
            self.power_divider = 100

        if self.power_tot < self.power_tot_prev + .0001 and self.power_tot > self.power_tot_prev - .0001:
            self.cont_counter += 1
        else:
            self.cont_counter = 0
        self.power_tot_prev = self.power_tot
        var = self.power_tot / self.Pout
        # print('power_tot: %f' % var)
        return self.power_tot / self.power_divider

    def power_bounds_fcn(self, x):
        if self.plotting_var == 'power':
            self.power_constraint = self.plotting_val

        # print('power fraction: %f' % (self.power_pred_tot(x) * self.power_divider / self.Pout))
        return (self.power_constraint - (self.power_pred_tot(x) * self.power_divider / self.Pout))

    def power_bounds_fcn1(self, x):
        if self.plotting_var == 'power':
            self.power_constraint = self.plotting_val

        # print('power fraction: %f' % (self.power_pred_tot(x) * self.power_divider / self.Pout))
        return (self.power_constraint - (self.power_pred_tot(x) * self.power_divider / self.Pout))

    def power_bounds_fcn2(self, x):
        if self.plotting_var == 'power':
            self.power_constraint = self.plotting_val

        # print('power fraction: %f' % (self.power_pred_tot(x) * self.power_divider / self.Pout))
        return -(self.power_constraint - (self.power_pred_tot(x) * self.power_divider / self.Pout))

    # TODO: Discuss with Skye, these are useful but hard coded to one set of x definitions
    def Rds1_bounds_minfcn(self, x):
        return x[0] - 0.05

    def Rds2_bounds_minfcn(self, x):
        # return x[1]-0.001
        return x[1] - 0.05

    def Rds1_bounds_maxfcn(self, x):
        return 3000 - x[0]

    def Rds2_bounds_maxfcn(self, x):
        return 3000 - x[1]

    # TODO: 4/2 changed x indices on these
    def fsw_bounds_minfcn(self, x):
        # return x[2] - .001
        return x[3] - 2

    def fsw_bounds_maxfcn(self, x):
        return 100 - x[3]

    # TODO: set min switching frequency here
    def delta_i_bounds_minfcn(self, x):
        return x[4] - 10

    def delta_i_bounds_maxfcn(self, x):
        return 10 - x[4]

    def minimize_fcn(self):
        results = dict()

        # TODO: can we make these functions take arguments and be custom for each converter
        # TODO: change these when switching between inverter and boost
        # If the user wants to set additional bounds, e.g. on the optimization variables, they may do so here.
        con_cobyla = [{'type': 'ineq', 'fun': self.bounds_table[self.plotting_var]},
                      {'type': 'ineq', 'fun': self.bounds_table[self.set_constraint]},
                      {'type': 'ineq', 'fun': self.Rds1_bounds_minfcn},
                      {'type': 'ineq', 'fun': self.Rds2_bounds_minfcn},
                      {'type': 'ineq', 'fun': self.fsw_bounds_minfcn},
                      {'type': 'ineq', 'fun': self.fsw_bounds_maxfcn},
                      {'type': 'ineq', 'fun': self.delta_i_bounds_minfcn},
                      {'type': 'ineq', 'fun': self.delta_i_bounds_maxfcn}
                      ]

        if self.plotting_var == 'power' or self.set_constraint == 'power':
            con_cobyla.append({'type': 'ineq', 'fun': self.bounds_table["power2"]})


        self.x0 = []
        # Make predictions for all components based on the optimization variables.
        # These functions are independent of the loss models, and only use the ML models.
        for fet_obj in self.fet_list:
            fet_obj.init_fet(self)
        for ind_obj in self.ind_list:
            ind_obj.init_ind(self)
        for cap_obj in self.cap_list:
            cap_obj.init_cap(self)

        self.optimization_table = {"power": self.power_pred_tot, "cost": self.cost_pred_tot, "area": self.area_pred_tot}

        rescobyla = minimize(self.optimization_table[self.opt_var], np.array(self.x0), # args=self,
                             method='COBYLA', constraints=con_cobyla,
                             # tol sets relevant solver-specific tolerance
                             # eps has to do with trust region constraint
                             # catol is the allowable constraint violation
                             # ftol is the step in the objective function that is allowable
                             # xtol is the step in the optimization variable array
                             options={'disp': True, 'maxiter': 1000, 'eps': 10 ** -4, 'catol': 10 ** -3, 'ftol': 10**-3, 'xtol': 10**-3})
                                #'tol' = 10**-6
        print(rescobyla.x)
        # TODO: change this when switching between topologies
        print("Un-normalized results:",
              self.fet_normalizer*rescobyla.x[0],
              self.fet_normalizer*rescobyla.x[1],
              self.fet_normalizer*rescobyla.x[2],
              self.ind_normalizer*rescobyla.x[3],
              rescobyla.x[4] / self.delta_i_normalizer,
              # self.cap_normalizer * rescobyla.x[5],
        )
        print('cost constraint: %f' % self.plotting_val)
        print('power: %f, cost: %f, area: %d' % (self.power_tot / self.Pout, self.Cost_tot, self.Area_tot))
        self.fun = rescobyla.fun
        if self.opt_var == 'power':
            self.fun = rescobyla.fun * self.power_divider / self.Pout

        if (rescobyla.status == 0) and rescobyla.maxcv < 0.005:
            self.status = 1
        else:
            self.status = rescobyla.status
        self.x_result = rescobyla.x

        return results


class OptimizerFet(OptimizerInit):
    def __init__(self, param_dict, fet_index):
        super().__init__(param_dict)
        self.Vdss = self.fets_df.loc[fet_index]['Vdss']
        self.Vgate = self.fets_df.loc[fet_index]['Vgate']
        self.Ron = 0
        self.Qg = 0
        self.Cost = 0
        self.Area = 0
        self.Coss = 0
        self.Vds_meas = 0
        self.tau_c = 0
        self.tau_rr = 0
        self.I_F = 0



    def init_fet(self, opt_obj):

        if self.fet_tech == 'MOSFET':
            self.p = 3
            self.t_rise = 20 * 10 ** -9  # for low-voltage Si grouping
            self.t_off = 10 * 10 ** -9  # for Si low-voltage grouping

            # for low-voltage Si grouping
        elif self.fet_tech == 'GaNFET':
            self.p = 1.5
            self.t_rise = 10 * 10 ** -9  # for low-voltage Si grouping
            self.t_off = 5 * 10 ** -9  # for Si low-voltage grouping

        # print(self.opt_var)
        # print(self.Ron)
        if opt_obj.opt_var == 'power':
            if opt_obj.plotting_var == 'cost':
                c1 = opt_obj.plotting_val / (2 * opt_obj.total_components)
                # what if optimizing as function of area constraint? will also be setting some cost constraint, can just
                # choose c1 based off that
            else:  # what happens if plotting_var == 'area'
                c1 = 1

        elif self.opt_var == 'cost':
            if self.plotting_var == 'power':
                self.I_Q1 = np.sqrt(self.dc) * self.converter_df.loc[0, 'Iout']
                self.I_Q2 = np.sqrt(1 - self.dc) * self.converter_df.loc[0, 'Iout']
                Rds = 0.1 * self.plotting_val * self.Pout / (self.I_Q1 ** 2)
                self.x0.append(Rds / self.fet_normalizer)
                return
            else:
                c1 = 1

            # do something where the initial Rds value is just some fraction of the power loss constraint?
        elif self.opt_var == 'area':
            # for plotting as function of changing power constraint
            if self.plotting_var == 'power':
                self.I_Q1 = np.sqrt(self.dc) * self.converter_df.loc[0, 'Iout']
                self.I_Q2 = np.sqrt(1 - self.dc) * self.converter_df.loc[0, 'Iout']
                Rds = 0.3 * self.plotting_val * self.Pout / (self.I_Q1 ** 2)
                self.x0.append(Rds / self.fet_normalizer)
                return

            # for plotting as function of changing cost constraint
            else:
                c1 = self.plotting_val / (2 * self.total_components)

            # could use an initial run of the original optimization w/ a high cost constraint to get initial values
        voltage = self.Vdss
        # file_name = '../mosfet_data/joblib_files/' + str(self.fet_tech) + '_models_' + 'Balanced' + 'Opt'
        #
        # fet_reg_RdsCost_model = fet_regression.load_models(['RdsCost_productOfCV'], file_name).reset_index()

        file_name = 'joblib_files/R_ds_initialization'
        fet_model = joblib.load(file_name)

        X = []
        if opt_obj.FET_type == 'N':
            X.extend([1, 0])
        else:
            X.extend([0, 1])
        if opt_obj.fet_tech == 'MOSFET':
            X.extend([1, 0, 0])
        elif opt_obj.fet_tech == 'SiCFET':
            X.extend([0, 1, 0])
        else:
            X.extend([0, 0, 1])
        X.extend(
            [np.log10(self.Vdss), np.log10(opt_obj.fet_normalizer * c1)])
        X = fet_regression.preproc(np.array(X).reshape(
            1, -1), 1)[0]
        Rds = 10 ** fet_model.predict(X.reshape(1, -1))[0]

        # X = fet_regression.preproc(np.array(X).reshape(
        #     1, -1), 1)[0]
        # Rds = (10 ** fet_reg_RdsCost_model.predict(np.array(
        #     fet_regression.preproc(np.array([np.log10(voltage), np.log10(c1)]).reshape(1, -1), opt_obj.degree))))

        Rds = [0.01]
        opt_obj.x0.append(Rds[0] / opt_obj.fet_normalizer)

        # opt_obj.x0.append(Rds[0] / opt_obj.fet_normalizer)

    def predict_fet(self, opt_obj, x):
        # Load the trained model for predicting Q_g and Unit_price based on R_on and V_dss
        # for other models, add joblib_files/ before full_dataset_models_chained, and comment out the area models, and SiC

        file_name = 'full_dataset_models_chained.joblib'
        # X should look like: ['N', 'P', 'MOSFET', 'SiCFET', 'GaNFET', log10[Vdss], log10[Rds]]
        X = []
        if opt_obj.FET_type == 'N':
            X.extend([1, 0])
        else:
            X.extend([0, 1])
        if opt_obj.fet_tech == 'MOSFET':
            X.extend([1, 0, 0])
        elif opt_obj.fet_tech == 'SiCFET':
            X.extend([0, 1, 0])
        else:
            X.extend([0, 0, 1])
        X.extend(
            [np.log10(self.Vdss), np.log10(opt_obj.fet_normalizer * self.Rds)])
        X = fet_regression.preproc(np.array(X).reshape(
            1, -1), 1)[0]
        fet_model = joblib.load(file_name)
        self.fet_model_df = pd.DataFrame(10 ** fet_model.predict(X.reshape(1, -1)),
                                         columns=['Qg', 'Cost'])

        # Load the trained model for predicting ['C_oss', 'Vds_meas', 'tau_c','tau_rr'] based on ['V_dss', 'R_ds', 'Q_g', 'Unit_price']
        file_name = 'joblib_files/pdf_dataset_models_chained.joblib'
        # X should look like: ['N', 'P', 'MOSFET', 'SiCFET', 'GaNFET', log10[Vdss], log10[Rds]]
        X = []
        if opt_obj.FET_type == 'N':
            X.extend([1, 0])
        else:
            X.extend([0, 1])
        if opt_obj.fet_tech == 'MOSFET':
            X.extend([1, 0]) # if including SiCFET = [1,0,0]
        elif opt_obj.fet_tech == 'SiCFET':
            X.extend([0, 1, 0])
        else:
            X.extend([0, 1]) # if including SiCFET = [0,0,1]
        X.extend(
            [np.log10(self.Vdss), np.log10(opt_obj.fet_normalizer * self.Rds), np.log10(self.fet_model_df.loc[0, 'Qg']),
             np.log10(self.fet_model_df.loc[0, 'Cost'])])
        X = fet_regression.preproc(np.array(X).reshape(
            1, -1), 1)[0]
        fet_model = joblib.load(file_name)
        self.fet_model_df[['Cossp1', 'Vds_meas', 'tau_c', 'tau_rr']] = 10 ** fet_model.predict(X.reshape(1, -1))[0]

        # Load the trained model for predicting ['Area'] based on ['V_dss', 'R_ds', 'Q_g', 'Unit_price']
        file_name = 'full_dataset_Pack_case.joblib'
        # X should look like: ['N', 'P', 'MOSFET', 'SiCFET', 'GaNFET', log10[Vdss], log10[Rds]]
        X = []
        if opt_obj.FET_type == 'N':
            X.extend([1, 0])
        else:
            X.extend([0, 1])
        if opt_obj.fet_tech == 'MOSFET':
            X.extend([1, 0, 0])
        elif opt_obj.fet_tech == 'SiCFET':
            X.extend([0, 1, 0])
        else:
            X.extend([0, 0, 1])
        X.extend(
            [np.log10(self.Vdss), np.log10(opt_obj.fet_normalizer * self.Rds),
             np.log10(self.fet_model_df.loc[0, 'Cost']), np.log10(self.fet_model_df.loc[0, 'Qg'])])
        X = fet_regression.preproc(np.array(X).reshape(
            1, -1), 1)[0]
        fet_model = joblib.load(file_name)
        self.fet_model_df['Area'] = 10 ** fet_model.predict(X.reshape(1, -1))[0]

        # set the attributes
        [self.Qg, self.Cost, self.Cossp1, self.Vds_meas, self.tau_c, self.tau_rr, self.Area] = self.fet_model_df.loc[
            0, ['Qg', 'Cost', 'Cossp1', 'Vds_meas', 'tau_c', 'tau_rr', 'Area']]

    def compute_Cdsq(self, opt_obj):

        # file_name = '../mosfet_data/joblib_files/' + str(opt_obj.fet_tech) + '_models' + '_' + 'C_ossp1' + '.joblib'
        # fet_model = joblib.load(file_name)[0]
        # # inputs to this model are: (from loss_and_physics.py) ['Unit_price', 'V_dss', 'R_ds', 'Q_g']
        # X = np.array([np.log10(self.Cost), np.log10(self.Vdss),
        #               np.log10(self.fet_normalizer * self.Rds), np.log10(self.Qg)])
        # Cossp1 = 10 ** (fet_model.predict(X.reshape(1, -1)))[0][0] * 10**-12
        # self.C_ossp1 = Cossp1

        gamma_eqn_dict = {'Si_low_volt': [0.0021, 0.251], 'Si_high_volt': [0.000569, 0.579],
                          'GaN_low_volt': [0.00062, 0.355],
                          'GaN_high_volt': [0.000394, 0.353], 'SiC': [0, 0.4509]}
        if opt_obj.fet_tech == 'MOSFET' and self.Vdss <= 200:
            self.category = 'Si_low_volt'
        elif opt_obj.fet_tech == 'MOSFET' and self.Vdss > 200:
            self.category = 'Si_high_volt'
        elif opt_obj.fet_tech == 'GaNFET' and self.Vdss <= 100:
            self.category = 'GaN_low_volt'
        elif opt_obj.fet_tech == 'GaNFET' and self.Vdss <= 200:
            self.category = 'GaN_high_volt'
        elif opt_obj.fet_tech == 'SiCFET':
            self.category = 'SiC'

        self.gamma = gamma_eqn_dict[self.category][0] * self.Vdss + gamma_eqn_dict[self.category][1]


        # return result

    def compute_kT(self, opt_obj):
        kT_eqn_dict = {'Si_low_volt': [0.233, 1.15], 'Si_high_volt': [0.353, 0.827],
                       'GaN_low_volt': [0.066, 1.34],
                       'GaN_high_volt': [0.148, 1.145], 'SiC': [0.572, -0.323]}

        if opt_obj.fet_tech == 'MOSFET' and self.Vdss <= 200:
            self.category = 'Si_low_volt'
        elif opt_obj.fet_tech == 'MOSFET' and self.Vdss > 200:
            self.category = 'Si_high_volt'
        elif opt_obj.fet_tech == 'GaNFET' and self.Vdss <= 100:
            self.category = 'GaN_low_volt'
        elif opt_obj.fet_tech == 'GaNFET' and self.Vdss > 100:
            self.category = 'GaN_high_volt'
        elif opt_obj.fet_tech == 'SiCFET':
            self.category = 'SiC'

        self.kT = kT_eqn_dict[self.category][0] * np.log10(self.Vdss) + kT_eqn_dict[self.category][1]

        # return kT

    # TODO: this should take IF as an input rather than assuming IF is Iout + deltaI (changed, Check w Skye)
    def compute_Qrr_est(self, opt_obj, IF):

        # Estimate Qrr
        # compute Qrr_est based on paper equation (1)

        # self.I_d2_off = opt_obj.converter_df.loc[0, 'Iout'] - (opt_obj.delta_i / opt_obj.delta_i_normalizer) * \
        #                 opt_obj.converter_df.loc[
        #                     0, 'Iout']
        # Qrr_est = self.Qrr * (self.I_d2_off / self.I_F) ** (1 / self.p)
        # self.Qrr_est = Qrr_est

        ### new Qrr estimation approach (see bottom of this file for compute_qrr test case example ###

        # self.Q_rr_ds = self.Q_rr
        # self.t_rr_ds = self.
        self.I_F = IF
        self.didt = 100e6

        # Takes as input: tau_c, tau_rr, and self.I_d2_off
        from scipy.optimize import least_squares
        # Estimate Qrr
        # compute Qrr_est based on paper equation (1)

        # self.I_d2_off = opt_obj.converter_df.loc[0, 'Iout'] - (opt_obj.delta_i / opt_obj.delta_i_normalizer) * \
        #                      opt_obj.converter_df.loc[
        #                          0, 'Iout']
        # Qrr_est = self.Qrr*(self.I_d2_

        # initial conditions assuming IF appx constant prior to diode turnoff
        self.Tm = 1 / (-1 / self.tau_c + 1 / self.tau_rr)  # recompute Tm with predicted tau_c and tau_rr
        self.qE0 = self.I_F * (self.tau_c + self.Tm)
        self.qM0 = self.I_F * self.tau_c

        # qE(t) solution, need to get zero of this
        def fun_T1(x):
            return (-self.tau_c ** 2 * self.didt) * np.exp(-x / self.tau_c) - self.didt * self.tau_c * x \
                   + self.qM0 + self.tau_c ** 2 * self.didt + self.Tm * (self.I_F - self.didt * x)

        T1_min = self.I_F / self.didt
        T1_max = 100 * T1_min
        x0 = (T1_max + T1_min) / 2
        # self.T1 = least_squares(fun_T1, x0 = (T1_max + T1_min)/2, bounds=([T1_min, T1_max]))
        self.T1 = fsolve(fun_T1, x0=x0)

        T0_predicted = self.I_F / self.didt
        I_rr = -(self.I_F - self.didt * self.T1[0])
        qMT1 = I_rr * self.Tm

        Q_rf = qMT1 * self.tau_rr / self.Tm
        Q_rs = I_rr * (self.T1[0] - T0_predicted) / 2

        Q_rr = Q_rf + Q_rs

        # calculated reverse-recovery charge, reverse current amplitude and reverse-recovery time
        self.Q_rr = Q_rr
        self.t_rr = I_rr / self.didt + (4 / 3) * np.log(4) * self.tau_rr
        self.I_rr = I_rr




class OptimizerInductor(OptimizerInit):
    def __init__(self, param_dict, ind_index):
        super().__init__(param_dict)
        self.I_rated = self.inds_df.loc[ind_index]['Current_rating']
        self.I_sat = 1.2 * self.inds_df.loc[ind_index]['Current_rating']
        self.Cost = 0
        self.R_dc = 0
        self.L = 0
        self.Height = 0
        self.Width = 0
        self.Length = 0

    def init_ind(self, opt_obj):
        if opt_obj.opt_var == 'power':
            if opt_obj.plotting_var == 'cost':
                c1 = opt_obj.plotting_val / (2 * opt_obj.total_components)
            else:
                c1 = 1
        elif self.opt_var == 'cost':
            if self.plotting_var == 'power':
                self.x0.append(500000 / self.ind_normalizer)
                return
            else:  # if plotting wrt area
                self.x0.append(100000 / self.ind_normalizer)
                return

            # how to get initial fsw guess from this? could choose a generic one and see how that goes to start
        elif self.opt_var == 'area':
            if self.plotting_var == 'power':
                self.x0.append(100000 / self.ind_normalizer)
                return
            else:
                self.x0.append(100000 / self.ind_normalizer)
                return

        Ind_fsw_prod = opt_obj.inds_df.loc[0]['Ind_fsw']

        file_name = 'joblib_files/fsw_initialization'
        # fet_reg_Energy_model = fet_regression.load_models(['Energy_Cost_product'], file_name)
        # fet_reg_Energy_model = fet_reg_Energy_model.reset_index()

        ind_model = joblib.load(file_name)
        X = fet_regression.preproc(np.array([np.log10(c1), np.log10(self.I_rated)]).reshape(1, -1), opt_obj.degree)
        Ind1 = (10 ** ind_model.predict(np.array(X))) / ((self.I_rated ** 2))

        fsw1=[50000]
        # add fsw as an optimization variable
        opt_obj.x0.append(fsw1[0] / opt_obj.ind_normalizer)
        # add delta_i as an optimization variable w/ initial value = 0.1
        # This is where you change starting guess of delta I
        opt_obj.x0.append(10)
        opt_obj.delta_i = opt_obj.x0[3]

    def predict_ind(self, opt_obj, x):

        file_name = 'inductor_models_chained.joblib'
        ind_model = joblib.load(file_name)
        opt_obj.set_optimization_variables(x)
        # opt_obj.component_computations(x)
        print(x)

        X = fet_regression.preproc(
            np.array([np.log10(self.I_rated), np.log10(self.I_sat), np.log10(self.L)]).reshape(1, -1),
            opt_obj.degree)[0]
        self.ind_model_df = pd.DataFrame(10 ** (ind_model.predict(X.reshape(1, -1))),
                                         columns=['Rdc', 'Unit_price [USD]', 'Area', 'fb [Hz]', 'b', 'Kfe', 'Alpha',
                                                  'Beta', 'Nturns', 'Ac [m^2]', 'Volume [mm^3]', 'Core Volume m^3'])

        [self.R_dc, self.Cost, self.Area, self.f_b, self.b, self.K_fe, self.Alpha, self.Beta, self.N_turns, self.A_c,
         self.Volume, self.Core_volume] = self.ind_model_df.loc[
            0, ['Rdc', 'Unit_price [USD]', 'Area', 'fb [Hz]', 'b', 'Kfe', 'Alpha', 'Beta', 'Nturns', 'Ac [m^2]',
                'Volume [mm^3]', 'Core Volume m^3']]
        self.Volume = self.Volume * 10 ** -9  # converting mm^3 to m^3 here


        # TODO: 4/3 change to fix cap issue?
        opt_obj.ind1.L = (opt_obj.converter_df.loc[0, 'Vin'] * opt_obj.dc) * (
                1 / (opt_obj.fsw * opt_obj.ind_normalizer)) / (
                              2 * (opt_obj.delta_i / opt_obj.delta_i_normalizer) * opt_obj.converter_df.loc[0, 'Iin'])
        opt_obj.ind1.delta_B = opt_obj.ind1.L * opt_obj.converter_df.loc[0, 'Iin'] * (
                opt_obj.delta_i / opt_obj.delta_i_normalizer) / (
                                    opt_obj.ind1.N_turns * opt_obj.ind1.A_c)


    def compute_IGSE(self, opt_obj):

        # now set the parameters to pass to Bailey's script
        # self.delta_B = self.L * opt_obj.converter_df.loc[0, 'Iout'] * (opt_obj.delta_i / opt_obj.delta_i_normalizer) / (
        #             self.N_turns * self.A_c)
        self.B_vector = [-self.delta_B, self.delta_B, -self.delta_B]
        Ts = 1 / (opt_obj.ind_normalizer * opt_obj.fsw)
        self.t_vector = [0, opt_obj.dc * Ts, Ts]
        # self.IGSE_total = 0
        self.IGSE_total = coreloss(self.t_vector, self.B_vector, self.Alpha, self.Beta, self.K_fe)

    def compute_Rac(self, opt_obj):
        from scipy.fft import fft
        self.Rac_total = 0
        # self.Rac_total += self.R_dc * opt_obj.converter_df.loc[0, 'Iout'] ** 2
        # print(f'Rac_total: {self.Rac_total}')
        for k in np.linspace(2, 11, num=10, dtype=int):
            f = k * opt_obj.ind_normalizer * opt_obj.fsw
            Rac = self.R_dc * max(1, (((k - 1) * f / self.f_b) ** self.b))
            # compute the waveshape to use with the fft
            # if change num=1000, how much difference in power loss do we get, how much difference in run-time, look at combos
            running_time = np.linspace(0, 1, num=20)
            current = running_time.copy()
            for i in range(len(running_time)):
                if running_time[i] < opt_obj.dc:
                    current[i] = opt_obj.converter_df.loc[0, 'Iout'] - (opt_obj.delta_i / opt_obj.delta_i_normalizer) * \
                                 opt_obj.converter_df.loc[0, 'Iout'] + (
                                             2 * (opt_obj.delta_i / opt_obj.delta_i_normalizer) *
                                             opt_obj.converter_df.loc[0, 'Iout'] * running_time[i]) / opt_obj.dc
                    # print(f'current[i] less: {current[i]}')
                else:
                    current[i] = opt_obj.converter_df.loc[0, 'Iout'] + (opt_obj.delta_i / opt_obj.delta_i_normalizer) * \
                                 opt_obj.converter_df.loc[0, 'Iout'] - (
                                             2 * (opt_obj.delta_i / opt_obj.delta_i_normalizer) *
                                             opt_obj.converter_df.loc[0, 'Iout'] * (running_time[i] - opt_obj.dc)) / (
                                             1 - opt_obj.dc)
                    # print(f'current[i] more: {current[i]}')


            # get the fast-fourier transform (fft) of the current signal I_rms
            current_harmonics = 2 * abs(fft(current) / len(current))
            # print(f'current harmonics: {current_harmonics}')
            Rac_k = Rac * current_harmonics[k - 1] ** 2 / 2
            # print(f'Rac_k: {Rac_k}')

            self.Rac_total += Rac_k

    def compute_Rac_boost(self, opt_obj):
        from scipy.fft import fft
        self.Rac_total = 0
        # self.Rac_total += self.R_dc * opt_obj.converter_df.loc[0, 'Iout'] ** 2
        # print(f'Rac_total: {self.Rac_total}')
        for k in np.linspace(2, 11, num=10, dtype=int):
            f = (k-1) * opt_obj.ind_normalizer * opt_obj.fsw
            Rac = self.R_dc * max(1, ((f / self.f_b) ** self.b))
            # compute the waveshape to use with the fft
            # if change num=1000, how much difference in power loss do we get, how much difference in run-time, look at combos
            running_time = np.linspace(0, 1, num=20)
            current = running_time.copy()
            # TODO: this is a stopgap measure, need a solution for Iin, Rac compute should be a more general function
            # already defined for QZVS boost
            if self.topology == 'boost':
                opt_obj.converter_df.loc[0, 'Iin'] = opt_obj.converter_df.loc[0, 'Iout'] / (1 - opt_obj.dc)
            for i in range(len(running_time)):
                if running_time[i] < opt_obj.dc:
                    current[i] = opt_obj.converter_df.loc[0, 'Iin'] - (
                                opt_obj.delta_i / opt_obj.delta_i_normalizer) * \
                                 opt_obj.converter_df.loc[0, 'Iin'] + (
                                         2 * (opt_obj.delta_i / opt_obj.delta_i_normalizer) *
                                         opt_obj.converter_df.loc[0, 'Iin'] * running_time[i]) / opt_obj.dc
                    # print(f'current[i] less: {current[i]}')
                else:
                    current[i] = opt_obj.converter_df.loc[0, 'Iin'] + (
                                opt_obj.delta_i / opt_obj.delta_i_normalizer) * \
                                 opt_obj.converter_df.loc[0, 'Iin'] - (
                                         2 * (opt_obj.delta_i / opt_obj.delta_i_normalizer) *
                                         opt_obj.converter_df.loc[0, 'Iin'] * (running_time[i] - opt_obj.dc)) / (
                                         1 - opt_obj.dc)
                    # print(f'current[i] more: {current[i]}')

            # get the fast-fourier transform (fft) of the current signal I_rms
            current_harmonics = 2 * abs(fft(current) / len(current))
            # print(f'current harmonics: {current_harmonics}')
            Rac_k = Rac * current_harmonics[k - 1] ** 2 / 2
            # print(f'Rac_k: {Rac_k}')

            self.Rac_total += Rac_k

        # print('done')

class OptimizerCapacitor(OptimizerInit):
    def __init__(self, param_dict, cap_index):
        super().__init__(param_dict)
        self.V_rated = self.caps_df.loc[cap_index]['Voltage_rating']
        self.V_dc = self.caps_df.loc[cap_index]['Vdc']
        self.Cost = 0
        self.Cap = 0
        self.Area = 0
        self.cap_index = cap_index

    def init_cap(self, opt_obj):
        # First, calculate capacitance, in units of [F]
        # self.compute_capacitance(opt_obj)
        opt_obj.set_optimization_variables(opt_obj.x0)
        # opt_obj.component_computations(opt_obj.x0)
        # Currently setting the temp. coefficient = X7R, as encoded via one-hot in training
        self.temp_coef_enc = 1
        if opt_obj.opt_var == 'power':
            if opt_obj.plotting_var == 'cost':
                c1 = opt_obj.plotting_val / (2 * opt_obj.total_components)
            else:
                c1 = 1
        elif self.opt_var == 'cost':
            if self.plotting_var == 'power':
                self.x0.append(500000 / self.ind_normalizer)
                return
            else:  # if plotting wrt area
                self.x0.append(100000 / self.ind_normalizer)
                return

            # how to get initial fsw guess from this? could choose a generic one and see how that goes to start
        elif self.opt_var == 'area':
            if self.plotting_var == 'power':
                self.x0.append(100000 / self.ind_normalizer)
                return
            else:
                self.x0.append(100000 / self.ind_normalizer)
                return

        file_name = 'joblib_files/cap_area_initialization_models_chained.joblib'
        # inputs: ['Capacitance','Rated_volt','Unit_price']
        # outputs: ['Size']

        cap_model = joblib.load(file_name)
        X = fet_regression.preproc(np.array([self.temp_coef_enc, np.log10(self.Capacitance), np.log10(self.V_rated), np.log10(c1)]).reshape(1, -1), opt_obj.degree)
        cap_area1 = (10 ** cap_model.predict(np.array(X))) # in units of [mm], so cap_normalizer right now

        opt_obj.x0.append(cap_area1[0][0] / opt_obj.cap_normalizer)
        # add cap_area1 as an optimization variable w/ initial value = 0.1
        # opt_obj.x0.append(1)
        # opt_obj.cap_area = opt_obj.x0[4]
        pass

    def predict_cap(self, opt_obj, x):
        # First predict the Cap.@0Vdc based on the Cap@Vdc (here Vdc = Vg or Vout)
        # inputs: (as seen in fet_regression.py dictionaries) ['Vrated [V]','Size','Vdc_meas [V]','Capacitance_at_Vdc_meas [uF]']
        # outputs: ['Capacitance_at_0Vdc [uF]']
        file_name = 'joblib_files/cap_pdf_params_models_chained.joblib'
        cap_model = joblib.load(file_name)

        opt_obj.set_optimization_variables(x)
        # opt_obj.component_computations(x)

        X = fet_regression.preproc(
            np.array([np.log10(self.V_rated), np.log10(opt_obj.x0[self.cap_index + 4]), np.log10(self.V_dc), self.Capacitance*10**6]).reshape(1, -1),
            opt_obj.degree)[0]
        self.cap_model_df = pd.DataFrame(10 ** (cap_model.predict(X.reshape(1, -1))),
                                         columns=['Cap_0Vdc'])

        [self.Cap_0Vdc] = 10**-6 * self.cap_model_df.loc[
            0, ['Cap_0Vdc']]

        # Then predict the cost based on the main_page parameters, and the newly predicted cap@0Vdc
        # inputs: (as seen in fet_regression.py dictionaries) ['Temp_coef_enc', 'Capacitance','Rated_volt','Size']
        # outputs: ['Unit_price']
        file_name = 'joblib_files/cap_main_page_params_models_chained.joblib'
        cap_model = joblib.load(file_name)

        X = fet_regression.preproc(
            np.array([self.temp_coef_enc, np.log10(self.Cap_0Vdc), np.log10(self.V_rated), np.log10(opt_obj.x0[self.cap_index + 4])]).reshape(1, -1),
            opt_obj.degree)[0]
        self.cap_model_df = pd.DataFrame(10 ** (cap_model.predict(X.reshape(1, -1))),
                                         columns=['Cost'])

        [self.Cost] = self.cap_model_df.loc[
            0, ['Cost']]

    # # Compute the required capacitance based on parameters. Currently computing based on buck equations.
    # # 0 is for input capacitor, 1 is for output capacitor.
    # def compute_capacitance(self, opt_obj):
    #     if self.cap_index == 0:
    #         # delta_Vc1 spec is delta_Vc1/Vg <= 5%, call it = 5%.
    #         delta_V = 0.05 * opt_obj.converter_df.loc[0, 'Vin']
    #         self.Capacitance = opt_obj.converter_df.loc[0, 'Iout'] * opt_obj.dc * (1 - opt_obj.dc) / (2 * (opt_obj.x0[2] * opt_obj.ind_normalizer) * delta_V)
    #
    #     if self.cap_index == 1:
    #         # delta_V spec is: delta_V/Vout <= 1%, call it = 1%.
    #         delta_V = 0.01 * opt_obj.converter_df.loc[0, 'Vout']
    #         self.Capacitance = opt_obj.delta_i * (1 / (opt_obj.x0[2] * opt_obj.ind_normalizer)) / (8 * delta_V)



def brute_force_find_test_case():
    # Step 2
    with open('optimizer_obj_test_values', 'rb') as optimizer_obj_file:
        # Step 3
        optimizer_obj = pickle.load(optimizer_obj_file)

    optimizer_obj.make_component_predictions_bf()


def loss_comparison_plotting(param_dict):

    # In power_pred_tot(), the user defines the loss function for their converter. x0 contains the initial starting
    # values for the optimization variables. Throughout, anywhere that is case-specific, make a new case statement and
    # write information in 'elif param_dict['topology'] == '<topology name>'.
    # Case statements must be set for the following:
    # 1. Setting the desired optimization variables based on x0.
    # 2. Computing physics-based loss modeling values, e.g. Cdsq or Qrr, trr.
    # 3. Computing the overall power loss model for all components, and writing the specific loss contributions, e.g.
    #    Rds_loss_Q1 = R^2 * Id1
    #    Q1_loss = Rds_loss_Q1 + Qg_loss_Q1
    #    Overall_loss = Q1_loss + Q2_loss + L1_loss
    # This must only be done for power loss, because cost and area loss assume we simply sum the cost and area contributions
    # of each component.

    # def power_pred_tot(x0, self):
        # x = x0


    warnings.filterwarnings('ignore')
    optimizer_obj = OptimizerInit(param_dict)

    runtime = []
    runtimes = []
    for tech in optimizer_obj.tech_list:
        start = time.time()

        param_dict['test_fet_tech'] = tech
        optimizer_obj.fet_tech = tech
        optimizer_obj.FET_type = 'N'
        optimizer_obj.param_dict = param_dict
        in_range = True
        optimizer_obj.slsqp_points = []
        optimizer_obj.previous_opt_var = optimizer_obj.opt_var_reset
        optimizer_obj.previous_variable_constraint = optimizer_obj.constraint_reset

        optimizer_obj.fet_list = []
        optimizer_obj.ind_list = []
        optimizer_obj.cap_list = []

        optimizer_obj.create_component_lists(param_dict)

        for plotting_val in optimizer_obj.plot_range_list:

            run_status = 1
            while run_status > 0:

                optimizer_obj.plotting_init()
                optimizer_obj.plotting_val = plotting_val

                starttime = time.time()

                # Initialize all components.
                for fet_obj in optimizer_obj.fet_list:
                    fet_obj.init_fet(optimizer_obj)
                for ind_obj in optimizer_obj.ind_list:
                    ind_obj.init_ind(optimizer_obj)
                    ind_obj.N_turns = 1
                    ind_obj.A_c = 1
                for cap_obj in optimizer_obj.cap_list:
                    cap_obj.init_cap(optimizer_obj)

                # It is up to the designer to initialize all their desired optimization variables here. Currently it is
                # set that all transistor Rds' are optimization variables, and fsw and delta_iL, and all capacitor areas.
                optimizer_obj.set_optimization_variables(optimizer_obj.x0)


                # Set the written function as the one seen by the optimization algorithm
                # optimizer_obj.power_pred_tot = power_pred_tot
                # setattr(optimizer_obj, 'power_pred_tot', power_pred_tot)

                # Run the minimization
                optimizer_obj.minimize_fcn()
                endtime = time.time()
                runtimes.append(endtime - starttime)

                # with open('optimizer_obj_test_values', 'wb') as optimizer_obj_file:
                #     # Step 3
                #     pickle.dump(optimizer_obj, optimizer_obj_file)

                # TODO: changed this to force it to store optimization results (should be ==1)
                if optimizer_obj.status:
                    # optimizer_obj.previous_opt_var = optimizer_obj.fun
                    # plotting_val = round(plotting_val, 1)
                    # optimizer_obj.previous_variable_constraint = plotting_val
                    getattr(optimizer_obj, tech + "_overall_points").append((plotting_val, optimizer_obj.fun))
                    # getattr(optimizer_obj, tech + "_freq_points").append((plotting_val, optimizer_obj.fsw))
                    # getattr(optimizer_obj, tech + "_delta_i_points").append((plotting_val, optimizer_obj.delta_i))
                    #
                    # getattr(optimizer_obj, tech + "_Q1_loss_points").append(
                    #     (plotting_val, optimizer_obj.Q1_loss / optimizer_obj.Pout))
                    # getattr(optimizer_obj, tech + "_Q2_loss_points").append(
                    #     (plotting_val, optimizer_obj.Q2_loss / optimizer_obj.Pout))
                    # getattr(optimizer_obj, tech + "_L1_loss_points").append(
                    #     (plotting_val, optimizer_obj.L1_loss / optimizer_obj.Pout))
                    getattr(optimizer_obj, tech + "_total_cost_points").append((plotting_val, optimizer_obj.Cost_tot))
                    getattr(optimizer_obj, tech + "_total_area_points").append((plotting_val, optimizer_obj.Area_tot))
                    getattr(optimizer_obj, tech + "_total_power_points").append(
                        (plotting_val, optimizer_obj.power_tot))

                    with open('optimizer_test_values/optimizer_obj_' + optimizer_obj.topology + '_' + optimizer_obj.fet_tech + '_' + str(
                            plotting_val) + '_' + str(optimizer_obj.area_constraint), 'wb') as optimizer_obj_file:
                        # Step 3
                        # optimizer_obj.power_pred_tot = None
                        pickle.dump(optimizer_obj, optimizer_obj_file)

                    run_status = 0
                else:
                    run_status = 0
                    plotting_val += 0.05 * plotting_val
                    optimizer_obj.previous_variable_constraint = plotting_val
                    # if plotting_val > optimizer_obj.plot_range_list[-1]:
                    #     getattr(optimizer_obj, tech + "_overall_points").append(
                    #         (plotting_val, optimizer_obj.previous_opt_var))
                    #     getattr(optimizer_obj, tech + "_freq_points").append(
                    #         (plotting_val, getattr(optimizer_obj, tech + "_freq_points")[-1][1]))
                    #     getattr(optimizer_obj, tech + "_Q1_loss_points").append(
                    #         (plotting_val, getattr(optimizer_obj, tech + "_Q1_loss_points")[-1][1]))
                    #     getattr(optimizer_obj, tech + "_Q2_loss_points").append(
                    #         (plotting_val, getattr(optimizer_obj, tech + "_Q2_loss_points")[-1][1]))
                    #     getattr(optimizer_obj, tech + "_L1_loss_points").append(
                    #         (plotting_val, getattr(optimizer_obj, tech + "_L1_loss_points")[-1][1]))
                    #     getattr(optimizer_obj, tech + "_total_cost_points").append(
                    #         (plotting_val, getattr(optimizer_obj, tech + "_total_cost_points")[-1][1]))
                    #     getattr(optimizer_obj, tech + "_total_area_points").append(
                    #         (plotting_val, getattr(optimizer_obj, tech + "_total_area_points")[-1][1]))
                    #     getattr(optimizer_obj, tech + "_total_power_points").append(
                    #         (plotting_val, getattr(optimizer_obj, tech + "_total_power_points")[-1][1]))
                    #
                    #     in_range = False
                    #     break
            if optimizer_obj.predict_components:
                optimizer_obj.make_component_predictions()

        end = time.time()
        runtime.append(end - start)

    for tech in optimizer_obj.tech_list:
        plt.plot(*zip(*getattr(optimizer_obj, tech + "_overall_points")), alpha=0.7)

    optimizer_obj.get_visualization_params()
    plt.legend(optimizer_obj.legend, fontsize=optimizer_obj.fontsize)
    plt.xlabel(optimizer_obj.xlabel, fontsize=optimizer_obj.fontsize)
    plt.ylabel(optimizer_obj.ylabel, fontsize=optimizer_obj.fontsize)
    plt.xticks(fontsize=optimizer_obj.fontsize)
    plt.yticks(fontsize=optimizer_obj.fontsize)
    # plt.ylim(0, .15)
    # plt.scatter(3, 0.0272, s=100)
    # pt1 = plt.annotate('Selected design', (3.3, .026), size=15)
    plt.grid(color='lightgrey', linewidth=1, alpha=0.4)
    plt.title(optimizer_obj.title)
    plt.show()
    # to use axis:
    # fig, ax = plt.subplots()
    # for tech in optimizer_obj.tech_list:
    #     ax.plot(*zip(*getattr(optimizer_obj, tech + "_overall_points")), alpha=0.7)
    # ax.set_xlim(0)
    # ax.set_xticklabels(['0', '', '0.05', '', '0.1', '', '0.15', '', '0.2'])
    # optimizer_obj.get_visualization_params()
    # ax.legend(optimizer_obj.legend, fontsize=15)
    # ax.set_xlabel(optimizer_obj.xlabel, fontsize=12)
    # ax.set_ylabel(optimizer_obj.ylabel, fontsize=12)
    # ax.tick_params(axis='both', which='major', labelsize=12)

    fig, ax = plt.subplots()
    i = 0
    colorlist = ['royalblue', 'forestgreen']
    for tech in optimizer_obj.tech_list:
        ax.plot(*zip(*getattr(optimizer_obj, tech + "_overall_points")), alpha=0.7, linewidth=3, color=colorlist[i])
        i += 1
    ax.set_xlim(0)
    ax.set_xticklabels(['0', '', '0.05', '', '0.1', '', '0.15', '', '0.2'], fontsize=12)
    optimizer_obj.get_visualization_params()
    ax.legend(optimizer_obj.legend, fontsize=15)
    ax.set_xlabel(optimizer_obj.xlabel, fontsize=12)
    ax.set_ylabel(optimizer_obj.ylabel, fontsize=12)
    ax.tick_params(axis='both', which='major', labelsize=12)

    # lab1 = plt.annotate('A',(2.96,.049), xytext = (2.9,.057))
    # lab2 = plt.annotate('B', (8.96, .0188), xytext=(8.9, .025))
    # p3 = plt.scatter(9, .0188, c='red', s=30, zorder=2)
    # p4 = plt.scatter(9, .01558, c='red', s=30, zorder=2)
    # p1 = plt.scatter(3, .028, c='red', s=30, zorder=2)
    # p2 = plt.scatter(3,.049,c='red',s=30,zorder=2)

def manual_preproc(X, poly_degree):
    if type(X) == np.array:
        a = X[:, 0]
        b = X[:, 1]
    elif type(X) == np.ndarray:
        a = X[0]
        b = X[1]
        return np.array([1, a ** poly_degree, b ** poly_degree, a ** (0.5 * poly_degree), b ** (0.5 * poly_degree),
                         1 / (a ** poly_degree), 1 / (b ** poly_degree)]).reshape(1, -1)
    else:
        a = X.loc[:, 'V_dss']
        b = X.loc[:, 'Unit price']

    array = np.ones((len(X), 1))
    array = np.c_[array, a ** poly_degree]
    array = np.c_[array, b ** poly_degree]
    array = np.c_[array, a ** (0.5 * poly_degree)]
    array = np.c_[array, b ** (0.5 * poly_degree)]
    array = np.c_[array, 1 / (a ** poly_degree)]
    array = np.c_[array, 1 / (b ** poly_degree)]

    return array


def converter_optimizer():
    param_dict = {'opt_var': 'power', 'plotting_var': 'cost', 'set_constraint': 'area', 'set_constraint_val': 400,
                  'example_num': 1, 'tech_list': ['MOSFET', 'GaNFET'], 'num_points': 2,
                  'plotting_range': [4.4, 10], 'predict_components': False}
    # param_dict = {'opt_var': 'power', 'plotting_var': 'cost', 'set_constraint': 'area', 'set_constraint_val': 300,
    #               'example_num': 4, 'tech_list': ['MOSFET', 'GaNFET'], 'num_points': 8,
    #               'plotting_range': [2.5, 6], 'predict_components': False}
    # ,'Pout_range': [20, 60]}
    loss_comparison_plotting(param_dict=param_dict)


# New Qrr estimation approach. Will incorporate into the actual optimization
# takes as input the Qrr from datasheet, IF from the datasheet, and t_rr from datasheet. Assumes S=100A/uS.
def Qrr_est_new(Q_rr_ds, t_rr_ds, I_F_ds):
    didt_ds = 100 * 10 ** 6  # initialize didt_ds to an assumed value of 100A/us
    Q_rr_ds = Q_rr_ds * 10 ** -9
    t_rr_ds = t_rr_ds * 10 ** -9
    tau_c = 0  # initialize tau_c
    tau_rr = 0  # initialize tau_rr
    t_rr = 0  # initialize non-datasheet (ds) t_rr

    # calculate I_rr so that t_rr matches t_rr_ds

    # upper limit for I_rr
    I_rr_max = np.sqrt(2 * didt_ds * Q_rr_ds * 0.95)
    I_rr = I_rr_max
    (t_rr, tau_c, tau_rr) = calculate_time_constants(Q_rr_ds, I_rr, didt_ds, I_F_ds)  # nested function

    if (t_rr > t_rr_ds):
        error = 1  # indicates unable to solve for time constants
        return

    I_rr_min = np.sqrt(2 * didt_ds * Q_rr_ds * 0.05)
    I_rr = I_rr_min
    (t_rr, tau_c, tau_rr) = calculate_time_constants(Q_rr_ds, I_rr, didt_ds, I_F_ds)
    if (t_rr < t_rr_ds):
        error = 2  # indicates unable to solve for time constants
        return

    t_rr_error = 1  # initialize error in calculated t_rr
    while t_rr_error > 0.001:  # perform bi-section search until error in t_rr is less than 0.1%
        I_rr = (I_rr_min + I_rr_max) / 2
        (t_rr, tau_c, tau_rr) = calculate_time_constants(Q_rr_ds, I_rr, didt_ds, I_F_ds)

        if t_rr < t_rr_ds:
            I_rr_max = I_rr
        else:
            I_rr_min = I_rr

        t_rr_error = np.abs(t_rr - t_rr_ds) / t_rr_ds

    Tm = 1 / (-1 / tau_c + 1 / tau_rr)  # by definition of Tm

    # store and return calculated time constants
    Tm = Tm  # turn this into self.Tm = Tm
    tau_c = tau_c
    tau_rr = tau_rr
    t_rr = t_rr  # calculated t_rr, should be within 0.1% of t_rr_ds
    I_rr = I_rr # should be close to I_rr_datasheet if given
    return tau_c, tau_rr


def compute_Qrr_new():
    # we want to return Q_rr and t_rr, because in P_loss,rr, all other quantities (Vg, IF, fs) are known
    # this will be a step after tau_c and tau_rr have been predicted, and are attributes of self. Will also take Tm as
    # an input, and IF is calculated as Iout-delta_i, and didt assumed = 100A/uS

    (Q_rr, t_rr) = reverse_recovery_charge()


def calculate_time_constants(Q_rr_ds, I_rr, didt_ds, I_F_ds):
    from scipy.optimize import fsolve

    Q_rf = Q_rr_ds - I_rr ** 2 / 2 / didt_ds
    tau_rr = Q_rf / I_rr
    T1 = (I_rr + I_F_ds) / didt_ds

    def fun_tau_c(x, I_rr, didt_ds, tau_rr,
                  T1):  # will need to take self as the first argument if inside optimization function
        # solve this equation for x
        return I_rr - didt_ds * (x - tau_rr) * (1 - np.exp(-T1 / x))

    tau_c = fsolve(fun_tau_c, 10 ** -9, args=(I_rr, didt_ds, tau_rr, T1))
    t_rr = I_rr / didt_ds + (4 / 3) * np.log(4) * tau_rr
    return (t_rr, tau_c[0], tau_rr)  # just make this self.t_rr in the actual function


# def reverse_recovery_charge():


class Qrr_test_case():
    def __init__(self):
        self.Q_rr_ds = 7.4 * 10 ** -6
        self.t_rr_ds = 468 * 10 ** -9
        self.I_F_ds = 35
        # self.tau_c = 5.12e-7
        # self.tau_rr = 8.65e-8
        self.tau_c = 2.5e-8
        self.tau_rr = 6.1e-9
        self.I_F = 4.5
        self.didt = 100e6

    def compute_Qrr_est(self):
        from scipy.optimize import least_squares
        # Estimate Qrr
        # compute Qrr_est based on paper equation (1)

        # self.I_d2_off = opt_obj.converter_df.loc[0, 'Iout'] - (opt_obj.delta_i / opt_obj.delta_i_normalizer) * \
        #                      opt_obj.converter_df.loc[
        #                          0, 'Iout']
        # Qrr_est = self.Qrr*(self.I_d2_

        # initial conditions assuming IF appx constant prior to diode turnoff
        self.Tm = 1 / (-1 / self.tau_c + 1 / self.tau_rr)  # recompute Tm with predicted tau_c and tau_rr
        self.qE0 = self.I_F * (self.tau_c + self.Tm)
        self.qM0 = self.I_F * self.tau_c

        # qE(t) solution, need to get zero of this
        def fun_T1(x):
            return (-self.tau_c ** 2 * self.didt) * np.exp(-x / self.tau_c) - self.didt * self.tau_c * x \
                   + self.qM0 + self.tau_c ** 2 * self.didt + self.Tm * (self.I_F - self.didt * x)

        T1_min = self.I_F / self.didt
        T1_max = 100 * T1_min
        x0 = (T1_max + T1_min)/2
        # self.T1 = least_squares(fun_T1, x0 = (T1_max + T1_min)/2, bounds=([T1_min, T1_max]))
        self.T1 = fsolve(fun_T1, x0=x0)

        T0_predicted = self.I_F / self.didt
        I_rr = -(self.I_F - self.didt * self.T1[0])
        qMT1 = I_rr * self.Tm

        Q_rf = qMT1 * self.tau_rr / self.Tm
        Q_rs = I_rr * (self.T1[0] - T0_predicted)/2

        Q_rr = Q_rf + Q_rs

        # calculated reverse-recovery charge, reverse current amplitude and reverse-recovery time
        self.Q_rr = Q_rr
        self.t_rr = I_rr / self.didt + (4 / 3) * np.log(4) * self.tau_rr
        self.I_rr = I_rr

if __name__ == '__main__':
    # Qrr_test = Qrr_test_case()
    # Qrr_test.compute_Qrr_est()
    # Qrr_est_new(29, 30, 10)
    # Qrr_est_new(7.4*10**-6, 468*10**-9, 35)
    # Qrr_est_new(13.2 * 10 ** -9, 21.1 * 10 ** -9, 10)
    # brute_force_find_test_case()

    # In param_dict, the user defines the information about the design, independent of information about the topology
    param_dict_buck = {'opt_var': 'power', 'plotting_var': 'cost', 'set_constraint': 'area', 'set_constraint_val': 800,
                  'example_num': 1, 'tech_list': ['MOSFET', 'GaNFET'], 'test_fet_tech': 'MOSFET', 'num_points': 2,
                  'plotting_range': [5, 11], 'predict_components': False, 'topology': 'buck'}
    # ,'Pout_range': [20, 60]}
    param_dict_boost = {'opt_var': 'power', 'plotting_var': 'cost', 'set_constraint': 'area', 'set_constraint_val': 800,
                       'example_num': 1, 'tech_list': ['MOSFET', 'GaNFET'], 'test_fet_tech': 'MOSFET', 'num_points': 10,
                       'plotting_range': [5, 11], 'predict_components': False, 'topology': 'boost'}
    param_dict_QZVS_boost = {'opt_var': 'power', 'plotting_var': 'cost', 'set_constraint': 'area', 'set_constraint_val': 750,
                             'example_num': 1, 'tech_list': ['MOSFET', 'GaNFET'], 'test_fet_tech': 'MOSFET', 'num_points': 10,
                             'plotting_range': [10,35], 'predict_components': False, 'topology': 'QZVS_boost'}
    param_dict_2L_inverter = {'opt_var': 'power', 'plotting_var': 'cost', 'set_constraint': 'area', 'set_constraint_val': 250,
                             'example_num': 1, 'tech_list': ['MOSFET', 'GaNFET'], 'test_fet_tech': 'MOSFET', 'num_points': 10,
                             'plotting_range': [8/4.0, 15/4.0], 'predict_components': False, 'topology': '2L_inverter'}
    param_dict_microinverter_combined = {'opt_var': 'power', 'plotting_var': 'cost', 'set_constraint': 'area', 'set_constraint_val': 1000,
                             'example_num': 1, 'tech_list': ['MOSFET', 'GaNFET'], 'test_fet_tech': 'MOSFET', 'num_points': 2,
                             'plotting_range': [50, 100], 'predict_components': False, 'topology': 'microinverter_combined'}

    active_param_dict = param_dict_microinverter_combined
    loss_comparison_plotting(param_dict=active_param_dict)

    for fet_tech in active_param_dict['tech_list']:
        costs = np.linspace(active_param_dict['plotting_range'][0], active_param_dict['plotting_range'][1], active_param_dict['num_points'])
        for cost_constraint in costs:
            with open('optimizer_test_values/optimizer_obj_' + active_param_dict['topology'] +'_' + fet_tech + '_' + str(
                cost_constraint) + '_' + str(active_param_dict['set_constraint_val']), 'rb') as optimizer_obj_file:
                print(fet_tech, cost_constraint)
                result = pickle.load(optimizer_obj_file)
                print('next')