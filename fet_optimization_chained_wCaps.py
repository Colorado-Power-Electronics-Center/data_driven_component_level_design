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
        # self.fet_tech = param_dict['test_fet_tech']
        self.fet_tech = 'MOSFET'
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
        self.ind_normalizer = self.normalizer_dict[
            'ind_normalizer']  # was 10 ** 6 for ind and 10 ** -2 for fets, for buck/boost examples
        self.fet_normalizer = self.normalizer_dict['fet_normalizer']
        self.delta_i_normalizer = 10 ** 1
        self.cap_normalizer = 10 ** 1
        self.power_normalizer = 10 ** 1

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
        self.legend = self.fet_tech
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
        #
        # self.area_tot = area
        # # print('area: %f' % area)
        return self.Area_tot

    def area_bounds_fcn(self, x):
        if self.plotting_var == 'area':
            self.area_constraint = self.plotting_val

        return (self.area_constraint - self.area_pred_tot(x)) / 10

    def get_params(self):
        from optimization_tool_case_statements import get_params_separate
        get_params_separate(self)

    def set_optimization_variables(self, x):
        from optimization_tool_case_statements import set_optimization_variables_separate
        set_optimization_variables_separate(self, x)

    def create_component_lists(self, param_dict):
        from optimization_tool_case_statements import create_component_lists_separate
        create_component_lists_separate(self, param_dict)

    def power_pred_tot(self, x):
        from optimization_tool_case_statements import power_pred_tot_separate
        computed_power = power_pred_tot_separate(self, x)
        return computed_power

    def power_bounds_fcn(self, x):
        if self.plotting_var == 'power':
            self.power_constraint = self.plotting_val

        # print('power fraction: %f' % (self.power_pred_tot(x) * self.power_divider / self.Pout))
        return (self.power_constraint - (self.power_pred_tot(x) * self.power_divider / self.Pout))

    def power_bounds_fcn1(self, x):
        # if self.plotting_var == 'power':
        #     self.power_constraint = self.plotting_val
        #
        # # print('power fraction: %f' % (self.power_pred_tot(x) * self.power_divider / self.Pout))
        # return (self.power_constraint - (self.power_pred_tot(x) * self.power_divider / self.Pout))
        if self.plotting_var == 'power':
            self.power_constraint = self.plotting_val

        # print('power fraction: %f' % (self.power_pred_tot(x) * self.power_divider / self.Pout))
        return -(self.power_constraint - (self.power_pred_tot(x) * self.power_divider / self.Pout))

    def power_bounds_fcn2(self, x):
        if self.plotting_var == 'power':
            self.power_constraint = self.plotting_val

        # print('power fraction: %f' % (self.power_pred_tot(x) * self.power_divider / self.Pout))
        return -(self.power_constraint - (self.power_pred_tot(x) * self.power_divider / self.Pout))

    def Rds1_bounds_minfcn(self, x):
        return x[0] - 0.005

    def Rds2_bounds_minfcn(self, x):
        return x[1] - 0.005

    def Rds1_bounds_maxfcn(self, x):
        return 300 - x[0]

    def Rds2_bounds_maxfcn(self, x):
        return 300 - x[1]

    def fsw_bounds_minfcn(self, x):
        return self.fsw - self.bounds_dict['fsw_min']
        # swap x[2] w/ self.fsw? and same for delta_i, if setting during set_optimization_variables?
        # return x[3] - 2

    def fsw_bounds_maxfcn(self, x):
        return self.bounds_dict['fsw_max'] - self.fsw
        # return 100 - x[3]

    def delta_i_bounds_minfcn(self, x):
        return self.delta_i - self.bounds_dict['delta_i_min']
        # return x[4] - 10

    def delta_i_bounds_maxfcn(self, x):
        return self.bounds_dict['delta_i_max'] - self.delta_i
        # return 10 - x[4]

    def Ron_comp_fcn(self, x):
        return x[0] - x[1]

    def minimize_fcn(self):
        results = dict()
        # con_cobyla = [{'type': 'ineq', 'fun': exec('self.' + self.plotting_var + '_bounds_fcn')},
        #               {'type': 'ineq', 'fun': exec('self.' + self.set_constraint + '_bounds_fcn')},
        #               {'type': 'ineq', 'fun': self.Rds1_bounds_minfcn},
        #               {'type': 'ineq', 'fun': self.Rds2_bounds_minfcn},
        #               {'type': 'ineq', 'fun': self.fsw_bounds_maxfcn}]
        #
        # setattr(self, self.plotting_var+"_bounds_fcn", self.constraint_fcn_plotting)
        # setattr(self, self.set_constraint+"_bounds_fcn", self.constraint_fcn_set)

        # If the user wants to set additional bounds, e.g. on the optimization variables, they may do so here.
        con_cobyla = [{'type': 'ineq', 'fun': self.bounds_table[self.plotting_var]},
                      {'type': 'ineq', 'fun': self.bounds_table[self.set_constraint]},
                      # {'type': 'ineq', 'fun': self.Rds1_bounds_minfcn},
                      # {'type': 'ineq', 'fun': self.Rds2_bounds_minfcn},
                      {'type': 'ineq', 'fun': self.fsw_bounds_minfcn},
                      {'type': 'ineq', 'fun': self.fsw_bounds_maxfcn},
                      {'type': 'ineq', 'fun': self.delta_i_bounds_minfcn},
                      {'type': 'ineq', 'fun': self.delta_i_bounds_maxfcn},

                      ]

        # con_cobyla = [{'type': 'ineq', 'fun': self.bounds_table[self.plotting_var]},
        #               {'type': 'ineq', 'fun': self.bounds_table[self.set_constraint]},
        #               # {'type': 'ineq', 'fun': self.Rds1_bounds_minfcn},
        #               # {'type': 'ineq', 'fun': self.Rds2_bounds_minfcn},
        #               {'type': 'ineq', 'fun': self.fsw_bounds_minfcn},
        #               {'type': 'ineq', 'fun': self.fsw_bounds_maxfcn},
        #               {'type': 'ineq', 'fun': self.delta_i_bounds_minfcn},
        #               {'type': 'ineq', 'fun': self.delta_i_bounds_maxfcn}
        #               ]

        ####### may need to put back in for plotting against power
        # if self.plotting_var == 'power' or self.set_constraint == 'power':
        #     con_cobyla.append({'type': 'ineq', 'fun': self.bounds_table["power2"]})

        self.x0 = []
        # Make predictions for all components based on the optimization variables.
        # These functions are independent of the loss models, and only use the ML models.
        for fet_obj in self.fet_list:
            fet_obj.init_fet(self)
        for ind_obj in self.ind_list:
            ind_obj.init_ind(self)
            ind_obj.N_turns = 1
            ind_obj.A_c = 1
        for cap_obj in self.cap_list:
            cap_obj.init_cap(self)
        self.set_optimization_variables(self.x0)

        self.optimization_table = {"power": self.power_pred_tot, "cost": self.cost_pred_tot, "area": self.area_pred_tot}

        rescobyla = minimize(self.optimization_table[self.opt_var], np.array(self.x0),  # args=self,
                             method='COBYLA', constraints=con_cobyla,
                             # options={'disp': True, 'maxiter': 1000, 'eps': 10 ** -4, 'catol': 10 ** -3,
                             #          'ftol': 10 ** -3, 'xtol': 10 ** -3}
                            options = {'disp': True, 'maxiter': 1000, 'eps': 10 ** -6, 'catol': 10 ** -6,
                   'ftol': 10 ** -6, 'xtol': 10 ** -6},


                             # options={'disp': True, 'tol': 10e-5, 'rhobeg': 10 ** -1}

                             # options={'disp': True, 'maxiter': 1000, 'ftol': 10**-11, 'xtol': 10**-1, 'tol': 10e-4, 'eps': 10 ** -1, 'catol': 10 ** -1}
                             )
        # ftol: 10e-7, catol: 10**-1
        # unknown keys: ftol, xtol, eps. rhobeg set = 1.0, rhoend = tol
        print(rescobyla.x)

        from optimization_tool_case_statements import print_unnormalized_results
        print_unnormalized_results(self, rescobyla)
        print('cost constraint: %f' % self.plotting_val)
        print('power: %f, cost: %f, area: %d' % (self.power_tot / self.Pout, self.Cost_tot, self.Area_tot))
        self.fun = rescobyla.fun
        if self.opt_var == 'power':
            self.fun = rescobyla.fun * self.power_divider / self.Pout

        if (rescobyla.status == 2 or rescobyla.status == 4) and rescobyla.maxcv < 0.1:
            self.status = 1
        else:
            self.status = rescobyla.status
        self.x_result = rescobyla.x

        return results


class OptimizerFet(OptimizerInit):
    def __init__(self, param_dict, fet_index, make_new_opt_var=True):
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
        self.make_new_opt_var = make_new_opt_var

    def init_fet(self, opt_obj):

        if not self.make_new_opt_var:
            return

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
                # self.I_Q1 = np.sqrt(self.dc) * self.converter_df.loc[0, 'Iout']
                # self.I_Q2 = np.sqrt(1 - self.dc) * self.converter_df.loc[0, 'Iout']
                # Rds = 0.1 * self.plotting_val * self.Pout / (self.I_Q1 ** 2)
                Rds = [self.initialization_dict['Rds_initialization']]
                opt_obj.x0.append(Rds[0] / opt_obj.fet_normalizer)
                return
            else:
                c1 = 1

            # do something where the initial Rds value is just some fraction of the power loss constraint?
        elif self.opt_var == 'area':
            # for plotting as function of changing power constraint
            if self.plotting_var == 'power':
                # self.I_Q1 = np.sqrt(self.dc) * self.converter_df.loc[0, 'Iout']
                # self.I_Q2 = np.sqrt(1 - self.dc) * self.converter_df.loc[0, 'Iout']
                # Rds = 0.3 * self.plotting_val * self.Pout / (self.I_Q1 ** 2)
                Rds = 0.5
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

        Rds = [self.initialization_dict['Rds_initialization']]
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
            X.extend([1, 0])  # if including SiCFET = [1,0,0]
        elif opt_obj.fet_tech == 'SiCFET':
            X.extend([0, 1, 0])
        else:
            X.extend([0, 1])  # if including SiCFET = [0,0,1]
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

    def compute_Qrr_est(self, opt_obj):

        # see bottom of file for test case on Qrr estimation

        self.didt = 100e6

        # Takes as input: tau_c, tau_rr, and self.I_d2_off
        from scipy.optimize import least_squares
        # Estimate Qrr
        # compute Qrr_est based on paper equation (1)

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
                # opt_obj.x0.append(500000 / self.ind_normalizer)
                fsw1 = [self.initialization_dict['fsw_initialization']]
                # add fsw as an optimization variable
                opt_obj.x0.append(fsw1[0] / opt_obj.ind_normalizer)
                # add delta_i as an optimization variable w/ initial value = 0.1
                opt_obj.x0.append(self.initialization_dict['delta_i_initialization'])
                opt_obj.delta_i = opt_obj.x0[3]
                return
            else:  # if plotting wrt area
                opt_obj.x0.append(100000 / self.ind_normalizer)
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
        fsw1 = Ind_fsw_prod / Ind1

        fsw1 = [self.initialization_dict['fsw_initialization']]
        # add fsw as an optimization variable
        opt_obj.x0.append(fsw1[0] / opt_obj.ind_normalizer)
        # add delta_i as an optimization variable w/ initial value = 0.1
        opt_obj.x0.append(self.initialization_dict['delta_i_initialization'])
        opt_obj.delta_i = opt_obj.x0[3]

    def predict_ind(self, opt_obj, x):
        # file_name = '../mosfet_data/joblib_files/inductor_models_' + 'Balanced' + 'Opt' + '_Dimension.joblib'
        # ind_model = joblib.load(file_name)[0]
        # self.L = (opt_obj.converter_df.loc[0, 'Vin'] - opt_obj.converter_df.loc[0, 'Vout']) * opt_obj.dc * (1/(opt_obj.fsw * opt_obj.ind_normalizer)) / (2 * opt_obj.delta_i/opt_obj.delta_i_normalizer)
        # # self.ind1 = self.inds_df.loc[ind_index]['Ind_fsw'] / (self.ind_normalizer * x[ind_index + self.offset])
        # X = fet_regression.preproc(
        #     np.array([np.log10(self.L), np.log10(self.I_rated)]).reshape(1, -1),
        #     opt_obj.degree)[0]
        # ind_model_df = pd.DataFrame(10 ** (ind_model.predict(X.reshape(1, -1))), columns=['Cost', 'DCR', 'Area'])

        file_name = 'inductor_models_chained.joblib'
        ind_model = joblib.load(file_name)
        opt_obj.set_optimization_variables(x)

        # opt_obj.component_computations(x)
        print(x)

        # self.ind1 = self.inds_df.loc[ind_index]['Ind_fsw'] / (self.ind_normalizer * x[ind_index + self.offset])

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

        # self.ind_vals.append(self.ind_model_df.to_dict())

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

        running_time = np.linspace(0, 1, num=20)
        current = running_time.copy()
        for i in range(len(running_time)):
            if running_time[i] < opt_obj.dc:
                current[i] = opt_obj.IL - (opt_obj.delta_i / opt_obj.delta_i_normalizer) * \
                             opt_obj.IL + (
                                     2 * (opt_obj.delta_i / opt_obj.delta_i_normalizer) *
                                     opt_obj.IL * running_time[i]) / opt_obj.dc
                # print(f'current[i] less: {current[i]}')
            else:
                current[i] = opt_obj.IL + (opt_obj.delta_i / opt_obj.delta_i_normalizer) * \
                             opt_obj.IL - (
                                     2 * (opt_obj.delta_i / opt_obj.delta_i_normalizer) *
                                     opt_obj.IL * (running_time[i] - opt_obj.dc)) / (
                                     1 - opt_obj.dc)
                # print(f'current[i] more: {current[i]}')

        # get the fast-fourier transform (fft) of the current signal I_rms
        current_harmonics = 2 * abs(fft(current) / len(current))

        # print(f'Rac_total: {self.Rac_total}')
        for k in np.linspace(2, 11, num=10, dtype=int):
            f = (k - 1) * opt_obj.ind_normalizer * opt_obj.fsw
            # f = k * opt_obj.ind_normalizer * opt_obj.fsw
            Rac = max(self.R_dc, (self.R_dc * (f / self.f_b) ** self.b))
            # compute the waveshape to use with the fft
            # if change num=1000, how much difference in power loss do we get, how much difference in run-time, look at combos

            # print(f'current harmonics: {current_harmonics}')
            Pac_k = Rac * current_harmonics[k - 1] ** 2 / 2
            # print(f'Rac_k: {Rac_k}')

            self.Rac_total += Pac_k

        print('done')


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
        opt_obj.ind1.N_turns = 1
        opt_obj.ind1.A_c = 1
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
                opt_obj.x0.append(500000 / self.ind_normalizer)
                return
            else:  # if plotting wrt area
                opt_obj.x0.append(100000 / self.ind_normalizer)
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
        X = fet_regression.preproc(
            np.array([self.temp_coef_enc, np.log10(self.Capacitance), np.log10(self.V_rated), np.log10(c1)]).reshape(1,
                                                                                                                     -1),
            opt_obj.degree)
        cap_area1 = (10 ** cap_model.predict(np.array(X)))  # in units of [mm], so cap_normalizer right now

        cap_area1 = [[10]]
        opt_obj.x0.append(cap_area1[0][0] / opt_obj.cap_normalizer)
        # add cap_area1 as an optimization variable w/ initial value = 0.1
        # opt_obj.x0.append(1)
        # opt_obj.cap_area = opt_obj.x0[4]

    def predict_cap(self, opt_obj, x):
        # First predict the Cap.@0Vdc based on the Cap@Vdc (here Vdc = Vg or Vout)
        # inputs: (as seen in fet_regression.py dictionaries) ['Vrated [V]','Size','Vdc_meas [V]','Capacitance_at_Vdc_meas [uF]']
        # outputs: ['Capacitance_at_0Vdc [uF]']
        file_name = 'joblib_files/cap_pdf_params_models_chained.joblib'
        cap_model = joblib.load(file_name)

        opt_obj.set_optimization_variables(x)

        # opt_obj.component_computations(x)

        X = fet_regression.preproc(
            np.array([np.log10(self.V_rated), np.log10(opt_obj.x0[self.cap_index + 4]), np.log10(self.V_dc),
                      np.log10(self.Capacitance * 10 ** 6)]).reshape(1, -1),
            opt_obj.degree)[0]
        self.cap_model_df = pd.DataFrame(10 ** (cap_model.predict(X.reshape(1, -1))),
                                         columns=['Cap_0Vdc'])

        [self.Cap_0Vdc] = 10 ** -6 * self.cap_model_df.loc[
            0, ['Cap_0Vdc']]

        # Then predict the cost based on the main_page parameters, and the newly predicted cap@0Vdc
        # inputs: (as seen in fet_regression.py dictionaries) ['Temp_coef_enc', 'Capacitance','Rated_volt','Size']
        # outputs: ['Unit_price']
        file_name = 'joblib_files/cap_main_page_params_models_chained.joblib'
        cap_model = joblib.load(file_name)

        X = fet_regression.preproc(
            np.array([self.temp_coef_enc, np.log10(self.Cap_0Vdc), np.log10(self.V_rated),
                      np.log10(opt_obj.x0[self.cap_index + 4])]).reshape(1, -1),
            opt_obj.degree)[0]
        self.cap_model_df = pd.DataFrame(10 ** (cap_model.predict(X.reshape(1, -1))),
                                         columns=['Cost'])

        [self.Cost] = self.cap_model_df.loc[
            0, ['Cost']]


def loss_comparison_plotting(param_dict, unit_test=False):
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

        # Set all components used in the design, using the associated component objects. It is up to the designer to
        # create indices for their desired components, based on how they understand them to be in their design.
        # Transistors: optimizer_obj.fetX = OptimizerFet(param_dict, X-1)
        # Inductors:   optimizer_obj.indX = OptimizerInductor(param_dict, X-1)
        # Capacitors:  optimizer_obj.capX = OptimizerCapacitor(param_dict, X-1)
        # Then, the designer has to extend the associated component list with each of their components:
        # optimizer_obj.fet_list.extend([optimizer_obj.fetX, optimizer_obj.fet(X+1), optimizer_obj.fet(X+2)])
        optimizer_obj.create_component_lists(param_dict)

        for plotting_val in optimizer_obj.plot_range_list:
            # if not in_range:
            #     break
            # if plotting_val < optimizer_obj.previous_variable_constraint:
            #     continue
            run_status = 1
            while run_status > 0:

                optimizer_obj.plotting_init()
                optimizer_obj.plotting_val = plotting_val

                starttime = time.time()

                # Initialize all components.
                # for fet_obj in optimizer_obj.fet_list:
                #     fet_obj.init_fet(optimizer_obj)
                # for ind_obj in optimizer_obj.ind_list:
                #     ind_obj.init_ind(optimizer_obj)
                #     ind_obj.N_turns = 1
                #     ind_obj.A_c = 1
                # for cap_obj in optimizer_obj.cap_list:
                #     cap_obj.init_cap(optimizer_obj)
                #
                # # It is up to the designer to initialize all their desired optimization variables here. Currently it is
                # # set that all transistor Rds' are optimization variables, and fsw and delta_iL, and all capacitor areas.
                # optimizer_obj.set_optimization_variables(optimizer_obj.x0)

                # Run the minimization
                optimizer_obj.minimize_fcn()
                endtime = time.time()
                runtimes.append(endtime - starttime)

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
                    if unit_test:
                        return optimizer_obj

                    with open(
                            'optimizer_test_values/optimizer_obj_' + optimizer_obj.topology + '_' + optimizer_obj.fet_tech + '_' + str(
                                    plotting_val) + '_' + str(optimizer_obj.area_constraint),
                            'wb') as optimizer_obj_file:
                        # Step 3
                        # optimizer_obj.power_pred_tot = None
                        pickle.dump(optimizer_obj, optimizer_obj_file)

                    run_status = 0
                else:
                    plotting_val += 0.05 * plotting_val
                    optimizer_obj.previous_variable_constraint = plotting_val

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
    plt.show()
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
    I_rr = I_rr  # should be close to I_rr_datasheet if given
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
        self.tau_c = 1.72e-8
        self.tau_rr = 5.83e-9
        self.I_F = 1.865
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


if __name__ == '__main__':
    # deleting change 7000
    # Qrr_test = Qrr_test_case()
    # Qrr_test.compute_Qrr_est()
    # Qrr_est_new(13, 22.1, 10)
    # Qrr_est_new(7.4*10**-6, 468*10**-9, 35)
    # Qrr_est_new(13.2 * 10 ** -9, 21.1 * 10 ** -9, 10)
    # brute_force_find_test_case()
    param_dict = {'opt_var': 'power', 'plotting_var': 'cost', 'set_constraint': 'area', 'set_constraint_val': 200,
                  'example_num': 1, 'tech_list': ['MOSFET', 'GaNFET'], 'num_points': 15,
                  'plotting_range': [2, 10], 'predict_components': False}
    # param_dict = {'opt_var': 'cost', 'plotting_var': 'power', 'set_constraint': 'area', 'set_constraint_val': 200,
    #               'example_num': 1, 'tech_list': ['MOSFET','GaNFET'], 'num_points': 15,
    #               'plotting_range': [.008, .2], 'predict_components': False}
    # param_dict = {'opt_var': 'area', 'plotting_var': 'power', 'set_constraint': 'cost', 'set_constraint_val': 15,
    #               'example_num': 1, 'tech_list': ['MOSFET', 'GaNFET'], 'num_points': 15,
    #               'plotting_range': [.012, .18], 'predict_components': False}
    # param_dict = {'opt_var': 'area', 'plotting_var': 'cost', 'set_constraint': 'power', 'set_constraint_val': .1,
    #               'example_num': 1, 'tech_list': ['MOSFET', 'GaNFET'], 'num_points': 15,
    #               'plotting_range': [1, 8], 'predict_components': False}

    # In param_dict, the user defines the information about the design, independent of information about the topology
    param_dict_buck = {'opt_var': 'power', 'plotting_var': 'cost', 'set_constraint': 'area', 'set_constraint_val': 800,
                       'example_num': 1, 'tech_list': ['MOSFET','GaNFET'], 'num_points': 15,
                       'plotting_range': [5, 11], 'predict_components': False, 'topology': 'buck'}
    # ,'Pout_range': [20, 60]}
    param_dict_boost = {'opt_var': 'power', 'plotting_var': 'cost', 'set_constraint': 'area', 'set_constraint_val': 800,
                        'example_num': 1, 'tech_list': ['MOSFET'], 'num_points': 15,
                        'plotting_range': [5, 11], 'predict_components': False, 'topology': 'boost'}
    # param_dict_boost = {'opt_var': 'power', 'plotting_var': 'cost', 'set_constraint': 'area', 'set_constraint_val': 800,
    #                     'example_num': 2, 'tech_list': ['MOSFET', 'GaNFET'], 'test_fet_tech': 'MOSFET', 'num_points': 2,
    #                     'plotting_range': [5, 9], 'predict_components': False, 'topology': 'boost'}
    param_dict_microinverter_combined = {'opt_var': 'power', 'plotting_var': 'cost', 'set_constraint': 'area',
                                         'set_constraint_val': 1000,
                                         'example_num': 1, 'tech_list': ['MOSFET'],
                                         'num_points': 8,
                                         'plotting_range': [10.67, 14.67], 'predict_components': False,
                                         'topology': 'microinverter_combined'}
    # fun change 5!!
    param_dict_buck_revised = {'opt_var': 'cost', 'plotting_var': 'power', 'set_constraint': 'area', 'set_constraint_val': 800,
                       'example_num': 1, 'tech_list': ['MOSFET','GaNFET'], 'num_points': 3,
                       'plotting_range': [0.05, 0.5], 'predict_components': False, 'topology': 'buck'}
    loss_comparison_plotting(param_dict=param_dict_buck_revised)
