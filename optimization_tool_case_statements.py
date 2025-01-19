'''

This file contains the topology-specific functions that are used by the optimization tool found in
fet_optimization_chained_wCaps.py.

'''
import numpy as np
import scipy
from fet_optimization_chained_wCaps import loss_comparison_plotting

def get_params_separate(self):
    if self.example_num == 1 and self.topology == 'buck':
        self.converter_data_dict = {'Vin': [48], 'Vout': [12], 'Iout': [5]}
        if self.fet_tech == 'MOSFET' or self.fet_tech == 'SiCFET':
            self.fet_data_dict = {'Vdss': [60, 60], 'Vgate': [10, 10]}
        elif self.fet_tech == 'GaNFET':
            self.fet_data_dict = {'Vdss': [60, 60], 'Vgate': [5, 5]}
        self.ind_data_dict = {'Current_rating': [10], 'Ind_fsw': [4.6], 'Current_sat': [1.2 * 10]}
        self.cap_data_dict = {
            'Voltage_rating': [1.4 * self.converter_data_dict['Vin'][0], 1.4 * self.converter_data_dict['Vout'][0]],
            'Vdc': [self.converter_data_dict['Vin'][0], self.converter_data_dict['Vout'][0]]}
        self.bounds_dict = {'fsw_min': 0.01, 'fsw_max': 10, 'delta_i_min': 0.1, 'delta_i_max': 7,
                            'comp_selection_delta_i_min': 0.1, 'comp_selection_delta_i_max': 7}
        self.normalizer_dict = {'ind_normalizer': 10**6, 'fet_normalizer': 10**-2}
        self.initialization_dict = {'Rds_initialization': 0.01, 'fsw_initialization': 100000, 'delta_i_initialization': 1}

    if self.example_num == 1 and self.topology == 'boost':
        self.converter_data_dict = {'Vin': [12], 'Vout': [48], 'Iout': [2]}
        if self.fet_tech == 'MOSFET' or self.fet_tech == 'SiCFET':
            self.fet_data_dict = {'Vdss': [60, 60], 'Vgate': [10, 10]}
        elif self.fet_tech == 'GaNFET':
            self.fet_data_dict = {'Vdss': [60, 60], 'Vgate': [5, 5]}
        self.ind_data_dict = {'Current_rating': [15], 'Ind_fsw': [4.6], 'Current_sat': [1.2 * 15]}
        self.cap_data_dict = {
            'Voltage_rating': [1.4 * self.converter_data_dict['Vin'][0], 1.4 * self.converter_data_dict['Vout'][0]],
            'Vdc': [self.converter_data_dict['Vin'][0], self.converter_data_dict['Vout'][0]]}
        self.bounds_dict = {'fsw_min': 0.01, 'fsw_max': 10, 'delta_i_min': 0.1, 'delta_i_max': 7, 'comp_selection_delta_i_min': 0.1, 'comp_selection_delta_i_max': 7}
        self.normalizer_dict = {'ind_normalizer': 10**6, 'fet_normalizer': 10**-2}
        self.initialization_dict = {'Rds_initialization': 0.01, 'fsw_initialization': 100000, 'delta_i_initialization': 1}


        # self.constraints_data_dict = {'area_constraint': [400], 'power_constraint': [
        #     0.6 * self.converter_data_dict['Vout'][0] * self.converter_data_dict['Iout'][0]],
        #                               'example_cost_range': [(2, 20)],
        #                               'example_power_range': [(0.25, 0.02)],
        #                               'weights': [[0, 1, 0]]}
        # self.dIL = 0.1 * self.converter_data_dict['Vout'][0] * self.converter_data_dict['Iout'][0] / \
        #            self.converter_data_dict['Vin'][0]
        # self.dVo = 0.1 * self.converter_data_dict['Vout'][0]

    if self.example_num == 2 and self.topology == 'boost':
        self.converter_data_dict = {'Vin': [12], 'Vout': [20], 'Iout': [4]}
        if self.fet_tech == 'MOSFET' or self.fet_tech == 'SiCFET':
            self.fet_data_dict = {'Vdss': [20, 30], 'Vgate': [10, 10]}
        elif self.fet_tech == 'GaNFET':
            self.fet_data_dict = {'Vdss': [20, 30], 'Vgate': [5, 5]}
        self.ind_data_dict = {'Current_rating': [8], 'Ind_fsw': [4.6], 'Current_sat': [1.2 * 8]}
        self.cap_data_dict = {
            'Voltage_rating': [1.4 * self.converter_data_dict['Vin'][0], 1.4 * self.converter_data_dict['Vout'][0]],
            'Vdc': [self.converter_data_dict['Vin'][0], self.converter_data_dict['Vout'][0]]}
        self.bounds_dict = {'fsw_min': 0.01, 'fsw_max': 10, 'delta_i_min': 0.1, 'delta_i_max': 7, 'comp_selection_delta_i_min': 0.1, 'comp_selection_delta_i_max': 7}
        self.normalizer_dict = {'ind_normalizer': 10**6, 'fet_normalizer': 10**-2}
        self.initialization_dict = {'Rds_initialization': 0.01, 'fsw_initialization': 100000, 'delta_i_initialization': 1}


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

    # Note: energy storage cap between boost and inverter is sized separately
    if self.example_num == 1 and self.topology == 'microinverter_combined':
        # DC bus voltage, RMS output voltage  and RMS output current
        self.converter_data_dict = {'Vin': [32.94], 'Iin': [13.05], 'Vbus': [60], 'Vout': [40],
                                    'Iout': [15.27 / np.sqrt(2)]}
        if self.fet_tech == 'MOSFET' or self.fet_tech == 'SiCFET':
            self.fet_data_dict = {'Vdss': [100, 100, 100, 100, 100, 100], 'Vgate': [10, 10, 10, 10, 10, 10]}
        elif self.fet_tech == 'GaNFET':
            self.fet_data_dict = {'Vdss': [100, 100, 100, 100, 100, 100], 'Vgate': [5, 5, 5, 5, 5, 5]}
        self.ind_data_dict = {'Current_rating': [30], 'Ind_fsw': [4.6], 'Current_sat': [1.2 * 30]}
        # self.cap_data_dict = {'Voltage_rating': [1.4*self.converter_data_dict['Vin'][0]],
        #                       'Vdc': [self.converter_data_dict['Vin'][0]]}
        self.cap_data_dict = {}
        self.bounds_dict = {'fsw_min': 7.5, 'fsw_max': 7.5, 'delta_i_min': 10, 'delta_i_max': 10, 'comp_selection_delta_i_min': 9, 'comp_selection_delta_i_max': 13}
        self.normalizer_dict = {'ind_normalizer': 10**4, 'fet_normalizer': 10**-3}
        self.initialization_dict = {'Rds_initialization': 0.001, 'fsw_initialization': 50000, 'delta_i_initialization': 10}


def set_optimization_variables_separate(self, x):
    if self.topology == 'buck':

        if (isinstance(x, np.ndarray) or isinstance(x, list)):
            # Make sure none of the optimization variables are <= 0 or will hit convergence issues
            for index in range(len(x)):
                if x[index] < 0.000000001:
                    x[index] = 0.0000000001

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
            self.cap2.Capacitance = (self.delta_i / self.delta_i_normalizer) * self.converter_df.loc[0, 'Iout'] * (
                        1 / (x[2] * self.ind_normalizer)) / (8 * delta_V)
            self.ind1.L = ((self.converter_df.loc[0, 'Vin'] - self.converter_df.loc[0, 'Vout']) * self.dc) * (
                    1 / (self.fsw * self.ind_normalizer)) / (
                                  2 * (self.delta_i / self.delta_i_normalizer) * self.converter_df.loc[0, 'Iout'])
            self.ind1.delta_B = self.ind1.L * self.converter_df.loc[0, 'Iout'] * (
                    self.delta_i / self.delta_i_normalizer) / (
                                        self.ind1.N_turns * self.ind1.A_c)
            self.IL = self.converter_df.loc[0, 'Iout']
            self.I_d2_off = self.converter_df.loc[0, 'Iout'] - (self.delta_i / self.delta_i_normalizer) * \
                            self.converter_df.loc[0, 'Iout']
            self.fet2.I_F = self.I_d2_off

        if not (isinstance(x, np.ndarray) or isinstance(x, list)):
            delta_V = 0.01 * self.converter_df.loc[0, 'Vout']
            self.cap2.Capacitance = (self.delta_i / self.delta_i_normalizer) * self.converter_df.loc[0, 'Iout'] * (
                    1 / (self.fsw * self.ind_normalizer)) / (8 * delta_V)
            # print('done')

    elif self.topology == 'boost':
        if not (isinstance(x, np.ndarray) or isinstance(x, list)):
            delta_V = 0.01 * self.converter_df.loc[0, 'Vout']
            self.cap2.Capacitance = self.converter_df.loc[0, 'Iout'] * self.dc * (
                    1 / (self.fsw * self.ind_normalizer)) / (2 * delta_V)
        if (isinstance(x, np.ndarray) or isinstance(x, list)):
            # Make sure none of the optimization variables are <= 0 or will hit convergence issues
            for index in range(len(x)):
                if x[index] < 0.000000001:
                    x[index] = 0.0000000001
            self.fet1.Rds = x[0]
            self.fet2.Rds = x[1]
            self.fsw = x[2]
            self.delta_i = x[3]
            if len(x) > 5:
                self.cap1.Area = x[4]
                self.cap2.Area = x[5]

            delta_V = 0.05 * self.converter_df.loc[0, 'Vin']
            self.cap1.Capacitance = self.converter_df.loc[0, 'Iout'] * self.dc / (
                    2 * (x[2] * self.ind_normalizer) * delta_V)
            # delta_V spec is: delta_V/Vout <= 1%, call it = 1%.
            delta_V = 0.01 * self.converter_df.loc[0, 'Vout']
            # self.cap2.Capacitance = self.converter_df.loc[0, 'Iin'] * (self.delta_i / self.delta_i_normalizer) * self.dc * (
            #         1 / (x[2] * self.ind_normalizer)) / (8 * delta_V)
            self.cap2.Capacitance = self.converter_df.loc[0, 'Iout'] * self.dc * (1 / (x[2] * self.ind_normalizer)) / (
                        2 * delta_V)
            self.converter_df.loc[0, 'Iin'] = self.converter_df.loc[0, 'Iout'] / (1 - self.dc)
            self.ind1.L = (self.converter_df.loc[0, 'Vin'] * self.dc) * (
                    1 / (self.fsw * self.ind_normalizer)) / (
                                  2 * (self.delta_i / self.delta_i_normalizer) * self.converter_df.loc[0, 'Iin'])
            self.ind1.delta_B = self.ind1.L * self.converter_df.loc[0, 'Iin'] * (
                    self.delta_i / self.delta_i_normalizer) / (
                                        self.ind1.N_turns * self.ind1.A_c)
            self.IL = self.converter_df.loc[0, 'Iin']
            self.I_d2_off = self.converter_df.loc[0, 'Iout'] - (self.delta_i / self.delta_i_normalizer) * \
                            self.converter_df.loc[0, 'Iout']
            self.fet2.I_F = self.I_d2_off

    elif self.topology == 'microinverter_combined':
        if not (isinstance(x, np.ndarray) or isinstance(x, list)):
            pass

        if (isinstance(x, np.ndarray) or isinstance(x, list)):
            # Make sure none of the optimization variables are <= 0 or will hit convergence issues
            for index in range(len(x)):
                if x[index] < 0.000000001:
                    x[index] = 0.0000000001

            self.fet1.Rds = x[0]
            self.fet2.Rds = x[1]
            self.fet3.Rds = x[2]
            self.fet4.Rds = x[2]
            self.fet5.Rds = x[2]
            self.fet6.Rds = x[2]
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
            self.IL = self.converter_df.loc[0, 'Iin']

from fet_optimization_chained_wCaps import  OptimizerInit, OptimizerFet, OptimizerInductor, OptimizerCapacitor
def create_component_lists_separate(self, param_dict):
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

    if param_dict['topology'] == 'microinverter_combined':
        self.fet1 = OptimizerFet(param_dict, 0)
        self.fet2 = OptimizerFet(param_dict, 1)
        self.fet3 = OptimizerFet(param_dict, 2)
        self.fet4 = OptimizerFet(param_dict, 3, make_new_opt_var=False)
        self.fet5 = OptimizerFet(param_dict, 4, make_new_opt_var=False)
        self.fet6 = OptimizerFet(param_dict, 5, make_new_opt_var=False)


        self.ind1 = OptimizerInductor(param_dict, 0)
        # self.cap1 = OptimizerCapacitor(param_dict, 0)

        self.fet_list.extend([self.fet1, self.fet2, self.fet3, self.fet4, self.fet5, self.fet6])
        self.ind_list.extend([self.ind1])
        # self.cap_list.extend([self.cap1])

        self.dc = 1 - (self.converter_df.loc[0, 'Vin'] / self.converter_df.loc[0, 'Vbus'])

def print_unnormalized_results(self, rescobyla):
    if (self.topology == 'buck') or (self.topology == 'boost'):
        print("Un-normalized results:",
              self.fet_normalizer * rescobyla.x[0],
              self.fet_normalizer * rescobyla.x[1],
              self.ind_normalizer * rescobyla.x[2],
              rescobyla.x[3] / self.delta_i_normalizer,
              self.cap_normalizer * rescobyla.x[4],
              self.cap_normalizer * rescobyla.x[5],
              )

    elif self.topology == 'microinverter_combined':
        print("Un-normalized results:",
              self.fet_normalizer * rescobyla.x[0],
              self.fet_normalizer * rescobyla.x[1],
              self.fet_normalizer * rescobyla.x[2],
              self.ind_normalizer * rescobyla.x[3],
              rescobyla.x[4] / self.delta_i_normalizer,
              )

def power_pred_tot_separate(self, x):
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
            self.set_optimization_variables(x)
        for cap_obj in self.cap_list:
            cap_obj.predict_cap(self, x)

    # 2. It is up to the designer to write the specific loss model here, depending on the specific components used
    # in their design. Object attributes can be created for any given component in the design by writing:
    # self.<component>.<attribute> = <value>. If the designer wants to change the physics-based loss models, they
    # may do so here, but will have to write another object function and call that.

    if self.topology == 'buck':

        self.fet1.I_d2_off = self.converter_df.loc[0, 'Iout'] - (self.delta_i / self.delta_i_normalizer) * \
                             self.converter_df.loc[
                                 0, 'Iout']

        # predict Cossp1 for the fet
        self.fet1.compute_Cdsq(self)
        self.fet1.Cdsq = \
            scipy.integrate.quad(
                lambda Vds: (1 / self.converter_data_dict['Vin'][0]) * self.fet1.Cossp1 * 10 ** -12 * (
                        0.1 * self.fet1.Vdss / Vds) ** self.fet1.gamma, 0, self.converter_data_dict['Vin'][0])[0]

        # Add the info to the df about kT, the factor to multiply Ron by based on Vdss
        self.fet1.compute_kT(self)
        self.fet2.compute_kT(self)

        # Compute Qrr,est based on I_F prediction and Qrr prediction
        if self.fet_tech != 'GaNFET':
            self.fet2.compute_Qrr_est(self)

        # compute values related to P_Qoff loss
        self.fet1.I_off_Q1 = self.converter_df.loc[0, 'Iout'] + (self.delta_i / self.delta_i_normalizer) * \
                             self.converter_df.loc[0, 'Iout']

        # self.fet_vals.append(fet_model_df.to_dict())

        power = 0

        self.I_Q1 = np.sqrt(self.dc) * self.converter_df.loc[0, 'Iout'] * np.sqrt(
            1 + (self.delta_i / self.delta_i_normalizer) ** 2 / 3)
        self.I_Q2 = np.sqrt(1 - self.dc) * self.converter_df.loc[0, 'Iout'] * np.sqrt(
            1 + (self.delta_i / self.delta_i_normalizer) ** 2 / 3)

        # compute the IGSE including Vc
        self.ind1.compute_IGSE(self)

        # compute Rac values
        self.ind1.compute_Rac(self)

    # 3. It is up to the designer to write the loss model for their topology. Separate losses by specific loss
    #    contribution and by component.
    if self.topology == 'buck':
        self.fet1.Rds_loss = self.I_Q1 ** 2 * self.fet1.kT * self.fet_normalizer * self.fet1.Rds
        self.fet1.Qg_loss = self.ind_normalizer * self.fsw * self.fet1.Qg * self.fet1.Vgate

        self.fet1.toff_loss = self.fet1.I_off_Q1 ** 2 * self.fet1.t_off ** 2 * self.ind_normalizer * self.fsw / (
                48 * (10 ** -12 * self.fet1.Cossp1))
        self.fet1.Cdsq_loss = self.ind_normalizer * self.fsw * self.fet1.Cdsq * self.converter_df.loc[
            0, 'Vin'] ** 2
        self.Q1_loss = self.fet1.Rds_loss + self.fet1.Qg_loss + self.fet1.toff_loss + self.fet1.Cdsq_loss

        self.fet2.Rds_loss = self.I_Q2 ** 2 * self.fet2.kT * self.fet_normalizer * self.fet2.Rds
        self.fet2.Qg_loss = self.ind_normalizer * self.fsw * self.fet2.Qg * self.fet2.Vgate

        self.Q2_loss = self.fet2.Rds_loss + self.fet2.Qg_loss

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

        R_pcb = 10 * 10 ** -3
        R_temp_multiplier = 0.75 * 2.3 / 1.724
        ILrms = self.converter_df.loc[0, 'Iout'] * np.sqrt(1 + (self.delta_i / self.delta_i_normalizer) ** 2 / 3)
        self.PCB_loss = R_pcb * ILrms ** 2 * R_temp_multiplier

        self.power_tot = self.Q1_loss + self.Q2_loss + self.L1_loss + self.PCB_loss

    if self.topology == 'boost':
        self.fet1.I_d2_off = self.converter_df.loc[0, 'Iin'] - (self.delta_i / self.delta_i_normalizer) * \
                             self.converter_df.loc[
                                 0, 'Iin']

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
        self.fet1.I_off_Q1 = self.converter_df.loc[0, 'Iin'] + (self.delta_i / self.delta_i_normalizer) * \
                             self.converter_df.loc[0, 'Iin']

        # self.fet_vals.append(fet_model_df.to_dict())

        power = 0

        self.I_Q1 = np.sqrt(self.dc) * self.converter_df.loc[0, 'Iin']
        self.I_Q2 = np.sqrt(1 - self.dc) * self.converter_df.loc[0, 'Iin']

        # compute the IGSE including Vc
        self.ind1.compute_IGSE(self)

        # compute Rac values
        self.ind1.compute_Rac(self)

    if self.topology == 'boost':
        ####### NEW LOSS MODEL: ###########
        # self.compute_loss() --> use this generalized method to compute the loss, so that can pull from the same
        # equation in the component selection method, see compute_kT for how
        self.fet1.Rds_loss = self.I_Q1 ** 2 * self.fet1.kT * self.fet_normalizer * self.fet1.Rds
        self.fet1.Qg_loss = self.ind_normalizer * self.fsw * self.fet1.Qg * self.fet1.Vgate

        self.fet1.toff_loss = self.fet1.I_off_Q1 ** 2 * self.fet1.t_off ** 2 * self.ind_normalizer * self.fsw / (
                48 * (10 ** -12 * self.fet1.Cossp1))
        self.Q1_loss = self.fet1.Rds_loss + self.fet1.Qg_loss + self.fet1.toff_loss

        self.fet2.Rds_loss = self.I_Q2 ** 2 * self.fet2.kT * self.fet_normalizer * self.fet2.Rds
        self.fet2.Qg_loss = self.ind_normalizer * self.fsw * self.fet2.Qg * self.fet2.Vgate
        self.fet2.Cdsq_loss = self.ind_normalizer * self.fsw * self.fet2.Cdsq * self.converter_df.loc[
            0, 'Vout'] ** 2

        self.Q2_loss = self.fet2.Rds_loss + self.fet2.Qg_loss + self.fet2.Cdsq_loss

        if self.fet_tech != 'GaNFET':
            self.fet2.Qrr_loss = self.converter_df.loc[0, 'Vout'] * (
                    self.fet2.Q_rr + (self.fet2.t_rr * (self.converter_df.loc[0, 'Iin'] -
                                                        ((self.converter_df.loc[
                                                            0, 'Iin']) * self.delta_i / self.delta_i_normalizer)))) \
                                 * self.ind_normalizer * self.fsw
            self.Q2_loss += self.fet2.Qrr_loss

        self.ind1.Rdc_loss = (self.converter_df.loc[0, 'Iout'] / (1 - self.dc)) ** 2 * self.ind1.R_dc
        self.ind1.Rac_loss = self.ind1.Rac_total
        self.ind1.IGSE_loss = self.ind1.Core_volume * self.ind1.IGSE_total
        self.L1_loss = self.ind1.Rdc_loss + self.ind1.Rac_loss + self.ind1.IGSE_loss

        R_pcb = 10 * 10 ** -3
        R_temp_multiplier = 0.75 * 2.3 / 1.724
        ILrms = self.converter_df.loc[0, 'Iin'] * np.sqrt(1 + (self.delta_i / self.delta_i_normalizer) ** 2 / 3)
        self.PCB_loss = R_pcb * ILrms ** 2 * R_temp_multiplier

        self.power_tot = self.Q1_loss + self.Q2_loss + self.L1_loss + self.PCB_loss
        print(f'power: {self.power_tot}')

    if self.topology == 'microinverter_combined':

        # Add the info to the df about kT, the factor to multiply Ron by based on Vdss
        self.fet1.compute_kT(self)
        self.fet2.compute_kT(self)

        # compute values related to P_Qoff loss - assumes 100% ripple so
        self.fet1.I_off_Q1 = self.converter_df.loc[0, 'Iin'] * (1+self.delta_i/self.delta_i_normalizer)

        power = 0

        self.I_Q1 = self.converter_df.loc[0, 'Iin'] * np.sqrt(self.dc) * np.sqrt(
            1 + (1 / 3) * (self.delta_i / self.delta_i_normalizer / self.converter_df.loc[0, 'Iin']) ** 2)
        self.I_Q2 = self.converter_df.loc[0, 'Iin'] * np.sqrt(1 - self.dc) * np.sqrt(
            1 + (1 / 3) * (self.delta_i / self.delta_i_normalizer / self.converter_df.loc[0, 'Iin']) ** 2)

        # compute the IGSE including Vc
        self.ind1.compute_IGSE(self)

        # compute Rac values
        self.ind1.compute_Rac(self)

        # equation in the component selection method, see compute_kT for how
        self.fet1.Rds_loss = self.fet_normalizer * self.fet1.Rds * self.fet1.kT * (self.I_Q1) ** 2
        self.fet1.Qg_loss = self.ind_normalizer * self.fsw * self.fet1.Qg * self.fet1.Vgate
        self.fet1.toff_loss = self.fet1.I_off_Q1 ** 2 * self.fet1.t_off ** 2 * self.ind_normalizer * self.fsw / (
                48 * (10 ** -12 * self.fet2.Cossp1))
        self.Q1_loss = self.fet1.Rds_loss + self.fet1.Qg_loss + self.fet1.toff_loss

        self.fet2.Rds_loss = self.fet_normalizer * self.fet2.Rds * self.fet1.kT * (self.I_Q2) ** 2
        self.fet2.Qg_loss = self.ind_normalizer * self.fsw * self.fet2.Qg * self.fet2.Vgate
        self.Q2_loss = self.fet2.Rds_loss + self.fet2.Qg_loss

        # Using RMS value of current
        self.ind1.Rdc_loss = self.converter_df.loc[0, 'Iin'] ** 2 * self.ind1.R_dc
        self.ind1.Rac_loss = self.ind1.Rac_total
        self.ind1.IGSE_loss = self.ind1.Core_volume * self.ind1.IGSE_total
        self.L1_loss = self.ind1.Rdc_loss + self.ind1.Rac_loss + self.ind1.IGSE_loss

        self.fet3.compute_kT(self)
        # Line frequency
        F = 60
        # TODO: microinverter hardcoded at 20 kHz, do not change
        inv_fsw = 20e3
        Ns = int(inv_fsw / F)

        total_E_Qrr = 0
        total_E_Off = 0
        total_E_On = 0

        for i in range(0, int(Ns / 2)):
            Iac = np.sqrt(2) * self.converter_df.loc[0, 'Iout'] * np.sin(2 * np.pi * (i / Ns))
            if self.fet_tech != 'GaNFET':
                self.fet3.I_d2_off = Iac
                self.fet3.I_F = Iac
                self.fet3.compute_Qrr_est(self)
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
        self.fet3.Rds_loss = 2 * self.converter_df.loc[0, 'Iout'] ** 2 * self.fet3.kT * self.fet3.Rds * self.fet_normalizer
        self.Q3_loss = self.fet3.toff_loss + self.fet3.Qrr_loss + self.fet3.Cdsq_loss + self.fet3.Qg_loss + self.fet3.Rds_loss

        self.power_tot = self.Q1_loss + self.Q2_loss + self.L1_loss + self.Q3_loss # include if we want to include the inductor losses, but for now we do not
        # self.power_tot = self.Q1_loss + self.Q2_loss + self.Q3_loss

    # print(x)
    # print(self.power_tot)
    if self.fet_tech == 'MOSFET':
        self.power_divider = 10
    if self.fet_tech == 'GaNFET':
        self.power_divider = 100

    return self.power_tot / self.power_divider

if __name__ == '__main__':
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
                  'example_num': 1, 'tech_list': ['MOSFET', 'GaNFET'], 'test_fet_tech': 'MOSFET', 'num_points': 2,
                  'plotting_range': [5, 11], 'predict_components': False, 'topology': 'buck'}
    # ,'Pout_range': [20, 60]}
    param_dict_boost = {'opt_var': 'power', 'plotting_var': 'cost', 'set_constraint': 'area', 'set_constraint_val': 800,
                       'example_num': 1, 'tech_list': ['MOSFET','GaNFET'], 'test_fet_tech': 'MOSFET', 'num_points': 2,
                       'plotting_range': [5,9], 'predict_components': False, 'topology': 'boost'}
    # param_dict_boost = {'opt_var': 'power', 'plotting_var': 'cost', 'set_constraint': 'area', 'set_constraint_val': 800,
    #                     'example_num': 2, 'tech_list': ['MOSFET', 'GaNFET'], 'test_fet_tech': 'MOSFET', 'num_points': 2,
    #                     'plotting_range': [5, 9], 'predict_components': False, 'topology': 'boost'}
    loss_comparison_plotting(param_dict=param_dict_buck)