import numpy as np
from fet_optimization_chained_wCaps import OptimizerInit
from component_selection import predict_components, compare_combinations
import pandas as pd

# Set the combo components to match what is used throughout the equations. Finally, set delta_i.
def set_combo_variables(comp_obj, component_combo2):
    if (comp_obj.opt_obj.topology == 'buck') or (comp_obj.opt_obj.topology == 'boost'):
        comp_obj.opt_obj.fet1 = component_combo2.combo[0]
        comp_obj.opt_obj.fet2 = component_combo2.combo[1]
        comp_obj.opt_obj.ind1 = component_combo2.combo[2]
        comp_obj.opt_obj.cap1 = component_combo2.combo[3]
        comp_obj.opt_obj.cap2 = component_combo2.combo[4]

        component_combo2.fet1 = component_combo2.combo[0]
        component_combo2.fet2 = component_combo2.combo[1]
        component_combo2.ind1 = component_combo2.combo[2]
        component_combo2.cap1 = component_combo2.combo[3]
        component_combo2.cap2 = component_combo2.combo[4]

        comp_obj.opt_obj.delta_i = component_combo2.ind1.delta_i

    elif comp_obj.opt_obj.topology == 'microinverter_combined':
        comp_obj.opt_obj.fet1 = component_combo2.combo[0]
        comp_obj.opt_obj.fet2 = component_combo2.combo[1]
        comp_obj.opt_obj.fet3 = component_combo2.combo[2]
        comp_obj.opt_obj.ind1 = component_combo2.combo[3]

        component_combo2.fet1 = component_combo2.combo[0]
        component_combo2.fet2 = component_combo2.combo[1]
        component_combo2.fet3 = component_combo2.combo[2]
        component_combo2.ind1 = component_combo2.combo[3]

        comp_obj.opt_obj.delta_i = component_combo2.ind1.delta_i


# Equations for computing attributes of each inductor in combo
def set_inductor_attributes(indX, opt_obj):
    if opt_obj.topology == 'buck':

        indX.delta_i = (opt_obj.converter_df.loc[0, 'Vin'] * (1 - opt_obj.dc) * opt_obj.dc) * (
                1 / (opt_obj.fsw * opt_obj.ind_normalizer)) / (
                               2 * (indX.L / opt_obj.delta_i_normalizer) * opt_obj.converter_df.loc[
                           0, 'Iout'])

        indX.delta_B = indX.L * opt_obj.converter_df.loc[0, 'Iout'] * (
                indX.delta_i / opt_obj.delta_i_normalizer) / (
                               indX.N_turns * indX.A_c)

        # TODO: figure out best place to update IL, as well as any Iin-related things for boost
        # opt_obj.IL =

    elif opt_obj.topology == 'boost':
        opt_obj.converter_df.loc[0, 'Iin'] = opt_obj.converter_df.loc[0, 'Iout'] / (1 - opt_obj.dc)
        indX.delta_i = (opt_obj.converter_df.loc[0, 'Vin'] * opt_obj.dc) * (
                1 / (opt_obj.fsw * opt_obj.ind_normalizer)) / (
                               2 * (indX.L / opt_obj.delta_i_normalizer) * opt_obj.converter_df.loc[
                           0, 'Iin'])

        indX.delta_B = indX.L * opt_obj.converter_df.loc[0, 'Iin'] * (
                indX.delta_i / opt_obj.delta_i_normalizer) / (
                               indX.N_turns * indX.A_c)

    elif opt_obj.topology == 'microinverter_combined':
        # this is where delta i is set (normalized, and as a proportion of IL)
        indX.delta_i = (opt_obj.converter_df.loc[0, 'Vin'] * opt_obj.dc) * (
                1 / (opt_obj.fsw * opt_obj.ind_normalizer)) / (
                               2 * (indX.L / opt_obj.delta_i_normalizer) * opt_obj.converter_df.loc[
                           0, 'Iin'])

        indX.delta_B = indX.L * opt_obj.converter_df.loc[0, 'Iin'] * (
                indX.delta_i / opt_obj.delta_i_normalizer) / (
                               indX.N_turns * indX.A_c)

    return indX

# Any arrays that need to be set to further filter component combos, making arrays for both the component values of each
# component and the required component values given the combo
def make_original_arrays(comp_obj):
    if comp_obj.opt_obj.topology == 'buck':
        comp_obj.C2_component_arr = np.array([item.cap2.Cap for item in comp_obj.combo_list])
        comp_obj.C2_required_arr = np.array([item.C2_tot for item in comp_obj.combo_list])

    elif comp_obj.opt_obj.topology == 'boost':
        comp_obj.C2_component_arr = np.array([item.cap2.Cap for item in comp_obj.combo_list])
        comp_obj.C2_required_arr = np.array([item.C2_tot for item in comp_obj.combo_list])

# Creating a new array to combine with the overall power array for which combos meet that requirement
def make_new_arrays(comp_obj):
    if comp_obj.opt_obj.topology == 'buck':
        comp_obj.new_C2_arr = comp_obj.C2_component_arr >= 0.5 * comp_obj.C2_required_arr

    elif comp_obj.opt_obj.topology == 'boost':
        comp_obj.new_C2_arr = comp_obj.C2_component_arr >= 1 * comp_obj.C2_required_arr

# Combine the arrays in make_new_arrays() to return a valid overall array
def make_valid_arrays(comp_obj):
    if comp_obj.opt_obj.topology == 'buck':
        comp_obj.valid_arr = np.logical_and(comp_obj.valid_arr, comp_obj.new_C2_arr)

    elif comp_obj.opt_obj.topology == 'boost':
        comp_obj.valid_arr = np.logical_and(comp_obj.valid_arr, comp_obj.new_C2_arr)

# Compute the values of additional requirements given each component combo
def vectorize_equations(comp_obj):
    if (comp_obj.opt_obj.topology == 'buck') or (comp_obj.opt_obj.topology == 'boost'):

        def compute_total_C2(component_combo2):
            set_combo_variables(comp_obj, component_combo2)

            comp_obj.opt_obj.delta_i = component_combo2.combo[2].delta_i

            OptimizerInit.set_optimization_variables(comp_obj.opt_obj, None)
            # component_combo2.C2_tot = comp_obj.opt_obj.cap2.Capacitance
            component_combo2.C2_tot = comp_obj.opt_obj.cap2.Capacitance


        vec_f = np.vectorize(compute_total_C2)
        vec_f(comp_obj.combo_list)

        print('done')

    # if comp_obj.opt_obj.topology == 'boost':
    #
    #     def compute_total_C2(component_combo2):
    #         comp_obj.delta_i = comp_obj.ind1.delta_i
    #         comp_obj.fsw = comp_obj.opt_obj.fsw
    #         comp_obj.dc = comp_obj.opt_obj.dc
    #         delta_v_max = 0.01 * comp_obj.opt_obj.converter_df.loc[0, 'Vout']
    #         component_combo2.C2_tot = (comp_obj.delta_i / comp_obj.opt_obj.delta_i_normalizer) * comp_obj.opt_obj.converter_df.loc[0, 'Iout'] * (
    #                     1 / (comp_obj.opt_obj.fsw * comp_obj.opt_obj.ind_normalizer)) / (8 * delta_v_max)
    #
    #         comp_obj.opt_obj.converter_df.loc[0, 'Iin'] = comp_obj.opt_obj.converter_df.loc[0, 'Iout'] / (1 - comp_obj.opt_obj.dc)
    #         component_combo2.C2_tot = comp_obj.opt_obj.converter_df.loc[0, 'Iout'] * comp_obj.opt_obj.dc * (
    #                     1 / (comp_obj.opt_obj.fsw * comp_obj.opt_obj.ind_normalizer)) / (2 * delta_v_max)

        # vec_f = np.vectorize(compute_total_C2)
        # vec_f(comp_obj.combo_list)
        #
        # print('done')


# Compute the total power loss for each component combo, using the equation set in power_pred_tot() inside
# fet_optimization_chained_wCaps.py
def compute_total_power(comp_obj):

    def compute_power(component_combo2):
        set_combo_variables(comp_obj, component_combo2)
        OptimizerInit.power_pred_tot(comp_obj.opt_obj, x=None)
        component_combo2.power_tot = comp_obj.opt_obj.power_tot


    # could put this back into vectorization, should work fine
    for combination in comp_obj.combo_list:
        compute_power(combination)

    # print('done')
    # vec_f = np.vectorize(compute_power)
    # vec_f(comp_obj.combo_list)

def determine_norm_scored_vars(opt_obj, fet_obj):
    if (opt_obj.topology == 'buck') or (opt_obj.topology == 'boost'):
        if fet_obj.fet_index == 0:
            vars_list = ['Unit_price', 'R_ds', 'Q_g', 'Pack_case', 'C_ossp1']  # only tau_c, tau_rr, and Cossp1 for Q2
        elif fet_obj.fet_index == 1:
            vars_list = ['Unit_price', 'R_ds', 'Q_g', 'Pack_case', 'tau_c', 'tau_rr',
                         'C_ossp1']  # only tau_c, tau_rr, and Cossp1 for Q2

    elif opt_obj.topology == 'microinverter_combined':
        if fet_obj.fet_index == 0:
            vars_list = ['Unit_price', 'R_ds', 'Q_g', 'Pack_case', 'C_ossp1'] # only tau_c, tau_rr, and Cossp1 for Q2
            # vars_list = ['R_ds', 'Q_g', 'C_ossp1'] # only tau_c, tau_rr, and Cossp1 for Q2

        elif fet_obj.fet_index == 1:
            vars_list = ['Unit_price', 'R_ds', 'Q_g', 'Pack_case', 'tau_c', 'tau_rr', 'C_ossp1'] # only tau_c, tau_rr, and Cossp1 for Q2
            # vars_list = ['R_ds', 'Q_g', 'tau_c', 'tau_rr', 'C_ossp1'] # only tau_c, tau_rr, and Cossp1 for Q2

        elif fet_obj.fet_index == 2:
            vars_list = ['Unit_price', 'R_ds', 'Q_g', 'Pack_case', 'tau_c', 'tau_rr', 'C_ossp1']
            # vars_list = ['R_ds', 'Q_g', 'tau_c', 'tau_rr', 'C_ossp1']


    return vars_list

def bounds_check(fsw_loss_list, opt_obj):
    # look at each combo, get rid of ones that don't meet the delta_i or fsw bounds requirements from the optimization
    indexes = []
    for index,item in enumerate(fsw_loss_list):
        if (item[0] < opt_obj.bounds_dict['fsw_min'] * opt_obj.ind_normalizer) or (item[0] > opt_obj.bounds_dict['fsw_max'] * opt_obj.ind_normalizer) or (item[2] < opt_obj.bounds_dict['comp_selection_delta_i_min']) or (item[2] > opt_obj.bounds_dict['comp_selection_delta_i_max']):
            # remove item from list, make sure setting new list permanently
            indexes.append(index)
    for index in sorted(indexes, reverse=True):
        del fsw_loss_list[index]
    return fsw_loss_list


def print_practical(self):
    if self.opt_obj.topology == 'buck' or self.opt_obj.topology == 'boost':
        print('Printing datasheet parameters for all selected components: Q1, Q2, L1, C1, C2:')
        print(
            f'Total Ploss/Pout: {self.test_combo.power_tot / self.test_combo.opt_obj.Pout}, Total cost: {self.test_combo.Cost_tot}, total area: {self.test_combo.Area_tot}')
        print(f'fet1 loss breakdown:')
        print(
            f'Rds losses: {self.test_combo.fet1.Rds_loss}, Qg losses: {self.test_combo.fet1.Qg_loss}, toff losses: {self.test_combo.fet1.toff_loss}')
        print(f'fet2 loss breakdown:')
        print(f'Rds losses: {self.test_combo.fet2.Rds_loss}, Qg losses: {self.test_combo.fet2.Qg_loss},'
              f'Cdsq losses: {self.test_combo.fet2.Cdsq_loss}')
        if self.test_combo.opt_obj.fet_tech == 'MOSFET':
            print(f'Qrr losses: {self.test_combo.fet2.Qrr_loss}')
        print(f'ind1 loss breakdown:')
        print(f'Rdc losses: {self.test_combo.ind1.Rdc_loss}, Rac losses: {self.test_combo.ind1.Rac_loss},'
              f'IGSE losses: {self.test_combo.ind1.IGSE_loss}')
        print(f'Loss contribution by components:')
        print(
            f'Q1: {self.test_combo.opt_obj.Q1_loss}, Q2: {self.test_combo.opt_obj.Q2_loss}, L1: {self.test_combo.opt_obj.L1_loss}, PCB: {self.test_combo.opt_obj.PCB_loss}, total sum: {self.test_combo.opt_obj.Q1_loss + self.test_combo.opt_obj.Q2_loss + self.test_combo.opt_obj.L1_loss + self.test_combo.opt_obj.PCB_loss}')

        # Cost breakdown of selected components:
        print('Cost breakdown of selected components')
        print(
            f'Q1: {self.test_combo.fet1.Cost}, Q2: {self.test_combo.fet2.Cost}, L1: {self.test_combo.ind1.Cost}, C1: {self.test_combo.cap1.Cost}, C2: {self.test_combo.cap2.Cost}, total: {self.test_combo.Cost_tot}')

        # Area breakdown of selected components:
        print('Area breakdown of selected components')
        print(
            f'Q1: {self.test_combo.fet1.Area}, Q2: {self.test_combo.fet2.Area}, L1: {self.test_combo.ind1.Area}, C1: {self.test_combo.cap1.Area}, C2: {self.test_combo.cap2.Area}, total: {self.test_combo.Area_tot}')

    if self.opt_obj.topology == 'microinverter_combined':
        print('Printing datasheet parameters for all selected components: Q1, Q2, Q3, L1:')
        print(
            f'Total Ploss/Pout: {self.test_combo.power_tot / self.test_combo.opt_obj.Pout}, Total cost: {self.test_combo.Cost_tot}, total area: {self.test_combo.Area_tot}')
        print(f'fet1 loss breakdown:')
        print(
            f'Rds losses: {self.test_combo.fet1.Rds_loss}, Qg losses: {self.test_combo.fet1.Qg_loss}, toff losses: {self.test_combo.fet1.toff_loss}')
        print(f'fet2 loss breakdown:')
        print(f'Rds losses: {self.test_combo.fet2.Rds_loss}, Qg losses: {self.test_combo.fet2.Qg_loss}'
              )
        print(f'fets 3-6 loss breakdown:')
        print(f'Rds losses: {self.test_combo.fet3.Rds_loss}, Qg losses: {self.test_combo.fet3.Qg_loss}, '
              f'toff losses: {self.test_combo.fet3.toff_loss}, Qrr losses: {self.test_combo.fet3.Qrr_loss},'
              f'Cdsq losses: {self.test_combo.fet3.Cdsq_loss}'
              )
        print(f'ind1 loss breakdown:')
        print(f'Rdc losses: {self.test_combo.ind1.Rdc_loss}, Rac losses: {self.test_combo.ind1.Rac_loss},'
              f'IGSE losses: {self.test_combo.ind1.IGSE_loss}')
        print(f'Loss contribution by components:')
        print(
            f'Q1: {self.test_combo.opt_obj.Q1_loss}, Q2: {self.test_combo.opt_obj.Q2_loss}, Q3: {self.test_combo.opt_obj.Q3_loss}, L1: {self.test_combo.opt_obj.L1_loss}, total sum: {self.test_combo.opt_obj.Q1_loss + self.test_combo.opt_obj.Q2_loss + self.test_combo.opt_obj.Q3_loss +self.test_combo.opt_obj.L1_loss}')


        # Cost breakdown of selected components:
        print('Cost breakdown of selected components')
        print(
            f'Q1: {self.test_combo.fet1.Cost}, Q2: {self.test_combo.fet2.Cost}, Q3: {self.test_combo.fet3.Cost}, L1: {self.test_combo.ind1.Cost}, total: {self.test_combo.Cost_tot}')

        # Area breakdown of selected components:
        print('Area breakdown of selected components')
        print(
            f'Q1: {self.test_combo.fet1.Area}, Q2: {self.test_combo.fet2.Area}, Q3: {self.test_combo.fet3.Area}, L1: {self.test_combo.ind1.Area}, total: {self.test_combo.Area_tot}')

def print_theoretical(self): # --> make case statements for each
    # Now print all the parameters of the optimized, theoretical components
    print('\n')
    if self.opt_obj.topology == 'buck' or self.opt_obj.topology == 'boost':
        print('Printing optimized parameters for all components: Q1, Q2, L1, C1, C2:')

        print(
            f'fet1: \n Vdss: {self.opt_obj.fet1.Vdss}, Rds: {self.opt_obj.fet1.Rds * self.opt_obj.fet_normalizer}, Qg: {self.opt_obj.fet1.Qg}, cost: {self.opt_obj.fet1.Cost}, '
            f'area: {self.opt_obj.fet1.Area}')
        print(
            f'fet2: \n Vdss: {self.opt_obj.fet2.Vdss}, Rds: {self.opt_obj.fet2.Rds * self.opt_obj.fet_normalizer}, Qg: {self.opt_obj.fet2.Qg}, '
            f'tau_c: {self.opt_obj.fet2.tau_c}, tau_rr: {self.opt_obj.fet2.tau_rr}, Cossp1: {self.opt_obj.fet2.Cossp1}, '
            f'cost: {self.opt_obj.fet2.Cost},'
            f'area: {self.opt_obj.fet2.Area}')
        print(f'ind1: \n L: {self.opt_obj.ind1.L}, Irated: {self.opt_obj.ind1.I_rated}, Rdc: {self.opt_obj.ind1.R_dc}, '
              f'Rac total: {self.opt_obj.ind1.Rac_total}, IGSE total: {self.opt_obj.ind1.IGSE_total}, core volume: {self.opt_obj.ind1.Core_volume}, '
              f'cost: {self.opt_obj.ind1.Cost}, area: {self.opt_obj.ind1.Area}')
        print(
            f'cap1: \n cap: {self.opt_obj.cap1.Cap_0Vdc}, Vrated: {self.opt_obj.cap1.V_rated}, area: {self.opt_obj.cap1.Area},'
            f'cost: {self.opt_obj.cap1.Cost}')
        print(
            f'cap2: \n cap: {self.opt_obj.cap2.Cap_0Vdc}, Vrated: {self.opt_obj.cap2.V_rated}, area: {self.opt_obj.cap2.Area},'
            f'cost: {self.opt_obj.cap2.Cost}')

        # And print all the theoretical loss contributions
        print('theoretical loss contributions:')
        # self.opt_obj.fet1.Rds_loss = self.opt_obj.I_Q1 ** 2 * self.opt_obj.fet1.kT * self.opt_obj.fet_normalizer * self.opt_obj.fet1.Rds
        # self.opt_obj.fet1.Qg_loss = self.opt_obj.ind_normalizer * self.opt_obj.fsw * self.opt_obj.fet1.Qg * self.opt_obj.fet1.Vgate
        # self.Q1_loss = self.opt_obj.fet1.Rds_loss + self.opt_obj.fet1.Qg_loss
        print(f'Q1 loss breakdown:')
        print(
            f'Rds losses: {self.opt_obj.fet1.Rds_loss}, Qg losses: {self.opt_obj.fet1.Qg_loss}, toff losses: {self.opt_obj.fet1.toff_loss}, Q1 total loss: {self.opt_obj.Q1_loss}')

        print('Q2 loss breakdown:')
        print(f'Rds losses: {self.opt_obj.fet2.Rds_loss}, Qg losses: {self.opt_obj.fet2.Qg_loss},'
              f'Cdsq losses: {self.opt_obj.fet2.Cdsq_loss}')
        if self.opt_obj.fet_tech == 'MOSFET':
            print(f'Qrr losses: {self.opt_obj.fet2.Qrr_loss}')
        print(f'Q2 total loss: {self.opt_obj.Q2_loss}')

        print(f'ind1 loss breakdown:')
        print(f'Rdc losses: {self.opt_obj.ind1.Rdc_loss}, Rac losses: {self.opt_obj.ind1.Rac_loss},'
              f'IGSE losses: {self.opt_obj.ind1.IGSE_loss}')

        print(f'Loss contribution by components:')
        print(
            f'Q1: {self.opt_obj.Q1_loss}, Q2: {self.opt_obj.Q2_loss}, L1: {self.opt_obj.L1_loss}, total sum: {self.opt_obj.Q1_loss + self.opt_obj.Q2_loss + self.opt_obj.L1_loss}')

        print('Cost breakdown by components:')
        print(
            f'Q1: {self.opt_obj.fet1.Cost}, Q2: {self.opt_obj.fet2.Cost}, L1: {self.opt_obj.ind1.Cost}, C1: {self.opt_obj.cap1.Cost}, C2: {self.opt_obj.cap2.Cost}, total: {self.opt_obj.Cost_tot}')

        print('Area breakdown by components:')
        print(
            f'Q1: {self.opt_obj.fet1.Area}, Q2: {self.opt_obj.fet2.Area}, L1: {self.opt_obj.ind1.Area}, C1: {self.opt_obj.cap1.Area}, C2: {self.opt_obj.cap2.Area}, total: {self.opt_obj.Area_tot}')

        # Print all the optimization variables themselves: Rds1, Rds2, fsw, delta_i (%)
        print('Printing the optimization variables:')
        print(
            f'Rds1: {self.opt_obj.fet1.Rds * self.opt_obj.fet_normalizer}, Rds2: {self.opt_obj.fet2.Rds * self.opt_obj.fet_normalizer}, '
            f'fsw: {self.opt_obj.fsw * self.opt_obj.ind_normalizer}, delta_i: {self.opt_obj.delta_i / self.opt_obj.delta_i_normalizer},'
            f'Cap_area1: {self.opt_obj.cap1.Area / self.opt_obj.cap_normalizer}, Cap_area2: {self.opt_obj.cap2.Area / self.opt_obj.cap_normalizer}')

    if self.opt_obj.topology == 'microinverter_combined':
        print('Printing optimized parameters for all components: Q1, Q2, Q3, L1:')

        print(
            f'fet1: \n Vdss: {self.opt_obj.fet1.Vdss}, Rds: {self.opt_obj.fet1.Rds * self.opt_obj.fet_normalizer}, Qg: {self.opt_obj.fet1.Qg}, cost: {self.opt_obj.fet1.Cost}, '
            f'area: {self.opt_obj.fet1.Area}')
        print(
            f'fet2: \n Vdss: {self.opt_obj.fet2.Vdss}, Rds: {self.opt_obj.fet2.Rds * self.opt_obj.fet_normalizer}, Qg: {self.opt_obj.fet2.Qg}, '
            f'cost: {self.opt_obj.fet2.Cost},'
            f'area: {self.opt_obj.fet2.Area}')
        print(
            f'fet3: \n Vdss: {self.opt_obj.fet3.Vdss}, Rds: {self.opt_obj.fet3.Rds * self.opt_obj.fet_normalizer}, Qg: {self.opt_obj.fet3.Qg}, '
            f'tau_c: {self.opt_obj.fet3.tau_c}, tau_rr: {self.opt_obj.fet3.tau_rr}, Cossp1: {self.opt_obj.fet3.Cossp1}, '
            f'cost: {self.opt_obj.fet3.Cost},'
            f'area: {self.opt_obj.fet3.Area}')
        print(f'ind1: \n L: {self.opt_obj.ind1.L}, Irated: {self.opt_obj.ind1.I_rated}, Rdc: {self.opt_obj.ind1.R_dc}, '
              f'Rac total: {self.opt_obj.ind1.Rac_total}, IGSE total: {self.opt_obj.ind1.IGSE_total}, core volume: {self.opt_obj.ind1.Core_volume}, '
              f'cost: {self.opt_obj.ind1.Cost}, area: {self.opt_obj.ind1.Area}')


        # And print all the theoretical loss contributions
        print('theoretical loss contributions:')
        # self.opt_obj.fet1.Rds_loss = self.opt_obj.I_Q1 ** 2 * self.opt_obj.fet1.kT * self.opt_obj.fet_normalizer * self.opt_obj.fet1.Rds
        # self.opt_obj.fet1.Qg_loss = self.opt_obj.ind_normalizer * self.opt_obj.fsw * self.opt_obj.fet1.Qg * self.opt_obj.fet1.Vgate
        # self.Q1_loss = self.opt_obj.fet1.Rds_loss + self.opt_obj.fet1.Qg_loss
        print(f'Q1 loss breakdown:')
        print(
            f'Rds losses: {self.opt_obj.fet1.Rds_loss}, Qg losses: {self.opt_obj.fet1.Qg_loss}, toff losses: {self.opt_obj.fet1.toff_loss}, Q1 total loss: {self.opt_obj.Q1_loss}')

        print('Q2 loss breakdown:')
        print(f'Rds losses: {self.opt_obj.fet2.Rds_loss}, Qg losses: {self.opt_obj.fet2.Qg_loss}')
        print(f'Q2 total loss: {self.opt_obj.Q2_loss}')

        print('Q3 loss breakdown:')
        print(f'Rds losses: {self.opt_obj.fet3.Rds_loss}, Qg losses: {self.opt_obj.fet3.Qg_loss},'
              f'Cdsq losses: {self.opt_obj.fet3.Cdsq_loss}')
        if self.opt_obj.fet_tech == 'MOSFET':
            print(f'Qrr losses: {self.opt_obj.fet3.Qrr_loss}')
        print(f'Q2 total loss: {self.opt_obj.Q3_loss}')

        print(f'ind1 loss breakdown:')
        print(f'Rdc losses: {self.opt_obj.ind1.Rdc_loss}, Rac losses: {self.opt_obj.ind1.Rac_loss},'
              f'IGSE losses: {self.opt_obj.ind1.IGSE_loss}')

        print(f'Loss contribution by components:')
        print(
            f'Q1: {self.opt_obj.Q1_loss}, Q2: {self.opt_obj.Q2_loss}, Q3: {self.opt_obj.Q3_loss}, L1: {self.opt_obj.L1_loss}, '
            f'total sum: {self.opt_obj.Q1_loss + self.opt_obj.Q2_loss + self.opt_obj.Q3_loss + self.opt_obj.L1_loss}')

        print('Cost breakdown by components:')
        print(
            f'Q1: {self.opt_obj.fet1.Cost}, Q2: {self.opt_obj.fet2.Cost}, Q3: {self.opt_obj.fet3.Cost}, L1: {self.opt_obj.ind1.Cost}, total: {self.opt_obj.Cost_tot}')

        print('Area breakdown by components:')
        print(
            f'Q1: {self.opt_obj.fet1.Area}, Q2: {self.opt_obj.fet2.Area}, Q3: {self.opt_obj.fet3.Area}, L1: {self.opt_obj.ind1.Area}, total: {self.opt_obj.Area_tot}')

        # Print all the optimization variables themselves: Rds1, Rds2, fsw, delta_i (%)
        print('Printing the optimization variables:')
        print(
            f'Rds1: {self.opt_obj.fet1.Rds * self.opt_obj.fet_normalizer}, Rds2: {self.opt_obj.fet2.Rds * self.opt_obj.fet_normalizer},'
            f'Rds3: {self.opt_obj.fet3.Rds * self.opt_obj.fet_normalizer}, '
            f'fsw: {self.opt_obj.fsw * self.opt_obj.ind_normalizer}, delta_i: {self.opt_obj.delta_i / self.opt_obj.delta_i_normalizer}')

    print('done')



if __name__ == '__main__':
    pd.set_option("display.max_rows", 100, "display.max_columns", 100)

    # data_cleaning()
    # ML_model_training()
    # resulting_parameter_plotting()
    # predict_components()
    compare_combinations()
    print('done')