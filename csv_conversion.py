'''
    This file contains the functions used to write data to the desired csv and to turn csv data into a df.
'''

import csv
import pandas as pd
import numpy as np

'''
    Turn the data from a csv file into a dataframe. Add column names later, this is a general function.    
''' 
def csv_to_mosfet(csv_file):
    df = pd.read_csv(csv_file)
    return df

def csv_to_df(csv_file):
    df = pd.read_csv(csv_file)
    return df
    
'''
    Write the data stored in the fet object as a line to the specified csv file. fet is the fet object and filename is
    a string.
'''      
def single_fet_to_csv(fet, filename):
    fields = ['Mfr_part_no', 'Unit_price', 'Mfr', 'Series','FET_type', 'Technology', 'V_dss', 'I_d', 'V_drive', 'R_ds', 'V_thresh','Q_g', 'V_gs', 'Input_cap', 'P_diss','Op_temp','Mount_type','Supp_pack','Pack_case','Q_rr','C_oss', 'I_F', 'Vds_meas']
    rows = []
    
    # Create a dataframe row of the object attributes, and write that row to the csv
    #print(fet.mfr_part_no,fet.unit_price, fet.mfr, fet.series, fet.fet_type, fet.technology, fet.V_dss, fet.I_d, fet.V_drive, fet.R_ds, fet.V_thresh, fet.Q_g, fet.V_gs, fet.input_cap, fet.P_diss, fet.op_temp, fet.mount_type, fet.supp_pack, fet.pack_case)
    row = [[fet.mfr_part_no, fet.unit_price, fet.mfr, fet.series, fet.fet_type, fet.technology, fet.V_dss, fet.I_d, fet.V_drive, fet.R_ds, fet.V_thresh, fet.Q_g, fet.V_gs, fet.input_cap, fet.P_diss, fet.op_temp, fet.mount_type, fet.supp_pack, fet.pack_case, fet.Q_rr, fet.Coss, fet.I_F, fet.Vds_meas]]
    #print(row)
    df = pd.DataFrame(row, columns=fields)
    df.to_csv(filename, mode='a',header=False)

'''
    Write the data stored in the fet object (downloaded, including all pdf info) as a line to the specified csv file. fet is the fet object and filename is
    a string.
'''


def single_fet_to_csv_downloaded(fet, filename):
    fields = ['Mfr_part_no', 'Datasheet', 'Unit_price', 'Mfr', 'Series', 'FET_type', 'Technology', 'V_dss', 'I_d', 'V_drive', 'R_ds',
              'V_thresh', 'Q_g', 'V_gs', 'Input_cap', 'P_diss', 'Op_temp', 'Mount_type', 'Supp_pack', 'Pack_case',
              'Q_rr', 't_rr', 'I_S', 'diFdt', 'I_F', 'C_oss', 'Vds_meas']
    rows = []

    # Create a dataframe row of the object attributes, and write that row to the csv
    # print(fet.mfr_part_no,fet.unit_price, fet.mfr, fet.series, fet.fet_type, fet.technology, fet.V_dss, fet.I_d, fet.V_drive, fet.R_ds, fet.V_thresh, fet.Q_g, fet.V_gs, fet.input_cap, fet.P_diss, fet.op_temp, fet.mount_type, fet.supp_pack, fet.pack_case)
    row = [[fet.mfr_part_no, fet.datasheet, fet.unit_price, fet.mfr, fet.series, fet.fet_type, fet.technology, fet.V_dss, fet.I_d,
            fet.V_drive, fet.R_ds, fet.V_thresh, fet.Q_g, fet.V_gs, fet.input_cap, fet.P_diss, fet.op_temp,
            fet.mount_type, fet.supp_pack, fet.pack_case, fet.Q_rr, fet.t_rr, fet.IS, fet.diFdt, fet.I_F, fet.Coss, fet.Vds_meas]]
    # print(row)
    df = pd.DataFrame(row, columns=fields)
    df.to_csv(filename, mode='a', header=False)

'''
    Takes the df, applies appropriate column labels, and turns the df into a csv with the specified name. Doesn't return
    anything but writes to the csv.
'''
def df_to_csv(df,file_name, component_type, pareto=True):
    # if component_type == 'fet':
    #     if pareto==True:
    #         fields = ['V_dss', 'Unit_price', 'R_ds', 'Q_g', 'Q_rr','FET_type','Technology']
    #     else:
    #         fields = ['Mfr_part_no', 'Unit_price', 'Mfr', 'Series','FET_type', 'Technology', 'V_dss', 'I_d', 'V_drive', 'R_ds', 'V_thresh','Q_g', 'V_gs', 'Input_cap', 'P_diss','Op_temp','Mount_type','Supp_pack','Pack_case']
    #     row = [[fet.Mfr_part_no, fet.Unit_price, fet.Mfr, fet.Series, fet.FET_type, fet.Technology, fet.V_dss, fet.I_d, fet.V_drive, fet.R_ds, fet.V_thresh, fet.Q_g, fet.V_gs, fet.Input_cap, fet.P_diss, fet.Op_temp, fet.Mount_type, fet.Supp_pack, fet.Pack_case]]
    # df = pd.DataFrame(row, columns=fields)
    df.to_csv(file_name, mode='a',header=False)


