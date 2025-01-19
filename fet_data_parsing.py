'''
    This file contains the functions for parsing the attributes once they are scraped. These types of functions include
    adding the right prefix value to the associated attributes, cleaning the attributes so they only contain usable
    values, and eliminating rows with unavailable data in an associated parsed column.
'''

import math
import re
import si_prefix
import numpy as np
from joblib import Memory
import pandas as pd

memory = Memory('cache', verbose=0)

prefix_dict = {'y': 1e-24,  # yocto
               'z': 1e-21,  # zepto
               'a': 1e-18,  # atto
               'f': 1e-15,  # femto
               'p': 1e-12,  # pico
               'n': 1e-9,  # nano
               'u': 1e-6,  # micro
               'µ': 1e-6,  # micro
               'μ': 1e-6,  # micro (alt)
               'm': 1e-3,  # mili
               'c': 1e-2,  # centi
               'd': 1e-1,  # deci
               '': 1,
               None: 1,
               'k': 1e3,  # kilo
               'M': 1e6,  # mega
               'G': 1e9,  # giga
               'T': 1e12,  # tera
               'P': 1e15,  # peta
               'E': 1e18,  # exa
               'Z': 1e21,  # zetta
               'Y': 1e24,  # yotta
               }


'''
    Returns the text split on the first letter, returns [number, everything else (first letter inclusive)]
'''
def split_on_letter(text):
    match = re.compile("[^\W\d]").search(text)
    return [text[:match.start()], text[match.start():]]


'''
    Removes non-alpha and non-numeric characters, returns the string without special characters
'''
def remove_special_chars(text):
    return re.sub('[^a-zA-Z0-9 \n\.]', '', text)

def remove_non_integers(text):
    return re.sub('[^0-9 .]','', text).split()

'''
    Takes an alphanumeric string, splits it into numeric and alpha parts, and returns the value adjusted by the
    necessary prefix, if a prefix exists
'''
def prefix_adj(text,inductance = False):
    if text == '0' or text == '0.0' or text=='nan':
        return 0.0
    # Get the prefix part of the entry
    split_var = split_on_letter(text)

    if inductance:
        #print(split_var)
        for k, v in prefix_dict.items():
            if k == (split_var[1][0]):
                if k == 'm' or k == 'n' or k == 'µ' or k == 'μ' or k == 'u' or k == 'p':
                    fixed_var = round(float(split_var[0]) * v, abs(si_prefix.split(v)[1]))
                else:
                    fixed_var = np.nan

                return fixed_var

    else:
        for k, v in prefix_dict.items():
            if k == (split_var[1][0].strip()):
                # fixed_var = round(float(split_var[0]) * v, abs(si_prefix.split(v)[1]))
                fixed_var = float(split_var[0]) * v
                return fixed_var
    return split_var[0]



'''
    Change zero floats to NaN, which are better used in dataframes
'''
def catch(func, *args):
    try:
        if func(*args) == 0.0:
            return np.nan
        else:
            return func(*args)
    except:
        return np.nan


'''
    The following are all the functions for parsing the attributes that are collected off the Digikey main site.
'''
def Unit_price_parse(entry):
    # return float(entry.split('$')[1])
    return float(entry)

def Series_parse(entry):
    return remove_special_chars(entry)


def FET_type_parse(entry):
    return remove_special_chars(entry.split('-')[0])


def V_dss_parse(entry):
    return float(prefix_adj(entry))


def Technology_parse(entry):
    return remove_special_chars(entry.split()[0])


def Q_g_parse(entry):
    return float(prefix_adj(entry))


def R_ds_parse(entry):
    return float(prefix_adj(entry))


def Input_cap_parse(entry):
    return float(prefix_adj(entry))

def Capacitance_parse(entry):
    return float(prefix_adj(entry))

def Rated_volt_parse(entry):
    return float(prefix_adj(entry))

def Size_parse(entry):
    size_entry = re.sub('[^0-9 \n\.]', '', entry).split()
    return float(size_entry[2]) * float(size_entry[3])

def Thickness_parse(entry):
    thickness_entry = re.sub('[^0-9 \n\.]', '', entry).split()
    return float(thickness_entry[1])


# fet_df.columns = ['Mfr_part_no', 'Unit_price', 'Mfr', 'Series', 'FET_type', 'Technology', 'V_dss', 'I_d', 'V_drive',
#                   'R_ds', 'V_thresh', 'Q_g', 'V_gs', 'Input_cap', 'P_diss', 'Op_temp', 'Mount_type', 'Supp_pack',
#                   'Pack_case', 'Q_rr']
#['Mfr_part_no','Unit_price','Mfr','Series','Capacitance','Rated_volt', 'Temp_coef','Size','Thickness']
# converter = lambda x : x*2 if x < 10 else (x*3 if x < 20 else x)
#

def V_gs_parse(entry):
    return float(remove_special_chars(entry.split('V')[0]))

def I_d_parse(entry):
    return float(prefix_adj(entry))

# def P_diss_parse(entry):
#     return float(prefix_adj(entry))

def Inductance_parse(entry):
    # if float(prefix_adj(entry,inductance=True)) > 1:
        #print('True')
    return float(prefix_adj(entry,inductance=True))

def Current_rating_parse(entry):
    return float(prefix_adj(entry))

def Current_sat_parse(entry):
    return float(prefix_adj(entry))

def DCR_parse(entry):
    return float(prefix_adj(entry.split(' ')[0]))

def Dimension_parse(entry):
    entry = remove_non_integers(entry)
    return float(float(entry[2]) * float(entry[3]))

def Height_parse(entry):
    entry = remove_non_integers(entry)
    return float(entry[1])

def Channel_type_parse(df):
    channel_type_dict = {'High-Side or Low-Side': 'HS_or_LS', 'Half-Bridge': 'HB', 'Low-Side': 'LS', 'High-Side and Low-Side': 'HS_or_LS',
                         'Full-Bridge': 'FB', 'High-Side': 'HS', '-':'np.nan', 'Half-Bridge, Low-Side':'HB_LS'}
    df = df.replace({'Channel_type': channel_type_dict})
    return df

def Num_driver_parse(entry):
    return float(entry)

def Gate_type_parse(entry):
    return entry

# def Supply_volt_parse(entry):
#     return entry
#
# def Peak_current_parse(entry):
#     return float(entry)

def Input_type_parse(entry):
    return entry

def High_side_volt_parse(entry):
    return float(prefix_adj(entry))

def Peak_current_source_parse(entry):
    return float(prefix_adj(entry.split(' ')[0]))

def Peak_current_sink_parse(entry):
    return float(prefix_adj(entry.split(' ')[1]))



'''
    For each attribute, parse and replace the elements in each column so that the data is usable.
'''
# @memory.cache
def initial_fet_parse(fet_df, attr_list):

    for attr in attr_list:
        if attr in ['Mfr', 'Mfr_part_no','Pack_case','Temp_coef','Driven_config']:
            continue
        elif attr in ['Channel_type']:
            fet_df = Channel_type_parse(fet_df)
        elif attr in ['Inductance']:
            fet_df.loc[:, attr] = fet_df.loc[:, attr].apply(
                lambda x: catch(eval(f'{attr.replace(" ", "")}' + '_parse'), x))
        elif attr in ['V_gs']:
            fet_df.loc[:, attr] = fet_df.loc[:, attr].apply(
                lambda x: catch(eval(f'{attr.replace(" ", "")}' + '_parse'), x))
        elif attr in ['Peak_current']:
            fet_df.loc[:, 'Peak_current_source'] = fet_df.loc[:, attr].apply(
                lambda x: catch(eval(f'{attr.replace(" ", "")}' + '_source_parse'), x))
            fet_df.loc[:, 'Peak_current_sink'] = fet_df.loc[:, attr].apply(
                lambda x: catch(eval(f'{attr.replace(" ", "")}' + '_sink_parse'), x))

        else:
            fet_df.loc[:, attr] = fet_df.loc[:, attr].apply(lambda x: catch(eval(f'{attr.replace(" ", "")}' + '_parse'), x))
    return fet_df


# @memory.cache
'''
    Eliminate unusable data in the desired attribute columns
'''
def column_fet_parse(fet_df, var_list):
    return fet_df.dropna(axis=0, subset=var_list)

