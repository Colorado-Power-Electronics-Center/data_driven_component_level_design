'''
    This file contains the functions used to scrape the component information off of the Digikey site.
    The functions look at both the information available on the site itself and the information available on the
    corresponding pdf of the datasheet.
'''

import numpy as np
import math
from tabula import read_pdf
import csv
import time
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from csv_conversion import single_fet_to_csv, single_fet_to_csv_downloaded
from collections import Counter
import os
import re
import logging
import requests
from bs4 import BeautifulSoup
import PyPDF2

from selenium.common.exceptions import StaleElementReferenceException
import selenium.common
from selenium.common import exceptions
# from webdriver_manager import driver
# from webdriver_manager.chrome import ChromeDriverManager
import pandas as pd
from pathlib import Path
# import pdfreader
# from pdfreader import PDFDocument, SimplePDFViewer
from multiprocessing.dummy import Pool
import pickle
# from tqdm import tqdm
# import typing as tp
# import pdfrw
# import PyPDF4
from joblib import Memory
import multiprocessing as mp
from multiprocessing.dummy import Pool
import io
from PyPDF2 import PdfFileReader
import tabula

logging.basicConfig(filename='log.log', filemode='w', encoding='utf-8', level=logging.DEBUG)
# memory = Memory('cache', verbose=0)

'''
    This function creates a class for the initial webdriver, starting at the first page. Takes as an argument the link
    to start the scraping on.
'''


class TransistorParser(object):
    def __init__(self, start_url):
        self.base_url = start_url

    def page_url(self, page):
        return self.base_url.format(page=page)

    @staticmethod
    def get_driver() -> webdriver:
        # Setup Selenium Preferences
        # chrome_options = Options()
        chrome_options = webdriver.ChromeOptions()

        # uncomment so you cant see the page open every time
        # chrome_options.add_argument('--window-size=1770,880')
        # chrome_options.add_argument("--window-position=-1820,480")
        prefs = {'download.default_directory': r'C:\Users\skyer\OneDrive\Documents\GitHub\Public_Component_Level_Power_Designer\xl_pdf_sets\\'}
        chrome_options.add_experimental_option('prefs', prefs)
        chrome_options.add_experimental_option("useAutomationExtension", False)
        chrome_options.add_experimental_option("excludeSwitches", ['enable-automation'])
        chrome_options.add_argument('--disable-blink-features=AutomationControlled')
        chrome_options.add_argument("--disable-extensions")

        # PROXY = "172.67.182.144"  # your proxy
        #
        # chrome_options.add_argument('--proxy-server=%s' % PROXY)
        # Open Selenium Driver
        driver = webdriver.Chrome('chromedriver2.exe', options=chrome_options)
        driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")

        driver.implicitly_wait(5)

        return driver


# @memory.cache()
'''
    This function goes to the initial url, finds the total number of pages that exist on the site, and returns that
    number as an int.
'''


def total_num_pages(start_url):
    # This function gets the driver and determines how many pages exist on the site
    transistor_parser = TransistorParser(start_url)
    driver = transistor_parser.get_driver()
    driver.get(transistor_parser.page_url(1))

    page_total = '/html/body/div[2]/main/section/div/div[2]/div/div[1]/div/div[1]/div/div[1]/div/div[2]/span'
    return_num = int(((driver.find_element_by_xpath(page_total).text).split('/'))[1])
    driver.quit()
    # print(return_num)
    return return_num


'''
    Takes the driver and number of component on the page and returns the fet object. Useful for easy assessment of 
    the component's information.
'''


def indv_part_info(driver, i):
    fet = DigikeyFet(driver, i)
    return fet


# uncomment so you don't have to reload the same info every time
# @memory.cache

# pd.set_option('display.width', 1000)
# pd.set_option('display.max_columns',100)

def is_integer(n):
    try:
        float(n)
    except ValueError:
        return False
    else:
        return float(n).is_integer()


def isfloat(num):
    try:
        float(num)
        if math.isnan(float(num)):
            return False
        return True
    except ValueError:
        return False


# Alternate way of scraping from pdfs, starting from excel spreadsheet with list of pdfs
import camelot
import pickle


def combine_downloaded_tables():
    component = 'cap'
    if component == 'cap':
        component_list_df = pd.DataFrame()
        directory = 'xl_pdf_sets/'
        for filename in os.listdir(directory):
            f = os.path.join(directory, filename)
            if os.path.isfile(f):
                if 'ceramic_capacitors (' in str(f) or 'ceramic_capacitors -' in str(f):
                    print(f)
                    data = pd.read_csv(f)
                    component_list_df = pd.concat([component_list_df, data], axis=0)
                component_list_df.to_csv('xl_pdf_sets\merged_capacitor_list_files.csv', index=False)

    if component == 'fet':
        component_list_df = pd.DataFrame()
        directory = 'xl_pdf_sets/'
        for filename in os.listdir(directory):
            f = os.path.join(directory, filename)
            if os.path.isfile(f):
                if 'single_fets__mosfets (' in str(f) or 'single_fets__mosfets -' in str(f) or '.tmp' in str(f):
                    print(f)
                    data = pd.read_csv(f)
                    component_list_df = pd.concat([component_list_df, data], axis = 0)
                component_list_df.to_csv('xl_pdf_sets\merged_component_list_files.csv', index=False)

import time
from multiprocessing import Process
def full_pdf_part_info_xl():
    component = 'cap'
    if component == 'cap':
        directory = 'xl_pdf_sets/merged_capacitor_list_files.csv'
        links_df = pd.read_csv(directory)
        overall_datasheet_lists = []

        for i in range(len(links_df)):
            link = links_df.iloc[i]['Datasheet']
            if 'st.com' in link:
                continue
            else:
                component_list = []
                tables_list = []
                component_list.append(links_df.iloc[i]['Mfr Part #'])
                component_list.append(links_df.iloc[i])
                component_list.append(tables_list)
                overall_datasheet_lists.append(component_list)

        with open("xl_pdf_sets/pickled_data_capacitors_no_sheets", "ab") as f:
            for ds_list in overall_datasheet_lists:
                pickle.dump(ds_list, f)

    directory = 'xl_pdf_sets/merged_component_list_files.csv'
    links_df = pd.read_csv(directory)

    good_links = 0
    bad_links = 0
    overall_datasheet_lists = []

    def scrape_datasheet_tables(links_df, i):
        link = links_df.iloc[i]['Datasheet']
        print(link)

        try:
            component_list = []
            tables_list = []
            tables = camelot.read_pdf(link, pages='all')
            print(tables)

            # first append the mfr part no and the series of the downloaded table information onto component_list
            component_list.append(links_df.iloc[i]['Mfr Part #'])
            component_list.append(links_df.iloc[i])

            # then append a list of all the combined dataframes of the tables from the downloaded table information
            for l in range(tables.n):
                print(tables[l].df)
                tables_list.append(tables[l].df)
            component_list.append(tables_list)

            # create a dictionary with the mfr part # as the key, and the values as a list with
            # {mfr_part_no1: [series of the row from the downloaded table, list of all dfs of the scraped pdf tables]}
            # overall_datasheet_dict[links_df.iloc[i]['Mfr Part #']] = [links_df.iloc[i], tables_list]
            overall_datasheet_lists.append(component_list)



        except:
            print('error decoding')

    # go through all the components in the .csv file, open the datasheet link, try to get all the tables from
    # all the pages, turn all the tables into a dataframe, add all the dataframes into one list to encompass
    # each component, add each component (as a list of df's) into the main list to be pickled
    for i in range(len(links_df)):
        link = links_df.iloc[i]['Datasheet']
        if 'st.com' in link:
            continue
        else:
            component_list = []
            tables_list = []
            component_list.append(links_df.iloc[i]['Mfr Part #'])
            component_list.append(links_df.iloc[i])
            component_list.append(tables_list)
            overall_datasheet_lists.append(component_list)

            ### ACTUAL SCRAPING PROCESS ###

            # p1 = Process(target = scrape_datasheet_tables(links_df, i), name='Process_scraping')
            # p1.start()
            # p1.join(timeout=60)
            # p1.terminate()

        # link = links_df.iloc[i]['Datasheet']
        # print(link)

        # if 'http' not in link.get('href'):
        #     link_href = 'http:' + link.get('href')
        # else:
        #     link_href = link.get('href')
        # # response = requests.get(link_href, headers=headers, timeout=10)
        # response = requests.get(link_href, timeout=10)

        # print("Downloading pdf file: ", j)

        # pdf = open('fetPdfData.pdf', 'wb')
        # pdf.write(response.content)
        # try:
        #     df = read_pdf("fetPdfData.pdf", pages='1-6', stream=True, encoding='unicode_escape')
        # except:
        #     df = read_pdf("fetPdfData.pdf", pages='all', stream=True, encoding='unicode_escape')
        #     # can replace pages=1-6 w/ 'all'
        # pdf.close()

        # try:
        #     tables_list = []
        #     tables = camelot.read_pdf(link, pages='1-10')
        #     print(tables)
        #
        #     for l in range(tables.n):
        #         print(tables[l].df)
        #         tables_list.append(tables[l].df)
        #
        #     overall_datasheet_lists.append(tables_list)
        #     good_links += 1
        #
        #
        #
        #
        # except:
        #     print('error decoding')
        #     bad_links += 1


            # tables.export('fetPdfData.csv', f='csv', compress=True)  # json, excel, html, markdown, sqlite
            # tables[0].df
            # tables[1].df


    # pickle the entire list of components one by one onto the specified file

    ### CHANGE FILE NAME IF INCLUDING TABLES, AND USE ABOVE FUNCTION ###
    with open("xl_pdf_sets/pickled_data_no_sheets", "ab") as f:
        for ds_list in overall_datasheet_lists:
            pickle.dump(ds_list, f)

    # pickle.dump(test2, f)
    # with open("C:/temp/test.pickle", "rb") as f:
    #     testout1 = pickle.load(f)
    #     testout2 = pickle.load(f)
    # try:
    #     df = read_pdf(link, stream=True, encoding='latin1', pages = '1-6')
    #     print(df)
    #     df = read_pdf(link, stream=True, pages = '1-6')
    #
    #     good_links += 1
    # except:
    #     print('error decoding')
    #     bad_links +=1
    #     continue

    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        if os.path.isfile(f):

            print(f)
            # xl_name = 'xl_pdf_sets/single_fets__mosfets.csv'
            # xl_name = 'xl_pdf_sets/excel_test.txt'


# figure out what components are with the current pickled tables on pickled_datasheets2, add them as a list to an overall
# list with the [mfr part #, series of table data, list of dataframes from the scraped pdfs
def attach_pickled_tables():
    new_overall_list = []

    unpickled_components = []

    directory = 'xl_pdf_sets/merged_component_list_files.csv'
    links_df = pd.read_csv(directory)
    with open("xl_pdf_sets/pickled_datasheets2", "rb") as f:
        while True:
            try:
                individual_component = []
                current_id = pickle.load(f)
                unpickled_components.append(current_id)
            except EOFError:
                print('Pickle ends')
                break

    # go through every set of tables, figure out what component it is, attach a list of [mfr part #, series, list of df tables]
    # onto a main list
    desired_width = 320
    pd.set_option('display.width', desired_width)
    pd.set_option('display.max_columns', 10)

    print(unpickled_components[0][6])
    to_be_pickled_component = []
    to_be_pickled_component.append(links_df[links_df['Mfr Part #'].str.contains('BSS7728N')].iloc[0]['Mfr Part #'])
    to_be_pickled_component.append(links_df[links_df['Mfr Part #'].str.contains('BSS7728N')].iloc[0])
    to_be_pickled_component.append(unpickled_components[0])
    new_overall_list.append(to_be_pickled_component)

    # then pickle this list of 3 items onto the 'pickle_datasheets' document
    with open("xl_pdf_sets/pickled_datasheets_from218", "ab") as f:
        for ds_list in new_overall_list:
            pickle.dump(ds_list, f)


    print(links_df[links_df['Mfr Part #'].str.contains('BSS7728N')].iloc[0])

    new_overall_list.append()


    directory = 'xl_pdf_sets/merged_component_list_files.txt'
    links_df = pd.read_csv(directory)

    with open("xl_pdf_sets/pickled_datasheets2", "rb") as f:
        while True:
            try:
                individual_component = []
                current_id = pickle.load(f)
                individual_component.append(links_df.iloc[0])

            except EOFError:
                print('Pickle ends')
                break


def find_t_rr():
    # load the pickled lists of all the dataframes for one file
    new_lists = []
    # with open("xl_pdf_sets/pickled_datasheets_from218", "rb") as f:
    # if want the full dataset without pdf parameters, use 'xl_pdf_sets/pickled_data_no_sheets'
    with open("xl_pdf_sets/pickled_datasheets2", "rb") as f:
        while True:
            try:
                current_id = pickle.load(f)
                new_lists.append(current_id)
            except EOFError:
                print('Pickle ends')
                break

    # If want to clear all contents from pickled file
    # open("xl_pdf_sets/pickled_datasheets", "w").close()

    for datasheet_set in new_lists:
        # first, create an object with all of the known attributes
        component_obj = DigikeyFet_downloaded(datasheet_set)

        # find trr, find diF/dt. then find Coss and Vdsmeas. then find Qrr and IF.
        try:
            df = datasheet_set[2]
            trr_success = False
            diFdt_success = False
            qrr_success = False
            i_f_success = False
            IS_success = False
            coss_success = False
            vds_success = False

            trr_value = np.NaN
            diFdt_value = np.NaN
            qrr_value = np.NaN
            i_f_value = np.NaN
            IS_value = np.NaN
            coss_value = np.NaN
            vds_value = np.NaN

            # first, find IS
            if not IS_success:
                try:

                    dfnum = -1
                    colname = 'colname'

                    for i in range(len(df)):
                        print('i=%d' % i)
                        # for temporary, set i=5 because that is where the information we want is located in example datasheet
                        # i=5
                        IS_interim_success = False
                        for col in df[i].columns:
                            # print(col)
                            IS_trial_list = ["IS"]
                            for IS_trial in IS_trial_list:
                                # print(IS_trial)
                                index = [idx for idx, s in enumerate(df[i][col]) if (IS_trial in s) and ('ISM' not in s)]
                                s = [s for idx, s in enumerate(df[i][col]) if (IS_trial in s) and ('ISM' not in s)]
                                print(index)
                                print(s)
                                if index != []:
                                    IS_interim_success = True
                                    IS_index = index
                                    break
                            if IS_interim_success:
                                break
                        # now have index and column where says IS, need to find where says A and then get one value before that
                        # df[i].iloc[IS_index, col]
                        if IS_interim_success:
                            value = [s for idx, s in enumerate(df[i].iloc[IS_index[0], :])]
                            for val in value:
                                if val == 'A':
                                    print(value.index(val))
                                    Amp_column_num = value.index(val)
                                    IS_val_col_num = df[i].columns[Amp_column_num - 1]
                                    IS_val = df[i].iloc[IS_index, df[i].columns[IS_val_col_num]]

                                    # check that IS_val is numeric, so that we know we actually have a value
                                    try:
                                        IS_val = float(IS_val[0])
                                        IS_success = True
                                    except:
                                        continue
                                    print(IS_val)

                            # now have IS_value, set this to be IF, later set as object attribute
                            if IS_success:
                                IS_value = IS_val
                                break
                            else:
                                IS_value = np.NaN
                        else:
                            continue

                except Exception as e:
                    print(e)
                    print('Exception getting IS')

            # Now do same thing to get qrr
            if not qrr_success:
                try:
                    for i in range(len(df)):
                        qrr_trial_list = ["QRR", "Q RR"]
                        for qrr_trial in qrr_trial_list:
                            for col in df[i].columns:
                                if df[i][col].astype(str).str.contains(qrr_trial, case=False).any():
                                    qrr_index = min(df[i][col][df[i][col].astype(str).str.contains(qrr_trial,
                                                                                                   case=False) == True].index)
                                    dfnum = i
                                    dfnum_qrr = i
                                    colname = col
                                    print(i)
                                    print(min(df[i][col][df[i][col].astype(str).str.contains(qrr_trial,
                                                                                             case=False) == True].index))


                                    collist = df[dfnum].columns.to_list()
                                    newlist = df[dfnum].loc[qrr_index].to_list()
                                    for value in newlist:
                                        ind = newlist.index(value)
                                        if ind > collist.index(colname):
                                            print(ind)
                                            try:
                                                qrr_value = float(re.sub(r'[- ]', '', value))
                                                if ('µC' in newlist) or ('?C' in newlist):
                                                    qrr_value = qrr_value * 10 ** 3
                                                print(qrr_value)
                                                qrr_success = True
                                                break
                                            except:
                                                if isfloat(value):
                                                    qrr_value = float(value)
                                                    if ('µC' in newlist) or ('?C' in newlist):
                                                        qrr_value = qrr_value * 10 ** 3
                                                    print(qrr_value)
                                                    qrr_success = True
                                                    break
                                                else:
                                                    continue

                except Exception as e:
                    print(e)
                    print('qrr error')


            # Now do the same thing to get dIF/dt
            if not diFdt_success:
                try:
                    diFdt_success = False
                    for i in range(len(df)):
                        # i = 5  # testing, for now, for example we know df[5] is where this info is
                        diFdt_trial_list = ["diF/dt", "di/dt"]
                        for diFdt_trial in diFdt_trial_list:
                            for col in df[i].columns:
                                if df[i][col].astype(str).str.contains(diFdt_trial, case=False).any():
                                    # diFdt_value = min(df[i][col][df[i][col].astype(str).str.contains(diFdt_trial,
                                    #                                                                case=False) == True])
                                    diFdt_index = min(df[i][col][df[i][col].astype(str).str.contains(diFdt_trial,
                                                                                                     case=False) == True].index)
                                    # dfnum = i
                                    colname = col
                                    diFdt_value = df[i].iloc[diFdt_index, colname]
                                    diFdt_value = ''.join(diFdt_value.split('\n'))
                                    # print(i)
                                    # print(min(df[i][col][df[i][col].astype(str).str.contains("qrr",
                                    #                                                          case=False) == True].index))
                                    #
                                    # collist = df[dfnum].columns.to_list()
                                    # newlist = df[dfnum].loc[qrr_index].to_list()
                                    # for value in newlist:
                                    #     ind = newlist.index(value)
                                    #     if ind > collist.index(colname):
                                    #         print(ind)
                                    #         try:
                                    #             qrr_value = float(re.sub(r'[- ]', '', value))
                                    if ('µs' in diFdt_value) or ('?s' in diFdt_value):
                                        diFdt_value = diFdt_value
                                    print(diFdt_value)
                                    diFdt_success = True
                                    break
                            if diFdt_success:
                                break



                except Exception as e:
                    print(e)
                    print('dIF/dt error')

            # then do same thing to  find trr
            if not trr_success:
                try:
                    trr_success = False
                    colname = 'colname'

                    for i in range(len(df)):
                        print('i=%d' % i)
                        # for temporary, set i=5 because that is where the information we want is located in example datasheet
                        # i = 5
                        trr_interim_success = False
                        for col in df[i].columns:
                            # print(col)
                            trr_trial_list = ["trr", "tRR"]
                            for trr_trial in trr_trial_list:
                                # print(IS_trial)
                                index = [idx for idx, s in enumerate(df[i][col]) if (trr_trial in s)]
                                s = [s for idx, s in enumerate(df[i][col]) if (trr_trial in s)]
                                print(index)
                                print(s)
                                if index != []:
                                    trr_interim_success = True
                                    print('trr inter success')
                                    trr_index = index
                                    break
                            if trr_interim_success:
                                break

                        # now have index and column where says trr, need to find where says ns and then get two values before that
                        # df[i].iloc[IS_index, col]
                        if trr_interim_success:
                            value = [s for idx, s in enumerate(df[i].iloc[trr_index[0], :])]
                            for val in value:
                                if val == 'ns':
                                    print(val)
                                    print(value.index(val))
                                    ns_column_num = value.index(val)
                                    trr_val_col_num = df[i].columns[ns_column_num - 2]
                                    trr_val = df[i].iloc[trr_index, df[i].columns[trr_val_col_num]]
                                    print(trr_val)

                                    # check that trr_val is numeric, so that we know we actually have a value
                                    try:
                                        trr_val = float(trr_val)
                                        trr_success = True
                                    except:
                                        continue
                                    # print(trr_val)

                            if trr_success:
                                # now have trr_value, set this to be trr, later set as object attribute
                                trr_value = trr_val
                                break
                            else:
                                trr_value = np.NaN

                        else:
                            continue


                except Exception as e:
                    print(e)
                    print('Exception getting trr')

            # Now do same thing to get IF -- if says IF=IS in datasheet, set IF = IS. Assume IF in same df as qrr
            if not i_f_success:
                try:
                    if_trial_list = ['I F', 'IF', 'I S', 'IS', 'I SD', 'ISD', 'I  = I', 'I =', 'I=', 'IF=IS']
                    # dfnum = 5  # for now, test case
                    dfnum = dfnum_qrr
                    for col in df[dfnum].columns:
                        for if_trial in if_trial_list:
                            try:
                                if any(k >= qrr_index - 2 for k in
                                       (df[dfnum][col].str.contains(if_trial, case=False) == True)[
                                           df[dfnum][col].str.contains(if_trial,
                                                                       case=False) == True].index):
                                    I_F_col_name = col
                                    print(col)
                                    I_F_index = \
                                        (df[dfnum][col].str.contains(if_trial, case=False) == True)[
                                            df[dfnum][col].str.contains(if_trial, case=False) == True].index
                                    print(I_F_index)
                                    m = max(val for val in I_F_index if val >= qrr_index - 2)
                                    i_f_value = df[dfnum].loc[m, I_F_col_name]
                                    i_f_value = ''.join(i_f_value.split('\n'))
                                    print('IF_value:')
                                    print(df[dfnum].loc[m, I_F_col_name])
                                    i_f_success = True
                                    break

                            except:
                                print('exception')
                            if i_f_success:
                                break
                except Exception as e:
                    print(e)
                    print('Exception getting IF')

            # now do similar to find Coss
            if not coss_success:
                try:
                    dfnum = -1
                    colname = 'colname'
                    for i in range(len(df)):
                        # i = 3
                        print('i=%d' % i)
                        for col in df[i].columns:
                            coss_trial_list = ["COSS", "C OSS", "Output capacitance"]
                            for coss_trial in coss_trial_list:
                                print(f"coss trial: {coss_trial}")
                                if df[i][col].astype(str).str.contains(coss_trial, case=False).any():
                                    coss_index = min(df[i][col][df[i][col].astype(str).str.contains(coss_trial,
                                                                                                    case=False) == True].index)
                                    dfnum = i
                                    colname = col  # this sets the column name containing the 'Coss' label, need to find next value greater than that
                                    # to get actual value, do this by indexing through the list of column names
                                    print(i)
                                    print(min(df[i][col][df[i][col].astype(str).str.contains(coss_trial,
                                                                                             case=False) == True].index))
                                    print("coss label location: ")
                                    print(df[i].iloc[coss_index, colname])
                                    collist = df[dfnum].columns.to_list()
                                    newlist = df[dfnum].loc[coss_index].to_list()
                                    for value in newlist:
                                        print(f"values in coss_index row: {value}")
                                        ind = newlist.index(value)
                                        if 'pF' in value:
                                            print('pF in value')
                                            coss_value = value
                                            coss_value = ''.join(coss_value.split('\n'))
                                            coss_success = True
                                            break
                                        if ind > collist.index(colname):
                                            print(ind)
                                            if (isfloat(value)):
                                                # print(value)
                                                coss_value = float(value)
                                                coss_value = ''.join(coss_value.split('\n'))
                                                print('Coss: %f' % coss_value)
                                                coss_success = True
                                                print('coss_success is True')
                                                coss_colname = colname
                                                break
                                        if coss_success:
                                            break

                                    if coss_success:
                                        break
                                if coss_success:
                                    break

                            if coss_success:
                                break

                        if coss_success:
                            break

                except:
                    print('Coss error')

            # now do similar to find Vds
            vds_success = False
            if not vds_success:
                try:
                    # keep i = dfnum from Coss, they will be in same table
                    # also, only check columns with column indices > colname
                    colname = colname

                    vds_trial_list = ["VDS", "V DS"]
                    # refer back to collist to check if column index > colname.
                    # also need to check that index (row) is > that of Coss label itself.
                    collist = df[dfnum].columns.to_list()
                    for vds_trial in vds_trial_list:
                        if df[dfnum][colname].astype(str).str.contains(vds_trial).any():
                            print('identified vds value in coss label/value')
                            dynamic_index = (df[dfnum][colname].str.contains(vds_trial, case=False) == True)[
                                df[dfnum][colname].str.contains(vds_trial, case=False) == True].index
                            if len([val for val in dynamic_index if val >= coss_index - 2]) > 0:

                                try:
                                    m = min(val for val in dynamic_index if val >= coss_index - 2)
                                    print(m)
                                    vds_value = df[dfnum].loc[m, colname]
                                    vds_value = ''.join(vds_value.split('\n'))
                                    vds_success = True
                                    break

                                except:
                                    m = min(val for val in dynamic_index if val >= coss_index)
                                    print(m)
                                    vds_value = df[dfnum].loc[m, colname]
                                    vds_value = ''.join(vds_value.split('\n'))
                                    vds_success = True
                                    break

                    for col in df[dfnum].columns:
                        # colindex = collist.index(col)
                        # print(f"col search index: {colindex}")
                        # colindex2 = collist.index(colname)
                        # print(f"colname of Coss label index: {colindex2}")

                        if collist.index(col) > collist.index(colname):
                            print('col index > colname index')
                            for vds_trial in vds_trial_list:
                                if (df[dfnum][col].astype(str).str.contains(vds_trial).any()):
                                    print('identified')
                                    dynamic_col_name = col
                                    print(dynamic_col_name)
                                    dynamic_index = (df[dfnum][col].str.contains(vds_trial, case=False) == True)[
                                        df[dfnum][col].str.contains(vds_trial, case=False) == True].index
                                    print(dynamic_index)
                                    list2 = [index_check for index_check in dynamic_index if (
                                                (index_check >= coss_index - 2) and not (
                                                    "RDS" in df[dfnum].iloc[index_check, col]))]
                                    print(f"{list2}")
                                    if (any([index_check for index_check in dynamic_index if (
                                            (index_check >= coss_index - 2) and not (
                                            "RDS" in df[dfnum].iloc[index_check, col]))])):
                                        print('vds_trial located')
                                        if len([val for val in list2 if val >= coss_index - 2]) > 0:
                                            try:
                                                m = min(val for val in list2 if val >= coss_index - 2)
                                                print(m)

                                            except:
                                                m = min(val for val in list2 if val >= coss_index)
                                                print(m)

                                            print('Vds: %s' % df[dfnum].loc[m, dynamic_col_name])
                                            vds_value = df[dfnum].loc[m, dynamic_col_name]
                                            vds_value = ''.join(vds_value.split('\n'))
                                            vds_success = True
                                            break

                        if vds_success:
                            break

                except Exception as e:
                    print(e)
                    print('Vds error')

            # now do similar to find Coss
            # if not coss_success:
            #     try:
            #         dfnum = -1
            #         colname = 'colname'
            #         for i in range(len(df)):
            #             print('i=%d' % i)
            #             for col in df[i].columns:
            #                 coss_trial_list = ["COSS", "C OSS", "Output capacitance"]
            #                 vds_trial_list = ["VDS", "V DS"]
            #                 for coss_trial in coss_trial_list:
            #                     if df[i][col].astype(str).str.contains(coss_trial, case=False).any():
            #                         coss_index = min(df[i][col][df[i][col].astype(str).str.contains(coss_trial,
            #                                                                                         case=False) == True].index)
            #                         dfnum = i
            #                         colname = col
            #                         print(i)
            #                         print(min(df[i][col][df[i][col].astype(str).str.contains(coss_trial,
            #                                                                                  case=False) == True].index))
            #                         for col in df[dfnum].columns:
            #                             for vds_trial in vds_trial_list:
            #                                 if df[dfnum][col].astype(str).str.contains(vds_trial).any():
            #                                     dynamic_col_name = col
            #                                     dynamic_index = (df[i][col].str.contains(vds_trial, case=False) == True)[
            #                                         df[i][col].str.contains(vds_trial, case=False) == True].index
            #                                     try:
            #                                         m = min(val for val in dynamic_index if val >= coss_index)
            #                                     except:
            #                                         m = min(val for val in dynamic_index if val >= coss_index - 1)
            #                                     print('Vds: %s' % df[i].loc[m, dynamic_col_name])
            #                                     vds_value = df[i].loc[m, dynamic_col_name]
            #                                     vds_value = ''.join(vds_value.split('\n'))
            #                                     vds_success = True
            #                                     break
            #                             collist = df[dfnum].columns.to_list()
            #                             newlist = df[dfnum].loc[coss_index].to_list()
            #                             for value in newlist:
            #                                 ind = newlist.index(value)
            #                                 if ind > collist.index(colname):
            #                                     print(ind)
            #                                     if (isfloat(value)):
            #                                         # print(value)
            #                                         coss_value = float(value)
            #                                         print('Coss: %f' % coss_value)
            #                                         coss_success = True
            #                                         break
            #
            #                         if dfnum != -1:
            #                             break
            #
            #     except:
            #         print('Coss or Vds error')
            #         continue

            # # then do something similar to find Vds_meas
            # if not vds_success:
            #     try:
            #         dfnum = -1
            #         colname = 'colname'
            #         for i in range(len(df)):
            #             print('i=%d' % i)
            #             for col in df[i].columns:
            #                 coss_trial_list = ["COSS", "C OSS", "Output capacitance"]
            #                 vds_trial_list = ["VDS", "V DS"]
            #                 for coss_trial in coss_trial_list:
            #                     if df[i][col].astype(str).str.contains(coss_trial, case=False).any():
            #                         coss_index = min(df[i][col][df[i][col].astype(str).str.contains(coss_trial,
            #                                                                                         case=False) == True].index)
            #                         dfnum = i
            #                         colname = col
            #                         print(i)
            #                         print(min(df[i][col][df[i][col].astype(str).str.contains(coss_trial,
            #                                                                                  case=False) == True].index))
            #                         for col in df[dfnum].columns:
            #                             for vds_trial in vds_trial_list:
            #                                 if df[dfnum][col].astype(str).str.contains(vds_trial).any():
            #                                     dynamic_col_name = col
            #                                     dynamic_index = \
            #                                     (df[i][col].str.contains(vds_trial, case=False) == True)[
            #                                         df[i][col].str.contains(vds_trial, case=False) == True].index
            #                                     try:
            #                                         m = min(val for val in dynamic_index if val >= coss_index)
            #                                     except:
            #                                         m = min(val for val in dynamic_index if val >= coss_index - 1)
            #                                     print('Vds: %s' % df[i].loc[m, dynamic_col_name])
            #                                     vds_value = df[i].loc[m, dynamic_col_name]
            #                                     vds_success = True
            #                                     break
            #                             collist = df[dfnum].columns.to_list()
            #                             newlist = df[dfnum].loc[coss_index].to_list()
            #                             for value in newlist:
            #                                 ind = newlist.index(value)
            #                                 if ind > collist.index(colname):
            #                                     print(ind)
            #                                     if (isfloat(value)):
            #                                         # print(value)
            #                                         coss_value = float(value)
            #                                         print('Coss: %f' % coss_value)
            #                                         coss_success = True
            #                                         break
            #
            #                         if dfnum != -1:
            #                             break
            #
            #     except:
            #         print('Vds error')
            #         continue


            try:
                component_obj.IS = IS_value
                component_obj.Q_rr = qrr_value
                component_obj.t_rr = trr_value
                component_obj.diFdt = diFdt_value
                component_obj.I_F = i_f_value
                component_obj.Coss = coss_value
                component_obj.Vds_meas = vds_value
            except:
                component_obj.IS = np.NaN
                component_obj.Q_rr = np.NaN
                component_obj.t_rr = np.NaN
                component_obj.diFdt = np.NaN
                component_obj.I_F = np.NaN
                component_obj.Coss = np.NaN
                component_obj.Vds_meas = np.NaN
        except:
            print('unable to get pdf parameters for element')
            component_obj.IS = np.NaN
            component_obj.Q_rr = np.NaN
            component_obj.t_rr = np.NaN
            component_obj.diFdt = np.NaN
            component_obj.I_F = np.NaN
            component_obj.Coss = np.NaN
            component_obj.Vds_meas = np.NaN

        try:
            single_fet_to_csv_downloaded(component_obj, 'csv_files/FET_pdf_tables_wt_rr_full.csv')
            print(component_obj.mfr_part_no)
        except:
            continue



            # # Now do the same thing to get qrr, trr, IF, and dIF/dt
            # try:
            #     for i in range(len(df)):
            #         trr_trial_list = ["trr"]
            #         diFdt_trial_list = ["diF/dt"]
            #         qrr_trial_list = ["QRR", "Q RR"]
            #         if_trial_list = ['I F', 'IF', 'I S', 'IS', 'I SD', 'ISD', 'I  = I', 'I =', 'I=']
            #         for qrr_trial in qrr_trial_list:
            #             if df[i][col].astype(str).str.contains(qrr_trial, case=False).any():
            #                 qrr_index = min(df[i][col][df[i][col].astype(str).str.contains("qrr",
            #                                                                                case=False) == True].index)
            #                 dfnum = i
            #                 colname = col
            #                 print(i)
            #                 print(min(df[i][col][df[i][col].astype(str).str.contains("qrr",
            #                                                                          case=False) == True].index))
            #
            #                 for col in df[dfnum].columns:
            #                     for if_trial in if_trial_list:
            #                         if df[dfnum][col].astype(str).str.contains(if_trial).any():
            #                             try:
            #                                 if any(k >= qrr_index - 2 for k in
            #                                        (df[dfnum][col].str.contains(if_trial, case=False) == True)[
            #                                            df[dfnum][col].str.contains(if_trial,
            #                                                                        case=False) == True].index):
            #                                     I_F_col_name = col
            #                                     I_F_index = \
            #                                         (df[dfnum][col].str.contains(if_trial, case=False) == True)[
            #                                             df[dfnum][col].str.contains(if_trial, case=False) == True].index
            #                                     m = min(val for val in I_F_index if val >= qrr_index - 2)
            #                                     print(df[dfnum].loc[m, I_F_col_name])
            #                                     i_f_value = df[dfnum].loc[m, I_F_col_name]
            #                                     i_f_success = True
            #
            #                                     break
            #                             except:
            #                                 continue
            #                 collist = df[dfnum].columns.to_list()
            #                 newlist = df[dfnum].loc[qrr_index].to_list()
            #                 for value in newlist:
            #                     ind = newlist.index(value)
            #                     if ind > collist.index(colname):
            #                         print(ind)
            #                         try:
            #                             qrr = float(re.sub(r'[- ]', '', value))
            #                             if ('µC' in newlist) or ('?C' in newlist):
            #                                 qrr = qrr * 10 ** 3
            #                             print(qrr)
            #                             qrr_success = True
            #                             break
            #                         except:
            #                             if isfloat(value):
            #                                 qrr = float(value)
            #                                 if ('µC' in newlist) or ('?C' in newlist):
            #                                     qrr = qrr * 10 ** 3
            #                                 print(qrr)
            #                                 qrr_success = True
            #                                 break
            #                             else:
            #                                 continue
            #
            # except Exception as e:
            #     print(e)
            #     print('qrr or IF error')
            #     continue
            #
            #
            #             # for col in df[i].columns:
            #         #     IS_trial_list = ['I F', 'IF', 'I S', 'IS', 'I SD', 'ISD', 'I  = I', 'I =', 'I=']
            #         #     for IS_trial in IS_trial_list:
            #                 # if df[i][col].astype(str).str.contains(IS_trial, case=False).any():
            #                 #     if df[i][col].astype(str).str.contains("ISM", case=False).any() == True:
            #                 #         continue
            #                 #     else:
            #                 #         IS_index = min(df[i][col][df[i][col].astype(str).str.contains(IS_trial,
            #                 #                                                                       case=False) == True].index)
            #                 #     break
            #
            #             # break
            #
            #             # have the index where the IS value is, now need to find the row where contains a value
            #             if any(k >= qrr_index - 2 for k in
            #                    (df[dfnum][col].str.contains(if_trial, case=False) == True)[
            #                        df[dfnum][col].str.contains(if_trial,
            #                                                    case=False) == True].index):
            #                 I_F_col_name = col
            #                 I_F_index = \
            #                     (df[dfnum][col].str.contains(if_trial, case=False) == True)[
            #                         df[dfnum][col].str.contains(if_trial, case=False) == True].index
            #                 m = min(val for val in I_F_index if val >= qrr_index - 2)
            #                 print(df[dfnum].loc[m, I_F_col_name])
            #                 i_f_value = df[dfnum].loc[m, I_F_col_name]
            #                 i_f_success = True
            #
            #             trr_trial_list = ["trr"]
            #             diFdt_trial_list = ["diF/dt"]
            #             qrr_trial_list = ["QRR", "Q RR"]
            #             if_trial_list = ['I F', 'IF', 'I S', 'IS', 'I SD', 'ISD', 'I  = I', 'I =', 'I=']
            #             for qrr_trial in qrr_trial_list:
            #                 if df[i][col].astype(str).str.contains(qrr_trial, case=False).any():
            #                     qrr_index = min(df[i][col][df[i][col].astype(str).str.contains("qrr",
            #                                                                                    case=False) == True].index)
            #                     dfnum = i
            #                     colname = col
            #                     print(i)
            #                     print(min(df[i][col][df[i][col].astype(str).str.contains("qrr",
            #                                                                              case=False) == True].index))
            #
            #                     for col in df[dfnum].columns:
            #                         for if_trial in if_trial_list:
            #                             if df[dfnum][col].astype(str).str.contains(if_trial).any():
            #                                 try:
            #                                     if any(k >= qrr_index - 2 for k in
            #                                            (df[dfnum][col].str.contains(if_trial, case=False) == True)[
            #                                                df[dfnum][col].str.contains(if_trial,
            #                                                                            case=False) == True].index):
            #                                         I_F_col_name = col
            #                                         I_F_index = \
            #                                         (df[dfnum][col].str.contains(if_trial, case=False) == True)[
            #                                             df[dfnum][col].str.contains(if_trial, case=False) == True].index
            #                                         m = min(val for val in I_F_index if val >= qrr_index - 2)
            #                                         print(df[dfnum].loc[m, I_F_col_name])
            #                                         i_f_value = df[dfnum].loc[m, I_F_col_name]
            #                                         i_f_success = True
            #
            #                                         break
            #                                 except:
            #                                     continue
            #                     collist = df[dfnum].columns.to_list()
            #                     newlist = df[dfnum].loc[qrr_index].to_list()
            #                     for value in newlist:
            #                         ind = newlist.index(value)
            #                         if ind > collist.index(colname):
            #                             print(ind)
            #                             try:
            #                                 qrr = float(re.sub(r'[- ]', '', value))
            #                                 if ('µC' in newlist) or ('?C' in newlist):
            #                                     qrr = qrr * 10 ** 3
            #                                 print(qrr)
            #                                 qrr_success = True
            #                                 break
            #                             except:
            #                                 if isfloat(value):
            #                                     qrr = float(value)
            #                                     if ('µC' in newlist) or ('?C' in newlist):
            #                                         qrr = qrr * 10 ** 3
            #                                     print(qrr)
            #                                     qrr_success = True
            #                                     break
            #                                 else:
            #                                     continue
            #
            # except Exception as e:
            #     print(e)
            #     print('qrr or IF error')
            #     continue

            #########################################
        #     dfnum = -1
        #     colname = 'colname'
        #     for i in range(len(df)):
        #         print('i=%d' % i)
        #         for col in df[i].columns:
        #             coss_trial_list = ["COSS", "C OSS", "Output capacitance"]
        #             vds_trial_list = ["VDS", "V DS"]
        #             for coss_trial in coss_trial_list:
        #                 if df[i][col].astype(str).str.contains(coss_trial, case=False).any():
        #                     coss_index = min(df[i][col][df[i][col].astype(str).str.contains(coss_trial,
        #                                                                                     case=False) == True].index)
        #                     dfnum = i
        #                     colname = col
        #                     print(i)
        #                     print(min(df[i][col][df[i][col].astype(str).str.contains(coss_trial,
        #                                                                              case=False) == True].index))
        #                     for col in df[dfnum].columns:
        #                         for vds_trial in vds_trial_list:
        #                             if df[dfnum][col].astype(str).str.contains(vds_trial).any():
        #                                 dynamic_col_name = col
        #                                 dynamic_index = (df[i][col].str.contains(vds_trial, case=False) == True)[
        #                                     df[i][col].str.contains(vds_trial, case=False) == True].index
        #                                 try:
        #                                     m = min(val for val in dynamic_index if val >= coss_index)
        #                                 except:
        #                                     m = min(val for val in dynamic_index if val >= coss_index - 1)
        #                                 print('Vds: %s' % df[i].loc[m, dynamic_col_name])
        #                                 vds_value = df[i].loc[m, dynamic_col_name]
        #                                 vds_success = True
        #                                 break
        #                         collist = df[dfnum].columns.to_list()
        #                         newlist = df[dfnum].loc[coss_index].to_list()
        #                         for value in newlist:
        #                             ind = newlist.index(value)
        #                             if ind > collist.index(colname):
        #                                 print(ind)
        #                                 if (isfloat(value)):
        #                                     # print(value)
        #                                     coss_value = float(value)
        #                                     print('Coss: %f' % coss_value)
        #                                     coss_success = True
        #                                     break
        #
        #                     if dfnum != -1:
        #                         break
        #
        # except:
        #     print('Coss or Vds error')
        #     continue

        # find qrr and IF (or IS)
        # try:
        #     qrr_success = False
        #     i_f_success = False
        #     dfnum = -1
        #     colname = 'colname'
        #
        #     for i in range(len(df)):
        #         print('i=%d' % i)
        #         for col in df[i].columns:
        #             qrr_trial_list = ["QRR", "Q RR"]
        #             if_trial_list = ['I F', 'IF', 'I S', 'IS', 'I SD', 'ISD', 'I  = I', 'I =', 'I=']
        #             for qrr_trial in qrr_trial_list:
        #                 if df[i][col].astype(str).str.contains(qrr_trial, case=False).any():
        #                     qrr_index = min(df[i][col][df[i][col].astype(str).str.contains("qrr",
        #                                                                                    case=False) == True].index)
        #                     dfnum = i
        #                     colname = col
        #                     print(i)
        #                     print(min(df[i][col][df[i][col].astype(str).str.contains("qrr",
        #                                                                              case=False) == True].index))
        #
        #                     for col in df[dfnum].columns:
        #                         for if_trial in if_trial_list:
        #                             if df[dfnum][col].astype(str).str.contains(if_trial).any():
        #                                 try:
        #                                     if any(k >= qrr_index - 2 for k in
        #                                            (df[dfnum][col].str.contains(if_trial, case=False) == True)[
        #                                                df[dfnum][col].str.contains(if_trial,
        #                                                                            case=False) == True].index):
        #                                         I_F_col_name = col
        #                                         I_F_index = (df[dfnum][col].str.contains(if_trial, case=False) == True)[
        #                                             df[dfnum][col].str.contains(if_trial, case=False) == True].index
        #                                         m = min(val for val in I_F_index if val >= qrr_index - 2)
        #                                         print(df[dfnum].loc[m, I_F_col_name])
        #                                         i_f_value = df[dfnum].loc[m, I_F_col_name]
        #                                         i_f_success = True
        #
        #                                         break
        #                                 except:
        #                                     continue
        #                     collist = df[dfnum].columns.to_list()
        #                     newlist = df[dfnum].loc[qrr_index].to_list()
        #                     for value in newlist:
        #                         ind = newlist.index(value)
        #                         if ind > collist.index(colname):
        #                             print(ind)
        #                             try:
        #                                 qrr = float(re.sub(r'[- ]', '', value))
        #                                 if ('µC' in newlist) or ('?C' in newlist):
        #                                     qrr = qrr * 10 ** 3
        #                                 print(qrr)
        #                                 qrr_success = True
        #                                 break
        #                             except:
        #                                 if isfloat(value):
        #                                     qrr = float(value)
        #                                     if ('µC' in newlist) or ('?C' in newlist):
        #                                         qrr = qrr * 10 ** 3
        #                                     print(qrr)
        #                                     qrr_success = True
        #                                     break
        #                                 else:
        #                                     continue
        #
        # except Exception as e:
        #     print(e)
        #     print('qrr or IF error')
        #     continue
        #




    # if not coss_success:
    #     coss = np.NaN
    # if not qrr_success:
    #     qrr = np.NaN
    # if not vds_success:
    #     vds_value = np.NaN
    # if not i_f_success:
    #     i_f_value = np.NaN
    #     # fet_pdf_df.iloc[k]['Coss'] = 'nan'
    #     # fet_pdf_info_dict['Qrr'][link.get('title').replace(" | Datasheet", "")] = 'nan'
    #     print('Invalid datasheet url exception')

    print('component scraping complete')

        # # first locate Coss and Vds
        # try:
        #     coss_success = False
        #     df = datasheet_set[2]
        #     dfnum = -1
        #     colname = 'colname'
        #     for i in range(len(df)):
        #         print('i=%d' % i)
        #         for col in df[i].columns:
        #             if df[i][col].astype(str).str.contains("COSS", case=False).any():
        #                 coss_index = min(df[i][col][df[i][col].astype(str).str.contains("COSS",
        #                                                                                 case=False) == True].index)
        #                 dfnum = i
        #                 colname = col
        #                 print(i)
        #                 print(min(df[i][col][df[i][col].astype(str).str.contains("COSS",
        #                                                                          case=False) == True].index))
        #                 for col in df[dfnum].columns:
        #                     if 'vds' in col.lower() or 'v ds' in col.lower():
        #                         vds_value = col
        #                         vds_success = True
        #                         break
        #                     list2 = [item for item in
        #                              (df[dfnum][col].astype(str).str.contains('VDS', case=False) == True)[
        #                                  df[dfnum][col].astype(str).str.contains('VDS', case=False) == True].index if
        #                              item >= coss_index]
        #                     # if df[dfnum][col].astype(str).str.contains('VDS').any() and (df[i][col].str.contains('VDS', case=False) == True)[
        #                     #         df[i][col].str.contains('VDS', case=False) == True].index[0] >= coss_index:
        #                     if len(list2) > 0:
        #                         dynamic_col_name = col
        #                         dynamic_index = (df[dfnum][col].str.contains('VDS', case=False) == True)[
        #                             df[dfnum][col].str.contains('VDS', case=False) == True].index
        #                         m = min(val for val in dynamic_index if val >= coss_index)
        #                         print(df[dfnum].loc[m, dynamic_col_name])
        #                         vds_value = df[dfnum].loc[m, dynamic_col_name]
        #                         vds_success = True
        #                         break
        #                 collist = df[dfnum].columns.to_list()
        #                 newlist = df[dfnum].loc[coss_index].to_list()
        #                 for value in newlist:
        #                     ind = newlist.index(value)
        #                     if ind > collist.index(colname):
        #                         print(ind)
        #                         if (isfloat(value)):
        #                             # print(value)
        #                             coss = float(value)
        #                             print(coss)
        #                             coss_success = True
        #                             break
        #                 if dfnum != -1:
        #                     break
        #             else:
        #                 coss_trial_list = ["C OSS", "Output capacitance"]
        #                 vds_trial_list = ["VDS", "V DS"]
        #                 for coss_trial in coss_trial_list:
        #                     if df[i][col].astype(str).str.contains(coss_trial, case=False).any():
        #                         coss_index = min(df[i][col][df[i][col].astype(str).str.contains(coss_trial,
        #                                                                                         case=False) == True].index)
        #                         dfnum = i
        #                         colname = col
        #                         print(i)
        #                         print(min(df[i][col][df[i][col].astype(str).str.contains(coss_trial,
        #                                                                                  case=False) == True].index))
        #                         for col in df[dfnum].columns:
        #                             for vds_trial in vds_trial_list:
        #                                 if df[dfnum][col].astype(str).str.contains(vds_trial).any():
        #                                     dynamic_col_name = col
        #                                     dynamic_index = (df[i][col].str.contains(vds_trial, case=False) == True)[
        #                                         df[i][col].str.contains(vds_trial, case=False) == True].index
        #                                     try:
        #                                         m = min(val for val in dynamic_index if val >= coss_index)
        #                                     except:
        #                                         m = min(val for val in dynamic_index if val >= coss_index - 1)
        #                                     print(df[i].loc[m, dynamic_col_name])
        #                                     vds_value = df[i].loc[m, dynamic_col_name]
        #                                     vds_success = True
        #                                     break
        #                             collist = df[dfnum].columns.to_list()
        #                             newlist = df[dfnum].loc[coss_index].to_list()
        #                             for value in newlist:
        #                                 ind = newlist.index(value)
        #                                 if ind > collist.index(colname):
        #                                     print(ind)
        #                                     if (isfloat(value)):
        #                                         # print(value)
        #                                         coss = float(value)
        #                                         print(coss)
        #                                         coss_success = True
        #                                         break
        #
        #                         if dfnum != -1:
        #                             break
        #
        # except:
        #     return None
        #
        #     #     if coss_success:
        #     #         # if dfnum != -1:
        #     #
        #     #         break
        #     #
        #     #
        #     #
        #     # collist = df[dfnum].columns.to_list()
        #     # newlist = df[dfnum].loc[coss_index].to_list()
        #     # for value in newlist:
        #     #     ind = newlist.index(value)
        #     #     if ind > collist.index(colname):
        #     #         print(ind)
        #     #         if (isfloat(value)):
        #     #             # print(value)
        #     #             coss = float(value)
        #     #             print(coss)
        #     #             coss_success = True
        #     #             break
        #
        #
        #


    #     newmeth = True
    #     # if 'EPC' in link.get('title'):
    #     if newmeth:
    #         qrr = 0
    #         dfnum = -1
    #         colname = 'colname'
    #
    #         for i in range(len(df)):
    #             print('i=%d' % i)
    #             for col in df[i].columns:
    #                 if df[i][col].astype(str).str.contains("qrr", case=False).any():
    #                     qrr_index = min(df[i][col][df[i][col].astype(str).str.contains("qrr",
    #                                                                                    case=False) == True].index)
    #                     dfnum = i
    #                     colname = col
    #                     print(i)
    #                     print(min(df[i][col][df[i][col].astype(str).str.contains("qrr",
    #                                                                              case=False) == True].index))
    #                     break
    #
    #                 elif df[i][col].astype(str).str.contains("q rr", case=False).any():
    #                     qrr_index = min(df[i][col][df[i][col].astype(str).str.contains("q rr",
    #                                                                                    case=False) == True].index)
    #                     dfnum = i
    #                     colname = col
    #                     print(i)
    #                     print(min(df[i][col][df[i][col].astype(str).str.contains("q rr",
    #                                                                              case=False) == True].index))
    #                     break
    #                 # if dfnum == i:
    #                 #     break
    #
    #         for col in df[dfnum].columns:
    #             keyword_list = ['I F', 'IF', 'I S', 'IS', 'I SD', 'ISD', 'I  = I', 'I =', 'I=']
    #             for keyword in keyword_list:
    #                 if df[dfnum][col].astype(str).str.contains(keyword).any():
    #                     try:
    #                         if any(k >= qrr_index - 2 for k in
    #                                (df[dfnum][col].str.contains(keyword, case=False) == True)[
    #                                    df[dfnum][col].str.contains(keyword,
    #                                                                case=False) == True].index):
    #                             I_F_col_name = col
    #                             I_F_index = (df[dfnum][col].str.contains(keyword, case=False) == True)[
    #                                 df[dfnum][col].str.contains(keyword, case=False) == True].index
    #                             m = min(val for val in I_F_index if val >= qrr_index - 2)
    #                             print(df[dfnum].loc[m, I_F_col_name])
    #                             i_f_value = df[dfnum].loc[m, I_F_col_name]
    #                             i_f_success = True
    #
    #                             break
    #                     except:
    #                         continue
    #
    #
    #         collist = df[dfnum].columns.to_list()
    #         newlist = df[dfnum].loc[qrr_index].to_list()
    #         for value in newlist:
    #             ind = newlist.index(value)
    #             if ind > collist.index(colname):
    #                 print(ind)
    #                 try:
    #                     qrr = float(re.sub(r'[- ]', '', value))
    #                     if ('µC' in newlist) or ('?C' in newlist):
    #                         qrr = qrr * 10 ** 3
    #                     print(qrr)
    #                     qrr_success = True
    #                     break
    #                 except:
    #                     if isfloat(value):
    #                         qrr = float(value)
    #                         if ('µC' in newlist) or ('?C' in newlist):
    #                             qrr = qrr * 10 ** 3
    #                         print(qrr)
    #                         qrr_success = True
    #                         break
    #                     else:
    #                         continue
    #
    # except:
    #     if not coss_success:
    #         coss = np.NaN
    #     if not qrr_success:
    #         qrr = np.NaN
    #     if not vds_success:
    #         vds_value = np.NaN
    #     if not i_f_success:
    #         i_f_value = np.NaN
    #         # fet_pdf_df.iloc[k]['Coss'] = 'nan'
    #         # fet_pdf_info_dict['Qrr'][link.get('title').replace(" | Datasheet", "")] = 'nan'
    #         print('Invalid datasheet url exception')


def full_pdf_part_info(start_url, total_pages):
    csv_name = 'csv_files/FET_pdf_tables_wVds_full.csv' # Note that last previous scraping was put onto this csv
    # import os
    # import digikey
    # from digikey.v3.productinformation import KeywordSearchRequest
    #
    # os.environ['DIGIKEY_CLIENT_ID'] = 'Zgw6OYINWKyePP7ROGenAGTqGC3SVlnI'
    # os.environ['DIGIKEY_CLIENT_SECRET'] = 'na2ggc2vowS1EfE3'
    # os.environ['DIGIKEY_CLIENT_SANDBOX'] = 'False'
    # os.environ['DIGIKEY_STORAGE_PATH'] = "C:"
    #
    # # Query product number
    # search_request = KeywordSearchRequest(keywords='CRCW080510K0FKEA', record_count=10)
    # result = digikey.keyword_search(body=search_request)
    # csv_name = 'csv_files/FET_pdf_tables_wt_rr_full.csv'

    component_type = 'fet'
    if component_type == 'fet':
        # total_pages right now is coming from the multiprocessing function, but could instead call total_num_pages()
        # to determine how many pages exist on the entire site, determining the number of times we will have to click
        # through.
        #   total_pages = total_num_pages(start_url)

        # Get the driver for the starting url
        transistor_parser = TransistorParser(start_url)
        driver = transistor_parser.get_driver()
        driver.get(transistor_parser.page_url(1))

        # Iterate through all the pages on the site. Start by downloading the page, then click to the next page
        for pg in range(450):
            print(pg)

            # find and click the 'download table' button on the page
            python_button = driver.find_element_by_xpath(
                '/html/body/div[2]/div/main/section/div[2]/div[1]/div/div[4]/div/button')
            driver.execute_script("arguments[0].click();", python_button)
            print('clicked')

            # find and click the 'download visible contents' button after opening the download table box
            try:
                python_button = driver.find_element_by_xpath(
                    '/html/body/div[7]/div[3]/div/div[2]/div/button')
            except:
                try:
                    python_button = driver.find_element_by_xpath('/html/body/div[12]/div[3]/div/div[2]/div/button')
                except:
                    python_button = driver.find_element_by_xpath('/html/body/div[7]/div[3]/div/div[2]/div/button')

            driver.execute_script("arguments[0].click();", python_button)
            print('clicked')

            # find and click the button to click to the next page
            if pg == 0:
                python_button = driver.find_element_by_xpath('/html/body/div[2]/div/main/section/div[2]/div[3]/div/div[3]/div/button[6]')
            else:
                python_button = driver.find_element_by_xpath(
                    '/html/body/div[2]/div/main/section/div[2]/div[3]/div/div[3]/div/button[8]')
            driver.execute_script("arguments[0].click();", python_button)
            print('clicked')




        ### stop here for updated scraping method ###
        for pg in range(total_pages):

            python_button = driver.find_element_by_xpath(
                '/html/body/div[2]/div/main/section/div[2]/div[3]/div/div[3]/div/button[8]')
            driver.execute_script("arguments[0].click();", python_button)
            print('clicked')
            python_button = driver.find_element_by_xpath(
                '/html/body/div[2]/div/main/section/div[2]/div[1]/div/div[4]/div/button')
            driver.execute_script("arguments[0].click();", python_button)
            print('clicked')

            python_button = driver.find_element_by_xpath(
                '/html/body/div[12]/div[3]/div/div[2]/div/button')
            driver.execute_script("arguments[0].click();", python_button)
            print('clicked')

            # if (i % 10 == 0):
            # driver.refresh()
            # Have an attempts variable to prevent the driver from becoming stale
            attempts = 0

            # while (attempts < 3):

            try:
                # Get all pdfs on page


                # # logging.info('Getting part numbers for page: {:}'.format(k))
                # # figure out how many components are on the page
                # table = driver.find_elements_by_xpath(
                #     '/html/body/div[2]/main/section/div[2]/div[2]/div/div[1]/table/tbody/tr')
                table = driver.find_elements_by_xpath('/html/body/div[2]/div/main/section/div[2]/div[2]/div/div[1]/table/tbody/tr')
                # # print("number of parts on page: %s" % len(table))
                #
                # print("Getting parts for page: %s" % str(i + 2))
                driver.implicitly_wait(5)

                # Go through every component on the page and add its information as a line to the specified csv
                for j in range(len(table)):
                    # driver.refresh()

                    # for j in range(len(2)):

                    j = j + 1

                    part_number_cell = driver.find_element_by_xpath(f'/html/body/div[2]/div/main/section/div[2]/div[2]/div/div[1]/table/tbody/tr[1]/td[2]/div/div[3]/a[1]')
                    part_url = part_number_cell.get_attribute('href')
                    part_no = part_number_cell.text

                    # except:  # exceptions.StaleElementReferenceException as e:
                    #     print("exception with part no./url")

                    fet = DigikeyFet(driver, j)


                    # Get the Qrr and Coss values and write onto csv
                    # from urllib import request
                    # response = request.urlopen(start_url).read()
                    # soup = BeautifulSoup(response, "html.parser")
                    # links = soup.find_all('a', href=re.compile(r'(.pdf)'))

                    import os
                    import digikey
                    from digikey.v3.productinformation import KeywordSearchRequest

                    os.environ['DIGIKEY_CLIENT_ID'] = 'vM3AaT34JW9LVugmrQ9F0nRSwhAvX00o'
                    os.environ['DIGIKEY_CLIENT_SECRET'] = 'U8I0zbnk1fLd6PcR'
                    os.environ['DIGIKEY_CLIENT_SANDBOX'] = 'True'
                    # os.environ['DIGIKEY_STORAGE_PATH'] = 'cache_dir'

                    # Query product number
                    dkpn = '296-6501-1-ND'
                    part = digikey.product_details(dkpn)

                    [part_no, fet.Q_rr, fet.Coss, fet.I_F, fet.Vds_meas, fet.t_rr, fet.didt] = pdf_part_info(part_url, part_no, csv_name)

                    # Go into the single fet's pdf datasheet and extract the Qrr value
                    try:
                        single_fet_to_csv(fet, csv_name)
                        print(fet.mfr_part_no)
                    except:
                        continue

                    # click to the next page
                    # try:
                    #     python_button = driver.find_element_by_xpath(
                    #         '/html/body/div[2]/main/section/div[2]/div[3]/div/div[3]/div/button[8]')
                    #     '/html/body/div[2]/main/section/div[2]/div[3]/div/div[3]/div/button[8]/span/svg'
                    #     driver.execute_script("arguments[0].click();", python_button)
                    #     print('clicked')
                    #
                    # except:  # nosuchelementexception
                    #     python_button = driver.find_element_by_xpath(
                    #         '/html/body/div[2]/main/section/div[2]/div[3]/div/div[3]/div/button[8]')
                    #     driver.execute_script("arguments[0].click();", python_button)
                    #     print('clicked')

            except:  # exceptions.StaleElementReferenceException as e:
                # print(e)
                print("Stale page exception")
                attempts += 1



import requests
from bs4 import BeautifulSoup
import io
from PyPDF2 import PdfFileReader

def pdf_part_info(part_url, part_no, csv_name):
    total_scraped = 0
    total_w_qrr = 0
    fet_pdf_df = pd.DataFrame(columns=['Mfr_part_no','Qrr','Coss'])
    fet_pdf_info_dict = {'Qrr': { },
         'Coss': { }}
    # for g in range(2):
    #     time.sleep(2)

    headers = {
        # "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36"
        # "User-Agent": "Mozilla/5.0 (iPad; CPU OS 12_2 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Mobile/15E148"
        # "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:98.0) Gecko/20100101 Firefox/98.0",
        # "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
        # "Accept-Language": "en-US,en;q=0.5",
        # "Accept-Encoding": "gzip, deflate",
        # "Connection": "keep-alive",
        # "Upgrade-Insecure-Requests": "1",
        # "Sec-Fetch-Dest": "document",
        # "Sec-Fetch-Mode": "navigate",
        # "Sec-Fetch-Site": "none",
        # "Sec-Fetch-User": "?1",
        # "Cache-Control": "max-age=0",

        "Authority": "www.digikey.com",
        "Method": "GET",
        "Path": "/classic/headerinfo.ashx?site=US&lang=en&cur=USD",
        "Scheme": "https",
        # "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36",
        "Accept": "*/*",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        # "Connection": "keep-alive",
        # "Upgrade-Insecure-Requests": "1",
        # "Sec-Fetch-Dest": "empty",
        # "Sec-Fetch-Mode": "cors",
        # "Sec-Fetch-Site": "same-origin",
        # "Sec-Fetch-User": "?1",
        # "Cache-Control": "no-cache",
        # "Cookie": "search=%7B%22usage%22%3A%7B%22dailyCount%22%3A1%2C%22lastRequest%22%3A%222023-06-13T21%3A50%3A06.863Z%22%7D%2C%22version%22%3A1%7D; ai_user=5tGarMh3VF7mc/0ORcxW94|2023-06-13T21:50:06.864Z; pf-accept-language=en-US; ping-accept-language=en-US; TS01c02def=01460246b61e51b41100d3c058e219f2a56fc3764e34626b976b0b64030c4c9c73d8dec0efe9f214ea13c78fa9fde960270b1dcc9b; TS0178f15d=01460246b61e51b41100d3c058e219f2a56fc3764e34626b976b0b64030c4c9c73d8dec0efe9f214ea13c78fa9fde960270b1dcc9b; TS01cba742=01460246b6a901ba9ecf1a8268b5138e12f9f2b6b3bb76d527b9bce0761725c1198b3d6b6fb5fa6cc9664ee6b9a59bbcdd9052a80e; TS01737bf0=01460246b6a901ba9ecf1a8268b5138e12f9f2b6b3bb76d527b9bce0761725c1198b3d6b6fb5fa6cc9664ee6b9a59bbcdd9052a80e; TS016ca11c=01460246b6fd1ea1bd0cb3136a716fb7f32b77e2aa98006d107a05528c9e49a22d193eee6f583f1f1cd471a008efdceefd32137f2c; TS01d47dae=01460246b6fd1ea1bd0cb3136a716fb7f32b77e2aa98006d107a05528c9e49a22d193eee6f583f1f1cd471a008efdceefd32137f2c; website#lang=en-US; TS0198bc0d=01460246b6a79bdcc8357c2360903a67bb833779247a842f1dedb99f4f724aff1db4608edddd519c334b85b7cc6c647622173cba16; TSc580adf4027=08374f23c1ab2000702bc07c1770935ddacc52167b9ef58e92c29752f00749f2bf417e4d69dc026d080c665e731130007d4a7371e80d64f54468d4b74b5145c5eefd2ec68c9b60b3a7f65cd3118fed06f89478b3c216a27b64a9f0ea2c57f611; ak_bmsc=F69B586C38A808BF7195DE66A97CC55C~000000000000000000000000000000~YAAQsFkhFz654rKIAQAAIty8thTTjxwL5gGRDJ7/DaIeOvd+y9QBwiGDA/0upW/L65F/KLNCpkQk29dSLEJB1He7fq5569p7SPhhDFcLHoZ/Etz9yinY7zb/p31xUCSW9KvFl2kOowL7TijvAwqveHa7Zz+p8oGsVUWLTv/4CYz4VgqgtfMi2YDlk14bCrAZ9Wl/VtIWEqbGPKdlxC1Wk1THd5YOi7ONUmComU5of35XrSe2NnjVjv2RXExa5/A9ZUYVXVyw+1pIghBw3AjN9uL89+Xr/fAJ2s/xds7GRziB5Vvo9GDYESU/h38cc8v4w933pIU3qkwTquc2iPehdBDwEkMmlH2AfWmvFwOC16skj+YGioLQrOWZRXlRPG6nolob6kvm4uSZ3Km+CaQbcRtE1m0ePyw=; EG-U-ID=B02f2ff4f0-2cb5-4cbd-a439-c723fd5d2b54; EG-S-ID=D32e9375e1-84ab-4f47-a7d0-8241a74bf859; __ssid=bd00a5c23355b4f1367cd361ead10e9; bm_sz=4CC46F15B4322A1709559BEFE1959ECC~YAAQsFkhF7bj4rKIAQAAeHa9thSj3Kk85r595VmFZY5j5siAnz+WYJzsk/bkSI57Gw51OAboWNBj5gNnHcUYhrbnJCqFpNjgxsmVKlZmUHlrTnFSJ0r0UYkK9KWhO7Z2uaA72GADCTRN/ZgZndaTH4E93NHc6ZeV0SoUclxhuzyX5G3No92W7yKXJX1ndQNfaTIloDIfFAu4OUyLSFtN35Jq18gPw5KqK+H99pwUC4N18jHVuyl4KgPkRkiRYL1vzCEhwvjROQCgz/7XrByBKAO5PX0yP9EFW6OrOcRCOFXVpZ8D0v96xlbGxyMq8Db8x08RLtO4ugG7Mhw=~3617077~3228995; utag_main=v_id:0188b6bcd787000241637445f6dd0506f007006700978$_sn:1$_se:2$_ss:0$_st:1686694865338$ses_id:1686693009289%3Bexp-session$_pn:2%3Bexp-session; _dd_s=; ai_session=lG5GUXCLArLOjo+1cp6Og7|1686695172612|1686695172612; TSd6475659027=08e0005159ab2000eba86a042edb6ae5874579eb9c21fa4ad9d1471a5ad1da628120bfb32e6ad30c08ada35499113000b1bf01141a7a5040d03d783bd909b405770228f69a0dbeb8e23153f306aa036e827e2edfa3c48183ff204270aab7e419; _abck=8AC2D5E0856EBABCE1320634A5C3E84A~0~YAAQsFkhF8bZ6bKIAQAA9eHdtgo1BYfLhekpeoYliAZGPargbwT0oJ5C7FGELRf0T3DfC1GRpvHRJNs9OgfSlH5RFOg+wPxrKWc4RMvGbVLh67PpLD74Z3TEVWFEqWd+7LyeEm8J6ntDCswdnnbRTeDvOO7xBg7GLrKeXEQDRCuVbIkMFKbPsXbGYmXqNC4rv7pKRUsZ2JsmT4uP5JGrDermGM7jpkBYtCerbTR0WBVPigqfH1755dYpa2TsGhunnAoJYsMq+oP1L318zdmzqSFlceEhO6czpWZAd5DpNA0mckm77BUNfQvTZyiE/elHP/xPQSUvpNW5rjUB3jpzNwutCeHEt37lkofZQ7nDJOwDTGDfV+Ad1ddkGLmg9sRudTAnCoPD6njJcQblDqrPIS4i2O9Vj7t8Zw==~-1~-1~-1; _evga_542b=9e06fe4b5eaa9c79.; bm_sv=3C371D74CCBDB9944FDBB1C5E45B3124~YAAQsFkhF9zZ6bKIAQAA3OLdthRIDcLQ7rO/qccv64Rvqzr5gzS+qw+DJgPH9SlzUSioFllqibyuPX7YX5M+/gOBoDR629Ei1n3lTI1icRbc2ZAatss4mhlo8zyPPxFNoRumj/EUy7OFqF8q1cHp/HJ2t36W8MpWUfeXZcGrjstZLVY/a/AbsXCLGfAGCxmCYalwJV2shX1oYunvrdj43EYzB8MNrg00v70RH1n4JPQJI2MGGTxNdEFFvyc3gnEUGQ==~1; dkc_tracker=3618507391988",
        "Cookie": "ai_user=DAr/D/21/U5oJqHdx6b8AQ|2022-01-26T16:13:16.077Z; SC_ANALYTICS_GLOBAL_COOKIE=6951bce66e534076b41f603bd3aa466f|False; _evga_542b=03fb1b6e751d8964.; LPVID=YwZjgwNGZmNDM2NDBjZTE4; _fbp=fb.1.1667439973219.55065029; __ssid=2a853296a98c5623f76d252d952db8e; _evga_8774=5cd0e1a0a0812399.; EG-U-ID=D861654a2d-55e5-46dc-a7b3-c943b576e528; _cs_c=0; _gcl_au=1.1.686436117.1683563489; sid=180144044511752900xLE4A6K3VCKCAK4NBVK6EP8X7ZX5A15ECCN3C044J93HU7B664LDOIRRAXJXQRKKT; ps-eudo-sid=%7b%22CustomerId%22%3a0%2c%22LoginId%22%3a0%2c%22CustomerClass%22%3a0%2c%22Currency%22%3a%22USD%22%2c%22OrderModel%22%3a-1%2c%22UserIdentity%22%3a0%7d; _gac_UA-88355857-46=1.1686088444.EAIaIQobChMIvt2h8tCv_wIV0X5MCh2GbQ0sEAAYASAAEgLXiPD_BwE; ken_gclid=EAIaIQobChMIvt2h8tCv_wIV0X5MCh2GbQ0sEAAYASAAEgLXiPD_BwE; _gcl_aw=GCL.1686088507.EAIaIQobChMIvt2h8tCv_wIV0X5MCh2GbQ0sEAAYASAAEgLXiPD_BwE; pf-accept-language=en-US; ping-accept-language=en-US; search=%7B%22usage%22%3A%7B%22dailyCount%22%3A75%2C%22lastRequest%22%3A%222023-06-15T22%3A33%3A52.471Z%22%7D%2C%22version%22%3A1%7D; bm_sz=441D2590D8CFDF4545854AA8E2CCE957~YAAQrlkhFw/2osCIAQAAy5gxwRRJD6du1r1ZsRJdq11wzN86m5V1g2P5iEM2CUNWtSkw4D9zcddrTLNE5bNgTzWhIumSa++E/Wv2Siwq9KRpingpwCQrkBHbnQ3byjeMjH32WO5XXHWa6gfheWEwMB//GkwZodRgnMlBHUhfwh9CAvOsBP0vHuQ95no1sY88k7wgJOUxOLC/OymQwf4bDYNVd/bH2gS2kpRgJi+qis08YrsRFEPbEFuluTOueEPXqC2r21vh9eOrvYygN0hRigimNNAE3uR/8mHuHn+ngeV1mFM/~4338756~3618117; dkuhint=false; TS01d47dae=01460246b6bc568f33f9d010b9af7fbeea5a3ee45b64039d6285c8b92dff17c29e1722a80e173113e13b9fa05e0c9126e1de79ea30; _cs_mk=0.40465992379328264_1686868434523; _gid=GA1.2.1404263820.1686868435; ak_bmsc=415707F49DB87754EA71F2E90836E114~000000000000000000000000000000~YAAQrlkhF+z3osCIAQAAsp8xwRS5uuDm8L2PAGxYtSqi9GztaY933P8lqLY9BJyTYQyCPuyFpT2G/JtXB9m7zL7L1G3Ulqg3MLztPvkH3UogI1GJIs67ZDCEu78ktBq+20BqxRt32UytE+8+TlevK8muogUy1yKI6YRBrWaxUS23DXA+NFpV6UKNl7rTFTfKe+oh4Zzb3qz144t5GYCUJOYxERpjxHO+IJ+77yli0ItHXfpT2Rksuz6DP79fwtLkk1nBbydzBtIyKTseNMzHA6ozcoNQudIZHL0NAseNdhTCZWOKQdBqCCY9ADdK33wKFdfNWATPPqHwEiUobodBA0pQDf0xv0mIHORIEOBIz6C2MFLXgj89uQaUGS9QyMGsgehEhauPWD6nWVSYjGpyGkx9XCnN+eSDBnEBmbjsxVYOu0V/3VrzdkU1ZTAzZ/YUm69tQT1CVBW7dcQPjZVhFAE6bA2iYGyRkBLx6tftXmc91Qae2LsaErTGpI/T; ln_or=eyI5NjYwMTciOiJkIn0%3D; EG-S-ID=D3cbfce911-056b-4a0b-842b-d58ca3797d57; TS01173021=01f9ef228dc6d369be3e0bfdf4fa0a6b4845c0c01908a60317da666a1bed62a3cac8ebcf330c9bd84a8f66be83999d8b24d4145615; TScaafd3c3027=08205709cbab20003f7e89c9835c5517d67c6154415d5fa868eed60bbffe902f16b409f0bd9109f008304145b3113000dd4be637054708bfa5efbebc223be6ac9dcc6b140391370516c36c3089ca472c3aeba04a5fc08fa77c86e3a3bb31600d; TS016ca11c=01460246b62682d17b517f61bf222503eb73b734381338eea215f49f8cb480d9cac89170fa789eb34d4d48adc975b385dc0c6a0744; TSd6475659027=08e0005159ab2000c180c42925f0eca0c2cfb15a70fc383edd85879e544dd30a80de645a4840bf2f08508f74331130009dd48fa2662e4ed8887d944f2915796b6a727c116026b0b2f89a3e76cc9de05c7fbb85f3aa5a0296358208a50ebab1a1; TS019f5e6c=01460246b6450c1ccf158a47890b39ee65f12caf58c19136b5ede4b4b81ad80b3fd15e6bf410d505ac463b26628b5070a9867a19eb; TS01c02def=01460246b613d152fbe2180b48cd7e04fd4b32316db8d9a71fd9c9c3102f3522a13c9d81361e1527887e498365e8bcd73c6ab5221c; TS0178f15d=01460246b613d152fbe2180b48cd7e04fd4b32316db8d9a71fd9c9c3102f3522a13c9d81361e1527887e498365e8bcd73c6ab5221c; TS01cba742=01460246b64a1bdd8f3fadcae333a6c72a6404dc5f7bcdbf97d1cc961fbd62e3733e92d7a5d0e7f83d776d07252e50a5effeaabec1; TS01737bf0=01460246b64a1bdd8f3fadcae333a6c72a6404dc5f7bcdbf97d1cc961fbd62e3733e92d7a5d0e7f83d776d07252e50a5effeaabec1; TSc580adf4027=08374f23c1ab2000b841176f25e968a10f80816fb1c788f95d58c7a4de786aef0462f03375a3e2c3083449114b113000e65742ca791dc90452e04573e3dc6e10b53f5d53ce8b72f4e27b47cce6283872369b0d52e67ca85b262dab49c5821519; _ga=GA1.1.2120151438.1643213620; QSI_HistorySession=; ai_session=CN/ows1iB8gXkeOs4G5kgR|1686868432473|1686869543993; utag_main=v_id:017e9729aaa40021f1330cdbe9b205072009a06a00978$_sn:13$_se:4$_ss:0$_st:1686871348674$ses_id:1686868433666%3Bexp-session$_pn:3%3Bexp-session; _gat_Production=1; _uetsid=b5f5a7b00bcc11ee9c92391e2e86db8b; _uetvid=ec8fa9a07ec211eca9bacfbfac7979c2; _cs_cvars=%7B%221%22%3A%5B%22Page%20Title%22%2C%22Part%20Detail%22%5D%2C%222%22%3A%5B%22Page%20Site%22%2C%22US%22%5D%2C%223%22%3A%5B%22Page%20Type%22%2C%22PS%22%5D%2C%224%22%3A%5B%22Page%20Sub%20Type%22%2C%22PD%22%5D%2C%225%22%3A%5B%22Page%20Content%20Group%22%2C%22Part%20Search%22%5D%2C%226%22%3A%5B%22PageContentSubGroup%22%2C%22Part%20Detail%22%5D%2C%227%22%3A%5B%22Page%20ID%22%2C%22PD%22%5D%2C%228%22%3A%5B%22Page%20Language%22%2C%22en%22%5D%2C%2210%22%3A%5B%22Customer%20Dimension%22%2C%22%7B%7D%22%5D%2C%2212%22%3A%5B%22Part%20Substitutes%22%2C%22False%22%5D%2C%2215%22%3A%5B%22Page%20State%22%2C%22Show%20Packaging%20Options%2C%20Parts%20In%20Stock%22%5D%2C%2216%22%3A%5B%22L1%20cat%22%2C%22Discrete%20Semiconductor%20Products%22%5D%7D; _cs_id=4d69db30-5708-af88-9efa-735ee6894932.1680795078.71.1686869549.1686868435.1.1714959078583; _cs_s=4.5.0.1686871349636; _abck=D202F97A26A9742F8DB67F6460C4FFA6~-1~YAAQpFkhF3wSM8GIAQAAyKxCwQofZb4jvKbAruLBpzYobUy1dQk07lFig3v18q4DJG5Cu9JLX7NI/L0zkxDmutPkHv2t2Rh0O4foV+674EHFkvMKs41e/sPZnzy+tMuXvFIQctW71j4od6Gib4bdof2X9QLkbui1uItzIZetePIgQVwLA9xNelc6kIKEfOa5ilc5pIjn0xXvh27rGd/xRROIeVIY/RmI4YYU8WVQvVgPdbfmlblyhNfFB4sz9rmRkKnKmMr70i9nhgWnPes5X2GJvs1G3N2jPCfso9EJtKVc7QsLS2f1po5avOkwa1gNA7bwtwXpmnM0rm+zyf7JOo5X2UB3KTLrF5hTIK/g6k6A37fBQBrxqCWgcOHZWGYpEvMlPtn74n9HTO1kNZQnsvYMpyoUVlolm+8=~-1~-1~-1; _dd_s=rum=0&expire=1686870447602; _ga_0434Z4NCVG=GS1.1.1686868434.29.1.1686869552.56.0.0; bm_sv=7BD1F06A1143EEDE5A2D350D243020BD~YAAQpFkhF5ESM8GIAQAA7a1CwRRaVR45Pdqh8h+XHRoDn5B5Cvxa53CsSrjkgEGSl8Y+Uy7EdFCJ/QoEG8WyxroxN/MeG8yLGx7JYjwpMcm+MEEb+T5ywFHtPgiWHplzmvIWd6Y9vma8VH3ByaIuCK6ksORd+bHTWLKx6hWWLEKFerGQwteEsnbqH5hcjBrqb/qBqxff8wfte5q/+mQjsgc0R8303wNpK+DFKT6p6aj33otz87R2Q8CywlRTz0QwdQY=~1; dkc_tracker=3618681769159",

        "Pragma": "no-cache",
        "Referer": "https://www.digikey.com/en/products/detail/nexperia-usa-inc/PSMN2R1-30YLEX/17084920",
        "Request-Id": "|4670266686cd4931aacbac5bfef51978.904be845769444c2",
        # "Referer": "https://www.digikey.com/en/products/detail/nexperia-usa-inc/PSMN2R1-30YLEX/17084920",
        "Sec-Ch-Ua": '"Not.A/Brand";v="8", "Chromium";v="114", "Google Chrome";v="114"',
        "Sec-Ch-Ua-Mobile": "?0",
        "Sec-Ch-Ua-Platform": '"Windows"',
        "Sec-Fetch-Dest": "empty",
        "Sec-Fetch-Mode": "cors",
        "Sec-Fetch-Site": "same-origin",
        "Traceparent": "00-4670266686cd4931aacbac5bfef51978-904be845769444c2-01"




    }

    read = requests.get(part_url)
    # read = requests.get(url,stream=True)
    soup = BeautifulSoup(read.text, 'html.parser')
    links = soup.find_all('a')
    k=0

    overall_success = False
    # Go through all the individual links to find which ones are pdfs (of the datasheet).
    for link in links:
        # print(link)
        if '.pdf' in link.get('href', []):
            # fet_pdf_df.iloc[k]['Mfr_part_no'] = link.get('title').replace(" | Datasheet", "")
            coss_success = False
            qrr_success = False
            vds_success = False
            i_f_success = False

            try:
                print(link)
                if 'http' not in link.get('href'):
                    link_href = 'http:' + link.get('href')
                else:
                    link_href = link.get('href')
                response = requests.get(link_href, headers=headers, timeout=10)
                #print("Downloading pdf file: ", j)

                # Write the contents of the pdf into the pdf component.pdf
                if os.path.exists('fetPdfData.pdf'):
                    os.remove('fetPdfData.pdf')
                if ('.pdf' in link_href) and (len(r"fetPdfData.pdf") < 40):
                    pdf = open('fetPdfData.pdf', 'wb')
                    pdf.write(response.content)
                    try:
                        df = read_pdf("fetPdfData.pdf", pages='1-6', stream=True, encoding= 'unicode_escape')
                    except:
                        df = read_pdf("fetPdfData.pdf", pages='all', stream=True, encoding= 'unicode_escape')
                        #can replace pages=1-6 w/ 'all'
                    pdf.close()

                    newmeth = True
                    # if 'EPC' in link.get('title'):
                    if newmeth:
                        coss = 0
                        dfnum = -1
                        colname = 'colname'
                        for i in range(len(df)):
                            print('i=%d' % i)
                            for col in df[i].columns:
                                if df[i][col].astype(str).str.contains("COSS", case=False).any():
                                    coss_index = min(df[i][col][df[i][col].astype(str).str.contains("COSS",
                                                                                                    case=False) == True].index)
                                    dfnum = i
                                    colname = col
                                    print(i)
                                    print(min(df[i][col][df[i][col].astype(str).str.contains("COSS",
                                                                                             case=False) == True].index))
                                    for col in df[dfnum].columns:
                                        if 'vds' in col.lower() or 'v ds' in col.lower():
                                            vds_value = col
                                            vds_success = True
                                            break
                                        list2 = [item for item in
                                                 (df[dfnum][col].astype(str).str.contains('VDS', case=False) == True)[
                                                     df[dfnum][col].astype(str).str.contains('VDS', case=False) == True].index if
                                                 item >= coss_index]
                                        # if df[dfnum][col].astype(str).str.contains('VDS').any() and (df[i][col].str.contains('VDS', case=False) == True)[
                                        #         df[i][col].str.contains('VDS', case=False) == True].index[0] >= coss_index:
                                        if len(list2) > 0:
                                            dynamic_col_name = col
                                            dynamic_index = (df[dfnum][col].str.contains('VDS', case=False) == True)[
                                                df[dfnum][col].str.contains('VDS', case=False) == True].index
                                            m = min(val for val in dynamic_index if val >= coss_index)
                                            print(df[dfnum].loc[m, dynamic_col_name])
                                            vds_value = df[dfnum].loc[m, dynamic_col_name]
                                            vds_success = True
                                            break
                                    collist = df[dfnum].columns.to_list()
                                    newlist = df[dfnum].loc[coss_index].to_list()
                                    for value in newlist:
                                        ind = newlist.index(value)
                                        if ind > collist.index(colname):
                                            print(ind)
                                            if (isfloat(value)):
                                                # print(value)
                                                coss = float(value)
                                                print(coss)
                                                coss_success = True
                                                break
                                    if dfnum != -1:
                                        break
                                else:
                                    coss_trial_list = ["C OSS", "Output capacitance"]
                                    vds_trial_list = ["VDS", "V DS"]
                                    for coss_trial in coss_trial_list:
                                        if df[i][col].astype(str).str.contains(coss_trial, case=False).any():
                                            coss_index = min(df[i][col][df[i][col].astype(str).str.contains(coss_trial,
                                                                                                            case=False) == True].index)
                                            dfnum = i
                                            colname = col
                                            print(i)
                                            print(min(df[i][col][df[i][col].astype(str).str.contains(coss_trial,
                                                                                                     case=False) == True].index))
                                            for col in df[dfnum].columns:
                                                for vds_trial in vds_trial_list:
                                                    if df[dfnum][col].astype(str).str.contains(vds_trial).any():
                                                        dynamic_col_name = col
                                                        dynamic_index = (df[i][col].str.contains(vds_trial, case=False) == True)[
                                                            df[i][col].str.contains(vds_trial, case=False) == True].index
                                                        try:
                                                            m = min(val for val in dynamic_index if val >= coss_index)
                                                        except:
                                                            m = min(val for val in dynamic_index if val >= coss_index - 1)
                                                        print(df[i].loc[m, dynamic_col_name])
                                                        vds_value = df[i].loc[m, dynamic_col_name]
                                                        vds_success = True
                                                        break
                                                collist = df[dfnum].columns.to_list()
                                                newlist = df[dfnum].loc[coss_index].to_list()
                                                for value in newlist:
                                                    ind = newlist.index(value)
                                                    if ind > collist.index(colname):
                                                        print(ind)
                                                        if (isfloat(value)):
                                                            # print(value)
                                                            coss = float(value)
                                                            print(coss)
                                                            coss_success = True
                                                            break

                                            if dfnum != -1:
                                                break

                            if coss_success:
                            # if dfnum != -1:

                                break




                        # coss = 0
                        # dfnum=-1
                        # colname = 'colname'
                        # for i in range(len(df)):
                        #     print('i=%d' % i)
                        #     for col in df[i].columns:
                        #         if df[i][col].astype(str).str.contains("COSS", case=False).any():
                        #             coss_index = min(df[i][col][df[i][col].astype(str).str.contains("COSS",
                        #                                                                             case=False) == True].index)
                        #             dfnum = i
                        #             colname = col
                        #             print(i)
                        #             print(min(df[i][col][df[i][col].astype(str).str.contains("COSS",
                        #                                                                      case=False) == True].index))
                        #             try:
                        #                 for col in df[i].columns:
                        #                     if df[i][col].astype(str).str.contains('VDS').any():
                        #                         dynamic_col_name = col
                        #                         dynamic_index = (df[i][col].str.contains('VDS', case=False) == True)[
                        #                             df[i][col].str.contains('VDS', case=False) == True].index
                        #                         break
                        #             except:
                        #                 continue
                        #
                        #             for index in dynamic_index:
                        #                 if index >= coss_index:
                        #
                        #                     print(df[i].loc[index, dynamic_col_name])
                        #                     vds_value = df[i].loc[index, dynamic_col_name][0]
                        #                     vds_success = True
                        #                     break
                        #
                        #     if vds_success:
                        #         break

                        # for i in range(len(df)):
                        #     print('i=%d' % i)
                        #     for col in df[i].columns:
                        #         if df[i][col].astype(str).str.contains("COSS", case=False).any():
                        #             coss_index = min(df[i][col][df[i][col].astype(str).str.contains("COSS",
                        #                                                                             case=False) == True].index)
                        #             dfnum = i
                        #             colname = col
                        #             print(i)
                        #             print(min(df[i][col][df[i][col].astype(str).str.contains("COSS",
                        #                                                                      case=False) == True].index))
                        #
                        #             ### Code for finding Vds, given it's somewhere around Coss
                        #             try:
                        #                 dynamic_col_name = 'Symbol Parameter'
                        #                 dynamic_index = \
                        #                 (df[i][dynamic_col_name].str.contains('DYNAMIC', case=False) == True)[
                        #                     df[i][dynamic_col_name].str.contains('DYNAMIC', case=False) == True].index[0]
                        #             except:
                        #                 continue
                        #
                        #             try:
                        #                 dynamic_col_name = 'Characteristic'
                        #                 dynamic_index = (df[i][dynamic_col_name].str.contains('DYNAMIC', case=False) == True)[
                        #                     df[i][dynamic_col_name].str.contains('DYNAMIC', case=False) == True].index[0]
                        #             except:
                        #                 continue
                        #             try:
                        #                 dynamic_col_name = 'Characteristic Symbol'
                        #                 dynamic_index = (df[i][dynamic_col_name].str.contains('DYNAMIC', case=False) == True)[
                        #                     df[i][dynamic_col_name].str.contains('DYNAMIC', case=False) == True].index[0]
                        #             except:
                        #                 continue
                        #             try:
                        #                 for col in df[i].columns:
                        #                     if df[i][col].astype(str).str.contains('VDS').any():
                        #                         dynamic_col_name = col
                        #                         dynamic_index = (df[i][col].str.contains('VDS', case=False) == True)[
                        #                             df[i][col].str.contains('VDS', case=False) == True].index[0]
                        #                         break
                        #             except:
                        #                 continue
 ##############
                                    # for index, row in df[i].iterrows():
                                    #     try:
                                    #         if df[i]['Symbol Parameter'].str.contains('DYNAMIC', case=False).any():
                                    #             dynamic_index = index
                                    #             break
                                    #             # if df[i].loc[index].str.contains("Dynamic", case=False).any():
                                    #             #     dynamic_index = index
                                    #         else:
                                    #             dynamic_index = 0
                                    #     except:
                                    #         dynamic_index = 0
                                    #
                                    #     try:
                                    #         if df[i]['Characteristic'].str.contains('DYNAMIC', case=False).any():
                                    #             dynamic_index = index
                                    #             break
                                    #             # if df[i].loc[index].str.contains("Dynamic", case=False).any():
                                    #             #     dynamic_index = index
                                    #         else:
                                    #             dynamic_index = 0
                                    #     except:
                                    #         dynamic_index = 0
                                    #
                                    #     try:
                                    #         if df[i]['Characteristic Symbol'].str.contains('DYNAMIC', case=False).any():
                                    #             dynamic_index = index
                                    #             break
                                    #             # if df[i].loc[index].str.contains("Dynamic", case=False).any():
                                    #             #     dynamic_index = index
                                    #         else:
                                    #             dynamic_index = 0
                                    #     except:
                                    #         dynamic_index = 0

                                        #     if ("DYNAMIC" in df[i].loc[index, 'Characteristic']) or (
                                        #             "Dynamic" in df[i].loc[index, 'Characteristic']) or (
                                        #             "dynamic" in df[i].loc[index, 'Characteristic']):
                                        #         dynamic_index = index
                                        #         break
                                        #     # if df[i].loc[index].str.contains("Dynamic", case=False).any():
                                        #     #     dynamic_index = index
                                        #     else:
                                        #         dynamic_index = 0
                                        # except:
                                        #     dynamic_index = 0
                                        #
                                        # try:
                                        #     if ("DYNAMIC" in df[i].loc[index, 'Characteristic  Symbol']) or (
                                        #                     "Dynamic" in df[i].loc[index, 'Characteristic  Symbol']) or (
                                        #                     "dynamic" in df[i].loc[index, 'Characteristic  Symbol']):
                                        #         dynamic_index = index
                                        #         break
                                        #
                                        # except:
                                        #     dynamic_index = 0
                                        #
                                        # try:
                                        #     if ("DYNAMIC" in df[i].loc[index, 'Characteristic Symbol']) or (
                                        #             "Dynamic" in df[i].loc[index, 'Characteristic Symbol']) or (
                                        #             "dynamic" in df[i].loc[index, 'Characteristic Symbol']):
                                        #         dynamic_index = index
                                        #         break
                                        #
                                        # except:
                                        #     dynamic_index = 0

                                    # for index, row in df[i].iterrows():
                                    #     if (df[i].loc[index, 'Characteristic'].find("DYNAMIC") != -1) or (df[i].loc[index, 'Characteristic'].find("Dynamic") != -1) or (df[i].loc[index, 'Characteristic'].find("dynamic") != -1):
                                    #         dynamic_index = index
                                    #     # if df[i].loc[index].str.contains("Dynamic", case=False).any():
                                    #     #     dynamic_index = index
                                    #     else:
                                    #         dynamic_index = 0

                            #         i=dfnum
                            #         for index, row in df[i].iterrows():
                            #             if index >= dynamic_index:
                            #                 if df[i].loc[index].str.contains("VDS", case=False).any() or df[i].loc[index].str.contains("VDS ", case=False).any():
                            #                     # print(index)
                            #                     vds_index = index
                            #                     try:
                            #                         column_name = df[i].loc[vds_index][
                            #                             df[i].loc[vds_index].str.contains("VDS", case=False) == True].index
                            #                     except:
                            #                         column_name = df[i].loc[vds_index][
                            #                             df[i].loc[vds_index].str.contains("VDS ",
                            #                                                               case=False) == True].index
                            #                     print(df[i].loc[vds_index, column_name])
                            #                     vds_value = df[i].loc[vds_index, column_name][0]
                            #                     vds_success = True
                            #                     break
                            #
                            #         # for index in [coss_index - 1, coss_index, coss_index + 1]:
                            #         #     if df[i].loc[index].str.contains("VDS").any():
                            #         #         print(index)
                            #         #         vds_index = index
                            #         #     column_name = df[i].loc[vds_index][df[i].loc[vds_index].str.contains("VDS") == True].index
                            #         #     print(df[i].loc[vds_index, column_name])
                            #         #
                            #         #
                            #         # for col in df[i].columns: #might have to do something with min to get the first one, is probably the right one:
                            #         #     print(df[i].loc[coss_index, col])
                            #         #     print(df[i].loc[coss_index].str.contains("VDS").any())
                            #         #     df[i].loc[coss_index][df[i].loc[coss_index].str.contains("VDS") == True].index
                            #
                            #         # break
                            # if dfnum == i:
                            #     break

                        # for value in df[dfnum].loc[coss_index]:
                        #     if (isfloat(value)):
                        #         # print(value)
                        #         coss = float(value)
                        #         print(coss)
                        #         coss_success=True
                        #         break

                        collist = df[dfnum].columns.to_list()
                        newlist = df[dfnum].loc[coss_index].to_list()
                        for value in newlist:
                            ind = newlist.index(value)
                            if ind > collist.index(colname):
                                print(ind)
                                if (isfloat(value)):
                                    # print(value)
                                    coss = float(value)
                                    print(coss)
                                    coss_success = True
                                    break

                    newmeth = True
                    # if 'EPC' in link.get('title'):
                    if newmeth:
                        qrr = 0
                        dfnum = -1
                        colname = 'colname'

                        for i in range(len(df)):
                            print('i=%d' % i)
                            for col in df[i].columns:
                                if df[i][col].astype(str).str.contains("qrr", case=False).any():
                                    qrr_index = min(df[i][col][df[i][col].astype(str).str.contains("qrr",
                                                                                                    case=False) == True].index)
                                    dfnum = i
                                    colname = col
                                    print(i)
                                    print(min(df[i][col][df[i][col].astype(str).str.contains("qrr",
                                                                                             case=False) == True].index))
                                    break

                                elif df[i][col].astype(str).str.contains("q rr", case=False).any():
                                    qrr_index = min(df[i][col][df[i][col].astype(str).str.contains("q rr",
                                                                                                   case=False) == True].index)
                                    dfnum = i
                                    colname = col
                                    print(i)
                                    print(min(df[i][col][df[i][col].astype(str).str.contains("q rr",
                                                                                             case=False) == True].index))
                                    break
                                # if dfnum == i:
                                #     break

                        for col in df[dfnum].columns:
                            keyword_list = ['I F', 'IF', 'I S', 'IS', 'I SD', 'ISD', 'I  = I', 'I =', 'I=']
                            for keyword in keyword_list:
                                if df[dfnum][col].astype(str).str.contains(keyword).any():
                                    try:
                                        if any(k >= qrr_index - 2 for k in
                                               (df[dfnum][col].str.contains(keyword, case=False) == True)[
                                                   df[dfnum][col].str.contains(keyword,
                                                                               case=False) == True].index):
                                            I_F_col_name = col
                                            I_F_index = (df[dfnum][col].str.contains(keyword, case=False) == True)[
                                                df[dfnum][col].str.contains(keyword, case=False) == True].index
                                            m = min(val for val in I_F_index if val >= qrr_index - 2)
                                            print(df[dfnum].loc[m, I_F_col_name])
                                            i_f_value = df[dfnum].loc[m, I_F_col_name]
                                            i_f_success = True

                                            break
                                    except:
                                        continue
                            # if df[dfnum][col].astype(str).str.contains('I F').any():
                            #     try:
                            #         if any(k >= qrr_index - 2 for k in
                            #                (df[dfnum][col].str.contains('I F', case=False) == True)[
                            #                    df[dfnum][col].str.contains('I F',
                            #                                                case=False) == True].index):
                            #             I_F_col_name = col
                            #             I_F_index = (df[dfnum][col].str.contains('I F', case=False) == True)[
                            #                 df[dfnum][col].str.contains('I F', case=False) == True].index
                            #             m = min(val for val in I_F_index if val >= qrr_index - 2)
                            #             print(df[dfnum].loc[m, I_F_col_name])
                            #             i_f_value = df[dfnum].loc[m, I_F_col_name]
                            #             i_f_success = True
                            #             break
                            #     except:
                            #         continue
                            #
                            # elif df[dfnum][col].astype(str).str.contains('IF').any():
                            #     try:
                            #         if any(k >= qrr_index - 2 for k in
                            #                (df[dfnum][col].str.contains('IF', case=False) == True)[
                            #                    df[dfnum][col].str.contains('IF',
                            #                                                case=False) == True].index):
                            #             I_F_col_name = col
                            #             I_F_index = (df[dfnum][col].str.contains('IF', case=False) == True)[
                            #                 df[dfnum][col].str.contains('IF', case=False) == True].index
                            #             m = min(val for val in I_F_index if val >= qrr_index - 2)
                            #             print(df[dfnum].loc[m, I_F_col_name])
                            #             i_f_value = df[dfnum].loc[m, I_F_col_name]
                            #             i_f_success = True
                            #
                            #             break
                            #     except:
                            #         continue
                            #
                            # elif df[dfnum][col].astype(str).str.contains(keyword).any():
                            #     try:
                            #         if any(k >= qrr_index - 2 for k in
                            #                (df[dfnum][col].str.contains(keyword, case=False) == True)[
                            #                    df[dfnum][col].str.contains(keyword,
                            #                                                case=False) == True].index):
                            #             I_F_col_name = col
                            #             I_F_index = (df[dfnum][col].str.contains(keyword, case=False) == True)[
                            #                 df[dfnum][col].str.contains(keyword, case=False) == True].index
                            #             m = min(val for val in I_F_index if val >= qrr_index - 2)
                            #             print(df[dfnum].loc[m, I_F_col_name])
                            #             i_f_value = df[dfnum].loc[m, I_F_col_name]
                            #             i_f_success = True
                            #
                            #             break
                            #     except:
                            #         continue
                        # if dfnum != -1:
                        #     break

                        # if dfnum == i:
                        #     break

                        collist = df[dfnum].columns.to_list()
                        newlist = df[dfnum].loc[qrr_index].to_list()
                        for value in newlist:
                            ind = newlist.index(value)
                            if ind > collist.index(colname):
                                print(ind)
                                try:
                                    qrr = float(re.sub(r'[- ]', '', value))
                                    if ('µC' in newlist) or ('?C' in newlist):
                                        qrr = qrr * 10 ** 3
                                    print(qrr)
                                    qrr_success = True
                                    break
                                except:
                                    if isfloat(value):
                                        qrr = float(value)
                                        if ('µC' in newlist) or ('?C' in newlist):
                                            qrr = qrr * 10 ** 3
                                        print(qrr)
                                        qrr_success = True
                                        break
                                    else:
                                        continue


            except:
                if not coss_success:
                    coss = np.NaN
                if not qrr_success:
                    qrr = np.NaN
                if not vds_success:
                    vds_value = np.NaN
                if not i_f_success:
                    i_f_value = np.NaN
                    # fet_pdf_df.iloc[k]['Coss'] = 'nan'
                    # fet_pdf_info_dict['Qrr'][link.get('title').replace(" | Datasheet", "")] = 'nan'
                    print('Invalid datasheet url exception')

            # print(fet.mfr_part_no)
            # print(fet.technology)
            # print(part_url)



            try:
                if not coss_success:
                    coss = np.NaN
                if not qrr_success:
                    qrr = np.NaN
                if not vds_success:
                    vds_value = np.NaN
                if not i_f_success:
                    i_f_value = np.NaN
                data_list = [part_no, qrr, coss, i_f_value, vds_value]
                overall_success = True
                print(data_list)
                return data_list
                with open('csv_files/' + csv_name, 'a', newline='') as f:
                    write = csv.writer(f)
                    write.writerow(data_list)
                f.close()

            except:
                continue

    if not overall_success:
        data_list = [part_no, np.nan, np.nan, np.nan, np.nan]
        return data_list

    # countQrr = Counter(fet_pdf_info_dict['Qrr'].values())
# countCoss = Counter(fet_pdf_info_dict['Coss'].values())

# Turn the dictionary into a dataframe w/ mfr part no. as the index and Coss, Qrr as column names

    print('done')




''' 
    This function takes the total number of pages as the number of times to click through the site (as a buffer clicks
    through 5 less pages to avoid error). First, the number of parts on that page are determined, and then for each part,
    the corresponding part info is scraped and added as an attribute to that part object. Then the value of Qrr is 
    attempted to be scraped via the datasheet of the component. The information contained in
    the fet object is then written to the specified csv. Nothing is returned, but all information is in the csv.
'''


def get_all_pages_and_part_nos(start_url, total_pages, pdf=False):
    # take out everything above line 133
    csv_name = 'fet_pdf_data.csv'
    component_type = 'fet'
    if component_type == 'fet':
        # total_pages right now is coming from the multiprocessing function, but could instead call total_num_pages()
        # to determine how many pages exist on the entire site, determining the number of times we will have to click
        # through.
        #   total_pages = total_num_pages(start_url)

        # Get the driver for the starting url
        transistor_parser = TransistorParser(start_url)
        driver = transistor_parser.get_driver()
        driver.get(transistor_parser.page_url(1))

        # Iterate through all the pages on the site
        for i in range(total_pages):
            # if (i % 10 == 0):
            #     driver.refresh()
            # Have an attempts variable to prevent the driver from becoming stale
            attempts = 0
            # current_page_num = '/ html / body / div[2] / main / section / div / div[2] / div / div[1] / div / div[1] / div / div[1] / div / div[2] / span'
            # print('Current page is: %s' % driver.find_elements_by_xpath(current_page_num))

            if pdf:
                while (attempts < 3):

                    try:
                        # logging.info('Getting part numbers for page: {:}'.format(k))
                        # figure out how many components are on the page
                        table = driver.find_elements_by_xpath(
                            '/html/body/div[2]/main/section/div/div[2]/div/div[2]//div/div[1]/table/tbody/tr')
                        # print("number of parts on page: %s" % len(table))

                        print("Getting parts for page: %s" % str(i + 2))
                        driver.implicitly_wait(5)

                        # click to the next page
                        try:
                            python_button = driver.find_element_by_xpath(
                                '/html/body/div[2]/main/section/div/div[2]/div/div[1]/div/div[1]/div/div[2]/div/button[6]')
                            driver.execute_script("arguments[0].click();", python_button)
                            print('clicked')

                        except:  # nosuchelementexception
                            python_button = driver.find_element_by_xpath(
                                '/html/body/div[2]/main/section/div/div[2]/div/div[1]/div/div[1]/div/div[2]/div/button[6]')
                            driver.execute_script("arguments[0].click();", python_button)
                            print('clicked')

                        # Go through every component on the page and add its information as a line to the specified csv
                        for j in range(len(table)):
                            j = j + 1
                            try:
                                part_number_cell = driver.find_element_by_xpath(
                                    f'/html/body/div[2]/main/section/div/div[2]/div/div[2]//div/div[1]/table/tbody/tr[{j}]/td[2]/div/div[3]/div[1]/a')
                                part_url = part_number_cell.get_attribute('href')
                                part_no = part_number_cell.text

                            except:  # exceptions.StaleElementReferenceException as e:
                                print("exception with part no./url")

                            # Create the object with the all the component information besides Qrr
                            fet = DigikeyFet(driver, j)
                            # print(fet.mfr_part_no)
                            # print(fet.technology)
                            # print(part_url)

                            # Go into the single fet's pdf datasheet and extract the Qgg value
                            if pdf:
                                import fet_pdf_scraper
                                if fet.technology != 'GaNFET (Gallium Nitride)':
                                    Q_rr = fet_pdf_scraper.pdf_scraper(part_url, j, component_type, fet)
                                    fet.Q_rr = Q_rr
                                    if not pd.isna(fet.Q_rr):
                                        single_fet_to_csv(fet, csv_name)
                                    print('element downloaded successfully')
                            else:
                                # Write all the attributes of the fet object onto the csv, line by line
                                single_fet_to_csv(fet, csv_name)
                                print('element downloaded successfully')


                    except:  # exceptions.StaleElementReferenceException as e:
                        # print(e)
                        print("Stale page exception")
                        attempts += 1

            else:

                # logging.info('Getting part numbers for page: {:}'.format(k))
                # figure out how many components are on the page
                table = driver.find_elements_by_xpath(
                    '/html/body/div[2]/main/section/div/div[2]/div/div[2]//div/div[1]/table/tbody/tr')
                # print("number of parts on page: %s" % len(table))

                print("Getting parts for page: %s" % str(i + 2))
                driver.implicitly_wait(5)

                # click to the next page
                try:
                    python_button = driver.find_element_by_xpath(
                        '/html/body/div[2]/main/section/div/div[2]/div/div[1]/div/div[1]/div/div[2]/div/button[6]')
                    driver.execute_script("arguments[0].click();", python_button)
                    print('clicked')

                except:  # nosuchelementexception
                    python_button = driver.find_element_by_xpath(
                        '/html/body/div[2]/main/section/div/div[2]/div/div[1]/div/div[1]/div/div[2]/div/button[6]')
                    driver.execute_script("arguments[0].click();", python_button)
                    print('clicked')

                # Go through every component on the page and add its information as a line to the specified csv
                for j in range(len(table)):
                    j = j + 1
                    try:
                        part_number_cell = driver.find_element_by_xpath(
                            f'/html/body/div[2]/main/section/div/div[2]/div/div[2]//div/div[1]/table/tbody/tr[{j}]/td[2]/div/div[3]/div[1]/a')
                        part_url = part_number_cell.get_attribute('href')
                        part_no = part_number_cell.text

                    except:  # exceptions.StaleElementReferenceException as e:
                        print("exception with part no./url")

                    # Create the object with the all the component information besides Qrr
                    fet = DigikeyFet(driver, j)
                    # print(fet.mfr_part_no)
                    # print(fet.technology)
                    # print(part_url)

                    # Go into the single fet's pdf datasheet and extract the Qgg value
                    try:
                        single_fet_to_csv(fet, csv_name)
                        print(fet.mfr_part_no)
                    except:
                        continue




                # print(len(return_fets))
        driver.quit()

        # No need to return anything, what we want has been written to the csv


'''
    This creates a class for each fet, and scrapes all the available information off of digikey.
'''


class DigikeyFet:
    def __init__(self, driver, i):
        try:
            self.mfr_part_no = driver.find_element_by_xpath(
                f'/html/body/div[2]/div/main/section/div[2]/div[2]/div/div[1]/table/tbody/tr[1]/td[2]/div/div[3]/a[1]').text

            # logging.info('Getting information for: {:}'.format(self.mfr_part_no))
            try:
                self.unit_price = driver.find_element_by_xpath(
                    f'/html/body/div[2]/div/main/section/div[2]/div[2]/div/div[1]/table/tbody/tr[{i}]/td[4]/div[1]/div[1]/strong').text
                self.mfr = driver.find_element_by_xpath(
                    f'/html/body/div[2]/div/main/section/div[2]/div[2]/div/div[1]/table/tbody/tr[{i}]/td[2]/div/div[3]/a[2]').text
                self.series = driver.find_element_by_xpath(
                    f'/html/body/div[2]/div/main/section/div[2]/div[2]/div/div[1]/table/tbody/tr[{i}]/td[5]').text
                self.fet_type = driver.find_element_by_xpath(
                    f'/html/body/div[2]/div/main/section/div[2]/div[2]/div/div[1]/table/tbody/tr[{i}]/td[8]').text
                self.technology = driver.find_element_by_xpath(
                    f'/html/body/div[2]/div/main/section/div[2]/div[2]/div/div[1]/table/tbody/tr[{i}]/td[9]').text
                self.V_dss = driver.find_element_by_xpath(
                    f'/html/body/div[2]/div/main/section/div[2]/div[2]/div/div[1]/table/tbody/tr[{i}]/td[10]').text
                self.I_d = driver.find_element_by_xpath(
                    f'/html/body/div[2]/div/main/section/div[2]/div[2]/div/div[1]/table/tbody/tr[{i}]/td[11]').text
                self.V_drive = driver.find_element_by_xpath(
                    f'/html/body/div[2]/div/main/section/div[2]/div[2]/div/div[1]/table/tbody/tr[{i}]/td[12]').text
                self.R_ds = driver.find_element_by_xpath(
                    f'/html/body/div[2]/div/main/section/div[2]/div[2]/div/div[1]/table/tbody/tr[{i}]/td[13]').text
                self.V_thresh = driver.find_element_by_xpath(
                    f'/html/body/div[2]/div/main/section/div[2]/div[2]/div/div[1]/table/tbody/tr[{i}]/td[14]').text
                self.Q_g = driver.find_element_by_xpath(
                    f'/html/body/div[2]/div/main/section/div[2]/div[2]/div/div[1]/table/tbody/tr[{i}]/td[15]').text
                self.V_gs = driver.find_element_by_xpath(
                    f'/html/body/div[2]/div/main/section/div[2]/div[2]/div/div[1]/table/tbody/tr[{i}]/td[16]').text
                self.input_cap = driver.find_element_by_xpath(
                    f'/html/body/div[2]/div/main/section/div[2]/div[2]/div/div[1]/table/tbody/tr[{i}]/td[17]').text
                self.P_diss = driver.find_element_by_xpath(
                    f'/html/body/div[2]/div/main/section/div[2]/div[2]/div/div[1]/table/tbody/tr[{i}]/td[19]').text
                self.op_temp = driver.find_element_by_xpath(
                    f'/html/body/div[2]/div/main/section/div[2]/div[2]/div/div[1]/table/tbody/tr[{i}]/td[20]').text
                self.mount_type = driver.find_element_by_xpath(
                    f'/html/body/div[2]/div/main/section/div[2]/div[2]/div/div[1]/table/tbody/tr[{i}]/td[21]').text
                self.supp_pack = driver.find_element_by_xpath(
                    f'/html/body/div[2]/div/main/section/div[2]/div[2]/div/div[1]/table/tbody/tr[{i}]/td[22]').text
                self.pack_case = driver.find_element_by_xpath(
                    f'/html/body/div[2]/div/main/section/div[2]/div[2]/div/div[1]/table/tbody/tr[{i}]/td[23]').text
                self.Q_rr = np.NaN
                # might need to adjust the following changes
                self.Coss = np.NaN
                self.I_F = np.NaN
                self.Vds_meas = np.NaN
                self.t_rr = np.NaN
                self.didt = np.NaN

            except:
                print("Component element exception")

        except:
            print("Stale element exception")
            return None

    # Function to close the driver
    def quit(self):
        self.driver.quit()
        self.driver = None

'''
    This creates a class for each fet based off the downloaded table information, after the tables in the pdf have been 
    opened and scraped and turned into a list of dataframes.
'''


class DigikeyFet_downloaded:
    def __init__(self, datasheet_set):
        try:
            self.mfr_part_no = datasheet_set[0]
            self.datasheet = datasheet_set[1]['Datasheet']
            self.unit_price = datasheet_set[1]['Price']
            self.stock = datasheet_set[1]['Stock']
            self.mfr = datasheet_set[1]['Mfr']
            self.series = datasheet_set[1]['Series']
            self.fet_type = datasheet_set[1]['FET Type']
            self.technology = datasheet_set[1]['Technology']
            self.V_dss = datasheet_set[1]['Drain to Source Voltage (Vdss)']
            self.I_d = datasheet_set[1]['Current - Continuous Drain (Id) @ 25°C']
            self.V_drive = datasheet_set[1]['Drive Voltage (Max Rds On, Min Rds On)']
            self.R_ds = datasheet_set[1]['Rds On (Max) @ Id, Vgs']
            self.V_thresh = datasheet_set[1]['Vgs(th) (Max) @ Id']
            self.Q_g = datasheet_set[1]['Gate Charge (Qg) (Max) @ Vgs']
            self.V_gs = datasheet_set[1]['Vgs (Max)']
            self.input_cap = datasheet_set[1]['Input Capacitance (Ciss) (Max) @ Vds']
            self.P_diss = datasheet_set[1]['Power Dissipation (Max)']
            self.op_temp = datasheet_set[1]['Operating Temperature']
            self.mount_type = datasheet_set[1]['Mounting Type']
            self.supp_pack = datasheet_set[1]['Supplier Device Package']
            self.pack_case = datasheet_set[1]['Package / Case']
            self.Q_rr = np.NaN
            self.I_F = np.NaN
            self.t_rr = np.NaN
            self.diFdt = np.NaN
            self.IS = np.NaN
            # might need to adjust the following changes
            self.Coss = np.NaN
            self.Vds_meas = np.NaN



        except:
            print("Component element exception")



'''
    Removes the desired pdf from the project.
'''


def remove_pdf(pdf_name):
    if os.path.exists(pdf_name):
        os.remove(pdf_name)  # one file at a time


if __name__ == '__main__':
    # attach_pickled_tables()
    testing_stage = 0
    find_t_rr()
    combine_downloaded_tables()
    scrape_all()