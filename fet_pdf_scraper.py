'''
    This file contains the functions for scraping off of pdfs, which is a more involved process than scraping from
    the front Digikey site.
'''

import threading
import pickle
import shutil
#from fet_digikey_main_scraper import get_all_pages_and_part_nos
import multiprocessing
import pandas as pd
import numpy as np
import re
from fet_data_parsing import column_fet_parse, initial_fet_parse
from fet_visualization import simple_df_plot2
import requests
from bs4 import BeautifulSoup
import PyPDF2
from fet_digikey_main_scraper import get_all_pages_and_part_nos, TransistorParser, full_pdf_part_info, full_pdf_part_info_xl, find_t_rr, combine_downloaded_tables


def scrape_individual_component(starting_link):
    get_all_pages_and_part_nos(starting_link)
    print('done')


def scrape_all():

    # This is a list of starting urls for the scraping function, so that we can use multiprocessing to scrape components
    # faster. The hashtag to the right of each link represents the corresponding page number of the link.
    # Currently using 3 processors, each running 500 times

    from starting_links_list import starting_links_list_1_100
    multiprocess(starting_links_list_1_100)

    with open('fets.pickle', 'rb') as f:
        fets = pickle.load(f)

    # Copy the file with the written links to save for safe keeping
    original = r'C:\Users\Skye\Github\dd_power_designer\mosfet_data\mosfet_data_wpdf.csv'
    target = r'C:\Users\Skye\Github\dd_power_designer\mosfet_data\mosfet_data_wpdfCopy.csv'

    shutil.copyfile(original, target)

'''
    This function takes all the starting url links and runs the scraping method concurrently with one another,
    using an amount of processors equal to the length of starting_links_list.
    The function tells you how long the scraping takes. Inside the function, the components are written line by line
    to the csv.
'''


def multiprocess(starting_links_list):

    starttime = time.time()
    processes = []
    for link_list in starting_links_list:

        # find_t_rr()
        full_pdf_part_info_xl()
        full_pdf_part_info(link_list, total_pages=1)
        combine_downloaded_tables()

        # This line is included in the instance we want to use multithreading as well, but this didn't work well with
        # the pdf scraping also included. If using multithreading, would break the list of 3 links into 9 links,
        # and would have each processor running, say, 3 links at a time using multithreading.
        # Otherwise, just use multiprocessing.
        #   p = multiprocessing.Process(target=multithread, args=(link_list,))
        #total_pages = 499
        #ended w/ 26700 components


    #     total_pages = 1
    #     p = multiprocessing.Process(target=full_pdf_part_info, args=(link_list,total_pages))
    #
    #     # p = multiprocessing.Process(target=get_all_pages_and_part_nos, args=(link_list,total_pages,True,))
    #     processes.append(p)
    #     p.start()
    #
    # for process in processes:
    #     process.join()

    print('Multiprocessing took {} seconds'.format(time.time() - starttime))


'''
    This function takes a starting link and calls the scraping function using multithreading. Watch out for timeout 
    and stale request errors.
'''


def multithread(link_list):
    threads = [threading.Thread(target=get_all_pages_and_part_nos, args=(url,)) for url in link_list]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()


'''
    Download cap. tables, combine all rows into one file.
'''
def capacitor_scraping():
    # download_capacitor_tables()
    # combine_downloaded_tables()
    full_pdf_part_info_xl()

'''
    Download all capacitor tables from main page, inside this function specify the starting url
'''
def download_capacitor_tables():
    # csv_name = 'csv_files/FET_pdf_tables_wVds_full.csv' # Note that last previous scraping was put onto this csv
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

    component_type = 'cap'
    if component_type == 'cap':
        start_url='https://www.digikey.com/en/products/filter/ceramic-capacitors/60?s=N4IgjCBcpg7FoDGUBmBDANgZwKYBoQB7KAbRABYAmS8gBngKprADYRHry52LOwBWHkypsOzABxDKAZlaSAugQAOAFyggAyioBOASwB2AcxABfEwUHQQSqJTDKbkMLVpmgA'        # total_pages right now is coming from the multiprocessing function, but could instead call total_num_pages()
        # to determine how many pages exist on the entire site, determining the number of times we will have to click
        # through.
        #   total_pages = total_num_pages(start_url)

        # Get the driver for the starting url
        transistor_parser = TransistorParser(start_url)
        driver = transistor_parser.get_driver()
        driver.get(transistor_parser.page_url(1))

        # Iterate through all the pages on the site. Start by downloading the page, then click to the next page
        for pg in range(2498):
            time.sleep(2)
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



'''
    This function changes function values that are 0.0 to be np.nan, for ease of use in dataframe manipulations.
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
    This function is useful to take a look at an individual pdf. As input, provide the link to the Digikey page that 
    the desired component is on as link, and the number of the component on the page as num_on_page (first element = 1)
'''


'''
    Takes the url of the individual component itself and finds the datasheets on it. Opens the datasheet, looks for
    the value specified by comp_param, and records that value as another entry in the line that gets written into the 
    csv. 
'''

import time
import os

def pdf_scraper(url, j, component_type, fet_obj):
    # add in something to wait, maybe get new driver?
    # try waiting until you see mfr part no in the links or something
    # Get all the links on the page
    for g in range(2):
        time.sleep(2)
        read = requests.get(url)
        soup = BeautifulSoup(read.text, 'html.parser')
        links = soup.find_all('a')

        # Go through all the individual links to find which ones are pdfs (of the datasheet).
        for link in links:
            if '.pdf' in link.get('href', []):
                try:
                    #print(link)
                    response = requests.get(link.get('href'))
                    #print("Downloading pdf file: ", j)

                    # Write the contects of the pdf into the pdf component.pdf
                    if os.path.exists('component'+str(j) +'.pdf'):
                        os.remove('component'+str(j) +'.pdf')
                    pdf = open('component'+str(j) +'.pdf', 'wb')
                    pdf.write(response.content)



                    # We have found the datasheet we wanted, break out to extract only the information we are looking for
                    break

                except:
                    print('Invalid datasheet url exception')

        # The following code is here in case we encounter EOF errors again.
        '''
        EOF_MARKER = b'%%EOF'
        file_name = 'component.pdf'
        #
        with open(file_name, 'rb') as f:
            contents = f.read()
        #
        # # check if EOF is somewhere else in the file
        # if EOF_MARKER in contents:
        #     # we can remove the early %%EOF and put it at the end of the file
        #     contents = contents.replace(EOF_MARKER, b'')
        #     contents = contents + EOF_MARKER
        #
        # else:
        #     # Some files really don't have an EOF marker
        #     # In this case it helped to manually review the end of the file
        #      # see last characters at the end of the file
        #     # printed b'\n%%EO%E'
        #     contents = contents[:-6] + EOF_MARKER
        #
        # with open(file_name.replace('.pdf', '') + '_fixed.pdf', 'wb') as f:
        #     contents = contents.replace(EOF_MARKER, b'')
        #     contents = contents + EOF_MARKER
        #     f.write(contents)
        '''

        try:
            #time.sleep(2)
            pdfReader = PyPDF2.PdfFileReader('component'+str(j) +'.pdf')

            # Check to see how many pages are in this pdf
            #   print(pdfReader.numPages)

        except:
            print('pdf not converted properly')
            return np.nan

    # Read every page in the pdf file and append it to a list of all the pages of the pdf
    pages_text = []
    for page in range(pdfReader.numPages):
        pageObj = pdfReader.getPage(page)
        # print(pageObj.extractText())
        single_page_text = pageObj.extractText()
        pages_text.append(single_page_text)



    # Now use the extracted text to find the section of text that contains the Qrr value.
    # Ideally, one method of extraction will work consistently for getting the desired parameter for all components.
    # For the case of Qrr, this involved finding where the text said 'Qrr', and 'nC', and taking all the text between.

    if component_type == 'fet':
        # Set the initial Qrr value to nan, until the actual Qrr value is discovered
        Qrr_section = np.nan

        # Iterate through each line and remove any instances of the newline character and blank spaces, then extract the
        # text from 'Qrr' to 'nC', which presumably contains the actual value of Qrr
        for page_i in range(len(pages_text)):
            line = pages_text[page_i].replace('\n', '')
            line = " ".join(line.split())

            # check if we are looking at the right data sheet
            if fet_obj.mfr_part_no[0:-3] in line:
                # Get Qrr value
                try:
                    new_line = re.split(r'Qrr', line)
                    Qrr_section = re.split(r'nC', new_line[1])[0]
                    print(Qrr_section)
                    break

                # If we couldn't find 'Qrr' and 'nC' on this page, try the next page
                except:
                    try:
                        new_line = re.split(r'QRR', line)
                        Qrr_section = re.split(r'nC', new_line[1])[0]
                        print(Qrr_section)
                        break
                    except:
                        try:
                            new_line = re.split(r'Qr', line)
                            Qrr_section = re.split(r'nC', new_line[1])[0]
                            print(Qrr_section)
                            break
                        except:
                            continue

            else:
                continue

        # For attemps at extracting Coss through individual manufacturer means, see 'pdf_extraction()' in Extra_code.py

        print('completed')
        print(Qrr_section)
        # Can remove a pdf once we have the text from it
        #   remove_pdf(pdf_name)
        # pdf = open('component.pdf', 'wb')
        # pdf.write("")
        # pdf.close()
        # Qrr_section is what gets added as the value to the fet object until later parsing
        return Qrr_section



def individual_pdf_find(page_link, num_on_page):
    mfr = 'onsemi'

    transistor_parser = TransistorParser(page_link)
    driver = transistor_parser.get_driver()
    driver.get(transistor_parser.page_url(1))
    i = num_on_page
    part_number_cell = driver.find_element_by_xpath(
        f'/html/body/div[2]/main/section/div/div[2]/div/div[2]//div/div[1]/table/tbody/tr[{i}]/td[2]/div/div[3]/div[1]/a')

    part_url = part_number_cell.get_attribute('href')
    part_no = part_number_cell.text

    Qrr_section = pdf_scraper(part_url, num_on_page,'fet')
    return Qrr_section

'''
    We have the following parsing functions here because they require more attention relevant to the scraping process.
    To get Qrr, for example, we have to take components on a case-by-manufacturer basis, in the end can group together
    to some extent. Unlike the other parameters that are available straight from the initial Digikey site, only in 
    very certain instances will we want to remove the components without Qrr values.
'''


'''
    This function parses an individual component's Qrr section text. It only has looked into the top 11 most common
    manufacturers, so checks that the component has one of these manufacturers first. Could add more manufacturers 
    later but the payoff diminishes in terms of how many components we can add. Some manufacturers had very specific 
    cases that were common enough to warrant their own checks, in order to keep as many components as possible.
'''


def Q_rr_parse(entry, Mfr_entry):
    print(entry)
    prefix = 10**-9
    prefix = 1
    # Check that the component is made by a manufacturer we have parsed through
    #if Mfr_entry in ['onsemi','Vishay Siliconix','Infineon Technologies','STMicroelectronics','IXYS','Diodes Incorporated','Fairchild Semiconductor','Nexperia USA Inc.','Alpha & Omega Semiconductor Inc.','Rohm Semiconductor','Toshiba Semiconductor and Storage']:
    #print(Mfr_entry)
    #print(entry)
    # first check if entry is already nan
    if str(entry) == 'nan':
        return np.nan
    # remove dash character first
    entry = re.sub('[-]', '', entry)

    if Mfr_entry in ['Alpha & Omega Semiconductor Inc.']:
        entry = typicalToMax_singleNum(entry, alpha_om = True)

        # check the case where we have nothing but the two numbers, the first being the typical value and the second
        # being the max. value, we want the typical value
        #entry = typicalToMax_singleNum(entry)
        print(entry)
        return float(entry)*prefix


    if Mfr_entry in ['Rohm Semiconductor'] and len(entry) > 20:
        entry = entry[-5:]
        entry = re.sub('[A-Za-z°) ]', '', entry)

        # check the case where we have nothing but the two numbers, the first being the typical value and the second
        # being the max. value, we want the typical value
        #entry = typicalToMax_singleNum(entry)
        print(entry)
        return float(entry)*prefix

    # Look at the niche cases first
    if Mfr_entry in ['Diodes Incorporated'] and re.sub('[Š¾. ]', '', entry).isdecimal() == True:
        entry = re.sub('[Š¾ ]', '', entry)

        # check the case where we have nothing but the two numbers, the first being the typical value and the second
        # being the max. value, we want the typical value
        entry = typicalToMax_singleNum(entry)
        print(entry)
        return entry*prefix

    if Mfr_entry in ['STMicroelectronics','IXYS','Nexperia USA Inc.','Toshiba Semiconductor and Storage','Infineon Technologies','onsemi'] and (re.sub('[A-Za-zŒ. ]', '', entry).isdecimal() == True):
        entry = re.sub('[A-Za-zŒ ]', '', entry)
        if (Mfr_entry in ['Infineon Technologies']) and len(entry) == 4:
            if float(entry[0:2]) < float(entry[2:]):
                entry = float(entry[0:2])
                print(entry)
                return entry * prefix
            else:
                entry = np.nan
                print(entry)
                return entry*prefix

        entry = typicalToMax_singleNum(entry)
        print(entry)
        return entry*prefix

    # Now look at the more general, common cases. There are:
    #   1. mu is in the entry, at the start, implying we need to make a special case of multiplying the value
    #      by 3 bc 3 times bigger than the values which have units of nC.
    #   2. we have just a bunch of digits, from which we need to take the typical value.

    # case 1:
    if 'µ' in entry:
        # Split on the mu character, see what we have left, will get a list of before an after, always want the before
        entry = entry.split('µ')[0]

        # Do the check for if greater than 3 characters and the first half is less than second half.
        # If so, check that the rest of the entry only contains numbers, because there were some cases with
        # other random characters that did not give the correct Qrr values when compared with the datasheets.
        entry = re.sub('[A-Za-z ]', '', entry)
        # We want to check that without the decimal there are only integers, but we don't want to remove the decimal
        # itself.
        if re.sub('[.]', '', entry).isdecimal() == True:
            entry = typicalToMax_singleNum(entry)
            # Adjust for the mu case
            entry = entry*10**3
        else:
            entry = np.nan

    # case 2:
    try:
        if re.sub('[. ]', '', entry).isdecimal() == True:
            entry = typicalToMax_singleNum(entry)
            print(entry)
            return entry*prefix
    except:
        pass
    # all other cases:
    else:
        entry = np.nan
    print(entry)
    return entry*prefix

    # If component not made by one of these manufacturers, we can't be confident in how to parse, so return nan.
    # else:
    #     return np.nan



'''
    This function looks at an entry of integers (and maybe decimal points) and checks if the first half of the digits
    is a lower value than the second half. If this is the case, we can be fairly sure that we have a typical value 
    followed by a maximum value, and we want the typical value. Otherwise, if the number is <4 digits just return the
    value itself, and if >9 digits, likely garbage so return nan.
'''


def typicalToMax_singleNum(entry, alpha_om = False):
    entry = re.sub('[ ]', '', entry)
    if alpha_om:
        parsing = True
        while parsing:
            try:
                if len(entry) in [1, 2, 3]:
                    entry = float(entry)
                    parsing = False
                    # return float(entry)
                elif (len(entry) == 4):
                    # check if the
                    # check if we have 2 numbers in a row, and if so choose the first, otherwise have a 4- or 5-digit number
                    if float(entry[0:2]) < float(entry[2:]):
                        entry = float(entry[0:2])
                        parsing = False
                    else:  # we probably don't have two numbers in a row, so return the full number
                        entry = float(entry)
                        parsing = False


                elif (len(entry) == 5):
                    # first check if the first val is a decimal, if so, want that decimal value
                    if '.' in entry[0:3]:
                        entry = float(entry[0:3])
                        parsing = False

                    # otherwise, check if have 2 numbers in a row and choose the first
                    elif float(entry[0:2]) < float(entry[2:]):
                        entry = float(entry[0:2])
                        parsing = False

                    else:  # we probably don't have two numbers in a row, so return the full number
                        entry = float(entry)
                        parsing = False

                elif (len(entry) == 6 or len(entry) == 7):
                    if float(entry[0:2]) < float(entry[2:4]) and float(entry[2:4]) < float(entry[4:]):
                        entry = float(entry[2:4])
                        parsing = False

                    elif float(entry[0:3]) < float(entry[3:]):
                        entry = float(entry[0:3])
                        parsing = False

                elif (len(entry) == 8 or len(entry) == 9):
                    if float(entry[0:3]) < float(entry[3:]):
                        entry = float(entry[0:4])
                        parsing = False
                    else:
                        entry = np.nan
                        parsing = False

                else:
                    entry = np.nan
                    parsing = False


            except:
                entry = np.nan
                parsing = False

        # print(entry)
        return (entry)

    # otherwise
    parsing = True
    while parsing:
        try:
            if len(entry) in [1,2,3]:
                entry = float(entry)
                parsing = False
                # return float(entry)
            elif (len(entry) == 4):
                # check if the
                # check if we have 2 numbers in a row, and if so choose the first, otherwise have a 4- or 5-digit number
                if float(entry[0:2]) < float(entry[2:]):
                    entry = float(entry[0:2])
                    parsing = False
                else:  # we probably don't have two numbers in a row, so return the full number
                    entry = float(entry)
                    parsing = False


            elif (len(entry) == 5):
                # first check if the first val is a decimal, if so, want that decimal value
                if '.' in entry[0:3]:
                    entry = float(entry[0:3])
                    parsing = False

                # otherwise, check if have 2 numbers in a row and choose the first
                elif float(entry[0:2]) < float(entry[2:]):
                    entry = float(entry[0:2])
                    parsing = False

                else:  # we probably don't have two numbers in a row, so return the full number
                    entry = float(entry)
                    parsing = False

            elif (len(entry) == 6 or len(entry) == 7):
                if float(entry[0:3]) < float(entry[3:]):
                    entry = float(entry[0:3])
                    parsing = False

            elif (len(entry) == 8 or len(entry) == 9):
                if float(entry[0:3]) < float(entry[3:]):
                    entry = float(entry[0:4])
                    parsing = False

            else:
                entry = np.nan
                parsing = False


        except:
            entry = np.nan
            parsing = False

    #print(entry)
    return(entry)

'''
    Takes as input a csv file that includes the params scraped from pdfs and parses the specified value pdf_param 
    (e.g. Qrr). If return_only == True, return only the components that have a corresponding pdf_param. Currently, have
    to go in and update the columns list as we add more parameters to the component list. Returns the parsed df.
'''


def parse_pdf_param(csv_file, pdf_param, component_type, return_only, drop_first = True):
    # Turn the csv into a df
    df = pd.read_csv(csv_file)
    if drop_first:
        df.drop(columns=df.columns[0],
                axis=1,
                inplace=True)
    if component_type == 'fet':
        df.columns = ['Mfr_part_no', 'Unit_price', 'Mfr', 'Series','FET_type', 'Technology', 'V_dss', 'I_d', 'V_drive', 'R_ds', 'V_thresh','Q_g', 'V_gs', 'Input_cap', 'P_diss','Op_temp','Mount_type','Supp_pack','Pack_case','Q_rr']
    #df = df.drop(labels='Idx', axis=1)

    # Call the function this way because we also have to send the manufacturer name as an input
    for index, row in df.iterrows():
        if pdf_param == 'Q_rr':
            entry = row['Q_rr']
            print(row['Mfr_part_no'])
            print(row['Mfr'])
            row['Q_rr'] = Q_rr_parse(entry, row['Mfr'])
            #print(row['Q_rr'])

    # Drop the components that don't have the corresponding parameter value
    if return_only:
        df = df.dropna(axis=0, subset=[pdf_param])

    # May not need the following line
    df = df[~pd.isna(df.Q_rr)]
    return df





'''
    Looks at the component information from a limited section of all manufacturers.
'''


def individual_mfr_parse(mfr_list, df):
    df = df[df['Mfr'].isin(mfr_list)]
    return df


import matplotlib.pyplot as plt
from fet_regression import reg_score_and_dump
if __name__ == '__main__':
    # Test edit 6
    pd.set_option("display.max_rows", None, "display.max_columns", None)
    scrape_all()
    # df = csv_to_mosfet('mosfet_data_wpdf3.csv')
    df = parse_pdf_param(csv_file='csv_files/mosfet_data_wpdf3.csv', pdf_param='Q_rr', component_type='fet', return_only=False)

    attr_list = ['Mfr_part_no', 'Unit_price', 'FET_type', 'Technology', 'V_dss', 'R_ds', 'Q_g', 'Input_cap',
                 'Pack_case']
    df = column_fet_parse(initial_fet_parse(df, attr_list), attr_list)
    # df['Q_rr'] = df[df['Q_rr'] != 0.0]
    # df = df[(df['FET_type'] == 'N')]
    # df = df[(df['Technology'].isin(['MOSFET']))]
    #RQ_scatter_cost(df, 'Unit_price','R_ds', 'Q_rr')
    plt.scatter(df.loc[:, 'V_dss'],
                df.loc[:, 'R_ds'] * df.loc[:, 'Q_rr'], marker='.', s=1.0,
                alpha=0.9)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Log[Vdss]')
    plt.ylabel('Log[Qrr]')
    plt.title('Qrr vs. Vdss for N-channel FETs')


    scrape_all()
    starttime = time.time()
    processes = []

    total_pages = 1400
    get_all_pages_and_part_nos('https://www.digikey.com/en/products/filter/transistors-fets-mosfets-single/278', total_pages)
    # p = multiprocessing.Process(target=get_all_pages_and_part_nos, args=('https://www.digikey.com/en/products/filter/transistors-fets-mosfets-single/278', total_pages))
    # processes.append(p)
    # p.start()
    #

    print('Multiprocessing took {} seconds'.format(time.time() - starttime))
    scrape_all()
    pd.set_option("display.max_rows", None, "display.max_columns", None)
    for m in range(25):
        Qrr_section = individual_pdf_find('https://www.digikey.com/en/products/filter/transistors-fets-mosfets-single/278?s=N4IgrCBcoA5Q7AGhDOkBMYC%2BWg',m+1)
    # Take a csv file with Qrr_section texts and parse it
    df = parse_pdf_param(csv_file='csv_files/mosfet_data_csv_wpdf.csv', pdf_param='Q_rr', component_type='fet', return_only=False)

    # Use the following section of code for debugging, when need to parse an individual manufacturer's component
    # information
    #   mfr_list = ['onsemi','Vishay Siliconix','Infineon Technologies','STMicroelectronics','IXYS','Diodes Incorporated',
    #             'Fairchild Semiconductor','Nexperia USA Inc.','Alpha & Omega Semiconductor Inc.','Rohm Semiconductor',
    #             'Toshiba Semiconductor and Storage']
    #   truncated_df = individual_mfr_parse(mfr_list, df)
    #   truncated_df = parse_pdf_param(truncated_df)

    # To scrape the data, currently for Qrr
    # scrape_all()

    # train and score the data for Qrr
    reg_score_and_dump(df, 'Q_rr')

    # plot the data
    simple_df_plot2(df, 'Unit_price', 'Q_rr')

   # parse all the rest of the parameters
    attr_list = ['Mfr_part_no', 'Unit_price', 'FET_type', 'Technology', 'V_dss', 'R_ds', 'Q_g', 'Input_cap',
              'Pack_case']
    ready_df = column_fet_parse(initial_fet_parse(df, attr_list), attr_list)