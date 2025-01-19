'''
    Complete tool walk-through.
    Broken into subsections:
        1. Data scraping
        2. Data cleaning
        3. ML model training
        4. Physics-based loss modeling
        5. Optimization tool
'''

import pandas as pd

'''
Step 1: Data scraping.
'''
from fet_pdf_scraper import scrape_all, capacitor_scraping

def data_scraping():
    scrape_all()
    capacitor_scraping()

'''
Step 2: Data cleaning.
'''
from fet_best_envelope import data_cleaning_full
def data_cleaning():
    data_cleaning_full()

'''
Step 3: ML model training.
'''
from fet_best_envelope import train_all
def ML_model_training():
    train_all()

'''
Step 4: Physics and loss modeling. -- add inductor loss modeling scripts here if needed
'''
from loss_and_physics import physics_groupings
def physics_based_grouping():
    physics_groupings()

'''
Step 5: Optimization tool. Plot results of parameter outcomes.
'''
from component_selection import resulting_parameter_plotting
from fet_optimization_chained_wCaps import loss_comparison_plotting
def optimize_converter():
    # resulting_parameter_plotting()
    loss_comparison_plotting()

'''
Step 6: Component prediction/return.
'''
from component_selection import predict_components
def component_prediction():
    predict_components()

if __name__ == '__main__':
    pd.set_option("display.max_rows", 100, "display.max_columns", 100)

    # data_scraping()
    # data_cleaning()
    ML_model_training()
    # physics_based_grouping()
    # optimize_converter()
    component_prediction()
    print('done')



