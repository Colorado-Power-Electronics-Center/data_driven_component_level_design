"""
    This file contains the functions for visualizing the fet data and models.
"""


import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn


''' This visualization function creates a histogram of the desired variable.
    Takes as input a dataframe of fets and the variable to be plotted. '''


def histogram_df(df, component_type, var):
    params = Visualization_parameters('histogram',component_type, var)
    params.get_params()
    column = df.loc[:, var].values
    if component_type == 'ind' and (var in ['Inductance','Current_rating','DCR','Unit_price']):
        column = np.log10(df.loc[:, var].values)
    if component_type == 'cap' and (var in ['Capacitance','Rated_volt','Size','Thickness','Unit_price']):
        column = np.log10(df.loc[:, var].values)
    if component_type == 'gd' and (var in ['Unit_price', 'Peak_current_source', 'Peak_current_sink', 'Pack_case']):
        column = np.log10(df.loc[:, var].values)
    bin_tot = params.num_bins
    plt.interactive(False)
    plt.style.use('ggplot')

    #plt.xlim(params.lim)
    plt.xlabel(params.label)
    plt.ylabel(params.hist_label)
    plt.title(params.plot_title)
    plt.hist(column, bins=bin_tot)
    plt.show()



''' N.B. For a code outline on having multiple sub-axes, or for dictionary-oriented parameters, see the function 
    scatter_separation in Extra_code.py'''

''' This visualization function plots multiple scatter plots at once of various variables against each other, either 
    with log or base 10 y scale. '''


def scatter_df(df, x_var, y_var,log_state_x=False, log_state_y=True):
    # set aside list to iterate through variables to plot
    x_var_list = ['V_dss', 'Unit_price']
    y_var_list = ['R_ds','Q_g']
    log_state_list = [True, False]
    for log_state_x in log_state_list:
        for log_state_y in log_state_list:
            for x_var in x_var_list:
                for y_var in y_var_list:
                    scatter_df(df, x_var, y_var, log_state_x, log_state_y)
    params_x = Visualization_parameters('histogram', x_var)
    params_x.get_params()
    params_y = Visualization_parameters('histogram', y_var)
    params_y.get_params()
    ax = df.plot.scatter(x=x_var, y=y_var, s=params_y.size, c=params_y.color, logx = log_state_x, logy=log_state_y,
                         xlim=params_x.lim, ylim=params_y.lim,
                         title='%s vs. %s, logx = %s, logy = %s'
                               % (params_x.label, params_y.label, log_state_x, log_state_y),
                         alpha=1)


# This class contains the parameters for various types of visualization functions and variables
class Visualization_parameters(object):
    def __init__(self, plot_type, component_type, variable,refactor=True):
        self.component_type = component_type
        self.plot_type = plot_type
        self.variable = variable
        self.refactor = refactor

    def get_params(self):
        if self.component_type == 'fet':
            if self.plot_type == 'histogram':
                if self.variable == 'R_ds':
                    self.plot_title = 'Rds Histogram'
                    self.num_bins = 500
                    self.lim = [0,0.5]
                    self.label = '$R_{ds}$ [Ω]'


                elif self.variable == 'Q_g':
                    self.plot_title = 'Qg Histogram'
                    self.num_bins = 50
                    self.lim = [0, 1]
                    self.label = '$Q_g$ [nC]'

                elif self.variable == 'Unit_price':
                    self.plot_title = 'Cost Histogram'
                    self.num_bins = 500
                    self.lim = [0, 18]
                    self.label = 'Cost [dollars]'
                self.hist_label = 'Quantity [# of components]'

            elif self.plot_type == 'scatter':

                if self.variable == 'R_ds':
                    if self.refactor == True:
                        self.lim = [0,2]
                        self.label = '$R_{ds}$ [mΩ]'
                        self.factor = 10**3
                    else:
                        self.lim = [0, 2]
                        self.label = '$R_{ds}$ [Ω]'
                        self.factor = 1

                if self.variable == 'V_dss':
                    self.lim = [10, 1000]
                    self.label = 'Vdss [V]'
                    self.factor = 1

                if self.variable == 'Unit_price':
                    self.lim = [0, 10]
                    self.label = 'Cost [dollars]'
                    self.factor = 1

                if self.variable == 'Q_g':
                    if self.refactor == True:
                        self.lim = [1 * 10 ** -9, 10 * 10 ** -7]
                        self.label = '$Q_g$ [nC]'
                        self.factor = 10**9
                    else:
                        self.lim = [1 * 10 ** -9, 10 * 10 ** -7]
                        self.label = '$Q_g$ [C]'
                        self.factor = 1

                if self.variable == 'Q_rr':
                    if self.refactor == True:
                        self.lim = [1 * 10 ** -9, 10 * 10 ** -7]
                        self.label = '$Q_{rr}$ [nC]'
                        self.factor = 10**9

                    else:
                        self.lim = [1 * 10 ** -9, 10 * 10 ** -7]
                        self.label = '$Q_rr$ [C]'
                        self.factor = 1


                self.size = 0.1
                self.color = 'red'

            elif self.plot_type == 'correlation':
                self.title = 'Correlation of MOSFET Parameters'

        if self.component_type == 'ind':
            if self.plot_type == 'histogram':
                if self.variable == 'Inductance':
                    self.plot_title = 'Inductance Histogram'
                    self.num_bins = 40
                    self.lim = [0, 0.02]
                    self.label = 'Log[Inductance] [H]'


                elif self.variable == 'Current_rating':
                    self.plot_title = 'Current rating Histogram'
                    self.num_bins = 50
                    self.lim = [0, 1]
                    self.label = 'Log[Current Rating] [A]'

                elif self.variable == 'Unit_price':
                    self.plot_title = 'Cost Histogram'
                    self.num_bins = 100
                    self.lim = [0, 18]
                    self.label = 'Log[Cost] [dollars]'

                elif self.variable == 'DCR':
                    self.plot_title = 'DC Resistance Histogram'
                    self.num_bins = 100
                    self.lim = [0, 300]
                    self.label = 'Log[DCR] [Ohms]'

                self.hist_label = 'Quantity [# of components]'

            elif self.plot_type == 'scatter':
                if self.variable == 'DCR':
                    self.lim = [0,2]
                    self.label = '$R_{dc}$ [mΩ]'
                    if self.refactor == True:
                        self.factor = 10**3
                    else:
                        self.factor = 1

                if self.variable == 'Unit_price':
                    self.lim = [0,2]
                    self.label = 'Cost [Dollars]'
                    self.factor = 1

                if self.variable == 'Current_rating':
                    self.lim = [0,2]
                    self.label = 'Current Rating [A]'
                    self.factor = 1

                if self.variable == 'Inductance':
                    self.lim = [0,2]
                    self.label = 'Inductance [uH]'
                    if self.refactor == True:
                        self.factor = 10**6
                    else:
                        self.factor = 1

                if self.variable == 'Unity':
                    self.lim=[0,2]
                    self.label = 'Unity'
                    self.factor = 1

                if self.variable == 'DCR_Inductance_product':
                    self.set_var = 'Current_rating'
                    self.set_var_min = 1
                    self.set_var_max = 2
                    self.plot_title = 'DCR*Inductance FOM vs. Inductance'

                if self.variable == 'DCR_Cost_product':
                    self.set_var = 'DCR'
                    self.set_var_min = .1
                    self.set_var_max = .3
                    self.plot_title = 'DCR*Cost FOM vs. Inductance'

                if self.variable == 'Dimension':
                    self.set_var = 'Energy'
                    self.set_var_min = 10**-4*1**2
                    self.set_var_max = 2*10**-4*1.1**2
                    self.label = 'Dimension [mm^2]'
                    self.plot_title = 'Dimension vs. Inductance'

            elif self.plot_type == 'correlation':
                self.title = 'Correlation of Inductor Parameters'

        if self.component_type == 'cap':
            if self.plot_type == 'histogram':
                if self.variable == 'Capacitance':
                    self.plot_title = 'Capacitance Histogram'
                    self.num_bins = 40
                    self.lim = [0, 0.02]
                    self.label = 'Log[Capacitance] [F]'


                elif self.variable == 'Rated_volt':
                    self.plot_title = 'Rated voltage Histogram'
                    self.num_bins = 100
                    self.lim = [0, 1]
                    self.label = 'Log[Rated voltage] [V]'

                elif self.variable == 'Unit_price':
                    self.plot_title = 'Cost Histogram'
                    self.num_bins = 100
                    self.lim = [0, 18]
                    self.label = 'Log[Cost] [$\$$]'

                elif self.variable == 'Size':
                    self.plot_title = 'Size Histogram'
                    self.num_bins = 20
                    self.lim = [0, 300]
                    self.label = 'Log[Size] [mm^2]'

                elif self.variable == 'Thickness':
                    self.plot_title = 'Thickness Histogram'
                    self.num_bins = 20
                    self.lim = [0, 300]
                    self.label = 'Log[Thickness] [mm]'

                elif self.variable == 'Temp_coef':
                    self.plot_title = 'Temperature coefficient Histogram'
                    self.num_bins = 100
                    self.lim = [0, 300]
                    self.label = 'Temp. coef.'

                self.hist_label = 'Quantity [# of components]'

        if self.component_type == 'gd':
            if self.plot_type == 'histogram':
                self.hist_label = 'Quantity [# of components]'
                if self.variable == 'Num_drivers':
                    self.plot_title = 'Number of drivers Histogram'
                    self.num_bins = 40
                    self.lim = [0, 0.02]
                    self.label = 'Number of drivers'
                if self.variable == 'Unit_price':
                    self.plot_title = 'Cost Histogram'
                    self.num_bins = 100
                    self.lim = [0, 0.02]
                    self.label = 'Cost [$\$$]'
                if self.variable == 'Peak_current_source':
                    self.plot_title = 'Peak current (source) Histogram'
                    self.num_bins = 50
                    self.lim = [0, 0.02]
                    self.label = 'Peak current (source) [A]'
                if self.variable == 'Peak_current_sink':
                    self.plot_title = 'Peak current (sink) Histogram'
                    self.num_bins = 50
                    self.lim = [0, 0.02]
                    self.label = 'Peak current (sink) [A]'
                if self.variable == 'Pack_case':
                    self.plot_title = 'Area Histogram'
                    self.num_bins = 50
                    self.lim = [0, 0.02]
                    self.label = 'Area [$mm^2$]'

def correlation_matrix(corr_df, attr_list, component):
    if component == 'fet':
        corr_df['1/Rds'] = 1 / corr_df['R_ds']
        corr_df['RdsQg_product'] = corr_df['R_ds'] * corr_df['Q_g']
        corr_df['RdsCost_product'] = corr_df['R_ds'] * corr_df['Unit_price']
        corr_df['RdsQrr_product'] = corr_df['R_ds'] * corr_df['Q_rr']
        # corr_df['Energy'] = corr_df['Inductance'] * corr_df['Current_rating'] ** 2
        # corr_df['Inductance*DCR'] = corr_df['Inductance'] * corr_df['DCR']
        corr_df['1/RdsQg_product'] = 1 / (corr_df['RdsQg_product'])
        corr_df['1/RdsCost_product'] = 1 / (corr_df['RdsCost_product'])
        corr_df['1/RdsQrr_product'] = 1 / (corr_df['RdsQrr_product'])

    if component == 'ind':
        # Add columns to the dataframe that represent other FOMs
        corr_df['Volume'] = corr_df['Volume [mm^3]']
        corr_df['Unit_price'] = corr_df['Unit_price [USD]']
        corr_df['DCR'] = corr_df['DCR [Ohms]']
        corr_df['Inductance'] = corr_df['Inductance [H]']
        corr_df['Current_rating'] = corr_df['Current_Rating [A]']

        # corr_df['Volume'] = corr_df['Dimension'] * corr_df['Height']
        corr_df['Volume*DCR'] = corr_df['Volume'] * corr_df['DCR']
        corr_df['Cost*DCR'] = corr_df['Unit_price'] * corr_df['DCR']
        corr_df['Energy'] = corr_df['Inductance'] * corr_df['Current_rating'] ** 2
        corr_df['Inductance*DCR'] = corr_df['Inductance'] * corr_df['DCR']
        corr_df['1/DCR'] = 1 / (corr_df['DCR'])
        corr_df['1/Inductance'] = 1 / (corr_df['Inductance'])
        corr_df['1/Current_rating'] = 1 / (corr_df['Current_rating'])
        corr_df['Log[DCR]'] = np.log10(corr_df['DCR'])
        corr_df['Log[Inductance]'] = np.log10(corr_df['Inductance'])
        corr_df['Log[Current_rating]'] = np.log10(corr_df['Current_rating'])
        corr_df['L/DCR'] = corr_df['Inductance']/corr_df['DCR']
        corr_df['L/Cost'] = corr_df['Inductance']/corr_df['Unit_price']
        corr_df['L/Volume'] = corr_df['Inductance']/corr_df['Volume']
        corr_df['L*Imax^2/DCR'] = corr_df['Inductance']*corr_df['Current_rating']**2/corr_df['DCR']
        corr_df['L*Imax^2/Cost'] = corr_df['Inductance']*corr_df['Current_rating']**2/corr_df['Unit_price']
        corr_df['L*Imax^2/Volume'] = corr_df['Inductance']*corr_df['Current_rating']**2/corr_df['Volume']
        corr_df['L*Imax^2/Area'] = corr_df['Inductance']*corr_df['Current_rating']**2/corr_df['Dimension']


        attr_list = ['Unit_price', 'Current_rating', 'DCR', 'Dimension', 'Inductance', 'Volume', 'Volume*DCR',
                     'Cost*DCR', 'Energy', 'Inductance*DCR', '1/DCR', '1/Inductance', '1/Current_rating', 'Log[DCR]', 'Log[Inductance]', 'Log[Current_rating]']


    # attr_list = ['V_dss', 'Unit_price', 'R_ds', 'Q_g', 'Q_rr', 'Pack_case', '1/Rds', 'RdsQg_product', 'RdsCost_product',
    #              'RdsQrr_product']

    corr_params = Visualization_parameters('correlation', component_type=component, variable=np.nan)
    corr_params.get_params()

    # plot the correlation matrix using seaborn
    corrMatrix = corr_df[attr_list].corr()
    sn.heatmap(corrMatrix, annot=True)
    plt.title(corr_params.title)
    plt.xticks(rotation=45)
    plt.show()
    print('done')

''' This visualization function takes various cost brackets and color-codes the component based on its cost, 
    then plots V_dss as a function of R_ds. '''


def scatter_cost_bracket(parsed_df, output_var1, output_var2=None):
    output_var1 = 'R_ds'

    cost_list = [0,0.5, 5.0, 50.0,10**6]
    color_list = ['#00da00','#ebe700','#eba400','#f50000','#800000']
    # for i in range(len(cost_list)-1):
    #     temp_df = parsed_df[(parsed_df['Unit_price'] >= cost_list[i])]
    #     temp_df = temp_df[(temp_df['Unit_price'] <= cost_list[i+1])]
    #
    #     x = np.array(temp_df['V_dss'])
    #     output_var1 = 'R_ds'
    #     y = np.array(temp_df[output_var1])
    #     if output_var2 != None:
    #         y = np.array(temp_df[output_var1]*temp_df[output_var2])
    #     m, b = np.polyfit(np.log10(x), np.log10(y), 1)
    #     plt.plot(np.log10(x), m * np.log10(x) + b)
    #     plt.scatter(np.log10(x), np.log10(y), marker='o', s=0.3, c=color_list[i], alpha=0.8)
    #     plt.xlabel('Log[Vdss] [V]')
    #     plt.ylabel('Log[%s] [mOhm]' % (output_var1))
    #     #ylim = [df[y_var].min(), df[y_var].max()],
    #     plt.title("%s vs. Vdss at cost range %s to %s" % (output_var1, cost_list[i], cost_list[i+1]))
    #     #plt.title('Rds * Qg FOM vs. Vdss for N-channel FETs, cost $0-$0.10')
    #     plt.clf()
    #fig1, (ax1, ax2) = plt.subplots(nrows=2,ncols=1)
    cost_list = [0.1, 1.0, 10.0, 100.0]
    color_list = []
    for i in range(len(parsed_df.index)):
        # print(parsed_df.iloc[i][' Unit Price'])
        price = parsed_df.iloc[i]['Unit_price']

        # print(split_on_letter(text)[0])

        if price < cost_list[0]:
            color_list.append('#00da00')
        elif cost_list[0] <= price < cost_list[1]:
            color_list.append('#ebe700')
        elif cost_list[1] <= price < cost_list[2]:
            color_list.append('#eba400')
        elif cost_list[2] <= price < cost_list[3]:
            color_list.append('#f50000')
        else:
            color_list.append('#800000')

    ax1 = parsed_df.plot.scatter(x="V_dss", y="R_ds", s=0.1, c=color_list, logy=True,
                                title='$R_{ds}$ at rated voltage $V_{dss}$', xlim=[0, 150], ylim=[10 ** -3, 10],
                                alpha=0.5)
    ax1.set_xlabel("$V_{dss}$ [V]")
    ax1.set_ylabel("$R_{ds}$ [Ohm]")
    ax1.clear()
    temp_df = parsed_df[(parsed_df['Unit_price'] <= cost_list[0])]
    x = np.array(temp_df['V_dss'])
    y = np.array(temp_df['R_ds'])
    coef = np.polyfit(x, y, 1)
    poly1d_fn = np.poly1d(coef)
    plt.plot(poly1d_fn(x))
    plt.plot(x, y, 'o')
    plt.clf()
    m, b = np.polyfit(np.log10(x), np.log10(y), 1)
    plt.plot(np.log10(x), m * np.log10(x) + b)
    plt.scatter(np.log10(x), np.log10(y), marker = 'o', s=0.3,c='#00da00',alpha = 0.8)
    plt.xlabel('Log[Vdss] [V]')
    plt.ylabel('Log[Rds*Qg] [mOhm * nC]')
    plt.title('Rds * Qg FOM vs. Vdss for N-channel FETs, cost $0-$0.10')
    # ax1 = temp_df.plot.scatter(x="V_dss", y="R_ds", s=0.1, c='#00da00', logy=True,
    #                             title='$R_{ds}$ at rated voltage $V_{dss}$', xlim=[0, 150], ylim=[10 ** -3, 10],
    #                             alpha=0.5)
    # ax1.set_xlabel("$V_{dss}$ [V]")
    # ax1.set_ylabel("$R_{ds}$ [Ohm]")


#def scatter_cost_bracket_FOM(parsed_df, output_var1, output_var2=None):

def FOM_scatter(df, x_var, y_var1, y_var2 = None,  y_var3 = None, y_var4 = None):
    x_var_params = Visualization_parameters('scatter', 'fet',x_var)
    x_var_params.get_params()
    y_var1_params = Visualization_parameters('scatter', 'fet',y_var1)
    y_var1_params.get_params()

    # If need to filter for certain types of inductors, do so here
    ready_df = df

    cost_list = [0.1, 1.0, 10.0, 100.0]
    color_list = []
    for i in range(len(ready_df.index)):
        # print(parsed_df.iloc[i][' Unit_Price'])
        price = ready_df.iloc[i]['Unit_price']

        # print(split_on_letter(text)[0])

        if price < cost_list[0]:
            color_list.append('#00da00')
        elif cost_list[0] <= price < cost_list[1]:
            color_list.append('#ebe700')
        elif cost_list[1] <= price < cost_list[2]:
            color_list.append('#eba400')
        elif cost_list[2] <= price < cost_list[3]:
            color_list.append('#f50000')
        else:
            color_list.append('#800000')

    plt.scatter(x_var_params.factor * ready_df.loc[:, x_var],
                y_var1_params.factor * ready_df.loc[:, y_var1],
                marker='v', s=20, c=color_list,
                alpha=0.8)

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Log[%s]' % x_var_params.label)
    plt.ylabel('Log[%s]' % (y_var1_params.label))
    plt.title('%s vs. %s for N-channel FETs' % (y_var1_params.label, x_var_params.label))
    plt.show()


def RQ_scatter_cost(fet_df, component_type, x_var, y_var1, y_var2 = None,  y_var3 = None, y_var4 = None, volt_lookup = None):
    # create the scatter object
    x_var_params = Visualization_parameters('scatter', component_type,x_var)
    x_var_params.get_params()
    y_var1_params = Visualization_parameters('scatter', component_type,y_var1)
    y_var1_params.get_params()

    if y_var2 != None:
        y_var2_params = Visualization_parameters('scatter', component_type, y_var2)
        y_var2_params.get_params()

    if y_var3 != None:
        y_var3_params = Visualization_parameters('scatter', component_type, y_var3)
        y_var3_params.get_params()

    if y_var4 != None:
        y_var4_params = Visualization_parameters('scatter', component_type, y_var4)
        y_var4_params.get_params()

    # look at certain voltages
    if volt_lookup != None:
        fet_df = fet_df[fet_df['V_dss'] == volt_lookup]
    ###Si FETS
    ready_df = fet_df[(fet_df['FET_type'] == 'N')]
    ready_df = ready_df[(ready_df['Technology'].isin(['MOSFET']))]
    # ready_df = ready_df[(ready_df['Technology'] == 'SiCFET')]
    cost_list = [0.1, 1.0, 10.0, 100.0]
    color_list = []
    for i in range(len(ready_df.index)):
        # print(parsed_df.iloc[i][' Unit_Price'])
        price = ready_df.iloc[i]['Unit_price']

        # print(split_on_letter(text)[0])

        if price < cost_list[0]:
            color_list.append('#00da00')
        elif cost_list[0] <= price < cost_list[1]:
            color_list.append('#ebe700')
        elif cost_list[1] <= price < cost_list[2]:
            color_list.append('#eba400')
        elif cost_list[2] <= price < cost_list[3]:
            color_list.append('#f50000')
        else:
            color_list.append('#800000')
    cmap = ['r', 'y', 'o', 'b', 'g']
    # get the RQ product as the y-value for each component
    import numpy as np
    if y_var4 != None:
        plt.scatter(x_var_params.factor * ready_df.loc[:, x_var],
                    y_var1_params.factor * ready_df.loc[:, y_var1] * y_var2_params.factor * ready_df.loc[:, y_var2] * y_var3_params.factor * ready_df.loc[:, y_var3] *
                    (y_var4_params.factor * ready_df.loc[:, y_var4])**3,
                    marker='.', s=40, c=color_list,
                    alpha=0.8)


    elif y_var3 != None and y_var4 == None:
        plt.scatter(x_var_params.factor * ready_df.loc[:, x_var],
                    y_var1_params.factor * ready_df.loc[:, y_var1] * y_var2_params.factor * ready_df.loc[:, y_var2] * y_var3_params.factor * ready_df.loc[:, y_var3],
                    marker='.', s=40, c=color_list,
                    alpha=0.8)
        # plt.xscale('log')
        # plt.yscale('log')
        # plt.xlabel('Log[%s]' % x_var_params.label)
        # plt.ylabel('Log[%s * %s]' % (y_var1_params.label, y_var2_params.label))
        # plt.title('%s * %s FOM vs. Vdss for N-channel FETs' % (y_var1_params.label, y_var2_params.label))

    elif y_var2 != None and y_var3 == None and y_var4 == None:
        plt.scatter(x_var_params.factor * ready_df.loc[:, x_var],
                    y_var1_params.factor * ready_df.loc[:, y_var1] * y_var2_params.factor * ready_df.loc[:, y_var2], marker='.', s=40,c=color_list,
                    alpha=0.9)
    elif y_var2 == None:
        plt.scatter(x_var_params.factor * ready_df.loc[:, x_var],
                    y_var1_params.factor * ready_df.loc[:, y_var1],
                    marker='.', s=40, c=color_list,
                    alpha=0.9)
    # plt.xscale('log')
    # plt.yscale('log')
    # plt.xlabel('Log[%s]' % x_var_params.label)
    # plt.ylabel('Log[%s]' % (y_var1_params.label))
    # plt.title('%s vs. Vdss for N-channel FETs' % (y_var1_params.label))

    ###SiCFETs
    ready_df = fet_df[(fet_df['FET_type'] == 'N')]
    ready_df = ready_df[(ready_df['Technology'].isin(['SiCFET','SiC']))]
    #ready_df = ready_df[(ready_df['Technology'] == 'SiCFET')]
    cost_list = [0.1, 1.0, 10.0, 100.0]
    color_list = []
    for i in range(len(ready_df.index)):
        # print(parsed_df.iloc[i][' Unit_Price'])
        price = ready_df.iloc[i]['Unit_price']

        # print(split_on_letter(text)[0])

        if price < cost_list[0]:
            color_list.append('#00da00')
        elif cost_list[0] <= price < cost_list[1]:
            color_list.append('#ebe700')
        elif cost_list[1] <= price < cost_list[2]:
            color_list.append('#eba400')
        elif cost_list[2] <= price < cost_list[3]:
            color_list.append('#f50000')
        else:
            color_list.append('#800000')
    cmap = ['r','y','o','b','g']
    # get the RQ product as the y-value for each component
    import numpy as np
    if y_var4 != None:
        plt.scatter(x_var_params.factor * ready_df.loc[:, x_var],
                    y_var1_params.factor * ready_df.loc[:, y_var1] * y_var2_params.factor * ready_df.loc[:, y_var2] * y_var3_params.factor * ready_df.loc[:, y_var3] *
                    (y_var4_params.factor * ready_df.loc[:, y_var4])**3,
                    marker='x', s=40, c=color_list,
                    alpha=0.9)


    elif y_var3 != None and y_var4 == None:
        plt.scatter(x_var_params.factor * ready_df.loc[:, x_var],
                    y_var1_params.factor * ready_df.loc[:, y_var1] * y_var2_params.factor * ready_df.loc[:, y_var2],
                    marker='x', s=40, c=color_list,
                    alpha=0.8)

    elif y_var2 != None and y_var3 == None and y_var4 == None:
        plt.scatter(x_var_params.factor * ready_df.loc[:, x_var],
                    y_var1_params.factor * ready_df.loc[:, y_var1] * y_var2_params.factor * ready_df.loc[:, y_var2],
                    marker='x', s=40, c=color_list,
                    alpha=0.8)

    elif y_var2 == None:
        plt.scatter(x_var_params.factor * ready_df.loc[:, x_var],
                    y_var1_params.factor * ready_df.loc[:, y_var1],
                    marker='x', s=40, c=color_list,
                    alpha=0.8)
        print('plot made')

    # if y_var1 == 'Q_rr' or y_var2 == 'Q_rr':
    #     plt.xscale('log')
    #     plt.yscale('log')
    #     plt.xlabel('Log[$V_{dss}$] [V]')
    #     plt.ylabel('Log[Qrr * $R_{ds}$] [nC * mΩ]')
    #     plt.title('Qrr*Rds FOM vs. Vdss for N-channel FETs')

    ###GaN FETs
    if 'Q_rr' not in [y_var1, y_var2, y_var3, y_var4]:
        ready_df = fet_df[(fet_df['FET_type'] == 'N')]
        ready_df = ready_df[(ready_df['Technology'].isin(['GaNFET']))]
        # ready_df = ready_df[(ready_df['Technology'] == 'SiCFET')]
        cost_list = [0.1, 1.0, 10.0, 100.0]
        color_list = []
        for i in range(len(ready_df.index)):
            # print(parsed_df.iloc[i][' Unit_Price'])
            price = ready_df.iloc[i]['Unit_price']

            # print(split_on_letter(text)[0])

            if price < cost_list[0]:
                color_list.append('#00da00')
            elif cost_list[0] <= price < cost_list[1]:
                color_list.append('#ebe700')
            elif cost_list[1] <= price < cost_list[2]:
                color_list.append('#eba400')
            elif cost_list[2] <= price < cost_list[3]:
                color_list.append('#f50000')
            else:
                color_list.append('#800000')
        cmap = ['r', 'y', 'o', 'b', 'g']
        # get the RQ product as the y-value for each component
        import numpy as np
        if y_var2 != None:
            plt.scatter(x_var_params.factor * ready_df.loc[:, x_var],
                        y_var1_params.factor * ready_df.loc[:, y_var1] * y_var2_params.factor * ready_df.loc[:, y_var2],
                        marker='v', s=40, c=color_list,
                        alpha=0.8)

        elif y_var2 == None:
            plt.scatter(x_var_params.factor * ready_df.loc[:, x_var],
                        y_var1_params.factor * ready_df.loc[:, y_var1],
                        marker='v', s=40, c=color_list,
                        alpha=0.8)

    ### overall plot settings
    if y_var2 == None:
        plt.xscale('log')

        plt.yscale('log')
        plt.xlabel('Log[%s]' % x_var_params.label)
        plt.ylabel('Log[%s]' % (y_var1_params.label))
        plt.title('%s vs. Vdss for N-channel FETs' % (y_var1_params.label))
        #plt.legend(['MOSFET','SiCFET','GaNFET'])
        plt.show()
        print('plot made')

    elif y_var2 != None and y_var3 == None and y_var4 == None:
        plt.xscale('log')

        plt.yscale('log')
        plt.xlabel('Log[%s]' % x_var_params.label)
        plt.ylabel('Log[%s * %s]' % (y_var1_params.label, y_var2_params.label))
        plt.title('%s  * %s vs. Vdss for N-channel FETs' % (y_var1_params.label, y_var2_params.label))
        # plt.legend(['MOSFET','SiCFET','GaNFET'])
        plt.show()
        print('plot made')

    elif y_var3 != None and y_var4 == None:
        plt.xscale('log')

        plt.yscale('log')
        plt.xlabel('Log[%s]' % x_var_params.label)
        plt.ylabel('Log[%s * %s * %s]' % (y_var1_params.label, y_var2_params.label, y_var3_params.label))
        plt.title('%s  * %s * %s vs. Vdss for N-channel FETs' % (y_var1_params.label, y_var2_params.label, y_var3_params.label))
        # plt.legend(['MOSFET','SiCFET','GaNFET'])
        plt.show()
        print('plot made')

    elif y_var4 != None:
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('Log[%s]' % x_var_params.label)
        plt.ylabel('Log[%s*%s*%s*(%s)^3]' % (
        y_var1_params.label, y_var2_params.label, y_var3_params.label, y_var4_params.label))
        plt.title('%s*%s*%s*(%s)^3 FOM vs. Vdss for N-channel FETs' % (
        y_var1_params.label, y_var2_params.label, y_var3_params.label, y_var4_params.label))
        plt.show()


    # ax = parsed_df.plot.scatter(x="V_dss", y="R_ds", s=0.1, c=color_list, logy=True,
    #                             title='$R_{ds}$ at rated voltage $V_{dss}$', xlim=[0, 150], ylim=[10 ** -3, 10],
    #                             alpha=0.5)
    # ax.set_xlabel("$V_{dss}$ [V]")
    # ax.set_ylabel("$R_{ds}$ [Ohm]")

def full_scatter_cost(fet_df, output_param):
    ###Si FETS
    ready_df = fet_df[(fet_df['FET_type'] == 'N')]
    ready_df = ready_df[(ready_df['Technology'].isin(['MOSFET']))]
    # ready_df = ready_df[(ready_df['Technology'] == 'SiCFET')]
    cost_list = [0.1, 1.0, 10.0, 100.0]
    color_list = []
    for i in range(len(ready_df.index)):
        # print(parsed_df.iloc[i][' Unit_Price'])
        price = ready_df.iloc[i]['Unit_price']

        # print(split_on_letter(text)[0])

        if price < cost_list[0]:
            color_list.append('#00da00')
        elif cost_list[0] <= price < cost_list[1]:
            color_list.append('#ebe700')
        elif cost_list[1] <= price < cost_list[2]:
            color_list.append('#eba400')
        elif cost_list[2] <= price < cost_list[3]:
            color_list.append('#f50000')
        else:
            color_list.append('#800000')
    cmap = ['r', 'y', 'o', 'b', 'g']
    # get the RQ product as the y-value for each component
    import numpy as np
    plt.scatter(ready_df.loc[:, 'V_dss'],
                ready_df.loc[:, output_param], marker='.', s=0.3,c=color_list,
                alpha=1)

    ###SiCFETs
    ready_df = fet_df[(fet_df['FET_type'] == 'N')]
    ready_df = ready_df[(ready_df['Technology'].isin(['SiCFET','SiC']))]
    #ready_df = ready_df[(ready_df['Technology'] == 'SiCFET')]
    cost_list = [0.1, 1.0, 10.0, 100.0]
    color_list = []
    for i in range(len(ready_df.index)):
        # print(parsed_df.iloc[i][' Unit_Price'])
        price = ready_df.iloc[i]['Unit_price']

        # print(split_on_letter(text)[0])

        if price < cost_list[0]:
            color_list.append('#00da00')
        elif cost_list[0] <= price < cost_list[1]:
            color_list.append('#ebe700')
        elif cost_list[1] <= price < cost_list[2]:
            color_list.append('#eba400')
        elif cost_list[2] <= price < cost_list[3]:
            color_list.append('#f50000')
        else:
            color_list.append('#800000')
    cmap = ['r','y','o','b','g']
    # get the RQ product as the y-value for each component
    import numpy as np
    plt.scatter(ready_df.loc[:, 'V_dss'],
                ready_df.loc[:, output_param], marker = 'x',s=10,c=color_list, alpha=1)

    ###GaN FETs
    ready_df = fet_df[(fet_df['FET_type'] == 'N')]
    ready_df = ready_df[(ready_df['Technology'].isin(['GaNFET']))]
    # ready_df = ready_df[(ready_df['Technology'] == 'SiCFET')]
    cost_list = [0.1, 1.0, 10.0, 100.0]
    color_list = []
    for i in range(len(ready_df.index)):
        # print(parsed_df.iloc[i][' Unit_Price'])
        price = ready_df.iloc[i]['Unit_price']

        # print(split_on_letter(text)[0])

        if price < cost_list[0]:
            color_list.append('#00da00')
        elif cost_list[0] <= price < cost_list[1]:
            color_list.append('#ebe700')
        elif cost_list[1] <= price < cost_list[2]:
            color_list.append('#eba400')
        elif cost_list[2] <= price < cost_list[3]:
            color_list.append('#f50000')
        else:
            color_list.append('#800000')
    cmap = ['r', 'y', 'o', 'b', 'g']
    # get the RQ product as the y-value for each component
    import numpy as np
    plt.scatter(ready_df.loc[:, 'V_dss'],
                ready_df.loc[:, output_param], marker='v', s=10,c=color_list,
                alpha=1)

    ###plot settings
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Log[Vdss] [V]')
    plt.ylabel('Log[Qg] [nC]')
    plt.title('Qg vs. Vdss for N-channel FETs')
    #plt.legend(['MOSFET','SiCFET','GaNFET'])
    plt.show()
    print('plot made')
    # ax = parsed_df.plot.scatter(x="V_dss", y="R_ds", s=0.1, c=color_list, logy=True,
    #                             title='$R_{ds}$ at rated voltage $V_{dss}$', xlim=[0, 150], ylim=[10 ** -3, 10],
    #                             alpha=0.5)
    # ax.set_xlabel("$V_{dss}$ [V]")
    # ax.set_ylabel("$R_{ds}$ [Ohm]")


''' This visualization function is a simple scatter plot of any y_var variable as a function of any 
    x_var variable. '''


def simple_df_plot(df, x_var, y_var):
    # Plot-specific characteristics for the various variables, as [title script, value limit]
    plot_vars = {'R_ds': ['$R_{ds}$', 20], 'Unit_price': ['Cost', 20.00], 'V_dss': ['$V_{dss}$', 500],
                 'Q_g': ['$Q_g$', 2 * 10 ** -7], 'Pack case': ['Area', 1000]}
    plotted_df = df
    plotted_df = plotted_df[plotted_df['FET_type'] == 'N']
    plotted_df = plotted_df[plotted_df['Technology'] == 'MOSFET']
    plt.scatter(plotted_df.loc[:, 'Unit_price'],
                (plotted_df.loc[:, 'R_ds']), s=0.3)
    plt.yscale('log')
    plt.show()
    plt.scatter(plotted_df.loc[:, 'Unit_price'],
                (plotted_df.loc[:, 'Q_g']), s=0.3)
    plt.yscale('log')
    plt.show()
    # Need to change the xlim and ylim depending on the variable we're looking at
    ax = plotted_df.plot.scatter(x=x_var, y=y_var, s=0.3, c='red', logy=True, xlim=[df[x_var].min(), df[x_var].max()],
                                 ylim=[df[y_var].min(), df[y_var].max()],
                                 title=("%s vs. %s" % (plot_vars[y_var][0], plot_vars[x_var][0])),
                                 alpha=1)
    ax.set_xlabel("%s" % plot_vars[x_var][0])
    ax.set_ylabel("%s" % plot_vars[y_var][0])
    plt.scatter(plotted_df.loc[:, 'Unit_price'],
                (plotted_df.loc[:, 'R_ds'].astype(float) * plotted_df.loc[:, 'Q_g'].astype(float)), s=0.3)
    plt.yscale('log')
    plt.show()

def simple_df_plot2(fet_df, x_var, y_var):
    plotted_df = fet_df
    plotted_df = plotted_df[plotted_df['FET_type'] == 'N']
    plotted_df = plotted_df[plotted_df['Technology'] == 'MOSFET']
    #plotted_df = plotted_df[plotted_df['V_dss'] == 30.0]
    plt.scatter(plotted_df.loc[:, x_var],
                (plotted_df.loc[:, y_var]), s=0.3)
    plt.yscale('log')
    #plt.xscale('log')
    plt.xlabel('Cost [$]')
    plt.ylabel('Log[Q_g] [Ohm]')
    plt.title('Qg vs Cost')
    plt.show()

def simple_df_plot3(fet_df,x_var,y_var):
    import numpy as np

    plotted_df = fet_df
    plotted_df = plotted_df[plotted_df['FET_type'] == 'N']
    plotted_df = plotted_df[plotted_df['Technology'] == 'MOSFET']
    plotted30_df = plotted_df[plotted_df['V_dss'] == 30.0]
    plt.scatter(plotted30_df.loc[:, x_var],
                np.log10(plotted30_df.loc[:, 'R_ds'] * plotted30_df.loc[:, 'Q_g']), s=0.3)
    plotted100_df = plotted_df[plotted_df['V_dss'] == 100.0]
    plt.scatter(plotted100_df.loc[:, x_var],
                np.log10(plotted100_df.loc[:, 'R_ds'] * plotted100_df.loc[:, 'Q_g']), s=0.3)
    plotted600_df = plotted_df[plotted_df['V_dss'] == 600.0]
    plt.scatter(plotted600_df.loc[:, x_var],
                np.log10(plotted600_df.loc[:, 'R_ds'] * plotted600_df.loc[:, 'Q_g']), s=0.3)
    #plt.yscale('log')
    #plt.xscale('log')
    plt.legend(['30V', '100V', '600V'])
    plt.xlabel('Log[Cost] [$]')
    plt.ylabel('Log[RQ product] [Coul]')
    plt.title('RQ product vs Cost at set voltages, Q2 area')
    plt.show()





    plotted_df = fet_df
    plotted_df = plotted_df[plotted_df['FET_type'] == 'N']
    plotted_df = plotted_df[plotted_df['Technology'] == 'MOSFET']
    plotted30_df = plotted_df[plotted_df['V_dss'] == 30.0]
    plt.scatter(plotted30_df.loc[:, x_var],
                (plotted30_df.loc[:, y_var]), s=0.3)
    plotted100_df = plotted_df[plotted_df['V_dss'] == 100.0]
    plt.scatter(plotted100_df.loc[:, x_var],
                (plotted100_df.loc[:, y_var]), s=0.3)
    plotted600_df = plotted_df[plotted_df['V_dss'] == 600.0]
    plt.scatter(plotted600_df.loc[:, x_var],
                (plotted600_df.loc[:, y_var]), s=0.3)
    plt.yscale('log')
    plt.xscale('log')
    plt.legend(['30V', '100V', '600V'])
    plt.xlabel('Log[Cost] [$]')
    plt.ylabel('Log[Q_g] [Coul]')
    plt.title('Qg vs Cost at set voltages, all areas')
    plt.show()



def area_plotting(ready_df):
    #here, can start adding in area filtering stuff, and try training on a subset of area, so filter to include the
    #bottom quarter of area
    ready_df = area_filter(ready_df)
    #bottom quarter
    #area_range = np.abs(ready_df['Pack_case'].max() - ready_df['Pack_case'].min())
    idx1 = 0
    idx2 = int(np.floor(len(ready_df)/4))
    idx3 = int(np.floor(len(ready_df)/2))
    idx4 = int(np.floor(len(ready_df)/(4/3)))
    idx5 = int(np.floor(len(ready_df)))

    #get the discrete set of areas, take the lowest quarter, second lowest quarter, etc.
    unique_areas = np.sort(ready_df['Pack_case'].unique())
    idx1 = 0
    idx2 = int(np.floor(len(unique_areas) / 4))
    idx3 = int(np.floor(len(unique_areas) / 2))
    idx4 = int(np.floor(len(unique_areas)/(4/3)))
    idx5 = int(np.floor(len(unique_areas)))
    new_df = ready_df[ready_df['Pack_case'].isin(unique_areas[idx4:idx5])]
    # now plot the area-filtered version using the previous visualization function 'scatter cost bracket'
    scatter_cost_bracket(new_df)
    reg_score_and_dump2(new_df, 'R_ds')

    #ready_df = ready_df[ready_df['Pack_case'].lt(ready_df['Pack_case'].min() + area_range/30)]
    #ready_df = ready_df.loc[(ready_df['Pack_case'].sort_values())[idx2:idx3].index]
    #ready_df = ready_df[ready_df['Pack_case'].lt(ready_df['Pack_case'].min() + area_range/30)]
    #simple_df_plot3(ready_df, 'Unit_price', 'RQ_product')
    new_df = ready_df[ready_df['Pack_case'].isin(unique_areas[idx2:idx3])]
    reg_score_and_dump2(new_df,'R_ds')
    new_df = ready_df[ready_df['Pack_case'].isin(unique_areas[idx3:idx4])]
    reg_score_and_dump2(new_df, 'R_ds')
    new_df = ready_df[ready_df['Pack_case'].isin(unique_areas[idx4:idx5])]
    reg_score_and_dump2(new_df, 'R_ds')
    # new_df = ready_df[ready_df['Pack_case'].isin(unique_areas[idx1:idx2])]
    # reg_score_and_dump(new_df, 'Q_g')
    #trained_models is a dataframe essentially
    trained_models,best_degrees = load_models(['Q_g'])
    simple_df_plot3(new_df, 'Unit_price', 'Q_g')
    model_plus_data(np.array(trained_models.loc[0]), ready_df, best_degrees[0], 'v', 'RQ_product')
    volt_set_list = [30, 100, 600]
    for volt_set in volt_set_list:
        new_df = ready_df[ready_df['V_dss'] == volt_set]
        simple_df_plot2(new_df, 'Unit_price', 'R_ds')
