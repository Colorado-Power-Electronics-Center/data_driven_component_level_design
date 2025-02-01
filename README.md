# Public_Component_Level_Power_Designer
Combined repository for the code related to the component-level data-driven power designer

	1. Dataset: 
		a. Scraped off Digi-key
		b. Component information on Si, SiC, GaN transistors, inductors, and capacitors
		c. Scraping the pdfs of transistors to get Coss, Vds,meas, Qrr, and IF reporting;

	2. Cleaning:
		a. Take dataset and turn into numerical or categorical form
		b. Transistors: 
			i. FET material
			ii. N- or P-channel
			iii. Vdss rating
			iv. Ron
			v. Qg
			vi. Qrr
			vii. I_F
			viii. Coss
			ix. Vds,meas
			x. Cost
			xi. Area
		c. Inductors:
			i. Fixed inductors
			ii. Current rating
			iii. Inductance
			iv. R_dc
			v. Area
			vi. Height
			vii. Saturation current
			viii. Cost
      			x. Steinmetz parameters: determine using estimation algorithms
		d. Capacitors:
  			i. Vrated
     			ii. Capacitance, nominal
			iii. Cost
   			iv. Area
		Note: Variables listed as max. vs. typical, on websites sometimes reported as one, sometimes the other

	3. Modeling/estimation techniques for complex loss model:
		a. K_T estimation based on Vdss, FET material grouping
		b. Tau_c, tau_rr estimation based on predictions, after first computing via Qrr, I_F, diF/dt
		c. Coss,0.1 estimate based on predicting Coss and Vds,meas and computing based on gamma, which is predicted based on Vdss, FET material grouping
		d. Rac of inductors based on predicted quantities and Matlab-based modeling script
		e. IGSE loss contribution based on predicted quantities and the iterative IGSE method
		f. Cap. @ 0Vdc based on cap @ Vdc data, given computed Vdc

	4. Model training:
		a. Outlier removal: Look at the trends with certain values with respect to other parameters, see if there are outliers with these correlations, rather than purely based on the mean
			i. Each output parameter w.r.t. each of the input variables
		b. Take Pareto-front on each dataset (training on the datasets we have all the information for):
			i. Transistors (main page): Vdss, Qg, Ron, Cost, Area
      			ii. Transistors (pdf params): Vdss, Qg, Ron, Cost, Area, Qrr, Coss
			iii. Inductors: Rdc, Area, Cost, Current rating
   			iv. Capacitors: Cost, Area
    		c. Initialization models of Ron for each fet, fsw, delta_i, and area of each capacitor
			i. Transistor predictions of Qg, tau_c, tau_rr Coss0.1, Cost, Area as outputs, with Vdss, Ron, FET type as inputs
			ii. Inductor predictions of Rdc, Area, Height, Cost, Steinmetz parameters as outputs, with Current rating, Saturated current, Inductance (from fsw) as inputs
			iii. Capacitor predictions of Cost, Area
			iv. Uses kernel-ridge chained regression as the base model
	  		Note: prediction results similar across algorithms, could just combine into kernel ridge, linear, KNN, RF, etc. without significant change in performance accuracy

	5. Optimization tool:
		a. Optimization variables: Ron1, Ron2, f_sw, delta_i, A_C1, A_C2
		b. Set bounds on each of these values via constraints
		c. Supply a power loss function, and then cost and area functions are just the sum of all the component cost and area predictions, respectively
		d. Supply which is the objective function, and values on the other two as constraints
		e. Currently optimizes by including transistors, inductors, and capacitors
		f. COBYLA optimization algorithm, which handles multivariables, constraints/bounds, and adjustable convergence metrics

  	6. Brute force optimization:
   		a. Scoring system: Using component parameters returned by tool, determine which components are closest or farthest less than by using normalization metric (x-x_opt)/x_max --> Note no absolute 
                   value
		b. Based on desired run-time, select n top components from database of each component in design
  		c. Run the brute-force optimization using the x components n-length component databases and same loss equations as in the tool
    	d. Select desired component combination from returned results
		e. Use printing functions to print all parameters of optimized components and selected components
    		Note: For model estimation methods, same approach is used as in the optimization tool
      		Note: To get as good a loss estimation as possible, brute-force method only looks at component databases with all parameters available

    Instructions for use: main_script.py walkthrough
    Note: the steps are listed in __main__.
    
    1. Data scraping: 
        Description: This is the code to get data from the main page websites of components. Different starting URLs are
        needed for the different components used in the design. 
	
        i) full_pdf_part_info_xl. This function goes to the starting main page, then downloads
        all the table data available by automating the click on the 'download table' button on the site. Here, 'table' is defined as the
	array of information from as sites like Digikey present it. 
	
        ii) full_pdf_part_info(): Opens the pdf and turns the data into tables of information. The data that is provided to the function
        is the pdf link as well as all the main page information. 
        Note where this scraping function ends, for ease of use, one could put a breakpoint here and terminate the function if the only
	goal was to scrape data. All downloaded tables are in a .csv in the default directory xl_pdf_sets/. 
	
        combine_downloaded_tables(): Go through all files in xl_pdf_sets, and combine onto one giant .csv, for ease of 
        going through later. Put onto xl_pdf_sets/merged_component_list_files.csv.
	
        full_pdf_part_info_xl(): If want to actually go into each of the pdfs, uncomment the part after 'actual scraping process'
        to go into scrape_datasheet_tables(). Here, combine onto the main list overall_datasheet_lists as a list of lists: 
        looks like [[Mfr_part_no1, [main_page_table_info1], df_of_scraped_pdf_info_tables1],[2],[3]]. Can select if want to
        do with or without pdf info, if not will have an empty list of df_of_scraped_pdf_info_tables. The final step here
        is to pickle overall_datasheets_lists onto xl_pdf_sets/pickled_data_no_sheets or xl_pdf_sets/pickled_datasheets.
	
        find_t_rr(): Now unpickle the overall_datasheets_lists and go through each component, trying to find the actual
        numerical values of the pdf quantities via the df's of all the scraped pdf pages. Currently these pickled files
        are on xl_pdf_sets/pickled_datasheets2. Create a DigikeyFet_downloaded object, where all attributes are put onto.
        Inside this function is flexible for the designer to adjust how to find the desired information, and where code would
        be placed to find and keep additional parameters. Note the edge cases and the differences bw manufacturers as to
        where certain values can be found. This part will also require some iteration--that is why another file version
        to look at is pickled_datasheets_from218, which helps to only look at a few components, and start checking cases.
        Go through trials of potential ways different manufacturers represent the information. Watch out for scraping false
        information. At the end, each component_obj is put onto csv_files/FET_pdf_tables_wt_rr_full.csv using single_fet_to_csv_downloaded().   
        FET_pdf_tables_wt_rr_full.csv is the final file for this section. 
	
        For inductors, a similar process can be used, except there are no pdfs that need to be opened, even though the
        datasheet links will still be scraped. Currently, an older version of the scraping process was used to get inductor 
        information, which is found on inductor_training_updatedAlgorithm.csv after being passed through Bailey's parameter
        estimation script.
	
        For capacitors, A similar scraping process can be used. 
        full_df_part_info(): Make sure to input the new starting link. Everything will once again be put into xl_pdf_sets/.
	
        combine_downloaded_tables(): Make sure component = 'cap'. Put onto xl_pdf_sets/merged_capacitor_list_files.csv.
	
        full_pdf_part_info_xl(): Set component = 'cap', and once again create a list of 
        [[Mfr_part_no1, [main_page_table_info1], df_of_scraped_pdf_info_tables1],[2],[3]]. Put onto xl_pdf_sets/pickled_data_capacitors_no_sheets.
        This is the last file needed before cleaning.
        

    2. Data cleaning:
        Description: This section goes through the data and puts the data into objects with attributes easily used for
        computations such as ML model training. Ensures numerical values. This function starts in main_script.py and 
        is called data_cleaning().
	
        Transistors: In data_cleaning_full(), comment out capacitor_cleaning(). Open csv_files/FET_pdf_tables_wt_rr_full.csv.
        Parse the data using initial_fet_parse(), then go through and find additional pdf parameter quantities more cleanly
        (the first part only finds the code snippet with the right value, this part specifically cleans it). Put the 
        cleaned fet data onto cleaned_fet_dataset2.
	
        Inductors: Most of this cleaning is done in Bailey's script.
	
        Capacitors: In data_cleaning_full(), uncomment capacitor_cleaning(). 
        The first part of this function is taking the numerical information manually gathered on some capacitors showing
        the capacitance at various Vdc values, and putting this data onto datasheet_graph_info/capacitor_pdf_data_cleaned.csv.
        Include the area of the capacitor, as this is relevant training info. This will later be used in a separate ML
        model training to get the predictions of cap @ Vdc based on cap @ 0Vdc.
        The next part is to clean the scraped capacitor main page information. Open xl_pdf_sets/pickled_data_capacitors_no_sheets.
        Then for each line, create a DigikeyCap_downloaded object with all of the capacitor attributes.    
        As an interim step, write data to csv_files/capacitor_data_class2_extended.csv. The next step is re-opening this
        csv and parsing the data using a call to initial_fet_parse(). This new df is put onto csv_files\capacitor_data.csv.

    3. Physics-based loss modeling:
        Description: This section is where to generate quantities used for physics-based loss modeling. As described in the
        intro description at the top of this file, the following quantities need to be estimated: fets: kT based on Vdss, 
        tau_c and tau_rr need to be computed based on the scraped Qrr, IF, diFdt, and trr quantities, Coss,0.1 estimation based
        on the physics-based groupings and normalized graph quantities. Inductors: Rac estimation based off matlab script,
        IGSE estimation based on Bailey's updated algorithm. Capacitors: Capacitance relationship between 0Vdc and other
        Vdc values.
        Inside main_script.py, go to physics_based_groupings(). 
	
        FETs:
        kT: Ron_plotting(): Select a specific grouping by setting grouping = 'group'. The dictionary values_dict has 
        all the components with their voltage rating and ratio kT of Ron at 20 deg. C divided by Ron at 0 deg. C (reported).
        When plotting these for a given grouping, use np.polyfit() to get a,b, which will give the values for the kT dictionary
        showing the linear equations. These values are used in kt_compute(), kT_eqn_dict.
	
        Coss,0.1: Coss_plotting(): Set the grouping by setting grouping = 'group'. Shows lists with the following order
        of entries: [voltage rating [V], Coss value [pF] at 10% voltage, Vdss at Coss reported measurement [V], Coss reported measurement [pF]].
        Use this data to plot normalized values, and get these slopes for each grouping. These a,b coefficients are used
        as the linear equations in gamma_compute().
	
        tau_c, tau_rr: These are found separately, most of the work is in fet_optimization_chained_wCaps.py. Qrr_est_new()
        shows taking the datasheet quantities of Qrr_ds, trr_ds, and IF_ds to compute tau_c and tau_rr. In the other direction,
        Can check that everything is working via Qrr_test_case() in __main__, where we set the specific tau_c, tau_rr, and
        then compute Qrr, trr for use in the loss equation. Prior to training, calls are made to Qrr_est_new() to get
        all the tau_c, tau_rr values for each real components. Inside the tool, the tool itself makes calls to these functions given predictions
        of tau_c and tau_rr to get Qrr, trr.
	
        Inductors: 
        All of the parameters needed for compute_Rac() and compute_IGSE() inside the optimization tool are determined
        via Bailey's functions.
	
        Capacitors: 
        Cap.@0Vdc: These predictions are a function of cap. voltage rating, cap. area, and nominal capacitance. The datapoints
        of the delta_C vs. Vdc are found in capacitor_pdf_data.csv, listed as [Label (see Mathematica doc esr_size_freq_cap_data.nb
        for encoded description/version), Mfr part no., Vrated, Area (w/ size code, inside functions will decode size codes
        into mm^2), Vdc_meas, deltaC at Vdc_meas, capacitance at 0Vdc. These points were collected manually and recorded 
        here to be used for training. 

    4. ML model training:
        Description: This section is where the ML model trainings occur. There are many models needed by the tool. To start,
        go into main_script.py and run ML_model_training(). Most of the ML model training code is in fet_best_envelope.py.
        For most parameters, it is also important to take the log10 of the data. reg_score_and_dump_cat() is the main
        function for training models. Inside this function, param_training_dict() is a dictionary with {training_category1: 
        {'inputs': inputs1, 'outputs': outputs1, 'file_name': filename1, 'order': order1}, {training_category2: {}}.
        Note that 'file_name' is where the trained ML models are put onto, and that it's important that the optimization
        tool is looking at the correct files and folders. 'order' is a numerically ascending list, necessary for the
        chained regression multi-output method used by the tool. This has to match the number of outputs. Uses cross-validation
        to score models, and various ML model algorithms can be commented or uncommented. Uncomment the specified line 
        towards the end to dump the trained models onto the specified joblib file. There is also commented-out code beneath
        for comparing the testing performance with the performance on the training data itself.
	
        Transistors:
        Inside train_all(), set parameter_training = 'main_page_params' or 'pdf_params'. The first few steps are generally
        the same, with slightly different parameters considered and datasets used. The first step removes
        outliers using the outlier_detect() function, on the parameters in parameter_list, w.r.t. Vdss. outlier_detect()
        is based on using the Mahalanobis distance. The second step takes the 
        Pareto front of the data on the specified dimensions of interest. Note that there are some dimensions that should
        not be included in the Pareto front, but that have to be specified in data_dims_keep to make sure they are retained
        by the dataset after taking the Pareto front. The third step uses one-hot encoding to assign variables to the 
        FET technology and channel type. Finally, the fet_training() function is used to train the three separate parameter 
        sets: main page, pdf params, and area. These three parameters are specified in the retrain_parameter argument of 
        fet_training(). The df should be supplied for this function. 
	
        main page parameters: After doing the above three main first steps, there are three different retrain parameters that 
        must be gone through to have all necessary joblib files currently used by the tool: FOMs (which is the main
        page general data), area (which is the KNN area predictions), and initialization (which is to get the initial
        estimates for the optimization parameters, and takes a slightly different set of inputs). 
            main page general data: If retrain_parameter == 'FOMs', go into reg_score_and_dump_cat() with the specified 
            inputs and outputs.
	    
            area prediction: If retrain_param == 'area', will go into area_training(). Dump the trained models onto 
            'full_dataset_Pack_case.joblib'.
	    
            initialization: If retrain_parameter == 'initialization', go into reg_score_and_dump_cat() and get starting 
            predictions for Rds as a function of other variables. 
	    
        pdf parameters: After doing the above three main first steps, also have some additional GaN data here for Coss and
        Vds,meas measurements. Also have some additional Qrr, IF, diFdt, and trr data a little farther down for transistors.
        Could also add any additional pdf datasheet data here if want to add that manually. The next new thing is computing 
        tau_c and tau_rr for all datapoints. This is done here by making a call to Qrr_est_new(). Dump this new file with
        available info onto 'cleaned_fet_dataset_pdf_params3'. Finally, go into fet_training() w/ arguments
        retrain_params = 'FOMs' and training_params = 'pdf_params'.
	
        Inductors: 
        main page/Steinmetz parameters, and initialization: Farther down in train_all() is a line for csv_file = 'csv_files/inductor_training_updatedAlgorithm.csv'.
        From this point, the Pareto-front is taken, and then into two reg_score_and_dump_cat() calls. The first has 
        training_params = 'inductor_params', which has all the inductor main page information and Steinmetz parameters.
        The second has trainin_params = 'fsw_initialization', which makes an initial prediction for L that can be used
        as a starting point estimate for fsw, one of the optimization variables. To get the delta_i initial guess, can use 
        the predicted quantities along with additional calculations.
	
        Capacitors: 
        Main page parameters: Inside main_script.py -> ML_model_training() -> train_all(), uncomment capacitor_cleaning().
        Inside here, note only 5 Class II temperature coefficients are included. Note that cap. @ 0Vdc is one of the inputs.
        After Pareto optimizing, send to reg_score_and_dump_cat() w/ training_params == 'cap_main_page_params'.
        Initialization: Right below the main page parameters call to reg_score_and_dump(), uncomment the line that is
        a call to reg_score_and_dump_cat() w/ training_params == 'cap_area_initialization'.
        Cap.@0Vdc: Right after the code sending the main page info for training via reg_score_and_dump_cat(), which takes
        a separate dataset with all the capacitance, delta C, and voltage info and sends to be trained with 
        training_params == 'cap_pdf_params'.
        At this point, all ML models needed by the tool have been trained.

    5. Run the optimization tool:
        Description: This section has the code for the build of the optimization tool to take the design information for
        the desired topology and determine optimal parameters according to the optimization algorithm. In main_script.py,
        this code is found in optimize_converter(), but it can also be useful to go straight to __main__ of
        fet_optimization_chained_wCaps.py to run through specific cases for the tool.
        Note that in optimize_converter() is code for plotting the optimized parameters, which can be a useful visualization.
        From optimize_converter(), takes into function where user can specify dictionary of design parameters. This 
        dictionary can be adjusted for different test cases, and as part of this, the user specifies their desired
        topology. At the end of this section ("Run the optimization tool") is more detail on how to create different
        topology cases.
	
        loss_comparison_plotting() takes the dictionary param_dict and runs the optimization. 
        There are 4 OptimizerInit object functions that must be declared by the designer. In the codebase, all 4 functions
        are found right after another. 
	
        The first is OptimizerInit.get_params(). In this function, set the initially known design variables for the specific 
        design example and topology (e.g. Vin, Vout, Iout, Vdss, etc.)
	
        The second is OptimizerInit.set_optimization_variables(). Set what component attribute each of the optimization 
        variables corresponds to, and set topology-specific initialization equations based off these variables. e.g.:
        self.fet1.Rds = x[0], self.fet2.Rds = x[1]. And then e.g. self.cap1.Capacitance = ..., self.ind1.deltaB = ...
	
        The third is OptimizerInit.create_component_lists(). Create objects for each component the user wants in their design, 
        and add to an overall list of each component type. The user must index their components corresponding to their 
        topology block diagram. e.g. self.fet1 = OptimizerFet(param_dict, 0), self.fet2 = OptimizerFet(param_dict, 1).
        And, self.fet_list.extend([self.fet1, self.fet2]), etc.
	
        The fourth is OptimizerInit.power_pred_tot(). Compute all physics-based loss modeling parameters, and define the power
        loss function for the topology given the created component objects. e.g. self.fet2.Compute_Cdsq(), then self.fet1.Rds_loss = ,
        then Q1_loss = self.fet1.Rds_loss + self.fet1.Qg_loss, then self.power_tot = self.Q1_loss + self.Q2_loss.
    
        Once all of these are set up, the tool starts with the OptimizerInit object optimizer_obj. From here, runs through various fet 
        technologies (Si, GaN, SiC) cases based on what is in optimizer_obj.tech_list and assumes N-channel FETs, this
        can be changed. Makes a call to optimizer_obj.create_component_lists(), which has three separate object possibilities, OptimizerFet for fets,
        OptimizerInd for inductors, and OptimizerCap for capacitors. For buck converter, considering 5 components:
        Q1 and Q2, the inductor, and the input and output capacitors. 
	
        If want to check cases for various plotting constraint values, which is a common occurence such as getting the 
        optimized power loss values for multiple cost constraints, set optimizer_obj.plot_range_list. optimizer_obj
        stays the overall object of use, but some of the object_obj attributes are reset with each iteration of
        plotting variable value and fet tech, and the results of each run are stored in lists such as 
        optimizer_obj.MOSFET_overall_points_list. The entire optimizer_obj for each specific cost and area and FET tech 
        constraint are pickled, 'optimizer_test_values_MOSFET_overall_points' as one example.
        Initializes values for all fet, ind, and cap objects, and sets the optimization variables.
	
        The main optimization function is found in optimizer_obj.minimize_fcn(). Inside minimize_fcn(), starts with
        con_cobyla, which is a dictionary of the constraints and bounds needed for the selected optimization algorithm,
        COBYLA. Various other constrained, bounded, multi-variate optimization algorithms exist, and would follow a similar structure, but 
        each has slightly different structure requirements. 
	
        Next, initialize all variables using obj.init_fet(), obj.init_ind(), and obj.init_cap(). 
        These functions use the pre-trained initialization models for all components, based on known quantities at the
        start of the optimization, and generate initial starting values for all of the optimization variables.
	
        Then set the optimization
        variables in the desired format of COBYLA: self.x0, a list of all the optimization variables, and make sure they match
        the actual object attributes as expected.
	
        Next, predict all component parameters based on the initialized optimization variables and other known quantities
        about the design, using obj.predict_fet(), obj.predict_ind(), and obj.predict_cap(). These functions use the
        pre-trained models on component parameters.
	
        Now the minimize function is used from the scipy.optimize package, imported at the top. See scipy.optimize
        documentation for more information on the arguments. Here is where method='COBYLA' is specified, and the
        constraints=con_cobyla are set, in addition to other parameters that can contribute to successful convergence.
        The first argument is the function to be minimized, and the second argument is the starting values of all
        optimization quantities. The first argument would be adjusted if the designer wants to minimize e.g. cost instead
        of power loss. con_cobyla would then have to be adjusted to match any desired constraints.
	
        cost function: self.cost_pred_tot(). First the algorithm checks that the constraints are met. The cost function runs
        through all components and sums their cost, returning the total.
	
        area function: self.area_pred_tot(). The other constraint (in the current example) is the area. The area function
        runs through all components and sums their area, returning the total.
	
        power loss function: self.power_pred_tot(). Makes predictions based on the latest updated values of the optimization
        variables, for each of the components. Then computes all physics-based loss-related quantities, e.g. Cdsq and
        Qrr, trr. Once all quantities have been generated, everything is needed for power loss computation. Goes through
        each component and computes each specific loss contribution, then sums all contributions for each component, then
        sums all components, returning the total power loss. It is desirable to break down the loss contributions within
        each component for viewing and analyzing the individual contributions after running the example, so that the entire
        optimization need not be run again.
        
        When the algorithm has reached convergence, it will terminate and the results will be on the rescobyla object. The status
        number will indicate whether or not the convergence was successful or not (see the algorithm documentation for 
        specifics on what the status numbers mean). optimizer_obj.status == 1 indicates a success. Now are back in the
        loss_comparison_plotting() function. Put the results onto lists for total power loss, cost, and area, and pickle
        the entire optimizer_obj object to view results at any time. Following is the code for plotting the results. There
        is also code to make the graph look nice, based on the parameters in obj,get_visualization_params().

    6. Determine real components via exhaustive search on filtered databases:
        Description: The final step is to select the actual components. The parameters used by the optimization algorithm
        are used to filter the database associated with each component in the design, to find components with as good or
        better parameters, and then perform an exhaustive search to find optimal component combinations. 

        From main_script.py, component_predictions() --> predict_components() (in component_selection.py). Unpickle the saved
        optimization data based on the specified constraints. Then given this object, go into make_component_predictions_bf().
        Then have a variety of functions that can be implemented (this part of the code could be cleaned up so that you
        don't have to uncomment the functions you want, could instead specify function arguments for each function call).
        optimization_case.filter_components(): Create predictor objects (fet_predictor, ind_predictor, etc.). 
	
        First, case.database_df_filter(). Here, using df.normalize_x(), score each component. A normalized scoring method is used to 
        take each parameter and see how close or far below it is
        from what the tool determined to be optimal. Then these scores are summed, and the top n components are selected.
        The value of n determines the total run-time for this step, where higher n results in better power loss performance,
        but takes more time to run. n=10 is currently selected for this step. Note that the values of component attributes
        are not considered in the context of the loss equations here, because they rely on values that are dependent on
        other components and therefor the specific combination. The exception is inductors. These loss-related quantities
        are computed up-front, and used to determine which is best. This function returns the database as a df of the top
        n components.
	
        Second, case.compute_loss_equations(). All this does is turn each object from the dataframe into a list easily used 
        for the matrix reduction of optimal combinations. Because we cannot compute all power loss equations for all 
        components prior to knowing the combinations, we are only setting up the lists here for some components, but for
        inductors there is a call to compute_Rac() and compute_IGSE(). After doing that, the database is filtered, and 
        compute_loss_equations_final() is what turns the list into a list for use in the matrix calculations.
	
        Third, after filtering the components, is case.optimize_combinations(). This creates meshgrids of all of the
        component databases, and 1. sums the costs and areas and check that they meet constraints, 2. checks that the output
        capacitance meets the requirement, 3. computes total power loss with component_combo.compute_total_power(), 
        4. sorts the power losses from lowest to highest, and prints all the manufacturer part numbers in order. Note that
        specific components can be listed by mfr part no. in lists 'unavailable_fets, unavailable_inds, etc.', so that if
        a specific component is not available (usually inductors), it can be skipped over.

        Once components have been selected, can include them in the following code inside predict_components(), where 
        there's cases shown for FET technology and design parameters (cost constraint, and how many components num_comps 
        were considered when filtering the databases). Here, fill in the list with the selected mfr part nos, then go
        into compare_parameters(). First, filters the database to select that specific mfr part no., then prints all the
        parameters of the selected components. Then can call optimize_combinations() (note the difference in certain
        cases when the function sees there's only one component), and then prints the loss and cost
        breakdown given those components. The next code block prints the parameters of each component based on what the
        tool returned, and then the next code block prints all of the theoretical loss contributions.

        Finally, optimizer_fsw(). This takes the selected components and runs through a generated list of potential
        switching frequencies, and sees which one gives the lowest power loss given the selected inductor.

Running the tool: The code has been structured such that the user does not need to enter the tool codebase or adjust the functions 
    of the tool itself in order to run their design. In order to create a new design, the user must go to 
    optimization_tool_case_statements.py and go to __main__(). Do the following:
    1. Add desired info to param_dict showing what kind of design runs to do.
    2. Update the following functions, adding case statements for the desired topology and operating conditions:
        i. get_params_separate()
        ii. set_optimization_variables_separate()
        iii. create_component_lists_separate()
        iv. power_pred_tot_separate()
        Additional functions that can be updated are:
        i. print_unnormalized_results()
    Descriptions of each function are included in the code. Once these functions have been updated, the user runs __main__().

Component selection: The code for selecting components given the results from the ML-based step of the tool are found in
    component_selection_case_statements.py. The user also needs to update functions here with their desired case statement for
    their design:
    i. set_combo_variables()
    ii. set_inductor_attributes()
    iii. make_oririginal_arrays()
    iv. make_new_arrays()
    v. make_valid_arrays()
    vi. vectorize_equations()
    vii. determine_norm_scored_vars()
    Additional functions that can be updated are:
    i. print_practical()
    ii. print_theoretical()

    The first process is the predict_components(). If the user desides to run this function, they update with their design 
    information, then the script will pull the optimized parameters from the ML-based step, and use these to return lists of 
    real, commercially available components after running the exhaustive search with a set n choices per component database 
    (major function here is make_component_predictions_bf()). Include this function in __main__() of 
    component_selection_case_statements.py and run __main__().
    
    The second process can be used to compare the selected component combination with the theoretical values from the ML-based
    step. The compare_combinations() function also needs the dict_index updates, as well as the set_combo_dict,
    in order to specify which design they are running, and which components they intend to use.
    
    To run, the user goes to __main__() and runs.
