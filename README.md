# Public_Component_Level_Power_Designer
Combined repository for the code related to the component-level data-driven power designer.
Split into three sections: PARTS OF THE TOOL, BACKEND CODE, FRONTEND CODE.
Note that the data scraping must be performed to use the tool, as the scraped data is not published, and 
subsequently the design-oriented parameters and ML models also need to be generated.
Also note that the code instructions are published here.

PARTS OF THE TOOL:

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

INSTRUCTIONS FOR BACKEND CODE: main_script.py walkthrough
Note: the steps are listed in __main__.
    
    1. Data scraping (__main__ --> data_scraping() ): 
        Description: This is the code to get data from the main page websites of components. Different starting URLs are
        needed for the different components used in the design. 
	
        i) full_pdf_part_info_xl. This function goes to the starting main page, then downloads
        all the table data available by automating the click on the 'download table' button on the site. Here, 'table' is defined as the
	array of information from as sites like Digikey present it. 
	
        ii) full_pdf_part_info(): Opens the pdf and turns the data into tables of information. The data that is provided to the function
        is the pdf link as well as all the main page information. 
        Note where this scraping function ends, for ease of use, one could put a breakpoint here and terminate the function if the only
	goal was to scrape data. All downloaded tables are in a .csv in the default directory xl_pdf_sets/. 
	
        iii) combine_downloaded_tables(): Goes through all files in xl_pdf_sets/ folder, and combines onto one giant .csv (to make it easier
	to iterate through later). Puts all of the component tables under their respective mft part no. label, onto the file 
        xl_pdf_sets/merged_component_list_files.csv.
	
        iv) full_pdf_part_info_xl(): If you want to actually go into each of the pdfs (sometimes useful for seeing what data is read in),
	uncomment the section after the functions listed in i)-iii),
        to enter scrape_datasheet_tables(). This function takes all of the scraped components and creates a list of lists stored in
	overall_datasheet_lists. 
        The list entries look like: [[Mfr_part_no1, [main_page_table_info1], df_of_scraped_pdf_info_tables1],[2],[3]].
	The final step in this function is to pickle overall_datasheets_lists onto xl_pdf_sets/pickled_data_no_sheets (if there is no pdf 
 	information) or xl_pdf_sets/pickled_datasheets (if there is pdf information).
	
        v) find_t_rr(): This function unpickles the overall_datasheets_lists and goes through each component, trying to find the actual
        numerical values of the pdf quantities via the dataframes (dfs) of all the scraped pdf pages. Currently these pickled files
        are on xl_pdf_sets/pickled_datasheets2. Creates a DigikeyFet_downloaded object, and all attributes are put onto this object.
        Inside this function the designer can adjust how to find the desired information (e.g. if they were searching for an additional
	parameter within the pdf information, they could add lines here). Note the edge cases and the differences bw manufacturers regarding
        where certain values can be found. This part also requires some iteration and work--a testing file version exists
        called pickled_datasheets_from218, which contains only a few components, and you can add components to this list to look at test
	cases easily and check cases. Watch out for scraping false
        information. At the end, each component_obj is put onto csv_files/FET_pdf_tables_wt_rr_full.csv using single_fet_to_csv_downloaded().   
        FET_pdf_tables_wt_rr_full.csv is the final file for this section. 
	
        For inductors, a similar process can be used, except there are no pdfs that need to be opened, even though the
        datasheet links will still be scraped. Currently, an older version of the scraping process was used to get inductor 
        information, which is found on inductor_training_updatedAlgorithm.csv after being passed through coreloss.py, the parameter
        estimation script.
	
        For capacitors, A similar scraping process can be used. 
        i) full_df_part_info(): Make sure to input the new starting link for capacitors. Everything will once again be put into xl_pdf_sets/.
	
        ii) combine_downloaded_tables(): Make sure component = 'cap'. Put onto xl_pdf_sets/merged_capacitor_list_files.csv.
	
        iii) full_pdf_part_info_xl(): Sets component = 'cap', and once again creates a list of 
        [[Mfr_part_no1, [main_page_table_info1], df_of_scraped_pdf_info_tables1],[2],[3]]. Put onto xl_pdf_sets/pickled_data_capacitors_no_sheets.
        This is the last file needed before cleaning.
        

    2. Data cleaning (__main__ --> data_cleaning() ):
        Description: This section goes through the data and cleans it, ultimately putting the data into objects with attributes easily used for
        computations such as ML model training. Ensures all quantities are numerical values. 
	
        Transistors: In data_cleaning_full(), opens csv_files/FET_pdf_tables_wt_rr_full.csv.
        Parses the data using initial_fet_parse(), then goes through and finds additional pdf parameter quantities more cleanly
        (the first part of the scraping process, as listed in find_trr() above, only finds the code snippet with the right value, the second part specifically cleans it). Puts the 
        cleaned fet data onto cleaned_fet_dataset2.
	
        Inductors: Most of this cleaning is done in coreloss.py.
	
        Capacitors: In data_cleaning_full() --> capacitor_cleaning(). 
        i) The first part of this function takes the numerical information manually gathered on a subset of capacitors with
        the capacitance at various Vdc values, and puts this data onto datasheet_graph_info/capacitor_pdf_data_cleaned.csv.
        The area of the capacitor also has to be included because this is relevant training info. This will later be used in a separate ML
        model training to get the predictions of cap @ Vdc based on cap @ 0Vdc.
        ii) The second part cleans the scraped capacitor main page information. Opens xl_pdf_sets/pickled_data_capacitors_no_sheets.
        Then for each line, creates a DigikeyCap_downloaded object with all of the capacitor attributes.    
        As an interim step, this function writes the data to csv_files/capacitor_data_class2_extended.csv. 
	iii) The third step re-opens the csv and parses the data using a call to initial_fet_parse(). This new df is put onto csv_files\capacitor_data.csv
 	and can then be used in ml model training.

    3. Physics-based loss modeling (__main__ --> physics_based_groupings() ):
        Description: This section is where the quantities used for physics-based loss modeling are generated. As described in the
        intro description at the top of this file, the following quantities need to be estimated: Transistors: kT based on Vdss, 
        tau_c and tau_rr need to be computed based on the scraped Qrr, IF, diFdt, and trr quantities, Coss,0.1 estimation based
        on the physics-based groupings and normalized graph quantities. Inductors: Rac estimation based off matlab script,
        IGSE estimation based on the updated algorithm in coreloss.py. Capacitors: Capacitance relationship between 0Vdc and other
        Vdc values.
	
        Transistors:
	
	        kT: Ron_plotting(): Select a specific grouping by setting grouping = 'group'. The dictionary values_dict has 
	        all the components with their voltage rating and ratio kT of Ron at 20 deg. C divided by Ron at 0 deg. C (reported).
	        When plotting these for a given grouping, use np.polyfit() to get a,b, which will give the values for the kT dictionary
	        showing the linear equations. These values are used in kt_compute() and kT_eqn_dict.
	
	        Coss,0.1: Coss_plotting(): Set the grouping by setting grouping = 'group'. Shows lists with the following order
	        of entries: [voltage rating [V], Coss value [pF] at 10% voltage, Vdss at Coss reported measurement [V], Coss reported measurement [pF]].
	        Use this data to plot normalized values, and gets the slopes for each grouping. These a,b coefficients are used
	        as the linear equations in gamma_compute().
		
	        tau_c, tau_rr: These are found in an algorithm found in fet_optimization_chained_wCaps.py:
	 		i) Qrr_est_new() takes the datasheet quantities of Qrr_ds, trr_ds, and IF_ds to compute tau_c and tau_rr. 
    			ii) Qrr_test_case() checks that everything is working, where you set the specific tau_c, tau_rr, and
	        	then compute Qrr, trr for use in the loss equation. Prior to training, calls are made to Qrr_est_new() to get
	        	all the tau_c, tau_rr values for each component. 
	  		Inside the tool, the tool itself makes calls to i) and ii) given predictions
	        	of tau_c and tau_rr to get Qrr, trr which is used in the loss function.
		
        Inductors: 
	
	        All of the parameters needed for compute_Rac() and compute_IGSE() inside the optimization tool are determined
	        via the functions in coreloss.py.
	
        Capacitors: 
	
	        Cap.@0Vdc: These predictions are a function of cap. voltage rating, cap. area, and nominal capacitance. The datapoints
	        of the delta_C vs. Vdc are found in capacitor_pdf_data.csv, listed as [Label (see Mathematica doc esr_size_freq_cap_data.nb
	        for encoded description/version), Mfr part no., Vrated, Area (w/ size code, inside functions will decode size codes
	        into mm^2), Vdc_meas, deltaC at Vdc_meas, capacitance at 0Vdc]. These points were collected manually and recorded 
	        here to be used for training. 

    4. ML model training (__main__ --> ML_model_training() ):
        Description: This section trains the various ML models used throughout the tool. 
	Most of the ML model training code is in fet_best_envelope.py.
        Note that it is also important to take the log10 of the data for many of the parameters. 
	
	reg_score_and_dump_cat(): the critical function for training models. Inside this function, 
	param_training_dict() is a dictionary with: 
 	{training_category1: {'inputs': inputs1, 'outputs': outputs1, 'file_name': filename1, 'order': order1}, {training_category2: {}}.
        	'file_name': where the trained ML models are put onto. It is important that the optimization
        	tool is looking at the correct files and folders. 
	 	'order': a numerically ascending list, necessary for the chained regression multi-output method used by the tool. 
   		This has to match the number of outputs. Uses cross-validation
        	to score models, and various ML model algorithms can be replaced (see end of function). There is also commented-out code beneath
        	for comparing the testing performance with the performance on the training data itself.
	
        Transistors:
        Inside train_all(), set parameter_training = 'main_page_params' or 'pdf_params'. For both options, the first few steps are similar,
	with slightly different parameters considered and datasets used (main page or pdf). 
 	
  	i) outlier_detect(): remove outliers on the parameters in parameter_list, w.r.t. Vdss. outlier_detect()
        is based on using the Mahalanobis distance. Next, this function takes the Pareto front of the data on the specified dimensions of interest. 
	Note that there are some dimensions that should not be included in the Pareto front, but that have to be specified in data_dims_keep to make sure they are retained
        by the dataset after taking the Pareto front. The third step uses one-hot encoding to assign variables to the 
        FET technology and channel type. 
	
	ii) fet_training(): trains the three separate parameter sets: main page, pdf params, and area. 
	These three parameters are specified in the retrain_parameter argument of fet_training(). The df should be supplied for this function. 
	
	        a) main page parameters: After the above steps, there are three different parameters that 
	        must be trained in order to have all necessary joblib files currently used by the tool: 
			1) FOMs (which is the main page general data)
	  		2) area
	    		3) initialization (which is to get the initial
	        	estimates for the optimization parameters, and takes a slightly different set of inputs). 
	            
		    1) main page general data: If retrain_parameter == 'FOMs', go into reg_score_and_dump_cat() with the specified 
	            inputs and outputs.
		    
	            2) area prediction: If retrain_param == 'area', will go into area_training(). Dump the trained models onto 
	            'full_dataset_Pack_case.joblib'.
		    
	            3) initialization: If retrain_parameter == 'initialization', go into reg_score_and_dump_cat() and get starting 
	            predictions for Rds as a function of other variables. 
	    
	        b) pdf parameters: When this is set as the argument, this trains ML models on pdf parameters. The files used here also include additional GaN data with Coss and
	        Vds,meas measurements, as well as some additional Qrr, IF, diFdt, and trr data a little farther down for various transistor types.
	        You could also add any manual additional pdf datasheet data here. tau_c and tau_rr are computed here for all datapoints. This is done here by calling Qrr_est_new(). 
		The data with this information is dumped onto 'cleaned_fet_dataset_pdf_params3'. After adding any additional data, go into fet_training() w/ arguments
	        retrain_params = 'FOMs' and training_params = 'pdf_params' to complete this training step.
	
        Inductors: 
        	a) main page/Steinmetz parameters
	 
	 	b) initialization
   
   		Farther down in train_all() is a line for csv_file = 'csv_files/inductor_training_updatedAlgorithm.csv'.
	        From this point, the Pareto-front is taken, and then two calls are made to reg_score_and_dump_cat() to train 1) and 2). The first has 
	        training_params = 'inductor_params', which has all the inductor main page information and Steinmetz parameters.
	        The second has trainin_params = 'fsw_initialization', which makes an initial prediction for L that can be used
	        as a starting point estimate for fsw, one of the optimization variables. To get the delta_i initial guess, can use 
	        the predicted quantities along with additional calculations.
	
        Capacitors: 
        	a) Main page parameters: Inside main_script.py -> ML_model_training() -> train_all() -> capacitor_cleaning().
	        Inside here, note only 5 Class II temperature coefficients are included. Note that cap. @ 0Vdc is one of the inputs.
	        After Pareto optimizing, calls to reg_score_and_dump_cat() w/ training_params == 'cap_main_page_params'.
	 
	        b) Initialization: Right below the main page parameters call to reg_score_and_dump(), go to the line that is
	        a call to reg_score_and_dump_cat() w/ training_params == 'cap_area_initialization'.
        
		c) Cap.@0Vdc: This code is found right after the code sending the main page info for training via reg_score_and_dump_cat(). 
  		Makes another call to reg_score_and_dump_cat() with training_params == 'cap_pdf_params', using
	        a separate dataset with all the capacitance, delta C, and voltage info.
	 
        At this point, all ML models needed by the tool have been trained.

    5. Run the optimization tool (__main__ --> optimize_converter() ):
        Description: This section has the code for the build of the optimization tool. The tool takes information about the design and
        the desired topology and determines optimal parameters according to the optimization algorithm. In main_script.py,
        this code is found in optimize_converter(), but it can also be useful to go straight to __main__ of
        fet_optimization_chained_wCaps.py to run through specific cases for the tool.
        Note that in optimize_converter() there is code for plotting the optimized parameters, which can be a useful visualization.
        Inside optimize_converter(), the user can specify dictionary of design parameters. This 
        dictionary can be adjusted for different test cases, and as part of this, the user specifies their desired
        topology. At the end of the ReadMe is more detail on how to run the tool for a user who simply wishes to use the frontend tool,
	rather than change the backend code.
	
        loss_comparison_plotting() is the main function, which takes the dictionary param_dict and runs the optimization. See 'INSTRUCTIONS
	FOR FRONTEND CODE' for the details on what separate template functions must be declared by the designer prior to using the tool.

	4 major important functions:
 
        1) OptimizerInit.get_params(): In this function, set the initially known design variables for the specific 
        design example and topology (e.g. Vin, Vout, Iout, Vdss, etc.)
	
        2) OptimizerInit.set_optimization_variables(): Set what component attribute each of the optimization 
        variables corresponds to, and set topology-specific initialization equations based off these variables. e.g.:
        self.fet1.Rds = x[0], self.fet2.Rds = x[1]. And then e.g. self.cap1.Capacitance = ..., self.ind1.deltaB = ...
	
        3) OptimizerInit.create_component_lists(): Create objects for each component the user wants in their design, 
        and add to an overall list of each component type. The user must index their components corresponding to their 
        topology block diagram. e.g. self.fet1 = OptimizerFet(param_dict, 0), self.fet2 = OptimizerFet(param_dict, 1).
        And, self.fet_list.extend([self.fet1, self.fet2]), etc.
	
        4) OptimizerInit.power_pred_tot(): Compute all physics-based loss modeling parameters, and define the power
        loss function for the topology given the created component objects. e.g. self.fet2.Compute_Cdsq(), then self.fet1.Rds_loss = ,
        then Q1_loss = self.fet1.Rds_loss + self.fet1.Qg_loss, then self.power_tot = self.Q1_loss + self.Q2_loss.
    
        Once all of these are set up, the tool starts with the OptimizerInit object optimizer_obj. From here, the tool runs through various fet 
        technologies (Si, GaN, SiC) cases based on what is in optimizer_obj.tech_list and assumes N-channel FETs (this
        can be changed). Makes a call to optimizer_obj.create_component_lists(), which has three separate object possibilities: OptimizerFet for fets,
        OptimizerInd for inductors, and OptimizerCap for capacitors. For the buck converter example, 5 components are considered:
        Q1 and Q2, the inductor, and the input and output capacitors. 

 	Important functions and objects, as seen in order of appearance when the tool is run:
	
        i) optimizer_obj.plot_range_list: Set this if you want to check cases for various plotting constraint values, which is a common occurence such as getting the 
        optimized power loss values at multiple cost constraints.
	
	ii) optimizer_obj stays the overall object of use, but some of the object_obj attributes are reset with each iteration of
        plotting variable value and fet tech, and the results of each run are stored in lists such as 
        optimizer_obj.MOSFET_overall_points_list. The entire optimizer_obj for each specific cost and area and FET tech 
        constraint are pickled, e.g. the objective function values could be 'optimizer_test_values_MOSFET_overall_points'.
        The optimization varaibles and fet, ind, and cap objects are stored on optimizer_obj. 
	
        iii) optimizer_obj.minimize_fcn(): The main optimization algorithm-specific function. Inside minimize_fcn(), the process starts with
        con_cobyla, which is a dictionary of the constraints and bounds needed for the selected optimization algorithm (currently the
        COBYLA algorithm). Various other constrained, bounded, multi-variate optimization algorithms exist, and would follow a similar structure, but 
        each has slightly different structure requirements. 
	
        iv) obj.init_fet(), obj.init_ind(), and obj.init_cap(): Initialize all variables using these functions. 
        These functions use the pre-trained initialization models for all components, based on known quantities at the
        start of the optimization, and generate initial starting values for all of the optimization variables.
	
        v) self.x0 (a list of all the optimization variables):  Set the optimization variables in the desired format of COBYLA. Note they need to match
        the actual object attributes that the functions they are used by would expect, e.g. proper order of magnitude.
	
        vi) obj.predict_fet(), obj.predict_ind(), and obj.predict_cap(): Next, use these functions to predict all component parameters based on the initialized optimization variables and other known quantities
        about the design. These functions use the pre-trained models trained on component parameters.
	
        vii) min(): Next, the minimize function is used from the scipy.optimize package (imported at the top of the script). See scipy.optimize
        documentation for more information on the arguments. Here is where method='COBYLA' is specified, and the
        constraints=con_cobyla are set, in addition to other parameters that can contribute to successful convergence.
        The first argument is the function to be minimized, and the second argument is the starting values of all
        optimization quantities. The first argument would need be adjusted if the designer wants to minimize e.g. cost instead
        of power loss. con_cobyla dictionary would then have to be adjusted to match any desired constraints.

	viii) First the algorithm checks that the constraints are met, and then computes the objective function. 
 
 	ix) Three metrics considered by the tool to be the constraints and/or objective function: Cost, Area, Power loss.
  
	        1) cost function: self.cost_pred_tot(). The cost function runs
	        through all components and sums their cost, returning the total.
		
	        2) area function: self.area_pred_tot(). The area function
	        runs through all components and sums their area, returning the total.
		
	        3) power loss function: self.power_pred_tot(). Makes predictions based on the latest updated values of the optimization
	        variables, for each of the components. Then computes all physics-based loss-related quantities, e.g. Cdsq and
	        Qrr, trr. Once all quantities have been generated, everything is needed for power loss computation. The function goes through
	        each component and computes each specific loss contribution, then sums all contributions for each component, then
	        sums all components, returning the total power loss. It is desirable to break down the loss contributions within
	        each component for viewing and analyzing the individual contributions after running the example, so that the entire
	        optimization need not be run again.
        
        x) rescobyla object: When the algorithm converges, it terminates and the results are stored on this object. The status
        number will indicate whether or not the convergence was successful or not (see the algorithm documentation for 
        specifics on what all the different status numbers mean). optimizer_obj.status == 1 indicates a success. 
	After converging, returns back to the
        loss_comparison_plotting() function. The results are put onto lists for total power loss, cost, and area, and
        the entire optimizer_obj object is pickled (to view results at any time). 
	
 	xi) obj.get_visualization_params(): Following the pickling code is the code for plotting the results and making 
	the graph look visually appealing, and these visualization parameters can be changed in obj.get_visualization_params().

    6. Determine real components via exhaustive search on filtered databases (__main__ --> component_prediction() ):
        Description: The final step is to select the actual components from the database of real, commercially available components. 
	The parameters used by the optimization algorithm
        are used to filter the database associated with each component in the design, to find components with as good or
        better parameters, and then perform an exhaustive search to find optimal component combinations given the complete equations. 

        From main_script.py, component_predictions() --> predict_components() (will enter component_selection.py). First, the saved
        optimization data for the design is unpickled and the object optimizer_obj containing all the information is used here. 
	The function uses this object inside make_component_predictions_bf().
        Then there are a variety of functions that can be implemented, as follows:
	
        1) optimization_case.filter_components(): Create predictor objects (fet_predictor, ind_predictor, cap_predictor). 
	
        2) case.database_df_filter(): Using df.normalize_x(), score each component. A normalized scoring method is used to 
        take each parameter and see how close or far below it is
        from what the tool determined to be optimal. Then these scores are summed, and the top n components are selected.
        The value of n determines the total run-time for this step, where higher n results in better power loss performance,
        but takes more time to run. n=10 is currently selected for this step. Note that the values of component attributes
        are considered only with respect to what the ML-based step determined to be optimal. For inductors, however, the loss-related quantities
        are computed up-front, and used to determine which is best. This function returns the database as a df of the top
        n components.
	
        3) case.compute_loss_equations(): This objective of this function is to turn each object from the dataframe into a list easily used 
        for matrix reduction of optimal combinations. Because we cannot compute all power loss equations for all 
        components prior to knowing the combinations, we are only setting up the lists here for some components (transistors and capacitors).
        Inductors are slightly different--the functions compute_Rac() and compute_IGSE() are called, and then the database is filtered, and 
        compute_loss_equations_final() turns the list into a list for use in the matrix calculations.
	
        4) case.optimize_combinations(): Performed after filtering the components using the above function. This functions creates meshgrids of all of the
        component databases, and 
		1. sums the costs and areas and check that they meet constraints, 
  		2. checks that the output capacitance meets the requirement, 
		3. computes total power loss with component_combo.compute_total_power(), 
        	4. sorts the power losses from lowest to highest, and prints all the manufacturer part numbers in order. Note that
        	specific components can be listed by mfr part no. in lists 'unavailable_fets, unavailable_inds, etc.', so that if
        	a specific component is not available (usually inductors), it can be skipped over.

        predict_components(): Once components have been selected, they can be added to the cases in predict_components().
	There are cases shown as examples for various FET technologies (Si, GaN) and design parameters (cost constraint, and how many components num_comps 
        were considered when filtering the databases). Manually fill in the list with the selected mfr part nos of the selected components, then enter
        into compare_parameters() to continue the analysis. 
	First, filters the database to select that specific mfr part no., then the function prints all the
        parameters of the selected components. 
	Second, call optimize_combinations() (note the difference in cases when the function sees there's only one component rather than considering combinations
 	of components), and this function prints the loss and cost
        breakdown given those selected components. In the following block of code, the parameters are printed of each component based on what the
        tool returned as the optimized/theoretical parameters, and then the next code block prints all of the theoretical losses broken down by individual contribution.

        5) optimizer_fsw(). This takes the selected components and runs through a generated list of potential
        switching frequencies, and returns the frequency that gives the lowest power loss given the selected components.

INSTRUCTIONS FOR FRONTEND CODE: separate templates walkthrough

	Running the tool: The code has been structured such that the user does not need to enter the tool codebase or adjust the functions 
	    of the tool itself in order to run their design. In order to create a new design, the user must go to 
	    optimization_tool_case_statements.py and go to __main__() and then do the following:
	    1. Add desired info to param_dict showing what kind of design runs to do.
	    2. Update the following functions, adding case statements for the desired topology and operating conditions:
	        i. get_params_separate()
	        ii. set_optimization_variables_separate()
	        iii. create_component_lists_separate()
	        iv. power_pred_tot_separate()
	        Additional functions that can be updated are:
	        i. print_unnormalized_results()
	    Descriptions of each function are included in the code as well as in step 5. of this README doc. Once these functions have been updated, the user runs __main__()
	    inside optimization_tool_case_statements.py.

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
	
	    Once case statements are included for these functions, there are two major processes that can be run:
	    
	    1) predict_components(): The user updates this function with their design 
	    information, then the script will pull the optimized parameters from the ML-based step, and use these to return lists of 
	    real, commercially available components after running the exhaustive search with a set n choices per component database 
	    (the major function to note here is make_component_predictions_bf()). Include predict_components() in __main__() of 
	    component_selection_case_statements.py and run __main__().
	    
	    2) compare_combinations(): This process can be used to compare the selected component combination with the theoretical values from the ML-based
	    step. To run this process, the compare_combinations() function also needs dict_index and set_combo_dict to be updated,
	    in order to specify which design they are running, and which components they intend to use, respectively.
	    
	    To run, the user goes to __main__() inside component_selection_case_statements.py and runs the script.
