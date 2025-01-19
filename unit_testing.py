import unittest as ut
from component_selection import make_component_predictions_bf, predict_components
from fet_optimization_chained_wCaps import loss_comparison_plotting
import pickle
import dill
import time
import __main__
from fet_optimization_chained_wCaps import OptimizerInit
__main__.OptimizerInit = OptimizerInit
from fet_optimization_chained_wCaps import OptimizerFet
__main__.OptimizerFet = OptimizerFet
from fet_optimization_chained_wCaps import OptimizerInductor
__main__.OptimizerInductor = OptimizerInductor
from fet_optimization_chained_wCaps import OptimizerCapacitor
__main__.OptimizerCapacitor = OptimizerCapacitor

# Test cases:
# Test 2 designs (low-cost Si- and GaN-based designed) for buck, boost, and microinverter.
# Test the optimization tool itself, the component selection, and the parameter printing. So 18 unit tests in total.





# Test to make sure the power loss computation stays the same, when given pre-selected components
# Will want to sync this up so that can just directly call predict_components()
class Test_singular_loss(ut.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.set_combo_dict = {'0': {'fet_tech': 'MOSFET', 'cost_constraint': 5.0, 'num_comps': 10, 'topology': 'boost',
                                'Mfr_part_nos_dict': {'fet_mfr_part_nos': ['PXN012-60QLJ', 'PXN012-60QLJ'],
                                                      'ind_mfr_part_nos': ['SRP1265C-8R2M'],
                                                      'cap_mfr_part_nos': ['GRM31CC72A475ME11L', 'CL21A226MAYNNNE']}},
                          '1': {'fet_tech': 'MOSFET', 'cost_constraint': 5.0, 'num_comps': 10, 'topology': 'buck',
                                'Mfr_part_nos_dict': {'fet_mfr_part_nos': ['PXN012-60QLJ', 'PXN012-60QLJ'],
                                                      'ind_mfr_part_nos': ['SRP1265C-8R2M'],
                                                      'cap_mfr_part_nos': ['GRM31CC72A475ME11L', 'CL21A226MAYNNNE']}},
                          '2': {'fet_tech': 'GaNFET', 'cost_constraint': 5.0, 'num_comps': 5, 'topology': 'buck',
                                'Mfr_part_nos_dict': {'fet_mfr_part_nos': ['EPC2052', 'EPC2052'],
                                                      'ind_mfr_part_nos': ['MMD-06CZ-R68M-V1-RU'],
                                                      'cap_mfr_part_nos': ['GRM31CC72A475ME11L', 'CL21A226MAYNNNE']}},
                          '3': {'fet_tech': 'GaNFET', 'cost_constraint': 5.0, 'num_comps': 10, 'topology': 'buck',
                                'Mfr_part_nos_dict': {'fet_mfr_part_nos': ['EPC2052', 'EPC2052'],
                                                      'ind_mfr_part_nos': ['MMD-06CZ-R68M-V1-RU'],
                                                      'cap_mfr_part_nos': ['GRM31CC72A475ME11L', 'CL21A226MAYNNNE']}},
                          '4': {'fet_tech': 'MOSFET', 'cost_constraint': 11.0, 'num_comps': 10, 'topology': 'buck',
                                'Mfr_part_nos_dict': {'fet_mfr_part_nos': ['SIS176LDN-T1-GE3', 'PXN012-60QLJ'],
                                                      'ind_mfr_part_nos': ['AMDLA2213Q-470MT'],
                                                      'cap_mfr_part_nos': ['GRM32EC72A106ME05L',
                                                                           'C3216X5R1E476M160AC']}},

                          '5': {'fet_tech': 'GaNFET', 'cost_constraint': 11.0, 'num_comps': 10, 'topology': 'buck',
                                'Mfr_part_nos_dict': {'fet_mfr_part_nos': ['EPC2204', 'EPC2204'],
                                                      'ind_mfr_part_nos': ['ASPIAIG-Q1010-6R8M-T'],
                                                      'cap_mfr_part_nos': ['GRM32EC72A106ME05L',
                                                                           'C3216X5R1E336M160AC']}},
                              '6': {'fet_tech': 'MOSFET', 'cost_constraint': 18.0, 'area_constraint': 1000,
                                    'num_comps': 10,
                                    'topology': 'microinverter_combined',
                                    'Mfr_part_nos_dict': {
                                        'fet_mfr_part_nos': ['SIDR668DP-T1-GE3', 'PSMN3R9-100YSFX', 'SIR104LDP-T1-RE3'],
                                        'ind_mfr_part_nos': ['SRP1510CA-4R7M'],
                                        'cap_mfr_part_nos': []}},
                          }

    def test_buck_Si(self):
        fet_tech = 'MOSFET'
        cost_constraint = 5.0  # cost constraint must be a float
        area_constraint = 800  # area constraint, could generalize this
        num_comps = 10
        topology = 'buck'

        with open(
                'optimizer_test_values/optimizer_obj_' + topology + '_' + fet_tech + '_' + str(
                    cost_constraint) + '_' + str(
                    area_constraint), 'rb') as optimizer_obj_file:
            # Step 3
            optimizer_obj = pickle.load(optimizer_obj_file)
            # optimizer_obj = OptimizerInit(**optimizer_obj_file.__dict__)

        start_opt = time.time()
        self.optimization_case = make_component_predictions_bf(optimizer_obj, 'pdf_params', cost_constraint, area_constraint,
                                                          num_comps)

        self.optimization_case.practical = True
        self.optimization_case.theoretical = True

        dict_index = '1'
        self.optimization_case.fet_tech = self.set_combo_dict[dict_index]['fet_tech']
        self.optimization_case.cost_constraint = self.set_combo_dict[dict_index]['cost_constraint']
        self.optimization_case.num_comps = self.set_combo_dict[dict_index]['num_comps']
        self.optimization_case.topology = self.set_combo_dict[dict_index]['topology']
        self.optimization_case.Mfr_part_nos_dict = self.set_combo_dict[dict_index]['Mfr_part_nos_dict']

        self.optimization_case.print_all_parameters()
        self.assertTrue(
            (self.optimization_case.opt_obj.power_tot > 2.99) and (self.optimization_case.opt_obj.power_tot < 3.0),
            'Incorrect pre-selected component loss buck Si computation')


    def test_buck_GaN(self):
        fet_tech = 'GaNFET'
        cost_constraint = 5.0  # cost constraint must be a float
        area_constraint = 800  # area constraint, could generalize this
        num_comps = 10
        topology = 'buck'

        with open(
                'optimizer_test_values/optimizer_obj_' + topology + '_' + fet_tech + '_' + str(
                    cost_constraint) + '_' + str(
                    area_constraint), 'rb') as optimizer_obj_file:
            # Step 3
            optimizer_obj = pickle.load(optimizer_obj_file)
            # optimizer_obj = OptimizerInit(**optimizer_obj_file.__dict__)

        start_opt = time.time()
        self.optimization_case = make_component_predictions_bf(optimizer_obj, 'pdf_params', cost_constraint, area_constraint,
                                                          num_comps)

        self.optimization_case.practical = True
        self.optimization_case.theoretical = True

        self.optimization_case.practical = True
        self.optimization_case.theoretical = True

        dict_index = '2'
        self.optimization_case.fet_tech = self.set_combo_dict[dict_index]['fet_tech']
        self.optimization_case.cost_constraint = self.set_combo_dict[dict_index]['cost_constraint']
        self.optimization_case.num_comps = self.set_combo_dict[dict_index]['num_comps']
        self.optimization_case.topology = self.set_combo_dict[dict_index]['topology']
        self.optimization_case.Mfr_part_nos_dict = self.set_combo_dict[dict_index]['Mfr_part_nos_dict']

        self.optimization_case.print_all_parameters()
        self.assertTrue(
            (self.optimization_case.opt_obj.power_tot > 4.16) and (self.optimization_case.opt_obj.power_tot < 4.17),
            'Incorrect pre-selected component loss buck GaN computation')

    def test_microinverter_combined_Si(self):
        fet_tech = 'MOSFET'
        cost_constraint = 18.0  # cost constraint must be a float
        area_constraint = 1000  # area constraint, could generalize this
        num_comps = 10
        topology = 'microinverter_combined'

        with open(
                'optimizer_test_values/optimizer_obj_' + topology + '_' + fet_tech + '_' + str(
                    cost_constraint) + '_' + str(
                    area_constraint), 'rb') as optimizer_obj_file:
            # Step 3
            optimizer_obj = pickle.load(optimizer_obj_file)
            # optimizer_obj = OptimizerInit(**optimizer_obj_file.__dict__)

        start_opt = time.time()
        self.optimization_case = make_component_predictions_bf(optimizer_obj, 'pdf_params', cost_constraint,
                                                               area_constraint,
                                                               num_comps)
        self.optimization_case.opt_obj.fsw = 10

        self.optimization_case.practical = True
        self.optimization_case.theoretical = True

        dict_index = '6'
        self.optimization_case.fet_tech = self.set_combo_dict[dict_index]['fet_tech']
        self.optimization_case.cost_constraint = self.set_combo_dict[dict_index]['cost_constraint']
        self.optimization_case.num_comps = self.set_combo_dict[dict_index]['num_comps']
        self.optimization_case.topology = self.set_combo_dict[dict_index]['topology']
        self.optimization_case.Mfr_part_nos_dict = self.set_combo_dict[dict_index]['Mfr_part_nos_dict']

        # self.optimization_case.compare_parameters()
        self.optimization_case.print_all_parameters()
        self.assertTrue(
            (self.optimization_case.opt_obj.power_tot > 10.47) and (self.optimization_case.opt_obj.power_tot < 10.48),
            'Incorrect pre-selected component loss Si microinverter_combined computation')

    @classmethod
    def tearDownClass(cls) -> None:
        print("tearDownClass completed")



### 6 tests for the exhaustive search combination check ###



# Test to make sure when checking combinations, we get the correct best combo
class Test_component_exhaustive_search(ut.TestCase):
    def test_buck_Si(self):
        fet_tech = 'MOSFET'
        cost_constraint = 5.0  # cost constraint must be a float
        area_constraint = 800  # area constraint, could generalize this
        num_comps = 2
        topology = 'buck'

        with open('optimizer_test_values/optimizer_obj_' + topology + '_' + fet_tech + '_' + str(
                cost_constraint) + '_' + str(area_constraint), 'rb') as optimizer_obj_file:
            # Step 3
            optimizer_obj = dill.load(optimizer_obj_file)
            # optimizer_obj = OptimizerInit(self, **optimizer_obj_file.__dict__)

        self.start_opt = time.time()
        optimizer_obj.fsw = 0.178056115
        self.optimization_case = make_component_predictions_bf(optimizer_obj, 'pdf_params', cost_constraint,
                                                               area_constraint,
                                                               num_comps)

        self.optimization_case.filter_components()
        end_opt = time.time()
        print(f'until optimize combinations: {(end_opt - self.start_opt)}')

        ### PUT THE FOLLOWING LINE BACK IN if more than one given component###
        self.optimization_case.optimize_combinations()

        self.assertTrue(
            (self.optimization_case.Ploss_div_Pout > 0.076) and (self.optimization_case.Ploss_div_Pout < 0.077),
            'Incorrect combo loss computation in buck Si low-cost')
        print('done')


    # Test to make sure when checking combinations, we get the correct best combo
    def test_buck_GaN(self):
        fet_tech = 'GaNFET'
        cost_constraint = 5.0  # cost constraint must be a float
        area_constraint = 800  # area constraint, could generalize this
        num_comps = 2
        topology = 'buck'

        with open('optimizer_test_values/optimizer_obj_' + topology + '_' + fet_tech + '_' + str(
                cost_constraint) + '_' + str(area_constraint), 'rb') as optimizer_obj_file:
            # Step 3
            optimizer_obj = dill.load(optimizer_obj_file)
            # optimizer_obj = OptimizerInit(self, **optimizer_obj_file.__dict__)

        self.start_opt = time.time()
        optimizer_obj.fsw = 2.316067213
        self.optimization_case = make_component_predictions_bf(optimizer_obj, 'pdf_params', cost_constraint,
                                                               area_constraint,
                                                               num_comps)

        self.optimization_case.filter_components()
        end_opt = time.time()
        print(f'until optimize combinations: {(end_opt - self.start_opt)}')

        ### PUT THE FOLLOWING LINE BACK IN if more than one given component###
        self.optimization_case.optimize_combinations()

        self.assertTrue(
            (self.optimization_case.Ploss_div_Pout > 0.085) and (self.optimization_case.Ploss_div_Pout < 0.086),
            'Incorrect combo loss computation in buck GaN low-cost')
        print('done')


    # Test to make sure when checking combinations, we get the correct best combo
    def test_boost_Si(self):
        fet_tech = 'MOSFET'
        cost_constraint = 5.0  # cost constraint must be a float
        area_constraint = 800  # area constraint, could generalize this
        num_comps = 8
        topology = 'boost'

        with open('optimizer_test_values/optimizer_obj_' + topology + '_' + fet_tech + '_' + str(
                cost_constraint) + '_' + str(area_constraint), 'rb') as optimizer_obj_file:
            # Step 3
            optimizer_obj = dill.load(optimizer_obj_file)
            # optimizer_obj = OptimizerInit(self, **optimizer_obj_file.__dict__)

        self.start_opt = time.time()
        optimizer_obj.fsw = 0.6
        self.optimization_case = make_component_predictions_bf(optimizer_obj, 'pdf_params', cost_constraint,
                                                               area_constraint,
                                                               num_comps)

        self.optimization_case.filter_components()
        end_opt = time.time()
        print(f'until optimize combinations: {(end_opt - self.start_opt)}')

        ### PUT THE FOLLOWING LINE BACK IN if more than one given component###
        self.optimization_case.optimize_combinations()

        self.assertTrue(
            (self.optimization_case.Ploss_div_Pout > 0.14) and (self.optimization_case.Ploss_div_Pout < 0.15),
            'Incorrect combo loss computation in boost Si low-cost')
        print('done')


    # Test to make sure when checking combinations, we get the correct best combo
    def test_boost_GaN(self):
        fet_tech = 'GaNFET'
        cost_constraint = 5.0  # cost constraint must be a float
        area_constraint = 800  # area constraint, could generalize this
        num_comps = 8
        topology = 'boost'

        with open('optimizer_test_values/optimizer_obj_' + topology + '_' + fet_tech + '_' + str(
                cost_constraint) + '_' + str(area_constraint), 'rb') as optimizer_obj_file:
            # Step 3
            optimizer_obj = dill.load(optimizer_obj_file)
            # optimizer_obj = OptimizerInit(self, **optimizer_obj_file.__dict__)

        self.start_opt = time.time()
        optimizer_obj.fsw = 0.6
        self.optimization_case = make_component_predictions_bf(optimizer_obj, 'pdf_params', cost_constraint,
                                                               area_constraint,
                                                               num_comps)

        self.optimization_case.filter_components()
        end_opt = time.time()
        print(f'until optimize combinations: {(end_opt - self.start_opt)}')

        ### PUT THE FOLLOWING LINE BACK IN if more than one given component###
        self.optimization_case.optimize_combinations()

        self.assertTrue(
            (self.optimization_case.Ploss_div_Pout > 0.046) and (self.optimization_case.Ploss_div_Pout < 0.047),
            'Incorrect combo loss computation in boost GaN low-cost')
        print('done')


    # Test to make sure when checking combinations, we get the correct best combo
    def test_microinverter_Si(self):
        fet_tech = 'MOSFET'
        cost_constraint = 18.0  # cost constraint must be a float
        area_constraint = 1000  # area constraint, could generalize this
        num_comps = 2
        topology = 'microinverter_combined'

        with open('optimizer_test_values/optimizer_obj_' + topology + '_' + fet_tech + '_' + str(
                cost_constraint) + '_' + str(area_constraint), 'rb') as optimizer_obj_file:
            # Step 3
            optimizer_obj = dill.load(optimizer_obj_file)
            # optimizer_obj = OptimizerInit(self, **optimizer_obj_file.__dict__)

        self.start_opt = time.time()
        optimizer_obj.fsw = 10 # 100kHz
        self.optimization_case = make_component_predictions_bf(optimizer_obj, 'pdf_params', cost_constraint,
                                                               area_constraint,
                                                               num_comps)

        self.optimization_case.filter_components()
        end_opt = time.time()
        print(f'until optimize combinations: {(end_opt - self.start_opt)}')

        ### PUT THE FOLLOWING LINE BACK IN if more than one given component###
        self.optimization_case.optimize_combinations()

        self.assertTrue(
            (self.optimization_case.Ploss_div_Pout > 0.29) and (self.optimization_case.Ploss_div_Pout < 0.30),
            'Incorrect combo loss computation in microinverter_combined Si low-cost')
        print('done')


    # Test to make sure when checking combinations, we get the correct best combo
    def test_microinverter_combined_GaN(self):
        fet_tech = 'GaNFET'
        cost_constraint = 18.0  # cost constraint must be a float
        area_constraint = 1000  # area constraint, could generalize this
        num_comps = 2
        topology = 'microinverter_combined'

        with open('optimizer_test_values/optimizer_obj_' + topology + '_' + fet_tech + '_' + str(
                cost_constraint) + '_' + str(area_constraint), 'rb') as optimizer_obj_file:
            # Step 3
            optimizer_obj = dill.load(optimizer_obj_file)
            # optimizer_obj = OptimizerInit(self, **optimizer_obj_file.__dict__)

        self.start_opt = time.time()
        optimizer_obj.fsw = 10
        self.optimization_case = make_component_predictions_bf(optimizer_obj, 'pdf_params', cost_constraint,
                                                               area_constraint,
                                                               num_comps)

        self.optimization_case.filter_components()
        end_opt = time.time()
        print(f'until optimize combinations: {(end_opt - self.start_opt)}')

        ### PUT THE FOLLOWING LINE BACK IN if more than one given component###
        self.optimization_case.optimize_combinations()

        self.assertTrue(
            (self.optimization_case.Ploss_div_Pout > 0.04) and (self.optimization_case.Ploss_div_Pout < 0.05),
            'Incorrect combo loss computation in microinverter_combined GaN low-cost')
        print('done')

    @classmethod
    def tearDownClass(cls) -> None:
        print("tearDownClass completed")


### 6 tests for testing the optimization tool, all topology cases ###


# Test to make sure the optimization tool returns the expected values
class Test_optimization_tool(ut.TestCase):
    def test_buck_Si(self):
        self.param_dict_buck = {'opt_var': 'power', 'plotting_var': 'cost', 'set_constraint': 'area',
                           'set_constraint_val': 800,
                           'example_num': 1, 'tech_list': ['MOSFET'], 'test_fet_tech': 'MOSFET',
                           'num_points': 2,
                           'plotting_range': [5, 11], 'predict_components': False, 'topology': 'buck'}

        optimizer_obj = loss_comparison_plotting(param_dict=self.param_dict_buck, unit_test=True)
        self.assertTrue((optimizer_obj.fun < 0.058) and (optimizer_obj.fun > 0.0579),
                        'Incorrect optimization loss in buck Si low-cost')

    def test_buck_GaN(self):
        self.param_dict_buck = {'opt_var': 'power', 'plotting_var': 'cost', 'set_constraint': 'area',
                           'set_constraint_val': 800,
                           'example_num': 1, 'tech_list': ['GaNFET'],
                           'num_points': 2,
                           'plotting_range': [5, 11], 'predict_components': False, 'topology': 'buck'}

        optimizer_obj = loss_comparison_plotting(param_dict=self.param_dict_buck, unit_test=True)
        self.assertTrue((optimizer_obj.fun > 0.22) and (optimizer_obj.fun < 0.23),
                        'Incorrect optimization loss in buck GaN low-cost')


    def test_boost_Si(self):
        self.param_dict_boost = {'opt_var': 'power', 'plotting_var': 'cost', 'set_constraint': 'area', 'set_constraint_val': 800,
                       'example_num': 1, 'tech_list': ['MOSFET'], 'num_points': 2,
                       'plotting_range': [5,9], 'predict_components': False, 'topology': 'boost'}

        optimizer_obj = loss_comparison_plotting(param_dict=self.param_dict_boost, unit_test=True)
        self.assertTrue((optimizer_obj.fun < 0.055) and (optimizer_obj.fun > 0.054),
                        'Incorrect optimization loss in boost Si low-cost')


    def test_boost_GaN(self):
        self.param_dict_boost = {'opt_var': 'power', 'plotting_var': 'cost', 'set_constraint': 'area', 'set_constraint_val': 800,
                       'example_num': 1, 'tech_list': ['GaNFET'], 'num_points': 2,
                       'plotting_range': [5,9], 'predict_components': False, 'topology': 'boost'}

        optimizer_obj = loss_comparison_plotting(param_dict=self.param_dict_boost, unit_test=True)
        self.assertTrue((optimizer_obj.fun < 0.38) and (optimizer_obj.fun > 0.37),
                        'Incorrect optimization loss in boost GaN low-cost')

    def test_microinverter_combined_Si(self):
        self.param_dict_microinverter_combined = {'opt_var': 'power', 'plotting_var': 'cost', 'set_constraint': 'area',
                                         'set_constraint_val': 1000,
                                         'example_num': 1, 'tech_list': ['MOSFET'],
                                         'num_points': 2,
                                         'plotting_range': [18, 100], 'predict_components': False,
                                         'topology': 'microinverter_combined'}

        optimizer_obj = loss_comparison_plotting(param_dict=self.param_dict_microinverter_combined, unit_test=True)
        self.assertTrue((optimizer_obj.fun > 0.027) and (optimizer_obj.fun < 0.028),
                        'Incorrect optimization loss in boost Si low-cost')


    def test_microinverter_combined_GaN(self):
        self.param_dict_microinverter_combined = {'opt_var': 'power', 'plotting_var': 'cost', 'set_constraint': 'area',
                                         'set_constraint_val': 1000,
                                         'example_num': 1, 'tech_list': ['GaNFET'],
                                         'num_points': 2,
                                         'plotting_range': [18, 100], 'predict_components': False,
                                         'topology': 'microinverter_combined'}

        optimizer_obj = loss_comparison_plotting(param_dict=self.param_dict_microinverter_combined, unit_test=True)
        self.assertTrue((optimizer_obj.fun > 0.059) and (optimizer_obj.fun < 0.06),
                        'Incorrect optimization loss in boost GaN low-cost')



def SuiteComponentSelection():
    component_selection = ut.TestSuite()
    component_selection.addTest(ut.makeSuite(Test_component_selection))

if __name__ == '__main__':
    # ut.main()
    loader = ut.TestLoader()
    loader.testMethodPrefix = "Test"
    SuiteComponentSelection()