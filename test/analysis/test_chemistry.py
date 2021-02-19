import unittest
import os
import matplotlib.pyplot as plt
import IDSimPy.analysis.chemistry as chem

class TestChemistry(unittest.TestCase):

	@classmethod
	def setUpClass(cls):
		data_base_path = os.path.join('test', 'analysis', 'data')

		cls.test_concentrations_a = os.path.join(data_base_path, 'qitSim_2019_04_scanningTrapTest',
		                                          'qitSim_2019_04_10_001_concentrations.txt')
		cls.test_concentrations_b = os.path.join(data_base_path, 'qitSim_2019_04_scanningTrapTest',
		                                          'qitSim_2019_04_10_002_concentrations.txt')
		cls.result_path = os.path.join('test', 'test_results')

	def test_basic_concentration_plotting(self):
		chem.plot_concentration_file(self.test_concentrations_a)
		plt.savefig(os.path.join(self.result_path,'qitSim_2019_04_10_001_concentrations.pdf'))
		plt.figure()
		chem.plot_concentration_file(self.test_concentrations_b)
		plt.savefig(os.path.join(self.result_path, 'qitSim_2019_04_10_002_concentrations.pdf'))
