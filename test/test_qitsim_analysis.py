import unittest
import os
import numpy as np
import IDSimF_analysis.qitsim_analysis as qa


class TestQitSimAnalysis(unittest.TestCase):

	@classmethod
	def setUpClass(cls):
		cls.sim_name = os.path.join('data', 'qitSim_2019_04_scanningTrapTest','qitSim_2019_04_04_001')
		cls.result_path = "test_results"

	def test_qit_stability_parameters(self):
		sparams = qa.qit_stability_parameters(100,200,0.5e6)
		self.assertAlmostEqual(sparams['lmco'],86.126419,places=5)

	def test_simple_simulation_readers(self):
		conf = qa.read_QIT_conf(self.sim_name+'_conf.json')
		self.assertEqual(conf['geometry_mode'],'scaled')
		fft_time,fft_dat = qa.read_FFT_record(self.sim_name)
		coc_time,coc_pos = qa.read_center_of_charge_record(self.sim_name)
		ionsinac_time,ionsinac = qa.read_ions_inactive_record(self.sim_name)

		self.assertEqual(np.shape(fft_time), (4001,))
		self.assertEqual(np.shape(fft_dat), (4001,))
		self.assertEqual(np.shape(coc_time), (4001,))
		self.assertEqual(np.shape(coc_pos), (4001, 3))
		self.assertEqual(np.shape(ionsinac_time), (4001,))
		self.assertEqual(np.shape(ionsinac), (4001,))

		self.assertAlmostEqual(fft_time[-1], 8e-5)
		self.assertAlmostEqual(fft_dat[2000], -28.8141)
		self.assertAlmostEqual(fft_dat[-1], -618.981)

		self.assertAlmostEqual(coc_time[-1], 8e-5)
		self.assertTrue(np.all(coc_pos[-1] == [2.89441e-05, 1.06989e-05, -0.000842188]))

		self.assertAlmostEqual(ionsinac_time[-1], 8e-5)
		self.assertEqual(ionsinac[1838], 376)
		self.assertEqual(ionsinac[-1], 800)

	def test_nonreactive_stability_scan(self):
		dat = qa.read_and_analyze_stability_scan(self.sim_name)
		self.assertEqual(len(dat), 4001)

		row = 1794
		self.assertAlmostEqual(dat.loc[row]['V_rf'], 334.55)
		self.assertAlmostEqual(dat.loc[row]['inactive_ions'], 175.0)
		self.assertAlmostEqual(dat.loc[row]['time'], 3.588e-05)
		self.assertEqual(dat.loc[row]['ions_diff'], 41)


	#self.assertEqual(np.shape(ions_inactive), (2, 2001))
