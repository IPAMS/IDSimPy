import unittest
import os
import numpy as np
import matplotlib.pylab as plt
import IDSimF_analysis.qitsim_analysis as qa


class TestQitSimAnalysis(unittest.TestCase):

	@classmethod
	def setUpClass(cls):
		cls.sim_name_staticrf = os.path.join('data', 'qitSim_2019_04_scanningTrapTest', 'qitSim_2019_04_10_001')
		cls.sim_name_scanned =  os.path.join('data', 'qitSim_2019_04_scanningTrapTest', 'qitSim_2019_04_10_002')
		cls.result_path = "test_results"

	def test_qit_stability_parameters(self):
		sparams = qa.qit_stability_parameters(100,200,0.5e6)
		self.assertAlmostEqual(sparams['lmco'],86.126419,places=5)

	def test_simple_simulation_readers(self):
		conf = qa.read_QIT_conf(self.sim_name_scanned + '_conf.json')
		self.assertEqual(conf['geometry_mode'],'scaled')
		fft_time,fft_dat = qa.read_FFT_record(self.sim_name_scanned)
		coc_time,coc_pos = qa.read_center_of_charge_record(self.sim_name_scanned)
		ionsinac_time,ionsinac = qa.read_ions_inactive_record(self.sim_name_scanned)

		n_ftsamples = 758
		self.assertEqual(np.shape(fft_time), (n_ftsamples,))
		self.assertEqual(np.shape(fft_dat), (n_ftsamples,1))
		self.assertEqual(np.shape(coc_time), (n_ftsamples,))
		self.assertEqual(np.shape(coc_pos), (n_ftsamples, 3))
		self.assertEqual(np.shape(ionsinac_time), (n_ftsamples,))
		self.assertEqual(np.shape(ionsinac), (n_ftsamples,))

		last_time = 7.57e-05
		self.assertAlmostEqual(fft_time[-1],  last_time)
		self.assertAlmostEqual(fft_dat[500], 94.9953)
		self.assertAlmostEqual(fft_dat[-1], 1002.44)

		self.assertAlmostEqual(coc_time[-1],  last_time)
		self.assertTrue(np.all(coc_pos[-1] == [8.75876e-07, -1.5714e-05, 0.000278084]))

		self.assertAlmostEqual(ionsinac_time[-1], last_time)
		self.assertEqual(ionsinac[568], 188)
		self.assertEqual(ionsinac[-1], 399)

	def test_stability_scan(self):
		n_samples = 758
		dat = qa.read_and_analyze_stability_scan(self.sim_name_scanned)
		self.assertEqual(len(dat), n_samples)

		row = 709
		self.assertAlmostEqual(dat.loc[row]['V_rf'], 474.64, places=2)
		self.assertEqual(dat.loc[row]['inactive_ions'], 328)
		self.assertAlmostEqual(dat.loc[row]['time'], 7.09e-05)
		self.assertEqual(dat.loc[row]['ions_diff'], 5)

	def test_stability_scan_analysis(self):
		# FIXME: test analysis with scanned trap sim
		qa.analyze_stability_scan(self.sim_name_scanned, result_path=self.result_path)

	def test_nonresolved_fft_simulation_analysis(self):
		fft_dat = qa.analyse_FFT_sim(self.sim_name_scanned, result_path=self.result_path)

		n_ftsamples = 758
		n_freqs = 379
		last_time = 7.57e-05
		self.assertEqual(len(fft_dat['freqs']),n_freqs)
		self.assertEqual(len(fft_dat['amplitude']), n_freqs)
		self.assertEqual(len(fft_dat['transient']), n_ftsamples)
		self.assertEqual(len(fft_dat['time']), n_ftsamples)

		self.assertAlmostEqual(fft_dat['time'][-1], last_time)

	def test_massresolved_fft_simulation_analysis(self):
		"""At least the mass resolved fft analysis should not produce an exception and should produce a basic plot"""
		fft_dat = qa.analyse_FFT_sim(self.sim_name_staticrf, result_path=self.result_path)
		n_ftsamples = 1001
		n_freqs = 500
		self.assertEqual(np.shape(fft_dat['amplitude']), (n_freqs,3))
		self.assertEqual(len(fft_dat['transient']), n_ftsamples)


