import unittest
import os
import numpy.testing as np_test
import IDSimPy as md_analysis


class TestQitSimAnalysis(unittest.TestCase):

	@classmethod
	def setUpClass(cls):
		data_base_path = os.path.join('test', 'analysis', 'data')
		cls.md_trajectory_file_gzip = os.path.join(data_base_path, 'position_output_Ar_2_spawn_traj.txt.gz')
		cls.md_trajectory_file_txt = os.path.join(data_base_path, 'position_output_Ar_2_spawn_traj.txt')
		cls.result_path = os.path.join('test', 'test_results')

	def test_md_read_txt(self):
		md_dat = md_analysis.read_md_collisions_trajectory_file(self.md_trajectory_file_txt, 'IDSIMF')
		self.assertEqual(len(md_dat), 10)
		np_test.assert_array_almost_equal(
			md_dat[1][1, :],
			[-1.11633e-09, -1.60196e-09, -1.56016e-09, 2.49931e-09, 1.17182e-15])

	def test_md_read_gzip(self):
		md_dat = md_analysis.read_md_collisions_trajectory_file(self.md_trajectory_file_gzip, 'IDSIMF')
		self.assertEqual(len(md_dat), 82)
		np_test.assert_array_almost_equal(
			md_dat[1][1, :],
			[-1.11633e-09, -1.60196e-09, -1.56016e-09, 2.49931e-09, 1.17182e-15])