import unittest
import os
import numpy as np
import IDSimF_analysis as ia


class TestVisualization(unittest.TestCase):

	@classmethod
	def setUpClass(cls):
		cls.test_hdf5_bare_fname = os.path.join('data', 'QIT_test_trajectory.hd5')
		cls.test_hdf5_aux_fname = os.path.join('data', 'QIT_test_trajectory_aux.hd5')
		cls.test_json_fname = os.path.join('data', 'test_trajectories.json')
		cls.result_path = "test_results"

	def test_basic_hdf5_trajectory_reading(self):

		tra = ia.read_hdf5_trajectory_file(self.test_hdf5_aux_fname)
		self.assertEqual(np.shape(tra['positions']), (600,3,41))
		self.assertEqual(np.shape(tra['additional_parameters']), (600, 9, 41))


	def test_basic_json_trajectory_reading(self):
		tra = ia.read_json_trajectory_file(self.test_json_fname)
		print(np.shape(tra['positions']))
		print(np.shape(tra['additional_parameters']))
		print(np.shape(tra['masses']))
