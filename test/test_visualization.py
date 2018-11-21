import unittest
import os
import numpy as np
import matplotlib.pyplot as plt
import IDSimF_analysis as ia


class TestVisualization(unittest.TestCase):

	@classmethod
	def setUpClass(cls):
		cls.test_fname = os.path.join('data', '2018_10_15_001.json_result.txt_trajectories.json')

	def test_basic_density_plotting(self):
		t_indd = 1
		traj_10_11 = ia.tra.read_trajectory_file(self.test_fname)
		ia.plot_density_z_vs_x(traj_10_11['positions'], t_indd,
			xedges=np.linspace(-1, 5, 500),
		    zedges=np.linspace(-3, 3, 100) )

		plt.title("test title")
		plt.xlabel("x label test")
		plt.show()
