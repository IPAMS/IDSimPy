import unittest
import os
import numpy as np
import matplotlib.pyplot as plt
import IDSimF_analysis as ia


class TestVisualization(unittest.TestCase):

	@classmethod
	def setUpClass(cls):
		cls.test_fname = os.path.join('data', 'test_trajectories.json')
		cls.test_projectName = os.path.join('data', 'test')
		cls.result_path = "test_results"

	def test_basic_density_plotting(self):
		t_indd = 1
		traj_10_11 = ia.tra.read_json_trajectory_file(self.test_fname)
		ia.plot_density_z_vs_x(traj_10_11['positions'], t_indd,
			xedges=np.linspace(-1, 5, 500),
		    zedges=np.linspace(-3, 3, 100) )

		plt.title("test title")
		plt.xlabel("x label test")
		plt.show()

	def test_basic_density_animation(self):
		projectNames = [self.test_projectName,self.test_projectName]
		masses = ['all','all']
		resultName = os.path.join(self.result_path, 'animation_test.mp4')
		ia.render_XZ_density_animation(projectNames, masses, resultName, nFrames=100, delay=1, s_lim=7,
		                               annotation="", mode="lin", compressed=False)

	def test_density_animation_with_custom_limits(self):
		projectNames = [self.test_projectName,self.test_projectName]
		masses = ['all','all']
		resultName = os.path.join(self.result_path, 'animation_test_2.mp4')
		ia.render_XZ_density_animation(projectNames, masses, resultName, nFrames=100, delay=1, s_lim=[-1, 5, -1, 1],
		                               n_bins=[100, 20],
		                               annotation="", mode="log", compressed=False)
