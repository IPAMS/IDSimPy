import unittest
import os
import numpy as np
import matplotlib.pyplot as plt
#from matplotlib import animation
import IDSimF_analysis.trajectory as tra
import IDSimF_analysis.visualization as vis


class TestVisualization(unittest.TestCase):

	@classmethod
	def setUpClass(cls):
		cls.test_json_trajectory = os.path.join('data', 'test_trajectories.json')
		cls.test_json_projectName = os.path.join('data', 'test')
		cls.test_reactive_projectName = os.path.join('data', 'qitSim_2019_04_scanningTrapTest',
		                                          'qitSim_2019_04_15_001')
		cls.test_hdf5_trajectory_a = os.path.join('data', 'qitSim_2019_04_scanningTrapTest',
		                                          'qitSim_2019_04_10_001_trajectories.hd5')
		cls.test_hdf5_trajectory_b = os.path.join('data', 'qitSim_2019_04_scanningTrapTest',
		                                          'qitSim_2019_04_10_002_trajectories.hd5')
		cls.test_hdf5_trajectory_c = os.path.join('data', 'qitSim_2019_04_scanningTrapTest',
		                                          'qitSim_2019_04_15_001_trajectories.hd5')
		cls.result_path = "test_results"

	def test_basic_density_plotting(self):
		t_indd = 1
		traj_10_11 = tra.read_json_trajectory_file(self.test_json_trajectory)
		vis.plot_density_z_vs_x(traj_10_11['positions'], t_indd,
			xedges=np.linspace(-1, 5, 500),
		    zedges=np.linspace(-3, 3, 100) )

		plt.title("test title")
		plt.xlabel("x label test")
		#plt.show()

	def test_basic_density_animation_with_json(self):
		projectNames = [self.test_json_projectName, self.test_json_projectName]
		masses = ['all','all']
		resultName = os.path.join(self.result_path, 'animation_test.mp4')
		vis.render_XZ_density_animation(projectNames, masses, resultName, nFrames=100, interval=1, s_lim=7,
		                               select_mode='mass',
		                               annotation="", mode="lin", file_type='json')

	def test_density_animation_with_custom_limits_with_json(self):
		projectNames = [self.test_json_projectName, self.test_json_projectName]
		masses = ['all','all']
		resultName = os.path.join(self.result_path, 'animation_test_2.mp4')
		vis.render_XZ_density_animation(projectNames, masses, resultName, nFrames=100, interval=1,
		                                select_mode='mass',
		                                s_lim=[-1, 5, -1, 1],
		                                n_bins=[100, 20],
		                                annotation="", mode="log", file_type='json')

	def test_low_level_reactive_density_animation_with_hdf5(self):
		tra_b = tra.read_hdf5_trajectory_file(self.test_hdf5_trajectory_b)
		tra_c = tra.read_hdf5_trajectory_file(self.test_hdf5_trajectory_c)

		self.assertRaises( #too many frames: n_frames*interval > trajectory length
			ValueError,
			vis.animate_z_vs_x_density_plot, [tra_c, tra_c], [0, 1], 71, 1)

		anim = vis.animate_z_vs_x_density_plot([tra_c, tra_c], [0, 1], 51, 1, s_lim=0.001, select_mode='substance')
		result_name = os.path.join(self.result_path, 'reactive_density_animation_test_1.mp4')
		anim.save(result_name, fps=20, extra_args=['-vcodec', 'libx264'])

		self.assertRaises( #time vector length differs
			ValueError,
			vis.animate_z_vs_x_density_plot, [tra_b, tra_c], [0, 0], 71, 1)


	def test_reactive_density_animation(self):
		projectNames = [self.test_reactive_projectName, self.test_reactive_projectName]
		substances = [0,1]
		resultName = os.path.join(self.result_path, 'reactive_density_animation_test_2.mp4')
		vis.render_XZ_density_animation(projectNames, substances, resultName, nFrames=51, interval=1,
		                                select_mode='substance',
		                                s_lim=0.001,annotation="", file_type='hdf5')