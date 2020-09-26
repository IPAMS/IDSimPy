import unittest
import os
import numpy as np
import matplotlib.pyplot as plt
import IDSimPy.analysis.trajectory as tra
import IDSimPy.analysis.visualization as vis

class TestVisualization_images(unittest.TestCase):

	@classmethod
	def setUpClass(cls):
		data_base_path = os.path.join('analysis', 'data')
		cls.test_json_trajectory = os.path.join(data_base_path, 'test_trajectories.json')
		cls.test_json_projectName = os.path.join(data_base_path, 'test')
		cls.test_reactive_projectName = os.path.join(data_base_path, 'qitSim_2019_04_scanningTrapTest',
		                                             'qitSim_2019_04_15_001')
		cls.test_hdf5_trajectory_a = os.path.join(data_base_path, 'qitSim_2019_04_scanningTrapTest',
		                                          'qitSim_2019_04_10_001_trajectories.hd5')
		cls.test_hdf5_trajectory_b = os.path.join(data_base_path, 'qitSim_2019_04_scanningTrapTest',
		                                          'qitSim_2019_04_10_002_trajectories.hd5')
		cls.test_hdf5_trajectory_c = os.path.join(data_base_path, 'qitSim_2019_04_scanningTrapTest',
		                                          'qitSim_2019_04_15_001_trajectories.hd5')
		cls.result_path = "test_results"

	def test_basic_density_plotting(self):
		time_step_index = 1
		traj_json = tra.read_json_trajectory_file(self.test_json_trajectory)
		vis.plot_density_xz(traj_json, time_step_index)
		plt.title("test title")
		plt.xlabel("x label test")
		result_name = os.path.join(self.result_path, 'test_density_plotting_01.png')
		plt.savefig(result_name)

		vis.plot_density_xz(traj_json, time_step_index, xedges=200, zedges=150)
		plt.title("test title")
		plt.xlabel("x label test")
		result_name = os.path.join(self.result_path, 'test_density_plotting_02.png')
		plt.savefig(result_name)

		traj_hdf5 = tra.read_legacy_hdf5_trajectory_file(self.test_hdf5_trajectory_a)
		vis.plot_density_xz(traj_hdf5, time_step_index,
		                    xedges=np.linspace(-0.01, 0.05, 300),
		                    zedges=np.linspace(-0.03, 0.03, 50),
		                    figsize=(10,5),
		                    axis_equal=False)

		plt.title("test title 2")
		plt.xlabel("x label")
		result_name = os.path.join(self.result_path, 'test_density_plotting_03.png')
		plt.savefig(result_name)

	def test_particle_path_plotting(self):
		traj_json = tra.read_json_trajectory_file(self.test_json_trajectory)
		result_name = os.path.join(self.result_path, 'test_particle_plotting_01')

		dat = [
			(traj_json, (1, 2, 3, 4), "multiple particles"),
			(traj_json, 5, "single particle"),
		]

		vis.plot_particle_traces(result_name, dat)
