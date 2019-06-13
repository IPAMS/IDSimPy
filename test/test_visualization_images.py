import unittest
import os
import numpy as np
import matplotlib.pyplot as plt
import IDSimF_analysis.trajectory as tra
import IDSimF_analysis.visualization as vis

class TestVisualization_images(unittest.TestCase):

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
		traj_json = tra.read_json_trajectory_file(self.test_json_trajectory)
		vis.plot_density_xz(traj_json['positions'], t_indd)
		plt.title("test title")
		plt.xlabel("x label test")
		resultName = os.path.join(self.result_path, 'test_density_plotting_01.png')
		plt.savefig(resultName)

		vis.plot_density_xz(traj_json['positions'], t_indd, xedges = 200, zedges = 150)
		plt.title("test title")
		plt.xlabel("x label test")
		resultName = os.path.join(self.result_path, 'test_density_plotting_02.png')
		plt.savefig(resultName)


		traj_hdf5 = tra.read_hdf5_trajectory_file(self.test_hdf5_trajectory_a)
		vis.plot_density_xz(traj_hdf5['positions'], t_indd,
		                    xedges=np.linspace(-0.01, 0.05, 300),
		                    zedges=np.linspace(-0.03, 0.03, 50),
		                    figsize=(10,5),
		                    axis_equal=False)

		plt.title("test title 2")
		plt.xlabel("x label")
		resultName = os.path.join(self.result_path, 'test_density_plotting_03.png')
		plt.savefig(resultName)


	def test_particle_path_plotting(self):
		traj_json = tra.read_json_trajectory_file(self.test_json_trajectory)
		resultName = os.path.join(self.result_path, 'test_particle_plotting_01')
		dat = [[traj_json,"json trajectory"]]
		vis.plot_particles_path(dat,resultName,[1,2])
