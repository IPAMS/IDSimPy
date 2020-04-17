import unittest
import os
import numpy as np
import IDSimF_analysis.trajectory as tra
import IDSimF_analysis.visualization as vis


class TestVisualizationAnimations(unittest.TestCase):

	@classmethod
	def setUpClass(cls):
		cls.test_json_trajectory = os.path.join('data', 'test_trajectories.json')
		cls.test_json_projectName = os.path.join('data', 'test')

		cls.test_reactive_projectName = os.path.join(
			'data', 'qitSim_2019_04_scanningTrapTest', 'qitSim_2019_04_15_001')

		cls.legacy_hdf5_trajectory_a = os.path.join(
			'data', 'qitSim_2019_04_scanningTrapTest', 'qitSim_2019_04_10_001_trajectories.hd5')

		cls.legacy_hdf5_trajectory_b = os.path.join(
			'data', 'qitSim_2019_04_scanningTrapTest', 'qitSim_2019_04_10_002_trajectories.hd5')

		cls.legacy_hdf5_trajectory_c = os.path.join(
			'data', 'qitSim_2019_04_scanningTrapTest', 'qitSim_2019_04_15_001_trajectories.hd5')

		cls.new_hdf5_variable_projectName = os.path.join(
			'data', 'qitSim_2019_07_variableTrajectoryQIT', 'qitSim_2019_07_22_001')

		cls.new_hdf5_static_projectName = os.path.join(
			'data', 'qitSim_2019_07_variableTrajectoryQIT',  'qitSim_2019_07_22_002')

		cls.result_path = "test_results"

	def test_scatter_animation_variable_hdf5_trajectory(self):
		result_name = os.path.join(self.result_path, 'hdf5_trajectory_animation_test_1')
		vis.render_scatter_animation(self.new_hdf5_variable_projectName, result_name, interval=1, alpha=0.5)

	def test_scatter_animation_static_hdf5_trajectory(self):
		result_name = os.path.join(self.result_path, 'hdf5_trajectory_animation_test_2')

		cl_param = np.zeros(1000)
		cl_param[500:] = 1

		vis.render_scatter_animation(
			self.new_hdf5_static_projectName, result_name,
			interval=1, alpha=0.5, color_parameter=cl_param)

	def test_scatter_animation_json_trajectory(self):
		result_name = os.path.join(self.result_path, 'scatter_animation_test_1')
		vis.render_scatter_animation(self.test_json_projectName, result_name, file_type='json')

	def test_scatter_animation_hdf5_trajectory(self):
		result_name = os.path.join(self.result_path, 'scatter_animation_test_2')

		self.assertRaises(
			ValueError, vis.render_scatter_animation, self.test_reactive_projectName,
			result_name, n_frames=100, interval=5, file_type='legacy_hdf5')

		vis.render_scatter_animation(
			self.test_reactive_projectName, result_name, interval=5, alpha=0.5,
			color_parameter="velocity x", file_type='legacy_hdf5')

	def test_basic_scatter_animation_low_level(self):
		tra_b = tra.read_legacy_hdf5_trajectory_file(self.legacy_hdf5_trajectory_b)
		anim = vis.animate_scatter_plot(tra_b)
		result_name = os.path.join(self.result_path, 'scatter_animation_test_3.mp4')
		anim.save(result_name, fps=20, extra_args=['-vcodec', 'libx264'])

	def test_complex_scatter_animation_low_level(self):
		tra_b = tra.read_legacy_hdf5_trajectory_file(self.legacy_hdf5_trajectory_b)

		anim = vis.animate_scatter_plot(
			tra_b, xlim=(-0.001, 0.001), ylim=(-0.0015, 0.0015), zlim=(-0.005, 0.005),
			color_parameter="chemical id", alpha=0.4)

		result_name = os.path.join(self.result_path, 'scatter_animation_test_4.mp4')
		anim.save(result_name, fps=20, extra_args=['-vcodec', 'libx264'])

		# test with manual coloring:
		c_id = np.array([0 if i< 200 else 1 for i in range(400)])
		anim = vis.animate_scatter_plot(
			tra_b, xlim=(-0.001, 0.001), ylim=(-0.0015, 0.0015), zlim=(-0.005, 0.005),
			color_parameter=c_id, alpha=0.4)

		result_name = os.path.join(self.result_path, 'scatter_animation_test_5.mp4')
		anim.save(result_name, fps=20, extra_args=['-vcodec', 'libx264'])

		# test with manual coloring and an array:
		c_id = [i for i in range(400)]
		anim = vis.animate_scatter_plot(
			tra_b, xlim=(-0.001, 0.001), ylim=(-0.0015, 0.0015), zlim=(-0.005, 0.005),
			color_parameter=c_id, alpha=0.4)

		result_name = os.path.join(self.result_path, 'scatter_animation_test_6.mp4')
		anim.save(result_name, fps=20, extra_args=['-vcodec', 'libx264'])

	def test_density_animation_low_level(self):
		traj_hdf5 = tra.read_legacy_hdf5_trajectory_file(self.legacy_hdf5_trajectory_a)
		anim = vis.animate_xz_density(
			traj_hdf5['positions'],
			xedges=np.linspace(-0.001, 0.001, 50),
			zedges=np.linspace(-0.001, 0.001, 50),
			figsize=(10, 5))

		result_name = os.path.join(self.result_path, 'density_animation_test_1.mp4')
		anim.save(result_name, fps=20, extra_args=['-vcodec', 'libx264'])

	def test_density_animation(self):
		result_name = os.path.join(self.result_path, 'density_animation_test_2')
		vis.render_xz_density_animation(self.test_reactive_projectName, result_name, file_type='legacy_hdf5')

		result_name = os.path.join(self.result_path, 'density_animation_test_3')
		vis.render_xz_density_animation(
			self.test_reactive_projectName, result_name, xedges=10,
			zedges=np.linspace(-0.004, 0.004, 100),
			axis_equal=False, file_type='legacy_hdf5')

		result_name = os.path.join(self.result_path, 'density_animation_test_4')
		vis.render_xz_density_animation(
			self.new_hdf5_static_projectName, result_name,
			xedges=40, zedges=40, axis_equal=True)

	def test_comparison_density_animation_with_json(self):
		project_names = [self.test_json_projectName, self.test_json_projectName]
		masses = [73, 55]
		result_name = os.path.join(self.result_path, 'animation_test_1')
		vis.render_xz_density_comparison_animation(
			project_names, masses, result_name, n_frames=100, interval=1, s_lim=7,
			select_mode='mass', annotation="", mode="lin", file_type='json')

	def test_comparison_animation_with_custom_limits_with_json(self):
		project_names = [self.test_json_projectName, self.test_json_projectName]
		masses = [73, 55]
		result_name = os.path.join(self.result_path, 'animation_test_2')
		vis.render_xz_density_comparison_animation(
			project_names, masses, result_name, n_frames=100, interval=1,
			select_mode='mass', s_lim=[-1, 5, -1, 1], n_bins=[100, 20],
			annotation="", mode="log", file_type='json')

	def test_low_level_reactive_density_comparison_animation_with_hdf5(self):
		tra_b = tra.read_legacy_hdf5_trajectory_file(self.legacy_hdf5_trajectory_b)
		tra_c = tra.read_legacy_hdf5_trajectory_file(self.legacy_hdf5_trajectory_c)

		self.assertRaises(  # too many frames: n_frames*interval > trajectory length
			ValueError,
			vis.animate_xz_density_comparison_plot, [tra_c, tra_c], [0, 1], 71, 1)

		anim = vis.animate_xz_density_comparison_plot([tra_c, tra_c], [0, 1], 51, 1, s_lim=0.001, select_mode='substance')
		result_name = os.path.join(self.result_path, 'reactive_density_animation_test_1.mp4')
		anim.save(result_name, fps=20, extra_args=['-vcodec', 'libx264'])

		self.assertRaises(  # time vector length differs
			ValueError,
			vis.animate_xz_density_comparison_plot, [tra_b, tra_c], [0, 0], 71, 1)

	def test_reactive_density_comparison_animation(self):
		project_names = [self.test_reactive_projectName, self.test_reactive_projectName]
		substances = [0, 1]
		result_name = os.path.join(self.result_path, 'reactive_density_animation_test_2')
		vis.render_xz_density_comparison_animation(
			project_names, substances, result_name, n_frames=51, interval=1, select_mode='substance',
			s_lim=0.001, annotation="", file_type='legacy_hdf5')
