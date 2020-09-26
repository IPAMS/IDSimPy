import unittest
import os
import numpy as np
import IDSimPy.analysis as ia


class TestTrajectory(unittest.TestCase):

	@classmethod
	def setUpClass(cls):
		data_base_path = os.path.join('analysis', 'data')
		cls.legacy_hdf5_bare_fname = os.path.join(data_base_path, 'QIT_test_trajectory.hd5')
		cls.legacy_hdf5_aux_fname = os.path.join(data_base_path, 'QIT_test_trajectory_aux.hd5')

		cls.new_hdf5_variable_fname = os.path.join(data_base_path, 'qitSim_2019_07_variableTrajectoryQIT',
		                                           'qitSim_2019_07_22_001_trajectories.hd5')

		cls.new_hdf5_static_fname = os.path.join(data_base_path, 'qitSim_2019_07_variableTrajectoryQIT',
		                                         'qitSim_2019_07_22_002_trajectories.hd5')

		cls.legacy_hdf5_reactive_fn_a = os.path.join(data_base_path, 'qitSim_2019_04_scanningTrapTest',
		                                             'qitSim_2019_04_10_002_trajectories.hd5')
		cls.legacy_hdf5_reactive_fn_b = os.path.join(data_base_path, 'qitSim_2019_04_scanningTrapTest',
		                                             'qitSim_2019_04_15_001_trajectories.hd5')

		cls.test_json_fname = os.path.join(data_base_path, 'test_trajectories.json')
		cls.result_path = "test_results"

	@classmethod
	def generate_test_trajectory(cls, n_ions, n_steps, static=True):
		times = np.linspace(0, 5, n_steps)
		additional_attribute_names = ('param1', 'param2', 'param3', 'chemical id')

		x_pos = np.arange(0, n_ions)

		if static:
			pos = np.zeros((n_ions, 3, n_steps))
			additional_attributes = np.zeros((n_ions, 4, n_steps))
		else:
			pos = [np.zeros((n_ions + i, 3)) for i in range(n_steps)]
			additional_attributes = [np.zeros((n_ions + i, 4)) for i in range(n_steps)]

		for ts in range(n_steps):
			add_frame = np.zeros((n_ions, 4))
			add_frame[-(ts+1):, 3] = 1
			add_frame[:, :2] = ts

			if static:
				additional_attributes[:, :, ts] = add_frame
				pos[:, 0, ts] = x_pos
				pos[:, 1, ts] = ts * 0.1
			else:
				additional_attributes[ts][:n_ions, :] = add_frame
				pos[ts][:n_ions, 0] = x_pos
				pos[ts][:n_ions, 1] = ts * 0.1

		result = ia.Trajectory(
			positions= pos,
			particle_attributes=additional_attributes,
			particle_attribute_names=additional_attribute_names,
			times=times
		)

		return result

	#  --------------- test Trajectory basics ---------------

	def test_trajectory_class_basic_instantiation_and_methods(self):
		n_timesteps = 5
		n_particles_static = 4
		times = np.linspace(0, 2, n_timesteps)
		times_wrong = np.linspace(0, 2, n_timesteps+1)

		attribute_names = ('chem_id', 'temp')

		pos_static = np.zeros((n_particles_static, 3, n_timesteps))
		pos_static[:, 0, 1] = 1.0
		pos_static[3, :, 2] = (5.0, 6.0, 7.0)
		attributes_static = np.zeros((n_particles_static, 2, n_timesteps))
		attributes_static[3, :, 2] = (10, 300)

		tra_static = ia.Trajectory(positions=pos_static, times=times,
		                           particle_attributes=attributes_static,
		                           particle_attribute_names=attribute_names)
		self.assertEqual(tra_static.is_static_trajectory, True)
		with self.assertRaises(ValueError):
			ia.Trajectory(positions=pos_static, times=times_wrong)

		pos_variable = [np.zeros((i+1, 3)) for i in range(n_timesteps)]
		pos_variable[3][:, 1] = 2.0
		attributes_variable = [np.zeros((i + 1, 2)) for i in range(n_timesteps)]
		attributes_variable[1][1, :] = (20, 350)

		tra_variable = ia.Trajectory(positions=pos_variable, times=times,
		                             particle_attributes=attributes_variable,
		                             particle_attribute_names=attribute_names)
		self.assertEqual(tra_variable.is_static_trajectory, False)
		with self.assertRaises(ValueError):
			ia.Trajectory(positions=pos_variable, times=times_wrong)

		self.assertEqual(len(tra_static), n_timesteps)
		self.assertEqual(len(tra_variable), n_timesteps)

		self.assertEqual(tra_static.n_particles, 4)
		with self.assertRaises(AttributeError):
			tra_variable.n_particles

		np.testing.assert_almost_equal(tra_static[1][0, :], (1.0, 0.0, 0.0))
		np.testing.assert_almost_equal(tra_static.get_positions(1)[0, :], (1.0, 0.0, 0.0))
		np.testing.assert_almost_equal(tra_variable[3][:, 1], (2.0, 2.0, 2.0, 2.0))
		np.testing.assert_almost_equal(tra_variable.get_positions(3)[:, 1], (2.0, 2.0, 2.0, 2.0))
		np.testing.assert_almost_equal(tra_variable[3][1, :], (0.0, 2.0, 0.0))

		np.testing.assert_almost_equal(tra_static.get_particle_attributes(2)[3, :], (10, 300))
		np.testing.assert_almost_equal(tra_variable.get_particle_attributes(1)[1, :], (20, 350))

		particle = tra_static.get_particle(3, 2)
		np.testing.assert_almost_equal(particle[0], (5.0, 6.0, 7.0))
		np.testing.assert_almost_equal(particle[1], (10, 300))

	#  --------------- test Trajectory reading from files ---------------

	def test_hdf5_trajectory_reading_variable_timesteps(self):
		tra = ia.read_hdf5_trajectory_file(self.new_hdf5_variable_fname)

		self.assertEqual(tra.file_version_id, 2)
		self.assertEqual(tra.is_static_trajectory, False)
		self.assertEqual(np.shape(tra[9]), (512, 3))
		self.assertAlmostEqual(tra[9][500, 2], -0.000135881)

	def test_hdf5_trajectory_reading_static_timesteps(self):
		tra = ia.read_hdf5_trajectory_file(self.new_hdf5_static_fname)

		self.assertEqual(tra.file_version_id, 2)
		self.assertEqual(tra.is_static_trajectory, True)
		self.assertEqual(tra.n_particles, 1000)
		self.assertEqual(np.shape(tra.positions), (1000, 3, 51))
		self.assertEqual(np.shape(tra.particle_attributes), (1000, 9, 51))
		self.assertAlmostEqual(tra.positions[983, 0, 9], -0.00146076)

	def test_legacy_hdf5_trajectory_reading(self):
		tra = ia.read_legacy_hdf5_trajectory_file(self.legacy_hdf5_aux_fname)
		self.assertEqual(tra.n_particles, 600)
		self.assertEqual(np.shape(tra.positions), (600, 3, 41))
		self.assertEqual(np.shape(tra.particle_attributes), (600, 9, 41))

	def test_basic_json_trajectory_reading(self):
		tra = ia.read_json_trajectory_file(self.test_json_fname)
		self.assertEqual(tra.positions.shape, (2000, 3, 101))
		self.assertEqual(tra.particle_attributes.shape, (2000, 1, 101))
		self.assertEqual(len(tra.optional_attributes['masses']), 2000)

	#  --------------- test Trajectory filtering ---------------

	def test_parameter_filter_with_synthetic_trajectory(self):
		tra_static = self.generate_test_trajectory(20, 15, static=True)
		tra_variable = self.generate_test_trajectory(20, 15, static=False)

		tra_filtered_static = ia.filter_attribute(tra_static, 'chemical id', 1)
		tra_filtered_variable = ia.filter_attribute(tra_variable, 'chemical id', 1)

		particle_static = tra_filtered_static.get_particle(2, 5)
		np.testing.assert_almost_equal(particle_static[0], (16.0, 0.5, 0.0))

		particle_variable = tra_filtered_variable.get_particle(1, 7)
		np.testing.assert_almost_equal(particle_variable[0], (13.0, 0.7, 0.0))


#  --------------- test Trajectory export ---------------