import unittest
import os
import numpy as np
import IDSimPy.analysis as ia


class TestTrajectory(unittest.TestCase):

	@classmethod
	def setUpClass(cls):
		data_base_path = os.path.join('test', 'analysis', 'data')
		hdf_v2_path = os.path.join(data_base_path, 'trajectory_v2')
		hdf_v3_path = os.path.join(data_base_path, 'trajectory_v3')
		cls.legacy_hdf5_bare_fname = os.path.join(data_base_path, 'QIT_test_trajectory.hd5')
		cls.legacy_hdf5_aux_fname = os.path.join(data_base_path, 'QIT_test_trajectory_aux.hd5')
		cls.legacy_hdf5_reactive_fn_a = os.path.join(hdf_v2_path, 'qitSim_2019_04_scanningTrapTest',
		                                             'qitSim_2019_04_10_002_trajectories.hd5')
		cls.legacy_hdf5_reactive_fn_b = os.path.join(hdf_v2_path, 'qitSim_2019_04_scanningTrapTest',
		                                             'qitSim_2019_04_15_001_trajectories.hd5')

		cls.hdf5_v2_variable_fname = os.path.join(hdf_v2_path, 'qitSim_2019_07_variableTrajectoryQIT',
		                                           'qitSim_2019_07_22_001_trajectories.hd5')

		cls.hdf5_v2_static_fname = os.path.join(hdf_v2_path, 'qitSim_2019_07_variableTrajectoryQIT',
		                                         'qitSim_2019_07_22_002_trajectories.hd5')

		cls.hdf5_v3_variable_fname = os.path.join(hdf_v3_path, 'qitSim_2019_07_variableTrajectoryQIT',
		                                           'qitSim_2019_07_22_001_trajectories.hd5')

		cls.hdf5_v3_static_fname = os.path.join(hdf_v3_path, 'qitSim_2019_07_variableTrajectoryQIT',
		                                         'qitSim_2019_07_22_002_trajectories.hd5')

		cls.test_json_fname = os.path.join(data_base_path, 'test_trajectories.json')
		cls.result_path = os.path.join('test', 'test_results')

	@classmethod
	def generate_test_trajectory(cls, n_ions, n_steps, static=True):
		times = np.linspace(0, 5, n_steps)
		additional_attribute_names_float = ('param1', 'param2', 'param3')
		additional_attribute_names_int = ('chemical id',)

		x_pos = np.arange(0, n_ions)

		if static:
			pos = np.zeros((n_ions, 3, n_steps))
			additional_attributes_float = np.zeros((n_ions, 3, n_steps))
			additional_attributes_int = np.zeros((n_ions, 1, n_steps), dtype=int)
		else:
			pos = [np.zeros((n_ions + i, 3)) for i in range(n_steps)]
			additional_attributes_float = [np.zeros((n_ions + i, 3)) for i in range(n_steps)]
			additional_attributes_int = [np.zeros((n_ions + i, 1), dtype=int) for i in range(n_steps)]

		for ts in range(n_steps):
			add_frame_float = np.zeros((n_ions, 3))
			add_frame_int = np.zeros((n_ions, 1), dtype=int)
			add_frame_int[-(ts+1):, 0] = 1
			add_frame_float[:, :2] = ts

			if static:
				additional_attributes_float[:, :, ts] = add_frame_float
				additional_attributes_int[:, :, ts] = add_frame_int
				pos[:, 0, ts] = x_pos
				pos[:, 1, ts] = ts * 0.1
			else:
				additional_attributes_float[ts][:n_ions, :] = add_frame_float
				additional_attributes_int[ts][:n_ions, :] = add_frame_int
				pos[ts][:n_ions, 0] = x_pos
				pos[ts][:n_ions, 1] = ts * 0.1

		particle_attributes = ia.ParticleAttributes(
			additional_attribute_names_float, additional_attributes_float,
			additional_attribute_names_int, additional_attributes_int)

		result = ia.Trajectory(
			positions=pos,
			particle_attributes=particle_attributes,
			times=times
		)

		return result

	#  --------------- test Trajectory basics ---------------

	def test_particle_attribute_class(self):
		float_names = ("X", "Y", "Z")
		int_names = ("A", "B")

		n_ions = 5
		n_steps = 3

		float_dat = np.zeros((n_ions, 3, n_steps))
		float_dat[1, 2, :] = np.arange(n_steps) * 0.1

		int_dat = np.zeros((n_ions, 2, n_steps))
		int_dat[2, 0, :] = np.arange(n_steps)

		int_dat_non_static = [np.zeros((n_ions, 2)) for i in range(n_steps)]

		with self.assertRaises(ValueError):
			ia.ParticleAttributes(float_names, float_dat, int_names, int_dat_non_static)

		p_attribs = ia.ParticleAttributes(float_names, float_dat, int_names, int_dat)

		self.assertEqual(p_attribs.attr_name_map['Z'], (True, 2))
		self.assertEqual(p_attribs.attr_name_map['B'], (False, 1))

		self.assertEqual(p_attribs.get('Z', 2)[1], 0.2)
		self.assertEqual(p_attribs.get('A', 1)[2], 1)

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
		                           particle_attributes=ia.ParticleAttributes(attribute_names, attributes_static))
		self.assertEqual(tra_static.is_static_trajectory, True)
		with self.assertRaises(ValueError):
			ia.Trajectory(positions=pos_static, times=times_wrong)

		pos_variable = [np.zeros((i+1, 3)) for i in range(n_timesteps)]
		pos_variable[3][:, 1] = 2.0
		attributes_variable = [np.zeros((i + 1, 2)) for i in range(n_timesteps)]
		attributes_variable[1][1, :] = (20, 350)

		tra_variable = ia.Trajectory(positions=pos_variable, times=times,
		                             particle_attributes=ia.ParticleAttributes(attribute_names, attributes_variable))
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

		np.testing.assert_almost_equal(tra_static.particle_attributes.get_attribs_for_particle(3, 2), (10, 300))
		np.testing.assert_almost_equal(tra_variable.particle_attributes.get_attribs_for_particle(1, 1), (20, 350))

		particle = tra_static.get_particle(3, 2)
		np.testing.assert_almost_equal(particle[0], (5.0, 6.0, 7.0))
		np.testing.assert_almost_equal(particle[1], (10, 300))

	#  --------------- test Trajectory reading from files ---------------

	def test_hdf5_v3_trajectory_reading_variable_timesteps(self):
		tra = ia.read_hdf5_trajectory_file(self.hdf5_v3_variable_fname)

		self.assertEqual(tra.file_version_id, 3)
		self.assertEqual(tra.is_static_trajectory, False)
		self.assertEqual(np.shape(tra[3]), (144, 3))
		self.assertAlmostEqual(tra[3][140, 2], 0.0027527006)

		# test reading of particle attributes:
		self.assertEqual(tra.particle_attributes.attribute_names,
		                 ['velocity x', 'velocity y', 'velocity z',
		                  'rf x', 'rf y', 'rf z',
		                  'spacecharge x', 'spacecharge y', 'spacecharge z',
		                  'global index'])

		particle_dat = tra.particle_attributes.get_attribs_for_particle(2, 10)
		np.testing.assert_allclose(particle_dat,
		                           [17.736734, 23.286898, -2014.0177,
		                            2.7090779E-17, 3.572873E-17, -7.157211E-16,
		                            1.0849945E-21, -2.89312E-21, -7.8152134E-20,
		                            2])

		self.assertEqual(type(particle_dat[8]), float)
		self.assertEqual(type(particle_dat[9]), int)

		global_index = tra.particle_attributes.get('global index', 5)
		self.assertEqual(global_index[0], 0)
		self.assertEqual(global_index[10], 10)
		self.assertEqual(len(global_index), 214)

		# test reading empty frame:
		velo_attribute = tra.particle_attributes.get('velocity x')
		self.assertEqual(np.shape(tra[0]), (0, 3))
		self.assertEqual(np.shape(velo_attribute[0]), (0,))
		self.assertEqual(np.shape(velo_attribute[1]), (48,))



		# test reading of start / splat data:
		ss_data = tra.start_splat_data
		np.testing.assert_allclose(ss_data.start_positions[9, :], [-2.8438022E-4, -1.1745411E-4, 0.0025111847])
		np.testing.assert_allclose(ss_data.splat_positions[9, :], [0.0, 0.0, 0.0])
		self.assertEqual(ss_data.splat_states[9, 0], 1)

		np.testing.assert_allclose(ss_data.splat_positions[35, :], [-9.886785E-5, -3.7777616E-5, 0.0049760267])
		self.assertEqual(ss_data.splat_states[35, 0], 2)

	def test_hdf5_v2_trajectory_reading_variable_timesteps(self):
		tra = ia.read_hdf5_trajectory_file(self.hdf5_v2_variable_fname)

		self.assertEqual(tra.file_version_id, 2)
		self.assertEqual(tra.is_static_trajectory, False)
		self.assertEqual(np.shape(tra[9]), (512, 3))
		self.assertAlmostEqual(tra[9][500, 2], -0.000135881)

	def test_hdf5_v2_trajectory_reading_static_timesteps(self):
		tra = ia.read_hdf5_trajectory_file(self.hdf5_v2_static_fname)

		self.assertEqual(tra.file_version_id, 2)
		self.assertEqual(tra.is_static_trajectory, True)
		self.assertEqual(tra.n_particles, 1000)
		self.assertEqual(np.shape(tra.positions), (1000, 3, 51))
		self.assertAlmostEqual(tra.positions[983, 0, 9], -0.00146076)

	def test_legacy_hdf5_trajectory_reading(self):
		tra = ia.read_legacy_hdf5_trajectory_file(self.legacy_hdf5_aux_fname)
		self.assertEqual(tra.n_particles, 600)
		self.assertEqual(np.shape(tra.positions), (600, 3, 41))
		self.assertEqual(tra.particle_attributes.number_of_attributes, 9)
		self.assertEqual(tra.particle_attributes.number_of_timesteps, 41)

	def test_basic_json_trajectory_reading(self):
		tra = ia.read_json_trajectory_file(self.test_json_fname)
		self.assertEqual(tra.positions.shape, (2000, 3, 101))
		self.assertEqual(tra.particle_attributes.number_of_timesteps, 101)
		self.assertEqual(tra.particle_attributes.number_of_attributes, 1)
		self.assertEqual(len(tra.optional_attributes[ia.OptionalAttribute.PARTICLE_MASSES]), 2000)

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

	def test_trajectory_selection_with_static_synthetic_trajectory(self):
		n_particles = 20
		n_steps = 10
		tra_static = self.generate_test_trajectory(n_particles, n_steps, static=True)

		static_selector = np.zeros(n_particles)
		static_selector[5:8] = 5.0

		tra_selected_static = ia.select(tra_static, static_selector, 5.0)
		self.assertEqual(tra_selected_static.is_static_trajectory, True)
		self.assertEqual(tra_selected_static.n_particles, 3)
		particle_selected_static = tra_selected_static.get_particle(1, 5)
		np.testing.assert_almost_equal(particle_selected_static[0], (6.0, 0.5, 0.0))
		self.assertEqual(particle_selected_static[1], [5.0, 5.0, 0.0, 0])

		static_non_numeric_selector = np.array(['no' for i in range(n_particles)], dtype=object)
		static_non_numeric_selector[5:8] = 'yes'
		tra_selected_static_non_num = ia.select(tra_static, static_non_numeric_selector, 'yes')
		self.assertEqual(tra_selected_static_non_num.is_static_trajectory, True)
		self.assertEqual(tra_selected_static_non_num.n_particles, 3)
		particle_selected_static_non_num = tra_selected_static_non_num.get_particle(1, 5)
		np.testing.assert_almost_equal(particle_selected_static_non_num[0], (6.0, 0.5, 0.0))
		self.assertEqual(particle_selected_static_non_num[1], [5.0, 5.0, 0.0, 0])

		variable_selector = [np.zeros(n_particles) for i in range(n_steps)]
		variable_selector[6][10:15] = 2.0
		variable_selector[6][5:8] = 5.0

		tra_selected_variable = ia.select(tra_static, variable_selector, 2.0)
		self.assertEqual(tra_selected_variable.is_static_trajectory, False)
		self.assertEqual(len(tra_selected_variable.get_positions(0)), 0)
		self.assertEqual(len(tra_selected_variable.get_positions(6)), 5)

		particle_selected_variable = tra_selected_variable.get_particle(2, 6)
		np.testing.assert_almost_equal(particle_selected_variable[0], (12.0, 0.6, 0.0))
		np.testing.assert_almost_equal(particle_selected_variable[1], (6.0, 6.0, 0.0, 0.0))

	def test_trajectory_selection_with_variable_synthetic_trajectory(self):
		n_particles = 20
		n_steps = 10
		tra_variable = self.generate_test_trajectory(n_particles, n_steps, static=False)

		static_selector = np.zeros(n_particles)

		with self.assertRaises(TypeError):
			ia.select(tra_variable, static_selector, 5.0)

		variable_selector = [
			np.zeros(tra_variable.get_n_particles(i)) for i in range(n_steps)]
		variable_selector[6][10:15] = 2.0
		variable_selector[6][5:8] = 5.0

		tra_selected_variable = ia.select(tra_variable, variable_selector, 2.0)
		self.assertEqual(tra_selected_variable.is_static_trajectory, False)
		self.assertEqual(len(tra_selected_variable.get_positions(0)), 0)
		self.assertEqual(len(tra_selected_variable.get_positions(6)), 5)

		particle_selected_variable = tra_selected_variable.get_particle(2, 6)
		np.testing.assert_almost_equal(particle_selected_variable[0], (12.0, 0.6, 0.0))
		np.testing.assert_almost_equal(particle_selected_variable[1], (6.0, 6.0, 0.0, 0.0))

#  --------------- test Trajectory analysis ---------------

	def test_center_of_charge_calculation_with_static_trajectory(self):
		tra = ia.read_hdf5_trajectory_file(self.hdf5_v2_static_fname)

		coc = ia.center_of_charge(tra)
		self.assertEqual(coc.shape[0], tra.n_timesteps)
		np.testing.assert_almost_equal(coc[3, :], (1.6910134945e-05, 3.038242994e-06, 3.465348754e-07))

		p_pos_frame = np.array(
			((-10, -10, -10),
			 ( 10,  10,  10),
			 (  0,   0,   0),
			 (-10,  10,  10),
			 ( 10, -10, -10)))

		p_pos = np.dstack((p_pos_frame, p_pos_frame))
		times = np.array((1.0, 2.0))

		synth_tra_static = ia.Trajectory(
			positions=p_pos,
			times=times
		)

		coc_synth_tra = ia.center_of_charge(synth_tra_static)

		self.assertEqual(synth_tra_static.is_static_trajectory, True)
		np.testing.assert_almost_equal(coc_synth_tra[0], (0.0, 0.0, 0.0))

		charge_weights = np.array((0, 10, 1, 0, 0))
		synth_tra_static.optional_attributes = {ia.OptionalAttribute.PARTICLE_CHARGES: charge_weights}

		coc_synth_tra_weighted = ia.center_of_charge(synth_tra_static)
		np.testing.assert_almost_equal(coc_synth_tra_weighted[0], (9.0909091, 9.0909091, 9.0909091))

	def test_center_of_charge_calculation_with_variable_trajectory(self):
		p_pos_1 = np.array(
			((-10, -10, -10),
			 (  0,   0,   0)))

		p_pos_2 = np.array(
			((-10, -10, -10),
			 (  0,   0,   0),
			 ( 10,  10,  10)))

		p_pos = [p_pos_1, p_pos_2]
		times = np.array((1.0, 2.0))

		synth_tra_variable = ia.Trajectory(
			positions=p_pos,
			times=times
		)

		coc_synth_tra = ia.center_of_charge(synth_tra_variable)

		self.assertEqual(synth_tra_variable.is_static_trajectory, False)
		np.testing.assert_almost_equal(coc_synth_tra[0], (-5.0, -5.0, -5.0))
		np.testing.assert_almost_equal(coc_synth_tra[1], (0.0, 0.0, 0.0))

	#  --------------- test Trajectory export / writing ---------------

	def test_static_trajectory_legacy_vtk_export(self):
		tra = ia.read_hdf5_trajectory_file(self.hdf5_v2_static_fname)
		vtk_export_path = os.path.join(self.result_path, 'vtk_export')
		if not os.path.exists(vtk_export_path):
			os.makedirs(vtk_export_path)
		ia.export_trajectory_to_vtk(tra, os.path.join(self.result_path, 'vtk_export', 'static_test'))



