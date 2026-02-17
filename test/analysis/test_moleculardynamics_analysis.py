import unittest
import os
import numpy.testing as np_test
import IDSimPy.analysis.moleculardynamics_analysis as md_analysis
import IDSimPy.analysis.visualization as vis



class TestMDCollisionsSimAnalysis(unittest.TestCase):

	@classmethod
	def setUpClass(cls):
		data_base_path = os.path.join('test', 'analysis', 'data')
		cls.md_trajectory_file_gzip = os.path.join(data_base_path, 'position_output_Ar_2_spawn_traj.txt.gz')
		cls.md_trajectory_file_txt = os.path.join(data_base_path, 'position_output_Ar_2_spawn_traj.txt')
		cls.md_trajectory_file_hdf5 = os.path.join(data_base_path, 'MD_collisions_multiatom_trajectories_N2.h5')
		cls.md_trajectory_file_hdf5_He = os.path.join(data_base_path, 'MD_collisions_multiatom_trajectories_He.h5')
		cls.result_path = os.path.join('test', 'test_results')

	def test_legacy_md_read_txt(self):
		md_dat = md_analysis.read_legacy_md_collisions_trajectory_file(self.md_trajectory_file_txt, 'IDSIMF')
		self.assertEqual(len(md_dat), 10)
		np_test.assert_array_almost_equal(
			md_dat[1][1, :],
			[-1.11633e-09, -1.60196e-09, -1.56016e-09, 2.49931e-09, 1.17182e-15])

	def test_legacy_md_read_gzip(self):
		md_dat = md_analysis.read_legacy_md_collisions_trajectory_file(self.md_trajectory_file_gzip, 'IDSIMF')
		self.assertEqual(len(md_dat), 82)
		np_test.assert_array_almost_equal(
			md_dat[1][1, :],
			[-1.11633e-09, -1.60196e-09, -1.56016e-09, 2.49931e-09, 1.17182e-15])

	def test_hdf_trajectory_read(self):
		md_dat = md_analysis.read_md_collisions_trajectory_file(self.md_trajectory_file_hdf5)

		self.assertEqual(len(md_dat), 5)

		tra1 = md_dat[0]
		self.assertEqual(tra1.name, 'trajectory1')
		self.assertEqual(tra1.n_atoms, [2, 3])
		self.assertEqual(tra1.column_names,
		                 ['time', 'dt', 'mol_a0_pos_x', 'mol_a0_pos_y', 'mol_a0_pos_z', 'mol_a1_pos_x',
		                  'mol_a1_pos_y', 'mol_a1_pos_z', 'bg_a0_pos_x', 'bg_a0_pos_y', 'bg_a0_pos_z',
		                  'bg_a1_pos_x', 'bg_a1_pos_y', 'bg_a1_pos_z', 'bg_a2_pos_x', 'bg_a2_pos_y',
		                  'bg_a2_pos_z'])

		self.assertEqual(tra1.trajectory.shape, (190, 17))
		self.assertAlmostEqual(tra1.trajectory[0, 0], 9.9999998e-18, places=20)
		self.assertAlmostEqual(tra1.trajectory[0, 2], 6.533315e-10, places=15)
		self.assertAlmostEqual(tra1.trajectory[0, 3], -3.2287795e-09, places=13)


	def test_md_trajectory_animation(self):
		vis.render_collision_trajectory_animation(self.md_trajectory_file_hdf5_He, 0, 0, 10)

