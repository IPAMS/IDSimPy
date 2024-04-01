import unittest
import os
import numpy as np
import IDSimPy as idsimpy


class TestIonCloudGeneration(unittest.TestCase):

	@classmethod
	def setUpClass(cls):
		cls.result_path = os.path.join('test', 'test_results')

	def test_cylinder_generation(self):

		cyl_r = 0.5
		cyl_z = 0.3
		cylinder_a = idsimpy.define_cylinder_x_dir(100, cyl_r, cyl_z, 1, 10)
		cylinder_b = idsimpy.define_cylinder_x_dir(100, cyl_r, cyl_z, 2, 100)


		cylinder_c = idsimpy.define_cylinder_z_dir(100, cyl_r, cyl_z, 1, 10)
		cylinder_d = idsimpy.define_cylinder_z_dir(100, cyl_r, cyl_z, 2, 100)

		cloud = np.vstack((cylinder_a, cylinder_b, cylinder_c, cylinder_d))

		result_file = os.path.join(self.result_path, 'ion_cloud_cylinder.csv')
		idsimpy.write_cloud_file(cloud, result_file)

	def test_grid_generation(self):

		cyl_r = 0.5

		grid_35 = idsimpy.define_xy_grid(10, 10, cyl_r, cyl_r, -1e-5, 0, 35)
		grid_37 = idsimpy.define_xy_grid(10, 10, cyl_r, cyl_r,  1e-5, 0, 37)

		cloud = np.vstack((grid_35, grid_37))

		result_file = os.path.join(self.result_path, 'ion_cloud_grid.csv')
		idsimpy.write_cloud_file(cloud, result_file)

	def test_block_generation(self):

		block_100 = idsimpy.define_origin_centered_block(100, 1, 2, 10, 100)
		block_200 = idsimpy.define_origin_centered_block(100, 2, 4, 5,  200)

		cloud = np.vstack((block_100, block_200))

		result_file = os.path.join(self.result_path, 'ion_cloud_block.csv')
		idsimpy.write_cloud_file(cloud, result_file)

	def test_fixed_kinetic_energy_setting(self):

		cloud = idsimpy.define_origin_centered_block(100, 1, 2, 10, 100)
		idsimpy.set_kinetic_energy_in_z_dir(cloud, 10)

		result_file = os.path.join(self.result_path, 'ion_cloud_fixed_ke.csv')
		idsimpy.write_cloud_file(cloud, result_file)

	def test_thermal_kinetic_energy_setting(self):
		cloud = idsimpy.define_origin_centered_block(100, 1, 2, 10, 100)
		idsimpy.add_thermalized_kinetic_energy(cloud, 10)

		result_file = os.path.join(self.result_path, 'ion_cloud_randomized_ke.csv')
		idsimpy.write_cloud_file(cloud, result_file)



