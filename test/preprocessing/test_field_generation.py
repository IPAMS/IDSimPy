import unittest
import os
import numpy as np
import IDSimPy.preprocessing.field_generation as fg


class TestFieldGeneration(unittest.TestCase):

	@classmethod
	def setUpClass(cls):
		cls.result_path = "test_results"

	def test_simple_field_generation(self):
		# define simple linear scalar field:
		grid_points = [[0, 2, 5, 15], [0, 2, 10], [0, 2, 5, 7, 10]]
		X, Y, Z = np.meshgrid(grid_points[0], grid_points[1], grid_points[2])
		S = X + Y + Z
		fields = [{'name': 'test_field', 'data': S}]
		dat = {"grid_points": grid_points, "meshgrid": [X, Y, Z], "fields": fields}
		fg.write_3d_scalar_fields_as_vtk_point_data(dat, os.path.join(self.result_path,
		                                                              'test_linear_scalar_field_01.vts'))

		fg.write_3d_scalar_fields_to_hdf5(dat, os.path.join(self.result_path,
		                                                    'test_linear_scalar_field_01.hdf5'))

		# define simple linear vector field:
		grid_points = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20], [-10, 0, 10], [0, 10]]
		X, Y, Z = np.meshgrid(grid_points[0], grid_points[1], grid_points[2])
		S_x = X + Y + Z
		S_y1 = np.zeros(np.shape(X)) + 5.0
		S_z1 = np.zeros(np.shape(X)) + 1.0
		S_y2 = np.zeros(np.shape(X)) + 15.0
		S_z2 = np.zeros(np.shape(X)) + 11.0

		fields = [
			{'name': 'test_vectorfield_1', 'data': [S_x, S_y1, S_z1]},
			{'name': 'test_vectorfield_2', 'data': [S_x, S_y2, S_z2]}
		]

		dat = {"grid_points": grid_points, "meshgrid": [X, Y, Z], "fields": fields}
		fg.write_3d_vector_fields_as_vtk_point_data(dat, os.path.join(self.result_path,
		                                                              'test_linear_vector_field_01.vts'))


class Buffer():
	def test_quadrupole_vector_field_generation(self):
		# define 2d axial symmetric pressure field and translate it to 3d cartesian:
		grid_r = [np.linspace(0, 0.01, 30)]
		grid_z = [np.linspace(0, 0.1, 200)]
		# grid_r = np.arange(0,20)
		# grid_z = 10

		R, Z = np.meshgrid(grid_r, grid_z)
		P = R * Z
		P[:, 1:10] = 0

		X, Y, Z, P3 = fg.transform_2d_axial_to_3d(R, Z, P)
		grid_points = [X[0, :, 0], Y[:, 0, 0], Z[0, 0, :]]
		grid_points_len = [len(X[0, :, 0]), len(Y[:, 0, 0]), len(Z[0, 0, :])]

		fields = [{'name': 'radial_pressure', 'data': P3}]
		dat = {"grid_points": grid_points, "fields": fields}
		fg.write_3d_scalar_fields_as_vtk_point_data(dat, os.path.join(self.result_path, 'quad_dev_pressure_radial.vts'))

		# define simple 3d pressure field for testing / development:
		dx = 0.4
		dyz = 0.1
		grid_points = [np.linspace(-dx, dx, 300), np.linspace(-dyz, dyz, 40), np.linspace(-dyz, dyz, 40)]

		X, Y, Z = np.meshgrid(grid_points[0], grid_points[1], grid_points[2])
		S = np.zeros(np.shape(X)) + 10.0  # (dx-X)*10
		fields = [{'name': 'pressure', 'data': S}]
		dat = {"grid_points": grid_points, "meshgrid": [X, Y, Z], "fields": fields}
		fg.write_3d_scalar_fields_as_vtk_point_data(dat, os.path.join(self.result_path, 'quad_dev_pressure_3d.vts'))

		# define simple 3d vector flow field for testing / development:
		S_x = ((dyz - np.abs(Y)) * (dyz - np.abs(Z))) * 150 / (dyz ** 2)
		S_y = np.zeros(np.shape(X))
		S_z = np.zeros(np.shape(X))
		S_xyz = (S_x, S_y, S_z)
		S_xyz_scaled = [fi * 2.0 for fi in S_xyz]

		#fields = [{'name': 'U', 'data': S_x}, {'name': 'V', 'data': S_y}, {'name': 'W', 'data': S_z}]
		fields = [
			{'name': 'velocity', 'data': S_xyz},
			{'name': 'velocity_scaled', 'data': S_xyz_scaled}
		]

		dat = {"grid_points": grid_points, "meshgrid": [X, Y, Z], "fields": fields}
		fg.write_3d_vector_fields_as_vtk_point_data(dat, os.path.join(self.result_path, 'quad_dev_flow_3d.vts'))

		# define simple 3d vector electrical field for testing / development:
		S_x = 0.0 + np.zeros(np.shape(X))
		S_y = Y * -10.0
		S_z = Z * -10.0

		fields = [{'name': 'electric field', 'data': (S_x, S_y, S_z)}]
		dat = {"grid_points": grid_points, "meshgrid": [X, Y, Z], "fields": fields}
		fg.write_3d_vector_fields_as_vtk_point_data(dat, os.path.join(self.result_path, 'quad_dev_field.vts'))


