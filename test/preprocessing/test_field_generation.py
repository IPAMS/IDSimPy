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

		# define simple linear vector field:
		grid_points = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20], [-10, 0, 10], [0, 10]]
		X, Y, Z = np.meshgrid(grid_points[0], grid_points[1], grid_points[2])
		S_x = X + Y + Z
		S_y1 = np.zeros(np.shape(X)) + 5.0
		S_z1 = np.zeros(np.shape(X)) + 1.0
		S_y2 = np.zeros(np.shape(X)) + 15.0
		S_z2 = np.zeros(np.shape(X)) + 11.0

		fields = [
			{'name': 'test_vectorfield_1',
			 'data': [{'name': 'X', 'data': S_x}, {'name': 'Y', 'data': S_y1}, {'name': 'Z', 'data': S_z1}]},
			{'name': 'test_vectorfield_2',
			 'data': [{'name': 'X', 'data': S_x}, {'name': 'Y', 'data': S_y2}, {'name': 'Z', 'data': S_z2}]}
		]

		dat = {"grid_points": grid_points, "meshgrid": [X, Y, Z], "fields": fields}
		fg.write_3d_vector_fields_as_vtk_point_data(dat, os.path.join(self.result_path,
		                                                              'test_linear_vector_field_01.vts'))


		#self.assertEqual(True, False)

