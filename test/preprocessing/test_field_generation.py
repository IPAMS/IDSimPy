import unittest
import os
import numpy as np
import IDSimPy as fg


class TestFieldGeneration(unittest.TestCase):

	@classmethod
	def setUpClass(cls):
		cls.result_path = os.path.join('test', 'test_results')

	def test_simple_scalar_field_generation(self):
		# define simple linear scalar field:
		grid_points = [[0, 2, 5, 15], [0, 2, 10], [0, 2, 5, 7, 10]]
		X, Y, Z = np.meshgrid(grid_points[0], grid_points[1], grid_points[2], indexing='ij')
		S = X + Y + Z
		fields = [{'name': 'test_field', 'data': S}]
		dat = {"grid_points": grid_points, "fields": fields}
		fg.write_3d_scalar_fields_as_vtk_point_data(dat, os.path.join(self.result_path,
		                                                              'test_linear_scalar_field_01.vts'))

		fg.write_3d_scalar_fields_to_hdf5(dat, os.path.join(self.result_path,
		                                                    'test_linear_scalar_field_01.h5'))

	def test_simple_vector_field_generation(self):

		# define simple linear vector field:
		grid_points = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20], [-10, 0, 10], [0, 10]]
		X, Y, Z = np.meshgrid(grid_points[0], grid_points[1], grid_points[2], indexing='ij')
		S_x = X + Y + Z
		S_y1 = np.zeros(np.shape(X)) + 5.0
		S_z1 = np.zeros(np.shape(X)) + 1.0
		S_y2 = np.zeros(np.shape(X)) + 15.0
		S_z2 = np.zeros(np.shape(X)) + 11.0

		fields = [
			{'name': 'test_vectorfield_1', 'data': [S_x, S_y1, S_z1]},
			{'name': 'test_vectorfield_2', 'data': [S_x, S_y2, S_z2]}
		]

		dat = {"grid_points": grid_points, "fields": fields}
		fg.write_3d_vector_fields_as_vtk_point_data(dat, os.path.join(self.result_path,
		                                                              'test_linear_vector_field_01.vts'))

		fg.write_3d_vector_fields_to_hdf5(dat, os.path.join(self.result_path,
		                                                    'test_linear_vector_field_01.h5'))


	def test_2d_3d_conversion_for_scalar_field(self):

		points_r = [np.linspace(0, 0.01, 30)]
		points_z = [np.linspace(0, 0.1, 200)]

		r_axi, z_ca = np.meshgrid(points_r, points_z, indexing='ij')

		v_axi = r_axi * z_ca * 1e2
		v_axi[1:10, :] = 0
		v_axi[1:3, :] = 0.1

		x_ca, y_ca, z_ca, v_ca = fg.transform_2d_axial_to_3d(r_axi, z_ca, v_axi)

		self.assertEqual(x_ca.shape, (60, 200, 60))

		# Check values on radial positions:
		self.assertAlmostEqual(v_ca[5, 100, 0], v_ca[24, 100, 0])
		self.assertAlmostEqual(v_ca[5, 100, 0], v_ca[0, 100, 5])

		#plt.contourf(np.transpose(V_ca[:,:,30]))
		#plt.colorbar()
		#plt.show()

		#plt.contourf(np.transpose(V_ca[:,100,:]))
		#plt.colorbar()
		#plt.show()

		#todo: Test convesion of axial symmetric vector field

	def test_quadrupole_vector_field_generation(self):

		def write_radial_pressure_field():
			# define 2d axial symmetric pressure field and translate it to 3d cartesian:
			points_r = [np.linspace(0, 0.01, 30)]
			points_z = [np.linspace(0, 0.1, 200)]

			r_axi, z_ca = np.meshgrid(points_r, points_z, indexing='ij')
			p_axi = r_axi * z_ca
			p_axi[:, 1:10] = 0

			x_ca, y_ca, z_ca, p_ca = fg.transform_2d_axial_to_3d(r_axi, z_ca, p_axi)
			grid_ca = [x_ca[:, 0, 0], y_ca[0, :, 0], z_ca[0, 0, :]]  # cartesian grid

			fields = [{'name': 'radial_pressure', 'data': p_ca}]
			dat = {"grid_points": grid_ca, "fields": fields}
			fg.write_3d_scalar_fields_as_vtk_point_data(dat,
			                                            os.path.join(self.result_path,
			                                                         'quad_dev_pressure_radial.vts'))

		def write_3d_fields():
			# define simple 3d pressure field for testing / development:
			dx = 0.4
			dyz = 0.1
			grid_ca = [np.linspace(-dx, dx, 300), np.linspace(-dyz, dyz, 40), np.linspace(-dyz, dyz, 40)]

			x_ca, y_ca, z_ca = np.meshgrid(grid_ca[0], grid_ca[1], grid_ca[2], indexing='ij')
			S = np.zeros(np.shape(x_ca)) + 10.0  # (dx-X)*10
			fields = [{'name': 'pressure', 'data': S}]
			dat = {"grid_points": grid_ca, "fields": fields}
			fg.write_3d_scalar_fields_as_vtk_point_data(dat, os.path.join(self.result_path, 'quad_dev_pressure_3d.vts'))

			# define simple 3d vector flow field for testing / development:
			S_x = ((dyz - np.abs(y_ca)) * (dyz - np.abs(z_ca))) * 150 / (dyz ** 2)
			S_y = np.zeros(np.shape(x_ca))
			S_z = np.zeros(np.shape(x_ca))
			S_xyz = (S_x, S_y, S_z)
			S_xyz_scaled = [fi * 2.0 for fi in S_xyz]

			fields = [
				{'name': 'velocity', 'data': S_xyz},
				{'name': 'velocity_scaled', 'data': S_xyz_scaled}
			]

			dat = {"grid_points": grid_ca, "fields": fields}
			fg.write_3d_vector_fields_as_vtk_point_data(dat, os.path.join(self.result_path, 'quad_dev_flow_3d.vts'))

			# define simple 3d vector electrical field for testing / development:
			S_x = 0.0 + np.zeros(np.shape(x_ca))
			S_y = y_ca * -10.0
			S_z = z_ca * -10.0

			fields = [{'name': 'electric field', 'data': (S_x, S_y, S_z)}]
			dat = {"grid_points": grid_ca, "fields": fields}
			fg.write_3d_vector_fields_as_vtk_point_data(dat, os.path.join(self.result_path, 'quad_dev_field.vts'))

		write_radial_pressure_field()
		write_3d_fields()


