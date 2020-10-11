import unittest
import os
import IDSimPy.preprocessing.comsol_import as ci
import IDSimPy.preprocessing.field_generation as fg


class TestFieldGeneration(unittest.TestCase):

	@classmethod
	def setUpClass(cls):
		cls.quad_rf_field = os.path.join('testfiles', 'transfer_quad_rf_field.csv.gz')
		cls.result_path = "test_results"
		cls.hdf5_vector_result_file = os.path.join(cls.result_path, 'comsol_vector_import.h5')
		cls.hdf5_scalar_result_file = os.path.join(cls.result_path, 'comsol_scalar_import.h5')

	def test_comsol_vector_field_import(self):
		dat = ci.import_comsol_3d_csv_grid(self.quad_rf_field)

		V_field = dat['fields'][0]
		E_x_field = dat['fields'][1]

		self.assertEqual(V_field['name'], '% V (V)')
		self.assertEqual(E_x_field['name'], '% es.Ex (V/m)')

		self.assertEqual(V_field['data'].shape, (160, 60, 60))
		self.assertAlmostEqual(V_field['data'][80, 10, 10], -0.994295)
		self.assertAlmostEqual(V_field['data'][40, 29, 29], -0.007102)

	def test_comsol_data_is_writeable_as_interpolated_grid(self):

		# import comsol data
		dat = ci.import_comsol_3d_csv_grid(self.quad_rf_field)

		# create electric vector field:
		e_x = dat['fields'][1]['data']
		e_y = dat['fields'][2]['data']
		e_z = dat['fields'][3]['data']
		e_field = (e_x, e_y, e_z)

		# create a new dat object with the vector field data
		dat_v_fields = [{'name': 'electric field', 'data':e_field}]
		dat_v = {'grid_points': dat['grid_points'], 'meshgrid': dat['meshgrid'], 'fields': dat_v_fields}

		# write vector field data to HDF5 file:
		fg.write_3d_vector_fields_to_hdf5(dat_v, self.hdf5_vector_result_file)

		# create n new dat object with the electric potential:
		dat_u_fields = [dat['fields'][0]]
		dat_u = {'grid_points': dat['grid_points'], 'meshgrid': dat['meshgrid'], 'fields': dat_u_fields}

		# write potential scalar field
		fg.write_3d_scalar_fields_to_hdf5(dat_u, self.hdf5_scalar_result_file)


