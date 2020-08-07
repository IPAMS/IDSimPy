import unittest
import os
import IDSimPy.preprocessing.comsol_import as ci


class TestFieldGeneration(unittest.TestCase):

	@classmethod
	def setUpClass(cls):
		cls.quad_rf_field = os.path.join('testfiles', 'transfer_quad_rf_field.csv.gz')
		cls.result_path = "test_results"


	def test_comsol_vector_field_import(self):
		dat = ci.import_comsol_3d_csv_grid(self.quad_rf_field)

		V_field = dat['fields'][0]
		E_x_field = dat['fields'][1]

		self.assertEqual(V_field['name'], '% V (V)')
		self.assertEqual(E_x_field['name'], '% es.Ex (V/m)')

		self.assertEqual(V_field['data'].shape, (160, 60, 60))
		self.assertAlmostEqual(V_field['data'][80, 10, 10], -0.994295)
		self.assertAlmostEqual(V_field['data'][40, 29, 29], -0.007102)


