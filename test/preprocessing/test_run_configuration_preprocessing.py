import unittest
import os
import IDSimPy as ip
import json

class TestInputFilePreprocessing(unittest.TestCase):

	@classmethod
	def setUpClass(cls):
		cls.input_path = os.path.join('test', 'testfiles')
		cls.result_path = os.path.join('test', 'test_results')

	def test_simple_scalar_field_generation(self):

		params = (
			(0.5, 'square', 1000),
			(0.1, 'sin', 2500),
			(0.2, 'bisin', 4500)
		)

		template_file = os.path.join(self.input_path, 'simulation_template_1.tmpl')
		result_basename = os.path.join(self.result_path, 'preprocess_test_1_')
		ip.generate_run_configurations_from_template(template_file, params, result_basename)

		result_file_1 = os.path.join(self.result_path, 'preprocess_test_1_01.json')
		result_file_2 = os.path.join(self.result_path, 'preprocess_test_1_02.json')
		with open(result_file_1) as json_file:
			data = json.load(json_file)
			self.assertEqual(data['sv_mode'], 'sin')
			self.assertAlmostEqual(data['cv_phase_shift'], 0.1)
			self.assertAlmostEqual(data['sv_Vmm-1'], 2500)

		with open(result_file_2) as json_file:
			data = json.load(json_file)
			self.assertEqual(data['sv_mode'], 'bisin')
			self.assertAlmostEqual(data['cv_phase_shift'], 0.2)
			self.assertAlmostEqual(data['sv_Vmm-1'], 4500)


