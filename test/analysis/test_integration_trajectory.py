import unittest
import os
import numpy as np
import IDSimPy.analysis as ia


class IntegrationTestTrajectory(unittest.TestCase):

	@classmethod
	def setUpClass(cls):
		data_base_path = os.path.join('test', 'analysis', 'integration', 'sim_results')
		cls.qitSim_2021_08_30_001_fn = os.path.join(data_base_path, 'qitSim_2021_08_30_001_trajectories.hd5')

	#  --------------- test importing of freshly generated trajectories ---------------

	def integration_test_qit_simulation(self):
		if not os.path.exists(self.qitSim_2021_08_30_001_fn):
			raise FileNotFoundError('Simulation qitSim_2019_07_22_001 not found, '
			                        'have you run "run_integrations_test_simulations.py"?')
		else:
			tra = ia.read_hdf5_trajectory_file(self.qitSim_2021_08_30_001_fn)

			self.assertEqual(tra.file_version_id, 3)
			self.assertEqual(tra.is_static_trajectory, False)

			particle_dat = tra.particle_attributes.get_attribs_for_particle(3, 20)
			self.assertAlmostEqual(particle_dat[1], -367.01282, delta=1e-5)
			self.assertEqual(particle_dat[9], 3)

			self.assertEqual(type(particle_dat[1]), float)
			self.assertEqual(type(particle_dat[9]), int)

			global_index = tra.particle_attributes.get('global index', 5)

			# test reading of start / splat data:
			ss_data = tra.start_splat_data
			np.testing.assert_allclose(ss_data.start_positions[2, :], [0.0003, 0.001, 0.001])
			np.testing.assert_allclose(ss_data.splat_positions[4, :], [0.0041480227, -0.0025845657, -9.2058815E-4])
			#self.assertEqual(ss_data.splat_states[35, 0], 2)

			#print(tra)
