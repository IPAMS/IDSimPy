# -*- coding: utf-8 -*-

import qitsim_analysis as qa


def translate_json_trajectories_to_vtk(json_file_name, vtk_file_base_name):
	"""
	Translates a ion trajectory file to set of legacy VTK ascii files
	:param json_file_name: the trajectory file to translate
	:param vtk_file_base_name: the base name of the vtk files to generate
	"""

	tr = qa.read_trajectory_file(json_file_name)

	header="""# vtk DataFile Version 2.0
BTree Test
ASCII
DATASET POLYDATA
POINTS """

	n_steps = len(tr["times"])

	for i in range(n_steps):
		vtk_file_name = vtk_file_base_name+"%05d"%i+".vtk"
		print(vtk_file_name)
		with open(vtk_file_name, 'w') as vtk_file:
			vtk_file.write(header+ str(tr["n_ions"])+" float\n")

			ion_positions = tr["positions"]
			for i_pos in ion_positions[:,:,i]:
				vtk_file.write(str(i_pos[0])+" "+str(i_pos[1])+" "+str(i_pos[2])+" \n")