# -*- coding: utf-8 -*-

import gzip
import json
import io
import h5py
import numpy as np

################## Trajectory input ######################

def read_legacy_trajectory_file(trajectoryFileName):
	"""
	Reads a legacy trajectory file and returns a legacy trajectory object

	Trajectory objects are dictionaries which contain three elements:
	trajectories: a vector which contains the x,y,z positions of all particles for all time steps
	(a vector of lists one vector entry per time step)
	times: vector of times of the individual time steps
	masses: the vector of particle masses

	:param trajectoryFileName: the file name of the file to read
	:return: the trajectory data dictionary
	"""
	if (trajectoryFileName[-8:] == ".json.gz"):
		with gzip.open(trajectoryFileName) as tf:
			tj = json.load(io.TextIOWrapper(tf))
	else:
		with open(trajectoryFileName) as tf:
			tj = json.load(tf)

	steps = tj["steps"]
	nIons = len(steps[0]["positions"])

	t = np.zeros([nIons,3,len(steps)])
	times = np.zeros(len(steps))
	for i in range(len(steps)):
		t[:,:,i] = np.array(steps[i]["positions"])
		times[i] = float(steps[i]["time"])

	masses = np.zeros([nIons])
	massesJson = tj["ionMasses"]
	for i in range(len(massesJson)):
		masses[i] = float(massesJson[i])
	return{"trajectories":t,"times":times,"masses":masses}


def read_json_trajectory_file(trajectoryFileName):
	"""
	Reads a trajectory file and returns a trajectory object

	Trajectory objects are dictionaries which contain three elements:
	trajectories: a vector which contains the x,y,z positions of all particles for all time steps
	(a vector of lists one vector entry per time step)
	times: vector of times of the individual time steps
	masses: the vector of particle masses

	:param trajectoryFileName: the file name of the file to read
	:return: the trajectory data dictionary
	"""
	if trajectoryFileName[-8:] == ".json.gz":
		with gzip.open(trajectoryFileName) as tf:
			tj = json.load(io.TextIOWrapper(tf))
	else:
		with open(trajectoryFileName) as tf:
			tj = json.load(tf)

	steps = tj["steps"]
	nIons = len(steps[0]["ions"])

	times = np.zeros(len(steps))
	positions = np.zeros([nIons,3,len(steps)])

	n_additional_parameters = len(steps[0]["ions"][0])-1
	additional_parameters = np.zeros([nIons,n_additional_parameters,len(steps)])

	for i in range(len(steps)):
		for j in range (nIons):
			positions[j,:,i] = np.array(steps[i]["ions"][j][0])
			additional_parameters[j,:,i] = np.array(steps[i]["ions"][j][1:])

		times[i] = float(steps[i]["time"])

	masses = np.zeros([nIons])
	masses_json = tj["ionMasses"]
	for i in range(len(masses_json)):
		masses[i] = float(masses_json[i])

	splat_times = np.zeros([nIons])
	splat_times_json = tj["splatTimes"]
	for i in range(len(splat_times_json)):
		splat_times[i] = float(splat_times_json[i])

	return{"positions":positions,
	       "additional_parameters":additional_parameters,
	       "times":times,
	       "masses":masses,
	       "n_particles":nIons,
	       "splat_times":splat_times}



def read_hdf5_trajectory_file(trajectory_file_name):
	##fixme: impletment HDF5 reader
	hdf5File = h5py.File(trajectory_file_name, 'r')

	tra_group = hdf5File['particle_trajectory']
	attribs = tra_group.attrs
	n_particles = attribs['number of particles'][0]
	n_timesteps = attribs['number of timesteps'][0]
	positions = tra_group['positions']
	times = tra_group['times']

	aux_parameters = None
	if 'aux_parameters' in tra_group.keys():
		aux_parameters_names = [name.decode('UTF-8') for name in attribs['auxiliary parameter names']]
		aux_parameters = tra_group['aux_parameters']

	return {"positions": positions,
	        "additional_parameters": aux_parameters,
	        "additional_names": aux_parameters_names,
	        "times": times,
	        "n_particles": n_particles}






################## Trajectory output / translation ######################


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

################## Data Processing Methods ######################

def filter_mass(positions,masses,massToFilter):
	"""
	Filters out trajectories of ions with a given mass

	:param positions: a positions vector from an imported trajectories object
	:type trajectory positions: positions vector from dict returned from readTrajectoryFile
	:param masses: a mass vector from an imported trajectories object
	:param massToFilter: the mass to filter for
	:return: a filtered positions vector
	"""
	mass_indexes = np.nonzero(masses == massToFilter)
	return positions[mass_indexes,:,:][0]


def center_of_charge(tr):
	"""
	Calculates the center of charge of a charged particle cloud

	:param tr: a trajectories vector from an imported trajectories object
	:type tr: trajectories vector from dict returned from readTrajectoryFile
	:return: vector of the spatial position of the center of mass
	"""
	nSteps = np.shape(tr)[2]
	coc = np.zeros([nSteps,3])
	for i in range(nSteps):
		xMean = np.mean(tr[:,0,i])
		yMean = np.mean(tr[:,1,i])
		zMean = np.mean(tr[:,2,i])
		coc[i,:] = np.array([xMean,yMean,zMean])
	return(coc)
