# -*- coding: utf-8 -*-

import gzip
import json
import io
import h5py
import numpy as np

__all__ = (
        'read_legacy_trajectory_file',
        'read_json_trajectory_file',
        'read_hdf5_trajectory_file',
        'read_legacy_hdf5_trajectory_file',
		'translate_json_trajectories_to_vtk',
		'filter_parameter',
		'center_of_charge')

################## Trajectory input ######################

def read_legacy_trajectory_file(trajectory_filename):
	"""
	Reads a legacy trajectory file and returns a legacy trajectory object

	Trajectory objects are dictionaries which contain three elements:
	trajectories: a vector which contains the x,y,z positions of all particles for all time steps
	(a vector of lists one vector entry per time step)
	times: vector of times of the individual time steps
	masses: the vector of particle masses

	:param trajectory_filename: the file name of the file to read
	:return: the trajectory data dictionary
	"""
	if (trajectory_filename[-8:] == ".json.gz"):
		with gzip.open(trajectory_filename) as tf:
			tj = json.load(io.TextIOWrapper(tf))
	else:
		with open(trajectory_filename) as tf:
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


def read_json_trajectory_file(trajectory_filename):
	"""
	Reads a trajectory file and returns a trajectory object

	Trajectory objects are dictionaries which contain three elements:
	trajectories: a vector which contains the x,y,z positions of all particles for all time steps
	(a vector of lists one vector entry per time step)
	times: vector of times of the individual time steps
	masses: the vector of particle masses

	:param trajectory_filename: the file name of the file to read
	:return: the trajectory data dictionary
	"""
	if trajectory_filename[-8:] == ".json.gz":
		with gzip.open(trajectory_filename) as tf:
			tj = json.load(io.TextIOWrapper(tf))
	else:
		with open(trajectory_filename) as tf:
			tj = json.load(tf)

	steps = tj["steps"]
	n_timesteps = len(steps)
	nIons = len(steps[0]["ions"])

	times = np.zeros(len(steps))
	positions = np.zeros([nIons,3,len(steps)])

	n_additional_parameters = len(steps[0]["ions"][0])-1
	additional_parameters = np.zeros([nIons,n_additional_parameters,n_timesteps])

	for i in range(n_timesteps):
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
	       "n_timesteps":n_timesteps,
	       "splat_times":splat_times,
	       "static_trajectory":True}



def read_hdf5_trajectory_file(trajectory_file_name):
	"""
	Reads a version 2 hdf5 trajectory file (which allows also exported simulation frames
	with variable number of particles.

	If the trajectory is static the trajectory is returned in a static format
	(three dimensional positions and additional_data arrays).
	If the trajectory is variable (the number of particles differ in the individual exported frames)
	the position and additional_data is given in lists of individual arrays for the
	individual time steps.

	:param trajectory_file_name: the name of the file to read
	:return: a trajectory dictionary
	"""
	hdf5File = h5py.File(trajectory_file_name, 'r')

	tra_group = hdf5File['particle_trajectory']
	attribs = tra_group.attrs

	file_version_id = attribs['file version'][0]

	n_timesteps = attribs['number of timesteps'][0]

	timesteps_group = tra_group['timesteps']
	times = tra_group['times']

	aux_parameters_names = None
	if 'auxiliary parameter names' in attribs.keys():
		aux_parameters_names = [name.decode('UTF-8') for name in attribs['auxiliary parameter names']]


	positions = []
	aux_parameters = []

	n_ion_per_frame = []
	for ts_i in range(n_timesteps):
		ts_group = timesteps_group[str(ts_i)]

		ion_positions = np.array(ts_group['positions'])

		n_ion_per_frame.append(np.shape(ion_positions)[0])
		positions.append(ion_positions)


		if aux_parameters_names:
			aux_parameters.append(np.array(ts_group['aux_parameters']))


	unique_n_ions = len(set(n_ion_per_frame))


	# if more than one number of ions are present in the frames, the trajectory is not static
	# and has variable frames
	# if the trajectory is static, transform the trajectory to the old format returned by
	# legacy hdf5 and json files to allow compatibility with the visualization methods
	if unique_n_ions > 1:
		static_trajectory = False
	else:
		static_trajectory = True
		positions = np.dstack(positions)


	result = {"positions": positions,
	          "times": np.array(times),
	          "n_timesteps": n_timesteps,
	          "file_version_id": file_version_id,
	          "static_trajectory": static_trajectory}

	if 'splattimes' in tra_group.keys():
		result['splattimes'] = np.array(tra_group['splattimes'])

	if static_trajectory:
		result['n_particles'] = n_ion_per_frame[0]

	if aux_parameters_names:
		aux_dat = np.array(aux_parameters)
		if static_trajectory:
			aux_dat = np.dstack(aux_dat)

		result["additional_parameters"] = aux_dat
		result["additional_names"] = aux_parameters_names

	return result


def read_legacy_hdf5_trajectory_file(trajectory_file_name):
	"""
	Reads a legacy hdf5 trajectory file (with static particles per exported simulation frame)

	:param trajectory_file_name: the name of the file to read
	:return: a trajectory dictionary
	"""
	hdf5File = h5py.File(trajectory_file_name, 'r')

	tra_group = hdf5File['particle_trajectory']
	attribs = tra_group.attrs
	n_particles = attribs['number of particles'][0]
	n_timesteps = attribs['number of timesteps'][0]
	positions = tra_group['positions']
	times = tra_group['times']

	result = {"positions": np.array(positions),
	          "times": np.array(times),
	          "n_particles": n_particles,
	          "n_timesteps": n_timesteps,
	          "static_trajectory":True}

	if 'aux_parameters' in tra_group.keys():
		aux_parameters_names = [name.decode('UTF-8') for name in attribs['auxiliary parameter names']]
		aux_parameters = tra_group['aux_parameters']
		result["additional_parameters"] = np.array(aux_parameters)
		result["additional_names"] = aux_parameters_names

	return result



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


def filter_parameter(positions, filter_param, value):
	"""
	Filters out trajectories of ions according to a given parameter vector

	:param positions: a positions vector from an imported trajectories object
	:type trajectory positions: positions vector from dict returned from readTrajectoryFile
	:param filter_param: a parameter from an imported trajectories object to filter for
		if a vector is provided it is assumed, that the filtering is stable across all timesteps
		if a two dimensional array is provided, the filtering is performed with individual filter param vectors
		for the timesteps
	:param value: the value to filter for
	:return: filtered particle positions
		if a filter vector is provided: Numpy array is returned with the particles, spatial dimensions and timesteps
		as dimensions
		if a filter matrix (individual filter param vectors for the individual timesteps) is provided:
		list of individual filterd position vectors for the individual timesteps
	"""

	# if filter_param is a vector: Same filtering for all timesteps
	if filter_param.ndim == 1:
		filtered_indexes = np.nonzero(filter_param == value)
		return positions[filtered_indexes,:,:][0]
	if filter_param.ndim == 2:
		# we have a different filter parameter vector per timestep
		# filtered particles per timestep could variate: generate a vector per timestep
		n_ts = np.shape(filter_param)[1]
		filtered_indexes = [np.nonzero(filter_param[:,i] == value) for i in range(n_ts)]
		result = [positions[filtered_indexes[i], :, i][0] for i in range(len(filtered_indexes))]
		return result
	else:
		raise ValueError('Filter parameter is not a vector nor a two dimensional array')


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
