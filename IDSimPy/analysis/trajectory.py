# -*- coding: utf-8 -*-

import gzip
import json
import io
import h5py
import numpy as np
from enum import Enum

__all__ = (
	'Trajectory',
	'OptionalAttribute',
	'read_json_trajectory_file',
	'read_hdf5_trajectory_file',
	'read_legacy_hdf5_trajectory_file',
	'translate_json_trajectories_to_vtk',
	'filter_attribute',
	'select',
	'center_of_charge')


class OptionalAttribute(Enum):
	"""
	Optional trajectory attributes identifiers: Distinct and extensible identifiers for optional trajectory
	data attributes.
	"""
	PARTICLE_MASSES = 1  #: Particle masses
	PARTICLE_CHARGES = 2  #: Particle charges


class Trajectory:
	"""
	An IDSimF particle simulation trajectory. The simulation trajectory combines the result of an IDSimF particle
	simulation in one object. The trajectory consists of the positions of simulated particles at the time steps of
	the simulation, the times of the time steps and optional attributes of the simulated particles.

	The trajectory can be "static" which means that the number of particles is not
	changing between the time steps.

	:ivar positions: Particle positions. The particles are stored in a different scheme, depending if the trajectory
		is static:

		* If the trajectory is static: **positions** is a ``numpy.ndarray`` with the shape ``[n ions, spatial
		  dimensions, n time steps]``. With 5 particles and 15 time steps the shape would be ``[5, 3, 15]``.
		* If the trajectory is not static: **positions** is a ``list`` of ``numpy.ndarray`` with the shape ``[spatial
		  dimensions, n ions]``

	:ivar times: Vector of simulated times for the individual time frames.
	:type times: numpy.ndarray
	:ivar n_timesteps: Number of time steps in the trajectory
	:type n_timesteps: int
	:var particle_attributes: Optional simulation result attributes for the simulated particles. Basically,
		particle attributes are a vector of numeric additional particle attributes, attached to every particle
		in every time step. Similarly to ``positions`` the shape depends if the trajectory is static or not:

		* If the trajectory is static: **particle_attributes** is a ``numpy.ndarray`` with the shape ``[n ions,
		  particle attribute, n time steps]``. With 5 particles, 4 additional numerical attributes (e.g. x,y,z velocity
		  and chemical id) and 15 time steps the shape would be ``[5, 4, 15]``.
		* If the trajectory is not static: **particle_attributes** is a ``list`` of ``numpy.ndarray`` with the shape
		  ``[particle attribute, n ions]``

	:ivar particle_attribute_names: Names of the particle attributes
	:type particle_attribute_names: list[str]
	:ivar splat_times: Vector of particle termination / splat times
	:type splat_times: numpy.ndarray
	:ivar optional_attributes: dictionary of optional / free form additional attributes for the trajectory
	:type optional_attributes: dict
	:ivar is_static_trajectory: Flag if the trajectory is static.
	:type is_static_trajectory: bool
	"""

	def __init__(self, positions=None, times=None, particle_attributes=None, particle_attribute_names=None,
	             splat_times=None, optional_attributes=None, file_version_id=0):
		"""
		Constructor: (for details about the shape of the parameters see the class docsting)

		:param positions: Particle positions
		:type positions: numpy.ndarray or list[numpy.ndarray]
		:param times: Times of the simulation time steps
		:type times: numpy.ndarray with shape ``[n timesteps, 1]``
		:param particle_attributes: Additional attributes for every particle for every time step
		:type particle_attributes: numpy.ndarray or list[numpy.ndarray]
		:param particle_attribute_names: Names of partricle attributes
		:type particle_attribute_names: tuple[str]
		:param splat_times: Particle termination / "splat" times
		:type splat_times: numpy.ndarray
		:param optional_attributes: Optional attributes dictionary
		:type optional_attributes: dict
		:param file_version_id: File version id
		:type file_version_id: int
		"""

		if type(positions) == np.ndarray:
			self.is_static_trajectory = True
			if len(positions.shape) != 3 or positions.shape[1] != 3:
				raise ValueError('Static positions have wrong shape')
			self.n_timesteps = positions.shape[2]
		elif type(positions) == list:
			self.is_static_trajectory = False
			self.n_timesteps = len(positions)
		else:
			raise TypeError('Wrong type for positions, has to be an numpy.ndarray or a list of numpy.ndarrays')

		if type(times) != np.ndarray:
			raise TypeError('Wrong type for times, a numpy vector is expected')

		if times.shape[0] != self.n_timesteps:
			raise ValueError('Times vector has wrong length')

		if particle_attributes is not None:
			if self.is_static_trajectory:
				if type(particle_attributes) != np.ndarray:
					raise ValueError('Additional parameter has wrong type for static trajectory')
				n_additional_attributes = particle_attributes.shape[1]
				self.particle_attributes = particle_attributes
			else:
				if type(particle_attributes) != list:
					raise ValueError('Additional parameter has wrong type for variable trajectory')
				n_additional_attributes = particle_attributes[0].shape[1]
				self.particle_attributes = particle_attributes

			if len(particle_attribute_names) != n_additional_attributes:
				raise ValueError('Additional parameter name vector has wrong length')

		if splat_times is not None:
			if self.is_static_trajectory:
				if type(splat_times) != np.ndarray:
					raise ValueError('Splat times vector has wrong type')
			else:
				raise ValueError('Currently, splat times are only supported for static trajectories')

		self.positions = positions
		self.times: np.ndarray = times
		self.particle_attributes = particle_attributes
		self.particle_attribute_names: list = particle_attribute_names
		self.splat_times = splat_times
		self.optional_attributes = optional_attributes
		self.file_version_id: int = file_version_id

	def __len__(self):
		return self.n_timesteps

	def __getitem__(self, timestep_index):
		if self.is_static_trajectory:
			return self.positions[:, :, timestep_index]
		else:
			return self.positions[timestep_index]

	def get_n_particles(self, timestep_index=None):
		"""
		Returns the static number of particles in a static trajectory

		:return: Number of particles in the static trajectory
		:rtype: int
		"""
		if self.is_static_trajectory:
			return self.positions.shape[0]
		else:
			if timestep_index is not None:
				return self.positions[timestep_index].shape[0]
			else:
				raise AttributeError("Time step independent number of ions is only defined for static trajectories")

	n_particles = property(get_n_particles)


	def get_positions(self, timestep_index):
		"""
		Get particle positions for a time step

		:param timestep_index: The index of the time step to get the positions for
		:type timestep_index: int
		:return: Array of particle positions for a time step. Dimensions are ``[n particles, spatial dimensions]``
		:rtype: numpy.ndarray
		"""
		return self[timestep_index]

	def get_particle_attributes(self, timestep_index):
		"""
		Get particle attributes for a time step

		:param timestep_index: The index of the time step to get the attributes for
		:type timestep_index: int
		:return: Array of particle attributes for a time step. Dimensions are ``[n particles, attributes]``
		:rtype: numpy.ndarray
		"""
		if self.is_static_trajectory:
			return self.particle_attributes[:, :, timestep_index]
		else:
			return self.particle_attributes[timestep_index]

	def get_particle(self, particle_index, timestep_index):
		"""
		Get particle values (positions and additional attributes) for a particle at a specified time step

		:param particle_index: The index of the particle
		:type particle_index: int
		:param timestep_index: The index of the time step
		:type timestep_index: int
		:return: Tuple with the position and the attribute vector for the particle at the selected
			time step
		:rtype: tuple of two numpy.ndarray
		"""
		if self.is_static_trajectory:
			pos = self.positions[particle_index, :, timestep_index]
			attributes = self.particle_attributes[particle_index, :, timestep_index]
		else:
			pos = self.positions[timestep_index][particle_index, :]
			attributes = self.particle_attributes[timestep_index][particle_index, :]

		return pos, attributes


# -------------- Trajectory input -------------- #


def read_json_trajectory_file(trajectory_filename):
	"""
	Reads a json trajectory file and returns a trajectory object

	:param trajectory_filename: File name of the file to read
	:type trajectory_filename: str
	:return: Trajectory object with trajectory data
	:rtype: Trajectory
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
	positions = np.zeros([nIons, 3, len(steps)])

	n_additional_parameters = len(steps[0]["ions"][0]) - 1
	additional_parameters = np.zeros([nIons, n_additional_parameters, n_timesteps])
	additional_parameters_names = ['attribute '+str(i+1) for i in range(n_additional_parameters)]

	for i in range(n_timesteps):
		for j in range(nIons):
			positions[j, :, i] = np.array(steps[i]["ions"][j][0])
			additional_parameters[j, :, i] = np.array(steps[i]["ions"][j][1:])

		times[i] = float(steps[i]["time"])

	masses = np.zeros([nIons])
	masses_json = tj["ionMasses"]
	for i in range(len(masses_json)):
		masses[i] = float(masses_json[i])

	splat_times = np.zeros([nIons])
	splat_times_json = tj["splatTimes"]
	for i in range(len(splat_times_json)):
		splat_times[i] = float(splat_times_json[i])

	optional_attributes = {OptionalAttribute.PARTICLE_MASSES: masses}

	result = Trajectory(
		positions=positions,
		times=times,
		particle_attributes=additional_parameters,
		particle_attribute_names=additional_parameters_names,
		optional_attributes=optional_attributes,
		splat_times=splat_times)

	return result


def read_hdf5_trajectory_file(trajectory_file_name):
	"""
	Reads a version 2 hdf5 trajectory file (which allows also exported simulation frames
	with variable number of particles.

	:param trajectory_file_name: Name of the file to read
	:type trajectory_file_name: str
	:return: Trajectory object with trajectory data
	:rtype: Trajectory
	"""
	hdf5file = h5py.File(trajectory_file_name, 'r')

	tra_group = hdf5file['particle_trajectory']
	attribs = tra_group.attrs

	file_version_id = attribs['file version'][0]

	n_timesteps = attribs['number of timesteps'][0]

	timesteps_group = tra_group['timesteps']
	times = tra_group['times']

	particle_attributes_names = None
	if 'auxiliary parameter names' in attribs.keys():
		particle_attributes_names = [name.decode('UTF-8') for name in attribs['auxiliary parameter names']]

	positions = []
	particle_attributes = []

	n_ion_per_frame = []
	for ts_i in range(n_timesteps):
		ts_group = timesteps_group[str(ts_i)]

		ion_positions = np.array(ts_group['positions'])

		n_ion_per_frame.append(np.shape(ion_positions)[0])
		positions.append(ion_positions)

		if particle_attributes_names:
			particle_attributes.append(np.array(ts_group['aux_parameters']))

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

	splat_times = None
	if 'splattimes' in tra_group.keys():
		splat_times = np.array(tra_group['splattimes'])

	particle_attributes_dat = None
	if particle_attributes_names:
		particle_attributes_dat = particle_attributes
		if static_trajectory:
			particle_attributes_dat = np.dstack(np.array(particle_attributes_dat))

	result = Trajectory(
		positions=positions,
		times=np.array(times),
		particle_attributes=particle_attributes_dat,
		particle_attribute_names=particle_attributes_names,
		splat_times=splat_times,
		file_version_id=file_version_id)

	return result


def read_legacy_hdf5_trajectory_file(trajectory_file_name):
	"""
	Reads a legacy hdf5 trajectory file (with static particles per exported simulation frame)

	:param trajectory_file_name: The name of the file to read
	:type trajectory_file_name: str
	:return: Trajectory object with trajectory data
	:rtype: Trajectory
	"""
	hdf5file = h5py.File(trajectory_file_name, 'r')

	tra_group = hdf5file['particle_trajectory']
	attribs = tra_group.attrs
	positions = tra_group['positions']
	times = tra_group['times']

	aux_parameters_names = None
	aux_parameters = None
	if 'aux_parameters' in tra_group.keys():
		aux_parameters_names = [name.decode('UTF-8') for name in attribs['auxiliary parameter names']]
		aux_parameters = np.array(tra_group['aux_parameters'])

	result = Trajectory(
		positions=np.array(positions),
		times=np.array(times),
		particle_attributes=aux_parameters,
		particle_attribute_names=aux_parameters_names,
		file_version_id=1)

	return result


# -------------- Trajectory output / translation -------------- #


def translate_json_trajectories_to_vtk(json_file_name, vtk_file_base_name):
	"""
	Translates a ion trajectory file to set of legacy VTK ascii files

	:param json_file_name: the trajectory file to translate
	:type json_file_name: str
	:param vtk_file_base_name: the base name of the vtk files to generate
	:type vtk_file_base_name: str
	"""

	tr = read_json_trajectory_file(json_file_name)

	header = """# vtk DataFile Version 2.0
BTree Test
ASCII
DATASET POLYDATA
POINTS """

	n_steps = len(tr["times"])

	for i in range(n_steps):
		vtk_file_name = vtk_file_base_name + "%05d" % i + ".vtk"
		print(vtk_file_name)
		with open(vtk_file_name, 'w') as vtk_file:
			vtk_file.write(header + str(tr["n_ions"]) + " float\n")

			ion_positions = tr["positions"]
			for i_pos in ion_positions[:, :, i]:
				vtk_file.write(str(i_pos[0]) + " " + str(i_pos[1]) + " " + str(i_pos[2]) + " \n")


# -------------- Data Processing Methods -------------- #


def filter_attribute(trajectory, attribute_name, value):
	"""
	Filters select ions according to a value of a specified particle attribute.
	The method takes a :py:class:`Trajectory` the name of a particle attribute and a value which
	is selected for and constructs a new :py:class:`Trajectory` from the filtered data.

	Currently optional trajectory attributes and splat times are **not** retained.

	This method returns always a variable, non static Trajectory.


	:param trajectory: Trajectory object with the trajectory data to filter for
	:type trajectory: Trajectory
	:param attribute_name: Name of a particle attribute to filter for
	:type attribute_name: str
	:param value: Value to filter for: This value is used as selector
	:return: A Trajectory object with filtered particle positions
	:rtype: Trajectory
	"""

	attribute_index = trajectory.particle_attribute_names.index(attribute_name)
	n_ts = trajectory.n_timesteps

	#  iterate through time steps and construct time step wise selected index arrays
	filtered_indexes = [
		np.nonzero(trajectory.get_particle_attributes(i)[:, attribute_index] == value)[0]
		for i in range(n_ts)
	]
	new_positions = [trajectory.get_positions(i)[filtered_indexes[i], :] for i in range(n_ts)]
	new_particle_attributes = [trajectory.get_particle_attributes(i)[filtered_indexes[i], :] for i in range(n_ts)]

	result = Trajectory(
		positions=new_positions,
		particle_attributes=new_particle_attributes,
		particle_attribute_names=trajectory.particle_attribute_names,
		times=trajectory.times)

	return result


def select(trajectory, selector_data, value):
	"""
	Selects simulated particles according to given value in an array of selector data and constructs a new
	:py:class:`Trajectory` with the selected data.

	This method is primarily intended to provide a flexible mechanism to select particles from a trajectory
	with a custom constructed parameter to be used for selection.


	:param trajectory: The trajectory object to be selected from
	:param selector_data: Selector data which assigns one value of a parameter to be used for selection
		to every particle.

		* if a vector (one dimensional array) is provided it is assumed, that the particle related parameter which is
		  filtered for is stable across all time steps. This is only possible for static :py:class:`Trajectory` objects.
		* An individual filtering for every time step is done when a list of selector data vectors, one per time step,
		  is provided. Every time step then has an vector of parameter values for the individual particles. This
		  mode is possible for static an variable :py:class:`Trajectory` objects.
	:type selector_data: numpy.ndarray or list of numpy.ndarray

	:param value: Value to select for: Particles with this value are selected from the trajectory object.
	:return: Trajectory with selected data
	:rtype: Trajectory
	"""

	# if selector is a vector: Same filtering for all time steps
	if type(selector_data) is np.ndarray and selector_data.ndim == 1:
		static_selector = True
		selected_indices = np.nonzero(selector_data == value)[0]

	elif type(selector_data) is list:
		# we have a different filter parameter vector per time step
		# filtered particles per time step could variate: generate a vector per time step
		static_selector = False
		n_ts = len(selector_data)
		selected_indices = [np.nonzero(selector_data[i] == value)[0] for i in range(n_ts)]

	else:
		raise TypeError('Wrong data type for selector_data. One dimensional numpy array or list of one dimensional '
		                'numpy arrays expected')

	new_splat_times = None
	if static_selector:
		if trajectory.is_static_trajectory:
			new_positions = trajectory.positions[selected_indices, :, :]
			new_particle_attributes = trajectory.particle_attributes[selected_indices, :, :]
			if trajectory.splat_times is not None:
				new_splat_times = trajectory.splat_times[selected_indices]
		else:
			raise TypeError('Variable trajectory can not be filtered with static selector_data')

	else:
		new_positions = [trajectory.get_positions(i)[selected_indices[i], :] for i in range(n_ts)]
		new_particle_attributes = [trajectory.get_particle_attributes(i)[selected_indices[i], :] for i in range(n_ts)]

	result = Trajectory(
		positions=new_positions,
		times=trajectory.times,
		particle_attribute_names=trajectory.particle_attribute_names,
		particle_attributes=new_particle_attributes,
		splat_times=new_splat_times
	)
	return result


def center_of_charge(tr):
	"""
	Calculates the center of charge of a charged particle cloud

	:param tr: Traject
	:type tr: trajectories vector from dict returned from readTrajectoryFile
	:return: vector of the spatial position of the center of mass
	:rtype: numpy.array
	"""
	nSteps = np.shape(tr)[2]
	coc = np.zeros([nSteps, 3])
	for i in range(nSteps):
		xMean = np.mean(tr[:, 0, i])
		yMean = np.mean(tr[:, 1, i])
		zMean = np.mean(tr[:, 2, i])
		coc[i, :] = np.array([xMean, yMean, zMean])
	return coc
