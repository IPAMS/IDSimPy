# -*- coding: utf-8 -*-

import gzip
import itertools
import numpy as np
import h5py
from dataclasses import dataclass


def read_legacy_md_collisions_trajectory_file(trajectory_filename, framework):
	"""
	Reads a legacy molecular collisions trajectory file (ASCII file)

	:param trajectory_filename: File name of the file to read
	:type trajectory_filename: str
	:param framework: Framework which generated the collisions file ("IDSIMF" for an IDSimF result)
	:type framework: str
	:return: List with individual MD collisions
	:rtype: list of numpy arrays
	"""
	prefixes = ['###', ' ###']

	if trajectory_filename[-7:] == ".txt.gz":
		file_open_fct = gzip.open
	else:
		file_open_fct = open

	with file_open_fct(trajectory_filename, 'rt') as tf:
		result = []
		if(framework == "IDSIMF"):
			for key, group in itertools.groupby(tf, lambda line: line.startswith(prefixes[0])):
				if not key:
					group_lines = [x[:-1] for x in group]
					group_lines_splitted = [line.split(",") for line in group_lines]
					trajectory_data = np.asfarray(group_lines_splitted)
					result.append(trajectory_data)
		else:
			for key, group in itertools.groupby(tf, lambda line: line.startswith(prefixes[1])):
				if not key:
					group_lines = [x[:-1] for x in group]
					group_lines_splitted = [line.split() for line in group_lines]
					trajectory_data = np.asfarray(group_lines_splitted)
					result.append(trajectory_data)

		return result

@dataclass
class MDTrajectory:
	name: str
	n_atoms: list[int]
	column_names: list[str]
	trajectory: np.ndarray

def read_md_collisions_trajectory_file(trajectory_filename):
	"""
	Reads a fully resolved molecular collisions trajectory file (HDF5 file)
	"""

	if trajectory_filename[-3:] != ".h5":
		raise ValueError("Only HDF5 trajectory (.h5) files are supported")


	with h5py.File(trajectory_filename, 'r') as hdf5file:
		trajectories_group = hdf5file['MD_trajectories']


		result = []
		for ds_name in trajectories_group:
			ds = trajectories_group[ds_name]
			attribs = ds.attrs
			traj = MDTrajectory(ds_name, attribs['number_of_atoms'].tolist(), attribs['column_names'].tolist(), np.array(ds))
			result.append(traj)

	return result




