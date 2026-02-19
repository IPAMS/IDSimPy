# -*- coding: utf-8 -*-

import gzip
import itertools
import numpy as np
import h5py
from dataclasses import dataclass


def read_legacy_md_collisions_trajectory_file(trajectory_filename, framework):
	"""
	Reads a legacy molecular collisions tra file (ASCII file)

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
	Reads a fully resolved molecular collisions tra file (HDF5 file)
	"""

	if trajectory_filename[-3:] != ".h5":
		raise ValueError("Only HDF5 tra (.h5) files are supported")


	with h5py.File(trajectory_filename, 'r') as hdf5file:
		trajectories_group = hdf5file['MD_trajectories']


		result = []
		for ds_name in trajectories_group:
			ds = trajectories_group[ds_name]
			attribs = ds.attrs
			traj = MDTrajectory(ds_name, attribs['number_of_atoms'].tolist(), attribs['column_names'].tolist(), np.array(ds))
			result.append(traj)

	return result


def export_trajectory_as_ovito_xyz(tra: MDTrajectory, file_name: str, scale= 1e10):
	"""
	Exports tra as an ovito xyz file.

	:param tra: Trajectory object to export
	:param file_name: File name of the ovito xyz file to write into
	:param scale: Scaling factor between the length scale in the trajectories and the ovito XYZ file
		(default is 1 Angström as unit in the xyz file)
	"""
	n_atoms_total= sum(tra.n_atoms)
	n_timesteps = tra.trajectory.shape[0]
	molec_names = [f'molecule_{i}' for i in range(tra.n_atoms[0])]
	bg_names = [f'bg_{i}' for i in range(tra.n_atoms[1])]


	with open(file_name, 'w') as xyz_file:
		for ti in range(n_timesteps):
			xyz_file.write(f'{n_atoms_total}\n')
			xyz_file.write(f'time step {ti}\n')

			col_i = 2
			for ai in range(tra.n_atoms[0]):
				xyz_file.write(f'{molec_names[ai]}    {tra.trajectory[ti, col_i]*scale}    {tra.trajectory[ti, col_i+1]*scale}    {tra.trajectory[ti, col_i+2]*scale}    {0}\n')
				col_i += 3

			for ai in range(tra.n_atoms[1]):
				xyz_file.write(f'{bg_names[ai]}    {tra.trajectory[ti, col_i]*scale}    {tra.trajectory[ti, col_i+1]*scale}    {tra.trajectory[ti, col_i+2]*scale}    {1}\n')
				col_i += 3







