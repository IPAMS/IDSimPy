# -*- coding: utf-8 -*-

"""
ion_cloud_generation: Generation of ion cloud initialization files
"""

import numpy as np
from btree.constants import *


def write_cloud_file(ion_cloud, filename):
	"""
	Writes an ion cloud to an ion cloud file
	:param ion_cloud: an np.array with the columns:
		[x pos, y pos, z pos, x velo, y velo, z velo, charge (in elem. charges), mass (in amu)]
	:param filename: name of the file in which the tabular ion cloud data is written to
	"""
	with open(filename, 'w') as file:
		for i_ion in range(np.shape(ion_cloud)[0]):
			line = ion_cloud[i_ion, :]
			for v in line:
				file.write(str(v) + ';')

			file.write('\n')


def velo_from_kinetic_energy(ke_eV, mass_amu):
	ke_J = ke_eV * JOULE_PER_EV
	m_kg = mass_amu * KG_PER_AMU
	v = np.sqrt(2.0 * ke_J / m_kg)
	return v


def random_sphere(radius, n_samples):
	# http://mathworld.wolfram.com/SpherePointPicking.html

	z = 2 * np.random.rand(n_samples) - 1  # uniform in -1, 1
	t = 2 * np.pi * np.random.rand(n_samples)  # uniform in 0, 2*pi
	x = np.sqrt(1 - z ** 2) * np.cos(t)
	y = np.sqrt(1 - z ** 2) * np.sin(t)
	coords = np.transpose(np.vstack([x, y, z])) * radius
	return coords


def set_kinetic_energy_in_z_dir(ion_cloud, ke):
	ion_cloud[:, 5] = velo_from_kinetic_energy(ke, ion_cloud[:, 7])
	return ion_cloud


def add_thermalized_kinetic_energy(ion_cloud, ke):
	n_ions = np.shape(ion_cloud)[0]
	thermal_velo_mag = velo_from_kinetic_energy(ke, ion_cloud[:, 7])
	thermal_velo = random_sphere(np.transpose([thermal_velo_mag]), n_ions)

	ion_cloud[:, 3:6] = ion_cloud[:, 3:6] + thermal_velo

	return ion_cloud


def define_xy_grid(n_x, n_y, w_x, w_y, o_x, o_y, mass):
	"""
	Defines a grid in the x-y direction (z=0)
	(grid is from -width to width)
	:param int n_x: ions in x direction
	:param int n_y: ions in y direction
	:param float w_x: width in x direction
	:param float w_y: width in y direction
	:param float o_x: offset in x direction
	:param float o_y: offset in y direction
	:param float mass: the mass of the ions in the grid
	:return: the ion cloud in an np.array with the structure as expected by write_cloud_file
	"""
	X = np.linspace(-w_x, w_x, n_x) + o_x
	Y = np.linspace(-w_y, w_y, n_y) + o_y

	result = np.zeros([n_x * n_y, 8])

	i = 0
	for x in X:
		for y in Y:
			result[i, :] = np.array([x, y, 0, 0, 0, 0, 1, mass])
			i += 1

	return result


def define_origin_centered_block(n_ions, w_x, w_y, w_z, mass):
	"""
	Defines a block of random ions around the coordinate system origin
	:param int n_ions: the number of ions in the block
	:param float w_x: the width in x direction
	:param float w_y: the width in y direction
	:param float w_z: the width in z direction
	:param float mass: the mass of the ions (in amu)
	:return: the ion cloud in an np.array with the structure as expected by write_cloud_file
	"""
	X = np.random.rand(n_ions, 3)
	V = np.zeros([n_ions, 3])
	M = np.zeros([n_ions, 1]) + mass
	C = np.zeros([n_ions, 1]) + 1

	X[:, 0] = (X[:, 0] * (2 * w_x)) - w_x
	X[:, 1] = (X[:, 1] * (2 * w_y)) - w_y
	X[:, 2] = (X[:, 2] * (2 * w_z)) - w_z

	result = np.hstack([X, V, C, M])
	return result


def define_cylinder_z_dir(n_ions, r, z, charge, mass):
	"""
	Defines a cylinder with the cylinder axis parallel to the z-axis and the center of one face of the cylinder on
	the origin of the coordinate system filled with random ions

	The disk cros section must be filled uniformly with ions, see
	http://mathworld.wolfram.com/DiskPointPicking.html for details and argument how to do this

	:param int n_ions: The number of ions in the cylinder
	:param float r: the radius of the cylinder
	:param z:
	:param charge:
	:param mass:
	:return:
	"""
	R = np.sqrt(np.random.rand(n_ions, 1)) * r
	phi = np.random.rand(n_ions, 1) * 2 * np.pi
	Z = np.random.rand(n_ions, 1) * z


	X = np.cos(phi) * R
	Y = np.sin(phi) * R

	V = np.zeros([n_ions, 3])
	C = np.zeros([n_ions, 1]) + charge
	M = np.zeros([n_ions, 1]) + mass

	result = np.hstack([X, Y, Z, V, C, M])
	return result


def write_xy_slice(n_ions, masses, w_x, w_y, filename):
	"""
	Writes a random slice of ions in xy-Direction to a file
	:param list[int] n_ions: a list of numbers of ions with ions in the slice
	:param list[float] masses: list of masses (in amu)
	:param float w_x: the width in x direction (in m)
	:param float w_y: the width in y direction (in m)
	:param str filename: the name of the file to write the resulting ion cloud to
	"""
	ion_cloud = []

	for i_m in range(0, len(masses)):

		if len(ion_cloud) == 0:
			ion_cloud = define_origin_centered_block(n_ions[i_m], w_x, w_y, 0.1 / 1000.0, masses[i_m])
		else:
			ion_cloud = np.vstack([
				ion_cloud,
				define_origin_centered_block(n_ions[i_m], w_x, w_y, 0.1 / 1000.0, masses[i_m])
			])

	write_cloud_file(ion_cloud, filename)
