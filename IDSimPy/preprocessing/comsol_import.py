# -*- coding: utf-8 -*-

import numpy as np
import gzip
import io


def import_comsol_3d_csv_grid(filename):
	"""
	Imports a csv file with multiple 3d scalar fields exported from comsol.

	  Result dictionary contains:

	    * grid_points: Array of vectors with the positions of the grid points in the x,y,z directions
	    * meshgrid: Array of meshgrid matrices as defined by numpy meshgrid
	    * fields: Array of scalar fields, which are dictionaries with a name and the scalar data in 'data'

	:param str filename: the file name to import
	:return: Result dictionary as defined above
	"""

	if filename[-3:] == ".gz":
		with gzip.open(filename, 'rt') as f:
			raw_dat_str = f.read()
	else:
		with open(filename, 'rt') as f:
			raw_dat_str = f.read()

	raw_dat_tokenized = raw_dat_str.split("% Data\n")

	raw_dat_header = raw_dat_tokenized[0]
	raw_dat_fields = raw_dat_tokenized[1:]

	grid_points = parse_spatial_dimensions(raw_dat_header)
	grid_dims = [len(x) for x in grid_points]

	fields = [parse_comsol_csv_data_chunk(raw_field, grid_dims) for raw_field in raw_dat_fields]
	meshgrid = np.meshgrid(grid_points[0], grid_points[1], grid_points[2], indexing='ij')

	return {"grid_points": grid_points, "meshgrid": meshgrid, "fields": fields}


# ---------------- utlilty functions / methods ----------------------


def parse_grid_vector(vec_str, delimiter=','):
	"""
	Parses a vector with spatial grid positions from a grid vector line from a comsol csv header.

	:param str vec_str: A string containing the header line
	:param str delimiter: the delimiter in the vector line
	:return: numpy array with the vector values
	"""
	vec_str_spl = vec_str.split(delimiter)
	vec_len = len(vec_str_spl)
	vec = np.zeros([vec_len])
	for i in range(vec_len):
		vec[i] = float(vec_str_spl[i])

	return (vec)


def parse_comsol_csv_data_chunk(raw_chunk, dims):
	"""
	Parses a raw data chunk (given as string) and returns a numpy array with the data values.

	:param str raw_chunk: The string containing the data chunk
	:param dims: 3 dim array with the number of points in the spatial (x,y,z) dimensions
	:type dims: tuple or numpy.Array
	:return: Dictionary with the field_name and the data of the field
	"""
	(field_name, raw_field) = raw_chunk.split('\n', 1)
	field_raw_dat = np.genfromtxt(io.BytesIO(raw_field.encode()), delimiter=',')

	x_len, y_len, z_len = dims
	field_dat = np.zeros([x_len, y_len, z_len])
	for k in range(z_len):
		for j in range(y_len):
			l_pos = j + k * y_len
			field_dat[:, j, k] = field_raw_dat[l_pos, :]

	return {"name": field_name, 'data': field_dat}


def parse_spatial_dimensions(header):
	"""
	Parses and returns the spatial dimensions of a comsol csv field file from the header of the
	comsol csv file.

	:param str header: The header (given as string)
	:return: Tuple of three vectors with the spatial point grid positions in x,y,z direction
	"""
	header_lines = header.split("\n")
	x_grid_str = header_lines[-4]
	y_grid_str = header_lines[-3]
	z_grid_str = header_lines[-2]

	x_grid_vec = parse_grid_vector(x_grid_str)
	y_grid_vec = parse_grid_vector(y_grid_str)
	z_grid_vec = parse_grid_vector(z_grid_str)

	return (x_grid_vec, y_grid_vec, z_grid_vec)
