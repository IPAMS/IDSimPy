# -*- coding: utf-8 -*-

import gzip
import itertools
import numpy as np


def read_md_collisions_trajectory_file(trajectory_filename):
	"""
	Reads a molecular collisions trajectory file

	:param trajectory_filename: File name of the file to read
	:type trajectory_filename: str
	:return: List with individual MD collisions
	:rtype: list of numpy arrays
	"""
	count = 0
	prefixes = ['###']
	limit = 15e-10 / 1e-10

	if trajectory_filename[-8:] == ".txt.gz":
		file_open_fct = gzip.open
	else:
		file_open_fct = open

	with file_open_fct(trajectory_filename) as tf:
		result = []
		for key, group in itertools.groupby(tf, lambda line: line.startswith(tuple(prefixes))):
			if not key:
				group_lines = [x[:-1] for x in group]
				group_lines_splitted = [line.split(",") for line in group_lines]
				trajectory_data = np.asfarray(group_lines_splitted)
				result.append(trajectory_data)

		return result
