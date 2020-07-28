# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pylab as plt
import pandas as pd


def read_concentration_file(filename):
	"""
	Reads a IDSimF RS concentration log / result file (tabular file with concentrations over
	time)
	:param filename: name of the file to read
	:type filename: str
	:return: Pandas DataFrame with the imported data
	:rtype: Pandas DataFrame
	"""
	df = pd.read_csv(filename, skiprows=1, delimiter=';').iloc[:, :-1].rename(columns=lambda x: x.strip())
	return df


def plot_concentration_file(filename, time_range=(0, 1)):
	"""
	Simple plot method: Reads and plots a IDSimF RS concentration log / result file
	:param filename: name of the file to plot
	:type filename: str
	:return: None
	"""
	df = read_concentration_file(filename)
	time = df['Time']
	print(time.shape[0])
	n_lines = time.shape[0]
	t_indices = range(int(time_range[0] * n_lines), int(time_range[1] * n_lines))
	for i, colname in enumerate(df.columns[2:]):
		col_i = i + 2
		plt.plot(time.iloc[t_indices], df.iloc[t_indices, col_i], label=colname)

	plt.ylabel('number of particles')
	plt.xlabel('time (s)')
	plt.legend()
