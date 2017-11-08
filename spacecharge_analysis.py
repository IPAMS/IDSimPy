import numpy as np
import matplotlib as mpl
from matplotlib import animation
import pylab as plt
import json
import gzip
import btree_analysis.qitsim_analysis as lq


def ion_radius_from_trajectories(positions, radius_center=[]):
	n_steps = np.shape(positions)[2]
	n_ions = np.shape(positions)[0]
	result = np.zeros([n_ions, n_steps])
	if len(radius_center) == 0:
		radius_center = np.zeros([n_steps, 3])
	for s in range(n_steps):
		result[:, s] = np.sqrt(
			(radius_center[s, 0] - positions[:, 0, s]) ** 2.0 +
			(radius_center[s, 1] - positions[:, 1, s]) ** 2.0 +
			(radius_center[s, 2] - positions[:, 2, s]) ** 2.0)

	return result


def RF_force_magnitude(ap):
	n_steps = np.shape(ap)[2]
	n_ions = np.shape(ap)[0]
	result = np.zeros([n_ions, n_steps])
	for s in range(n_steps):
		result[:, s] = np.sqrt(ap[:, 0, s] ** 2.0 + ap[:, 1, s] ** 2.0 + ap[:, 2, s] ** 2.0)

	return (result)


def space_charge_force_magnitude(ap):
	n_steps = np.shape(ap)[2]
	n_ions = np.shape(ap)[0]
	result = np.zeros([n_ions, n_steps])
	for s in range(n_steps):
		result[:, s] = np.sqrt(ap[:, 3, s] ** 2.0 + ap[:, 4, s] ** 2.0 + ap[:, 5, s] ** 2.0)

	return (result)

def space_charge_force_z_direction(ap):
	n_steps = np.shape(ap)[2]
	n_ions = np.shape(ap)[0]
	result = np.zeros([n_ions, n_steps])
	for s in range(n_steps):
		result[:, s] = ap[:, 5, s]

	return (result)

def space_charge_force_average(ap):
	n_steps = np.shape(ap)[2]
	result = np.zeros([3, n_steps])
	for s in range(n_steps):
		result[:, s] = np.mean(ap[:, 3:6, s], axis=0)

	return result


def ion_ensemble_average(dat):
	"""Calculates the averaged time series for an ion ensemble"""
	result = np.mean(dat, axis=0)
	return result


def average_electric_force_time_series(projectName, mass):
	tj = lq.read_trajectory_file(projectName + "_trajectories.json.gz")

	i_pos = tj["positions"]
	i_masses = tj["masses"]
	i_ap = tj["additional_parameters"]

	i_pos_mfiltered = lq.filter_mass(i_pos, i_masses, mass)
	i_ap_mfiltered = lq.filter_mass(i_ap, i_masses, mass)

	rf_force_mag = RF_force_magnitude(i_ap_mfiltered)
	sc_force_mag = space_charge_force_magnitude(i_ap_mfiltered)

	rf_avg_mag = ion_ensemble_average(rf_force_mag)
	sc_avg_mag = ion_ensemble_average(sc_force_mag)

	sc_avg = space_charge_force_average(i_ap_mfiltered)

	center_of_charge = lq.center_of_charge(i_pos_mfiltered)
	cloud_radius = ion_ensemble_average(ion_radius_from_trajectories(i_pos_mfiltered, radius_center=center_of_charge))

	return (rf_avg_mag, sc_avg_mag, sc_avg, center_of_charge, cloud_radius, tj["times"])


def compare_average_electric_force(projectName, mass_1, mass_2, t_steps=None):
	rf_avg_mag_1, sc_avg_mag_1, sc_avg_1, coc_1, cloud_radius_1, times_1 = average_electric_force_time_series(
		projectName, mass_1)
	rf_avg_mag_2, sc_avg_mag_2, sc_avg_2, coc_2, cloud_radius_2, times_2 = average_electric_force_time_series(
		projectName, mass_2)

	if not t_steps:
		t_steps = np.arange(len(times_1))

	fig_size = [11, 12]

	plt.figure(figsize=fig_size)
	plt.subplot(6, 1, 1)
	plt.plot(times_1[t_steps], rf_avg_mag_1[t_steps])
	plt.plot(times_2[t_steps], rf_avg_mag_2[t_steps])
	plt.ylabel("rf avg mag")
	plt.subplot(6, 1, 2)
	plt.plot(times_1[t_steps], sc_avg_mag_1[t_steps])
	plt.plot(times_2[t_steps], sc_avg_mag_2[t_steps])
	plt.ylabel("sc avg mag")
	plt.subplot(6, 1, 3)
	plt.plot(times_1[t_steps], sc_avg_1[0, t_steps], label="x")
	plt.plot(times_1[t_steps], sc_avg_1[1, t_steps], label="y")
	plt.plot(times_1[t_steps], sc_avg_1[2, t_steps], label="z")
	plt.ylabel("sc avg (1)")
	plt.legend()
	plt.subplot(6, 1, 4)
	plt.plot(times_1[t_steps], sc_avg_2[0, t_steps], label="x")
	plt.plot(times_1[t_steps], sc_avg_2[1, t_steps], label="y")
	plt.plot(times_1[t_steps], sc_avg_2[2, t_steps], label="z")
	plt.ylabel("sc avg (2)")
	plt.legend()
	plt.subplot(6, 1, 5)
	plt.plot(times_1[t_steps], coc_1[t_steps, 2])
	plt.plot(times_2[t_steps], coc_2[t_steps, 2])
	plt.ylabel("z pos avg")
	plt.subplot(6, 1, 6)
	plt.plot(times_1[t_steps], cloud_radius_1[t_steps])
	plt.plot(times_2[t_steps], cloud_radius_2[t_steps])
	plt.ylabel("avg cloud radius")
	plt.subplots_adjust(hspace=0.35)
	# plt.tight_layout()


def animate_simulation_z_vs_x_spacecharge_density(dat, nFrames, interval,
												fileMode='video',
												analysis_mode='space_charge_magnitude',
												sLim=3, nBins=100,
												h_min=1e-17,
												h_max=1e-15,
												alphaFactor=1, colormap=plt.cm.viridis,
												annotateString=""):
	"""
	Space charge density plot
	"""

	times = dat["times"]
	i_pos = dat["positions"]
	i_ap = dat["additional_parameters"]

	rf_force = RF_force_magnitude(i_ap)
	sc_force = space_charge_force_magnitude(i_ap)
	sc_force_z_dir = space_charge_force_z_direction(i_ap)

	if fileMode == 'video':
		fig = plt.figure(figsize=[10, 10])
	elif fileMode == 'singleFrame':
		fig = plt.figure(figsize=[6, 6])

	xedges = np.linspace(-sLim, sLim, nBins)
	zedges = np.linspace(-sLim, sLim, nBins)
	H = np.random.rand(len(xedges), len(zedges))
	ax = plt.axes(ylim=(zedges[0], zedges[-1]), xlim=(xedges[0], xedges[-1]))

	im1 = ax.imshow(H, interpolation='nearest', origin='low', alpha=1, vmin=0, vmax=10, cmap="Reds",
					extent=[xedges[0], xedges[-1], zedges[0], zedges[-1]])

	text_time = ax.annotate("TestText", xy=(0.02, 0.96), xycoords="figure fraction",
							horizontalalignment="left",
							verticalalignment="top",
							fontsize=20);

	plt.xlabel("r (mm)")
	plt.ylabel("z (mm)")
	fillChannel = np.ones([len(xedges) - 1, len(zedges) - 1])

	def animate(i):
		tsNumber = i * interval
		x = i_pos[:, 0, tsNumber]
		z = i_pos[:, 2, tsNumber]

		if analysis_mode == 'space_charge_magnitude':
			weights = sc_force[:, tsNumber]
		elif analysis_mode == 'space_charge_z_direction':
			weights = np.abs(sc_force_z_dir[:, tsNumber])
		elif analysis_mode == 'rf_force':
			weights = rf_force[:, tsNumber]


		h, xedges2, zedges2 = np.histogram2d(z, x, bins=(xedges, zedges), weights=weights)
		h_dens, xedges2, zedges2 = np.histogram2d(z, x, bins=(xedges, zedges))
		# h_max = 1e-15 #np.max(h)
		# h_min = 1e-17 #np.min(h)

		abs_dens = h_dens / np.max(h_dens)

		img_data_RGB = colormap((h - h_min) / h_max)
		#nonzero = np.nonzero(abs_dens > 0)
		#dens = np.zeros(np.shape(rel_dens))
		#dens[nonzero] = 1.0
		img_data_RGB[:, :, 3] = abs_dens * alphaFactor

		im1.set_array(img_data_RGB)
		text_time.set_text("t=" + str(times[tsNumber]) + u"µs" + " " + annotateString)

		return im1

	# call the animator.  blit=True means only re-draw the parts that have changed.
	if fileMode == 'video':
		anim = animation.FuncAnimation(fig, animate, frames=nFrames, blit=False)
		return (anim)
	elif fileMode == 'singleFrame':
		animate(nFrames)
		return (fig)
