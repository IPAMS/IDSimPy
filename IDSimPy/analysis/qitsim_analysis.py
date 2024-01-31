# -*- coding: utf-8 -*-

# NOTE: This analysis module is outdated and has to be updated. Many things will not work currently.
# TODO: Update QIT simulation analysis

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import commentjson
import os
from matplotlib import animation
from .constants import *
from . import trajectory as tra

################## Utility Methods  ######################
def qit_stability_parameters(
		m_amu,  # = 80    # ion mass [Da]
		Vrf,  # = 300    # voltage 0-p
		frf_Hz,  # rf frequency [Hz]
		r0=0.01  # ring electrode radius [m])
):

#####################
	# calculate secular frequency and period of the ion
	m_kg = m_amu * KG_PER_AMU
	frf_rad = 2*np.pi*frf_Hz
	fion = (np.sqrt(2) * ELEMENTARY_CHARGE * Vrf) / (4 * np.square(np.pi) * np.square(r0) * frf_Hz * m_kg)

	# calculate q value for the ion
	qz = 8 * ELEMENTARY_CHARGE * Vrf / (m_kg * 2 * np.square(r0) * np.square(frf_rad))

	# calculate cut off voltage for the ion, stability limit is q = 0.908
	Vcutoff = Vrf / qz * 0.908

	# calculate low mass cut off (lmco), satbility limit is q = 0.908
	lmco = (m_amu * qz / 0.908)

	# calculate pseudopotential (Dehmelt) for the ion
	D = qz * Vrf / 8

	return {"fion": fion, "qz": qz, "Vcutoff": Vcutoff, "lmco": lmco, "D": D}


################## Data Read Methods ######################

def read_QIT_conf(conf_file_name):
	"""
	Reads and parses the QIT simulation configuration file
	:param conf_file_name: the filename of the simulation configuration file
	:return: a parsed dictionary with the configuration parameters
	"""
	with open(conf_file_name) as jsonFile:
		confJson = commentjson.load(jsonFile)

	return (confJson)


def read_FFT_record(project_path):
	"""
	Reads a fft record file, which contains the exported mirror charge current on some detector electrodes

	:param project_path: path of the simulation project
	:return: two vectors: the time and simulated mirror charge from the fft file
	"""
	dat = np.loadtxt(project_path + "_fft.txt")
	t = dat[:, 0]
	z = dat[:, 1:]
	return t,z

def read_ions_inactive_record(project_path):
	"""
	Reads an ions inactive record file, which contains the number of inactive ions over the time

	:param project_path: path of the simulation project
	:return: two vectors: the time and the number of inactive ions
	"""
	dat = np.loadtxt(project_path + "_ionsInactive.txt")
	t = dat[:, 0]
	ions_inactive = dat[:, 1]
	return t, ions_inactive

def read_center_of_charge_record(project_path):
	"""
	Reads a center of charge (coc) record file, which contains the mean position of the charged particle cloud over the time

	:param project_path: path of the simulation project
	:return: the time vector and the mean positions of the charged particle cloud from the coc file in a two dim matrix
	"""
	dat = np.loadtxt(project_path + "_averagePosition.txt")
	t = dat[:, 0]
	pos = dat[:, 1:]
	return t,pos


def read_and_analyze_stability_scan(project_path, t_range=(0, 1)):
	"""
	Reads and analyzes an ions inactive record.

	:param project_path: path of the simulation project
	:type project_path: str
	:param t_range: a time
	:type t_range: tuple of floats with t_start, t_stop
	:return: a pandas dataframe with
		the trap RF amplitude (V_rf),
		the time,
		the number of inactive ions
		and the differential number of inactive ions (ion termination per timestep)
	"""
	time, inactive_ions = read_ions_inactive_record(project_path)

	with open(project_path + "_conf.json") as jsonFile:
		conf_json = commentjson.load(jsonFile)

	V_rf_start = conf_json["V_rf_start"]
	V_rf_end = conf_json["V_rf_end"]
	V_rf = np.linspace(V_rf_start, V_rf_end, len(time))

	n_samples = len(time)
	i_start = int(t_range[0] * n_samples)
	i_stop = int(t_range[1] * n_samples)
	i_range = np.arange(i_start, i_stop)

	df = pd.DataFrame({"time": time, "inactive_ions": inactive_ions, "V_rf": V_rf})
	df_slice = df.loc[i_range]
	ions_diff = np.append(np.diff(df_slice["inactive_ions"]), 0)
	df_slice["ions_diff"] = ions_diff

	return (df_slice)

################## Data Processing Methods ######################

def reconstruct_transient_from_trajectories(dat):
	"""
	Reconstructs a QIT transient (center of charge position in detection, z, direction) from trajectory data
	:param dat: imported trajectories object
	:type dat: dict returned from readTrajectoryFile
	:return: two vectors: a time vector and the center of charge in z direction
	"""
	times = dat["times"] / 1
	nSteps = len(times)
	tr = dat["positions"]
	result = np.zeros([nSteps, 1])
	for i in range(nSteps):
		r = np.sqrt(tr[:, 0, i] ** 2.0 + tr[:, 1, i] ** 2.0 + tr[:, 2, i] ** 2.0)
		valid = np.nonzero(r < 5)
		result[i, 0] = np.mean(tr[valid, 2, i])
	return times,result


def calculate_FFT_spectrum(t, z):
	"""
	Calculates the spectrum via fft from a given transient
	:param t: the time vector of the transient
	:param z: the mean position of the charged particle cloud in detection direction
	:return: two vectors: the frequency and the intensity vectors
	"""
	n = len(t)  # length of the signal
	Fs = float(n) / (t[-1] - t[0])
	Y = np.fft.fft(z,axis=0) / n  # fft computing and normalization
	Y = Y[range(n // 2)]

	k = np.arange(n)
	T = n / Fs
	frq = k / T  # two sides frequency range
	frq = frq[range(n // 2)]  # one side frequency range

	return frq,abs(Y)


################## High Level Simulation Project Processing Methods ######################

def analyse_FFT_sim(project_path, freq_start=0.0, freq_stop=1.0, amp_mode="lin",
                    load_mode="fft_record", title=None, result_path=None, plot_result='export',
                    figsize=(20, 5), titlepos=(0.1, 0.94)):
	"""
	Analyses a transient of a QIT simulation and calculates/plots the spectrum from it

	:param project_path: base path of a qit simulation
	:param freq_start: start of the plotted frequency window (normalized)
	:param freq_stop: stop/end of the plotted frequency window (normalized)
	:param amp_mode: fft spectrum plot mode, linear or logarithmic, options: "lin" or "log"
	:param load_mode: load a recorded fft record file
	("fft_record") or reconstruct transient from trajectories ("reconstruct_from_trajectories")
	:param title: Optional title string (if None a title is autogenerated)
	:param result_path: Optional export path for the result plots
	:param plot_result: Controls if a plot is generated and exported, options are
		+ 'export' a plot is generated, exported and returned in the return dictionary
		+ 'plot' a plot is generated and and returned in the return dictionary
	:return: a dictionary with the frequencies and amplitudes of the spectrum,
	the time vector and amplitude of the transient and the Figure object of the plot (if applicable)
	"""
	with open(project_path + "_conf.json") as jsonFile:
		confJson = commentjson.load(jsonFile)

	if load_mode == "fft_record":
		t, z = read_FFT_record(project_path)
	elif load_mode == "legacy_fft_record":
		dat = np.loadtxt(project_path + "_fft.txt")
		t = dat[:, 0]
		z = dat[:, 3]
	elif load_mode == "center_of_charge_record":
		t, z = read_center_of_charge_record(project_path)
	elif load_mode == "reconstruct_from_trajectories":
		tr = tra.read_trajectory_file(project_path + "_trajectories.json.gz")
		t, z = reconstruct_transient_from_trajectories(tr)
		t = t*1e-6
		print(t)

	frq, Y = calculate_FFT_spectrum(t, z)

	freqsPl= range(int(len(frq) * freq_start), int((len(frq) * freq_stop)))

	fig = None
	if plot_result == 'export' or plot_result == 'plot':
		fig, ax = plt.subplots(1, 2, figsize=figsize, dpi=50)
		ax[0].plot(t, z, 'C1')
		ax[0].set_xlabel('Time (s)')
		ax[0].set_ylabel('Amplitude (arb.)')

		#ax[1].semilogy(frq[freqsPl],abs(Y[freqsPl]),'r') # plotting the spectrum
		if amp_mode == "lin":
			ax[1].plot(frq[freqsPl]/1000,abs(Y[freqsPl]), 'C0') # plotting the spectrum
		elif amp_mode == "log":
			ax[1].semilogy(frq[freqsPl]/1000, abs(Y[freqsPl]), 'C0')  # plotting the spectrum logarithmic
		ax[1].set_xlabel('Freq (kHz)')
		ax[1].set_ylabel('Amplitude (arb.)')

		projectName = project_path.split("/")[-1]

		print(f'Title {title}')
		if title is None:
			if "partial_pressures_Pa" in confJson:
				background_pressure_str = str(confJson["partial_pressures_Pa"])
			else:
				background_pressure_str = str(confJson["background_gas_pressure_Pa"])

			if "collision_gas_masses_amu" in confJson:
				collision_gas_mass_str = str(confJson["collision_gas_masses_amu"])
			else:
				collision_gas_mass_str = str(confJson["collision_gas_mass_amu"])

			titlestring = projectName + f" p: {background_pressure_str} Pa, coll. gas mass: {collision_gas_mass_str} u"

			if "space_charge_factor" in confJson:
				titlestring = titlestring + ", space charge factor:"+'%6g' % (confJson["space_charge_factor"])

			fig.suptitle(titlestring, x=titlepos[0], y=titlepos[1],  fontsize=17, horizontalalignment='left')
		else:
			fig.suptitle(title, x=titlepos[0], y=titlepos[1],  fontsize=17, horizontalalignment='left')

		if plot_result == 'export':
			if result_path:
				result_project_path = os.path.join(result_path, projectName)
			else:
				result_project_path = projectName

			plt.tight_layout()
			plt.savefig(result_project_path + "_fftAnalysis.pdf", format="pdf")
			plt.savefig(result_project_path + "_fftAnalysis.png", format="png", dpi=180)

	return{"freqs":frq[freqsPl], "amplitude":abs(Y[freqsPl]), "time":t, "transient":z, "figure":fig}


def analyze_stability_scan(project_path, window_width=0, t_range=[0, 1], result_path=None):
	with open(project_path + "_conf.json") as jsonFile:
		confJson = commentjson.load(jsonFile)

	V_rf_start = confJson["V_rf_start"]
	V_rf_end = confJson["V_rf_end"]

	projectName = project_path.split("/")[-1]

	titlestring = projectName + " p " + str(confJson["partial_pressures_Pa"]) + " Pa, c. gas " + str(
		confJson["collision_gas_masses_amu"]) + " amu, "
	titlestring = titlestring + "spc:" + '%4g, ' % (confJson["space_charge_factor"])
	titlestring = titlestring + "RF: " + str(V_rf_start) + " to " + str(V_rf_end) + " V @ " + str(
		confJson["f_rf"] / 1000.0) + " kHz"

	t_end = confJson["sim_time_steps"]*confJson["dt"]
	dUdt = (V_rf_end - V_rf_start) / t_end
	titlestring = titlestring + (' (%2g V/s), ' % (dUdt))
	if confJson['excite_mode'] != 'off':
		titlestring = titlestring + str(confJson["excite_potential"]) + " V exci."

	project = [[project_path, ""]]
	plot_fn = project_path
	if result_path:
		plot_fn = os.path.join(result_path,projectName)

	plot_fn += "_ionEjectionAnalysis"

	analyze_stability_scan_comparison(project, plot_fn, window_width=window_width, titlestring=titlestring,t_range=t_range)


def analyze_stability_scan_comparison(projects, plot_fn, mode="absolute", window_width=0, titlestring="",t_range=[0,1]):
	fig = plt.figure(figsize=[15, 5])
	ax1 = fig.add_subplot(1, 2, 1)
	ax2 = fig.add_subplot(1, 2, 2)
	for pr in projects:
		dat = read_and_analyze_stability_scan(pr[0],t_range=t_range)

		if mode == "normalized":
			dat["inactive_ions"] = dat["inactive_ions"] / np.max(dat["inactive_ions"])
			dat["ions_diff"] = dat["ions_diff"] / np.max(dat["ions_diff"])

		if window_width > 0:
			dat = dat.rolling(window_width).mean()

		if len(pr) > 2:
			dat["V_rf"] = dat["V_rf"] + pr[2]

		ax1.plot(dat["time"], dat["inactive_ions"])
		ax2.plot(dat["V_rf"], dat["ions_diff"], alpha=0.9, label=pr[1])

	ax1.set_xlabel("time (ms)")
	ax2.set_xlabel("$U_{rf}$ (V)")

	if mode == 'absolute':
		ax1.set_ylabel("# ions ejected")
		ax2.set_ylabel("ion ejection rate")
	if mode == 'normalized':
		ax1.set_ylabel("fraction of ions ejected")
		ax2.set_ylabel("normalized ion ejection rate")

	if len(projects)>1:
		plt.legend()

	if len(titlestring)>1:
		plt.figtext(0.1, 0.94, titlestring, fontsize=12)

	plt.savefig(plot_fn + ".pdf", format="pdf")
	plt.savefig(plot_fn + ".png", format="png", dpi=100)


def center_of_charges_from_simulation(dat, species_masses, t_range=[]):
	"""
	Calculates the center of charges for two species, defined by their mass, from a simulation
	:param dat: imported trajectories object
	:type dat: dict returned from readTrajectoryFile
	:param species_masses: two element list with two particle masses
	:type species_masses: list
	:param t_range: a list / vector with time step indices to export the center of charges for
	:type t_range: list
	:return: dictionary with the time vector in "t", and the center of species a and b and of the whole
	ion cloud in dat in "cocA", "cocB" and "cocAll"
	:rtype dictionary
	"""
	masses = dat["masses"]
	times = dat["times"]

	if len(t_range)==0:
		t_range=range(len(times))
	else:
		times=times[t_range]

	tr = dat["positions"][:,:, t_range]

	cocA = tra.center_of_charge(tra.filter_parameter(tr, masses, species_masses[0]))
	cocB = tra.center_of_charge(tra.filter_parameter(tr, masses, species_masses[1]))
	cocAll = tra.center_of_charge(tr)

	return{"t":times,"cocA":cocA,"cocB":cocB,"cocAll":cocAll}

def plot_average_z_position(sim_projects, masses, compressed=True):
	"""
	Plots a comparison plot of averaged z-positions for individual masses in a qit simulation

	:param list[str] sim_projects: a list of simulation project names
	(the configuration file of the simulation is expected to have the same base-name)
	:param list[float] masses: a list of masses to draw the plot for
	:param bool compressed: flag if trajectory file is gzip compressed
	"""
	n_projects = len(sim_projects)
	fig_size = [12, 2 * n_projects]
	plt.figure(figsize=fig_size)

	if compressed:
		file_ext =  "_trajectories.json.gz"
	else:
		file_ext = "_trajectories.json"

	print(n_projects)
	for si in range(n_projects):
		tj = tra.read_trajectory_file(sim_projects[si] + file_ext)
		conf = read_QIT_conf(sim_projects[si] + "_conf.json")

		i_pos = tj["positions"]
		i_masses = tj["masses"]
		# i_ap = tj["particle_attributes"]
		times = tj["times"]

		plt.subplot(n_projects, 1, si + 1)
		for mass in masses:
			i_pos_mfiltered = tra.filter_parameter(i_pos, i_masses, mass)
			# i_ap_mfiltered = lq.filter_parameter(i_ap, i_masses, mass)
			coc = tra.center_of_charge(i_pos_mfiltered)
			plt.plot(times, coc[:, 2], label=str(mass))

		title_str = sim_projects[si] + ", "
		if "V_rf_start" in conf:
			title_str += str(conf["V_rf_start"])
		else:
			title_str += str(conf["V_rf"])

		title_str += "V " + str(conf["f_rf"] / 1000) + "kHz RF, scf=" + str(conf["space_charge_factor"])
		plt.title(title_str)
		plt.legend()
	plt.xlabel("t (microseconds)")
	plt.tight_layout()

def animate_simulation_center_of_masses_z_vs_x(dat, masses, n_frames, interval, frame_length, zlim=3, xlim=0.1):
	"""
	Animate the center of charges of the ion clouds in a QIT simulation in a z-x projection. The center of charges
	are rendered as a trace with a given length (in terms of simulation time steps)

	:param dat: imported trajectories object
	:type dat: dict returned from readTrajectoryFile
	:param masses: two element list with two particle masses to render the center of charges for
	:type masses: list
	:param n_frames: number of frames to export
	:param interval: interval in terms of time steps in the input data between the animation frames
	:param frame_length: length of the trace of the center of charges (in terms of simulation time steps)
	:param zlim: limits of the rendered spatial section in z direction
	:param xlim: limits of the rendered spatial section in x direction
	:return: an animation object with the rendered animation
	"""
	fig = plt.figure()
	ax = plt.axes(ylim=(-zlim, zlim), xlim=(-xlim, xlim))
	lA, = ax.plot([], [], lw=2)
	lB, = ax.plot([], [], lw=2)
	lall, = ax.plot([], [], lw=2)

	# animation function.  This is called sequentially
	def animate(i):
		#x = np.linspace(0, 2, 1000)
		#y = np.sin(2 * np.pi * (x - 0.01 * i))
		tRange=np.arange(0 + i * interval, frame_length + i * interval)
		d = center_of_charges_from_simulation(dat, masses, t_range=tRange)
		z = d["cocA"][:,2]
		x = d["cocA"][:,0]
		lA.set_data(x,z)
		z = d["cocB"][:,2]
		x = d["cocB"][:,0]
		lB.set_data(x,z)
		z = d["cocAll"][:,2]
		x = d["cocAll"][:,0]
		lall.set_data(x,z)

		return lA,lB,lall

	# call the animator.  blit=True means only re-draw the parts that have changed.
	anim = animation.FuncAnimation(fig, animate,
	                               frames=n_frames, blit=True)
	# call our new function to display the animation
	return(anim)

### Phase Space Analysis ###

def plot_phase_space_frame(tr, timestep):
	print(len(tr['times']))
	pos = tr['positions']
	ap = tr['particle_attributes']
	print(np.shape(pos))
	plt.subplot(1, 2, 1)
	plt.scatter(pos[:, 0, timestep], ap[:, 0, timestep], s=0.5, alpha=0.5)
	plt.subplot(1, 2, 2)
	plt.scatter(pos[:, 2, timestep], ap[:, 2, timestep], s=0.5, alpha=0.5)


def plot_phase_space_trajectory(tr, pdef):
	print(len(tr['times']))
	pos = tr['positions']
	ap = tr['particle_attributes']
	print(np.shape(pos))
	for pi in pdef:
		plt.scatter(pos[pi, 0, :], ap[pi, 0, :], s=10, alpha=1)


def animate_phase_space(tr, result_name, xlim=None, ylim=None, numframes=None, alpha=1.0, mode="radial"):
	fig = plt.figure(figsize=(13, 5))
	pos = tr.positions
	ap = tr.particle_attributes
	velocity_x = ap.get('velocity x')
	velocity_y = ap.get('velocity y')
	velocity_z = ap.get('velocity z')
	masses = tr.optional_attributes['Particle Masses']

	if not numframes:
		numframes = tr.n_timesteps

	fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

	scat1 = ax1.scatter(pos[:, 0, 0], velocity_x[:, 0], s=10, alpha=alpha, c=masses)

	if mode == "radial":
		ax1.set_xlabel("radial position")
		ax1.set_ylabel("radial velocity")
	elif mode == "cartesian":
		ax1.set_xlabel("x position")
		ax1.set_ylabel("x velocity")

	if ylim:
		ax1.set_ylim(ylim[0])
	else:
		if mode == "radial":
			r_velo = np.sqrt(velocity_x ** 2.0 + velocity_y ** 2.0)
			ax1.set_ylim((np.min(r_velo), np.max(r_velo)))
		elif mode == "cartesian":
			ax1.set_ylim((np.min(velocity_x), np.max(velocity_x)))


	if xlim:
		ax1.set_xlim(xlim[0])
	else:
		if mode == "radial":
			r_dist = np.sqrt(pos[:, 0, :] ** 2.0 + pos[:, 1, :] ** 2.0)
			ax1.set_xlim((np.min(r_dist), np.max(r_dist)))
		elif mode == "cartesian":
			ax1.set_xlim((np.min(pos[:, 0, :]), np.max(pos[:, 0, :])))


	scat2 = ax2.scatter(pos[:, 2, 0], velocity_z[:, 0], s=10, alpha=alpha, c=masses)
	ax2.set_xlabel("z position")
	ax2.set_ylabel("z velocity")

	if ylim:
		ax2.set_ylim(ylim[1])
	else:
		ax2.set_ylim((np.min(velocity_z), np.max(velocity_z)))

	if xlim:
		plt.xlim(xlim[1])
	else:
		plt.xlim((np.min(pos[:, 2, :]), np.max(pos[:, 2, :])))

	ani = animation.FuncAnimation(fig, update_phase_space_plot, frames=range(numframes),
	                              fargs=(pos, velocity_x, velocity_y, velocity_z, scat1, scat2, mode))
	ani.save(result_name + "_phaseSpace.mp4", fps=20, extra_args=['-vcodec', 'libx264'])


def update_phase_space_plot(i, pos, velocity_x, velocity_y, velocity_z, scat1, scat2, mode):

	if mode == "radial":
		r_dist = np.sqrt(pos[:, 0, i] ** 2.0 + pos[:, 1, i] ** 2.0)
		r_velo = np.sqrt(velocity_x[:, i] ** 2.0 + velocity_y[:, i] ** 2.0)
		scat1.set_offsets(np.transpose(np.vstack([r_dist, r_velo])))
	elif mode == "cartesian":
		scat1.set_offsets(np.transpose(np.vstack([pos[:, 0, i], velocity_x[:, i]])))

	scat2.set_offsets(np.transpose(np.vstack([pos[:, 2, i], velocity_z[:, i]])))
	return scat1, scat2


def render_phase_space_animation(pname, result_name, file_type='hdf5', ylim=None, xlim=None, numframes=None, alpha=1.0, mode="cartesian"):

	tr = tra.read_trajectory_file_for_project(pname, file_type)
	animate_phase_space(tr, result_name, ylim=ylim, xlim=xlim,alpha=alpha,numframes=numframes,mode=mode)