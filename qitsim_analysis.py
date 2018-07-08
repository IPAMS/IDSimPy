# -*- coding: utf-8 -*-
import numpy as np
import pylab as plt
import pandas as pd
import json
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

def read_QIT_conf(confFileName):
	"""
	Reads and parses the QIT simulation configuration file
	:param confFileName: the filename of the simulation configuration file
	:return: a parsed dictionary with the configuration parameters
	"""
	with open(confFileName) as jsonFile:
		confJson = json.load(jsonFile)

	return (confJson)


def read_FFT_record(projectPath):
	"""
	Loads a fft record file, which contains the exported mirror charge current on some detector electrodes

	:param projectPath: path of the simulation project
	:return: two vectors: the time and simulated mirror charge from the fft file
	"""
	dat = np.loadtxt(projectPath + "_fft.txt")
	t = dat[:, 0]
	z = dat[:, 1]
	return t,z

def read_ions_inactive_record(projectPath):
	"""
	Loads an ions inactive record file, which contains the number of inactive ions over the time

	:param projectPath: path of the simulation project
	:return: two vectors: the time and the number of inactive ions
	"""
	dat = np.loadtxt(projectPath + "_ionsInactive.txt")
	t = dat[:, 0]
	ions_inactive = dat[:, 1]
	return t, ions_inactive

def read_center_of_charge_record(projectPath):
	"""
	Loads a center of charge (coc) record file, which contains the mean position of the charged particle cloud over the time

	:param projectPath: path of the simulation project
	:return: two vectors: the time and the mean position of the charged particle cloud from the coc file
	"""
	dat = np.loadtxt(projectPath + "_averagePosition.txt")
	t = dat[:, 0]
	z = dat[:, 3]
	return t,z


def read_and_analyze_stability_scan(projectName,t_range=[0,1]):
	time, inactive_ions = read_ions_inactive_record(projectName)

	with open(projectName + "_conf.json") as jsonFile:
		conf_json = json.load(jsonFile)

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
	Y = np.fft.fft(z) / n  # fft computing and normalization
	Y = Y[range(n // 2)]

	k = np.arange(n)
	T = n / Fs
	frq = k / T  # two sides frequency range
	frq = frq[range(n // 2)]  # one side frequency range

	return frq,abs(Y)



################## High Level Simulation Project Processing Methods ######################

def analyse_FFT_sim(projectPath,freqStart=0.0,freqStop=1.0,ampMode="lin",loadMode="fft_record"):
	"""
	Analyses a transient of a QIT simulation and calculates/plots the spectrum from it

	:param projectPath: base path of a qit simulation
	:param freqStart: start of the plotted frequency window (normalized)
	:param freqStop: stop/end of the plotted frequency window (normalized)
	:param ampMode: fft spectrum plot mode, linear or logarithmic, options: "lin" or "log"
	:param loadMode: load a recorded fft record file
	("fft_record") or reconstruct transient from trajectories ("reconstruct_from_trajectories")
	:return: a dictionary with the frequencies and amplitudes of the spectrum and
	the time vector and amplitude of the transient
	"""
	with open(projectPath + "_conf.json") as jsonFile:
		confJson = json.load(jsonFile)

	if loadMode == "fft_record":
		t,z = read_FFT_record(projectPath)
	elif loadMode == "center_of_charge_record":
		t,z = read_center_of_charge_record(projectPath)
	elif loadMode == "reconstruct_from_trajectories":
		tr = tra.read_trajectory_file(projectPath + "_trajectories.json.gz")
		t,z = reconstruct_transient_from_trajectories(tr)
		t= t*1e-6
		print(t)

	frq,Y = calculate_FFT_spectrum(t, z)

	fig, ax = plt.subplots(1, 2,figsize=[20,5],dpi=50)
	ax[0].plot(t,z)
	ax[0].set_xlabel('Time (s)')
	ax[0].set_ylabel('Amplitude (arb.)')

	freqsPl= range(int(len(frq)*freqStart),int((len(frq)*freqStop)))

	#ax[1].semilogy(frq[freqsPl],abs(Y[freqsPl]),'r') # plotting the spectrum
	if ampMode == "lin":
		ax[1].plot(frq[freqsPl]/1000,abs(Y[freqsPl]),'r') # plotting the spectrum
	elif ampMode == "log":
		ax[1].semilogy(frq[freqsPl]/1000, abs(Y[freqsPl]), 'r')  # plotting the spectrum logarithmic
	ax[1].set_xlabel('Freq (kHz)')
	ax[1].set_ylabel('Amplitude (arb.)')

	projectName = projectPath.split("/")[-1]
	if "space_charge_factor" in confJson:
		titlestring = projectName+" p:"+str(confJson["background_pressure"])+" Pa, c gas mass:"+str(confJson["collision_gas_mass_amu"])+" amu, "
		titlestring = titlestring + "space charge factor:"+'%6g' % (confJson["space_charge_factor"])
	else:
		titlestring = projectName+" p:"+str(confJson["background_pressure"])+" Pa"


	#titlestring = titlestring + ", nIons:"+str(confJson["n_ions"])+ ", ion masses:"+str(confJson["ion_masses"])
	plt.figtext(0.1,0.94,titlestring,fontsize=17)

	plt.savefig(projectName+"_fftAnalysis.pdf",format="pdf")
	plt.savefig(projectName+"_fftAnalysis.png",format="png",dpi=180)

	return({"freqs":frq[freqsPl],"amplitude":abs(Y[freqsPl]),"time":t,"transient":z})


def analyze_stability_scan(projectName,window_width=0,t_range=[0,1]):
	with open(projectName + "_conf.json") as jsonFile:
		confJson = json.load(jsonFile)

	V_rf_start = confJson["V_rf_start"]
	V_rf_end = confJson["V_rf_end"]

	titlestring = projectName + " p " + str(confJson["background_pressure"]) + " Pa, c. gas " + str(
		confJson["collision_gas_mass_amu"]) + " amu, "
	titlestring = titlestring + "spc:" + '%4g, ' % (confJson["space_charge_factor"])
	titlestring = titlestring + "RF: " + str(V_rf_start) + " to " + str(V_rf_end) + " V @ " + str(
		confJson["f_rf"] / 1000.0) + " kHz"

	t_end = confJson["sim_time_steps"]*confJson["dt"]
	dUdt = (confJson["V_rf_end"] - confJson["V_rf_start"]) / t_end
	titlestring = titlestring + (' (%2g V/s), ' % (dUdt))
	titlestring = titlestring + str(confJson["excite_pulse_potential"]) + " V exci."

	project = [[projectName, ""]]
	plot_fn = projectName + "_ionEjectionAnalysis"
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


def center_of_charges_from_simulation(dat,speciesMasses,tRange=[]):
	"""
	Calculates the center of charges for two species, defined by their mass, from a simulation
	:param dat: imported trajectories object
	:type dat: dict returned from readTrajectoryFile
	:param speciesMasses: two element list with two particle masses
	:type speciesMasses: list
	:param tRange: a list / vector with time step indices to export the center of charges for
	:type tRange: list
	:return: dictionary with the time vector in "t", and the center of species a and b and of the whole
	ion cloud in dat in "cocA", "cocB" and "cocAll"
	:rtype dictionary
	"""
	masses = dat["masses"]
	times = dat["times"]

	if len(tRange)==0:
		tRange=range(len(times))
	else:
		times=times[tRange]

	tr = dat["positions"][:,:,tRange]

	cocA = tra.center_of_charge(tra.filter_mass(tr,masses,speciesMasses[0]))
	cocB = tra.center_of_charge(tra.filter_mass(tr,masses,speciesMasses[1]))
	cocAll = tra.center_of_charge(tr)

	return{"t":times,"cocA":cocA,"cocB":cocB,"cocAll":cocAll}

def plot_average_z_position(sim_projects, masses,compressed=True):
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
		# i_ap = tj["additional_parameters"]
		times = tj["times"]

		plt.subplot(n_projects, 1, si + 1)
		for mass in masses:
			i_pos_mfiltered = tra.filter_mass(i_pos, i_masses, mass)
			# i_ap_mfiltered = lq.filter_mass(i_ap, i_masses, mass)
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

def animate_simulation_center_of_masses_z_vs_x(dat,masses,nFrames,interval,frameLen,zLim=3,xLim=0.1):
	"""
	Animate the center of charges of the ion clouds in a QIT simulation in a z-x projection. The center of charges
	are rendered as a trace with a given length (in terms of simulation time steps)

	:param dat: imported trajectories object
	:type dat: dict returned from readTrajectoryFile
	:param masses: two element list with two particle masses to render the center of charges for
	:type masses: list
	:param nFrames: number of frames to export
	:param interval: interval in terms of time steps in the input data between the animation frames
	:param frameLen: length of the trace of the center of charges (in terms of simulation time steps)
	:param zLim: limits of the rendered spatial section in z direction
	:param xLim: limits of the rendered spatial section in x direction
	:return: an animation object with the rendered animation
	"""
	fig = plt.figure()
	ax = plt.axes(ylim=(-zLim, zLim), xlim=(-xLim, xLim))
	lA, = ax.plot([], [], lw=2)
	lB, = ax.plot([], [], lw=2)
	lall, = ax.plot([], [], lw=2)

	# animation function.  This is called sequentially
	def animate(i):
		#x = np.linspace(0, 2, 1000)
		#y = np.sin(2 * np.pi * (x - 0.01 * i))
		tRange=np.arange(0+i*interval,frameLen+i*interval)
		d = center_of_charges_from_simulation(dat,masses,tRange=tRange)
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
								   frames=nFrames, blit=True)
	# call our new function to display the animation
	return(anim)

def plot_phase_space_frame(tr, timestep):
	print(len(tr['times']))
	pos = tr['positions']
	ap = tr['additional_parameters']
	print(np.shape(pos))
	plt.subplot(1, 2, 1)
	plt.scatter(pos[:, 0, timestep], ap[:, 0, timestep], s=0.5, alpha=0.5)
	plt.subplot(1, 2, 2)
	plt.scatter(pos[:, 2, timestep], ap[:, 2, timestep], s=0.5, alpha=0.5)


def plot_phase_space_trajectory(tr, pdef):
	print(len(tr['times']))
	pos = tr['positions']
	ap = tr['additional_parameters']
	print(np.shape(pos))
	for pi in pdef:
		plt.scatter(pos[pi, 0, :], ap[pi, 0, :], s=10, alpha=1)


def animate_phase_space(tr, resultName, xlim=None, ylim=None, numframes=None,alpha=1.0, mode="radial"):
	fig = plt.figure(figsize=(13, 5))
	pos = tr['positions']
	ap = tr['additional_parameters']
	masses = tr['masses']

	if not numframes:
		numframes = len(tr['times'])

	plt.subplot(1, 2, 1)
	scat1 = plt.scatter(pos[:, 0, 0], ap[:, 0, 0], s=10, alpha=alpha, c=masses)


	if mode == "radial":
		plt.xlabel("radial position")
		plt.ylabel("radial velocity")
	elif mode == "cartesian":
		plt.xlabel("x position")
		plt.ylabel("x velocity")


	if ylim:
		plt.ylim(ylim[0])
	else:
		if mode == "radial":
			r_dist = np.sqrt(pos[:, 0, :] ** 2.0 + pos[:, 1, :] ** 2.0)
			plt.ylim((np.min(r_dist), np.max(r_dist)))
		elif mode == "cartesian":
			plt.ylim((np.min(ap[:, 0, :]), np.max(ap[:, 0, :])))


	if xlim:
		plt.xlim(xlim[0])
	else:
		if mode == "radial":
			r_velo = np.sqrt(ap[:, 0, :] ** 2.0 + ap[:, 1, :] ** 2.0)
			plt.xlim((np.min(r_velo), np.max(r_velo)))
		elif mode == "cartesian":
			plt.xlim((np.min(pos[:, 0, :]), np.max(pos[:, 0, :])))

	plt.subplot(1, 2, 2)
	scat2 = plt.scatter(pos[:, 2, 0], ap[:, 2, 0], s=10, alpha=alpha, c=masses)
	plt.xlabel("z position")
	plt.ylabel("z velocity")

	if ylim:
		plt.ylim(ylim[1])
	else:
		plt.ylim((np.min(ap[:, 2, :]), np.max(ap[:, 2, :])))

	if xlim:
		plt.xlim(xlim[1])
	else:
		plt.xlim((np.min(pos[:, 2, :]), np.max(pos[:, 2, :])))

	ani = animation.FuncAnimation(fig, update_phase_space_plot, frames=range(numframes),
	                              fargs=(pos, ap, scat1, scat2,mode))
	ani.save(resultName + "_phaseSpace.mp4", fps=20, extra_args=['-vcodec', 'libx264'])


def update_phase_space_plot(i, pos, ap, scat1, scat2, mode):

	if mode == "radial":
		r_dist = np.sqrt(pos[:, 0, i] ** 2.0 + pos[:, 1, i] ** 2.0)
		r_velo = np.sqrt(ap[:, 0, i] ** 2.0 + ap[:, 1, i] ** 2.0)
		scat1.set_offsets(np.transpose(np.vstack([r_dist, r_velo])))
	elif mode == "cartesian":
		scat1.set_offsets(np.transpose(np.vstack([pos[:, 0, i], ap[:, 0, i]])))

	scat2.set_offsets(np.transpose(np.vstack([pos[:, 2, i], ap[:, 2, i]])))
	return scat1, scat2


def render_phase_space_animation(pname,ylim=None,xlim=None,numframes=None,alpha=1.0,compressed=True,mode="cartesian"):

	if compressed:
		tr = tra.read_trajectory_file(pname + "_trajectories.json.gz")
	else:
		tr = tra.read_trajectory_file(pname + "_trajectories.json")
	animate_phase_space(tr, pname, ylim=ylim, xlim=xlim,alpha=alpha,numframes=numframes,mode=mode)