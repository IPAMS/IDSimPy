# -*- coding: utf-8 -*-

import numpy as np
import pylab as plt
import pandas as pd
import json
import gzip
import io
import base64
from .constants import *
from tempfile import NamedTemporaryFile
from matplotlib import animation



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

def read_legacy_trajectory_file(trajectoryFileName):
	"""
	Reads a legacy trajectory file and returns a legacy trajectory object

	Trajectory objects are dictionaries which contain three elements:
	trajectories: a vector which contains the x,y,z positions of all particles for all time steps
	(a vector of lists one vector entry per time step)
	times: vector of times of the individual time steps
	masses: the vector of particle masses

	:param trajectoryFileName: the file name of the file to read
	:return: the trajectory data dictionary
	"""
	if (trajectoryFileName[-8:] == ".json.gz"):
		with gzip.open(trajectoryFileName) as tf:
			tj = json.load(io.TextIOWrapper(tf))
	else:
		with open(trajectoryFileName) as tf:
			tj = json.load(tf)

	steps = tj["steps"]
	nIons = len(steps[0]["positions"])

	t = np.zeros([nIons,3,len(steps)])
	times = np.zeros(len(steps))
	for i in range(len(steps)):
		t[:,:,i] = np.array(steps[i]["positions"])
		times[i] = float(steps[i]["time"])

	masses = np.zeros([nIons])
	massesJson = tj["ionMasses"]
	for i in range(len(massesJson)):
		masses[i] = float(massesJson[i])
	return{"trajectories":t,"times":times,"masses":masses}


def read_trajectory_file(trajectoryFileName):
	"""
	Reads a trajectory file and returns a trajectory object

	Trajectory objects are dictionaries which contain three elements:
	trajectories: a vector which contains the x,y,z positions of all particles for all time steps
	(a vector of lists one vector entry per time step)
	times: vector of times of the individual time steps
	masses: the vector of particle masses

	:param trajectoryFileName: the file name of the file to read
	:return: the trajectory data dictionary
	"""
	if trajectoryFileName[-8:] == ".json.gz":
		with gzip.open(trajectoryFileName) as tf:
			tj = json.load(io.TextIOWrapper(tf))
	else:
		with open(trajectoryFileName) as tf:
			tj = json.load(tf)

	steps = tj["steps"]
	nIons = len(steps[0]["ions"])

	times = np.zeros(len(steps))
	positions = np.zeros([nIons,3,len(steps)])

	n_additional_parameters = len(steps[0]["ions"][0])-1
	additional_parameters = np.zeros([nIons,n_additional_parameters,len(steps)])

	for i in range(len(steps)):
		for j in range (nIons):
			positions[j,:,i] = np.array(steps[i]["ions"][j][0])
			additional_parameters[j,:,i] = np.array(steps[i]["ions"][j][1:])

		times[i] = float(steps[i]["time"])

	masses = np.zeros([nIons])
	masses_json = tj["ionMasses"]
	for i in range(len(masses_json)):
		masses[i] = float(masses_json[i])

	splat_times = np.zeros([nIons])
	splat_times_json = tj["splatTimes"]
	for i in range(len(splat_times_json)):
		splat_times[i] = float(splat_times_json[i])

	return{"positions":positions,
	       "additional_parameters":additional_parameters,
	       "times":times,
	       "masses":masses,
	       "n_ions":nIons,
	       "splat_times":splat_times}


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

def filter_mass(positions,masses,massToFilter):
	"""
	Filters out trajectories of ions with a given mass

	:param positions: a positions vector from an imported trajectories object
	:type trajectory positions: positions vector from dict returned from readTrajectoryFile
	:param masses: a mass vector from an imported trajectories object
	:param massToFilter: the mass to filter for
	:return: a filtered positions vector
	"""
	mass_indexes = np.nonzero(masses == massToFilter)
	return positions[mass_indexes,:,:][0]


def center_of_charge(tr):
	"""
	Calculates the center of charge of a charged particle cloud

	:param tr: a trajectories vector from an imported trajectories object
	:type tr: trajectories vector from dict returned from readTrajectoryFile
	:return: vector of the spatial position of the center of mass
	"""
	nSteps = np.shape(tr)[2]
	coc = np.zeros([nSteps,3])
	for i in range(nSteps):
		xMean = np.mean(tr[:,0,i])
		yMean = np.mean(tr[:,1,i])
		zMean = np.mean(tr[:,2,i])
		coc[i,:] = np.array([xMean,yMean,zMean])
	return(coc)

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

################## Simple Plot Methods ######################

def plot_particles_path(trajectories, pl_filename, p_indices, plot_mark='*-',time_range=(0,1)):
	"""
	Plots the paths of a selection of particles in a x,z and y,z projection
	:param tr: trajectory input data
	:type tr: list of lists of trajectory dictionaries from read_trajectory_file and an according label
	:param pl_filename: the basename of the plot image files to create
	:param p_indices:
	:type p_indices: list of integers
	:param plot_mark: matplotlib plot format string which is used for the path-plots
	:type plot_mark: str
	:param time_range: range of times to plot (given as a fraction between 0 and 1)
	:type time_range: tuple of two floats between 0 and 1
	"""
	fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

	for tr in trajectories:
		times = tr[0]['times']
		n_times = len(times)
		pos = tr[0]['positions']

		i_start = int(n_times*time_range[0])
		i_stop = int(n_times * time_range[1])
		i_range = np.arange(i_start,i_stop)

		for p in p_indices:
			p_pos = pos[p, :, i_range]
			ax1.plot(p_pos[:, 0], p_pos[:, 2], plot_mark)
			ax2.plot(p_pos[:, 1], p_pos[:, 2], plot_mark, label=tr[1] + ' p ' + str(p))

	ax1.set_xlabel('x')
	ax1.set_ylabel('z')
	ax2.set_xlabel('y')
	ax2.set_ylabel('z')
	ax2.legend()

	plt.tight_layout()
	plt.savefig(pl_filename + '.pdf', format='pdf')


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
		tr = read_trajectory_file(projectPath + "_trajectories.json.gz")
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

	cocA = center_of_charge(filter_mass(tr,masses,speciesMasses[0]))
	cocB = center_of_charge(filter_mass(tr,masses,speciesMasses[1]))
	cocAll = center_of_charge(tr)

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
		tj = read_trajectory_file(sim_projects[si] + file_ext)
		conf = read_QIT_conf(sim_projects[si] + "_conf.json")

		i_pos = tj["positions"]
		i_masses = tj["masses"]
		# i_ap = tj["additional_parameters"]
		times = tj["times"]

		plt.subplot(n_projects, 1, si + 1)
		for mass in masses:
			i_pos_mfiltered = filter_mass(i_pos, i_masses, mass)
			# i_ap_mfiltered = lq.filter_mass(i_ap, i_masses, mass)
			coc = center_of_charge(i_pos_mfiltered)
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
		d = centerOfChargesSimulation(dat,masses,tRange=tRange)
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

def plot_density_z_vs_x(trajectories,timeIndex):
	"""
	Renders an density plot in a z-x projection
	:param trajectories: a trajectories vector from an imported trajectories object
	:type trajectories: trajectories vector from dict returned from readTrajectoryFile
	:param timeIndex: index of the time step to render
	"""
	xedges = np.linspace(-10,10,80)
	zedges = np.linspace(-10,10,80)
	x = trajectories[:,0,timeIndex]
	z = trajectories[:,2,timeIndex]
	H, xedges, yedges = np.histogram2d(z,x, bins=(xedges, zedges))
	fig = plt.figure(figsize=(7, 7))


	ax = fig.add_subplot(111)
	ax.set_title('NonUniformImage: interpolated')
	im = plt.image.NonUniformImage(ax, interpolation='bilinear')
	xcenters = xedges[:-1] + 0.5 * (xedges[1:] - xedges[:-1])
	zcenters = zedges[:-1] + 0.5 * (zedges[1:] - zedges[:-1])
	im.set_data(xcenters, zcenters, H)
	ax.images.append(im)
	ax.set_xlim(xedges[0], xedges[-1])
	ax.set_ylim(zedges[0], zedges[-1])
	ax.set_aspect('equal')
	plt.show()


def animate_simulation_z_vs_x_density(dat,masses,nFrames,interval,
										fileMode='video',
										mode="lin",
										sLim=3,nBins=100,
										alphaFactor = 1,colormap = plt.cm.coolwarm,
										annotateString=""):
	"""
	Animate the center of charges of the ion clouds in a QIT simulation in a z-x projection. The center of charges
	are rendered as a trace with a given length (in terms of simulation time steps)

	:param dat: imported trajectories object
	:type dat: dict returned from readTrajectoryFile
	:param masses: two element list with two particle masses to render the particle densities for
	:type masses: list
	:param nFrames: number of frames to export
	:param interval: interval in terms of time steps in the input data between the animation frames
	:param fileMode: render either a video ("video") or single frames as image files ("singleFrame")
	:param mode: scale density linearly ("lin") or logarithmically ("log")
	:param sLim: spatial limits of the rendered spatial domain (given as distance from the origin of the coordinate system)
	:param nBins: number of density bins in the spatial directions
	:param alphaFactor: blending factor for graphical blending the densities of the two species
	:param colormap: a colormap for the density rendering (a pure species will end up on one side of the colormap)
	:param annotateString: an optional string which is rendered into the animation as annotation
	:return: animation object or figure (depends on the file mode)
	"""


	times = dat[0]["times"]
	datA = filter_mass(dat[0]["positions"],dat[0]["masses"],masses[0])
	datB = filter_mass(dat[1]["positions"],dat[1]["masses"],masses[1])

	if fileMode=='video':
		fig = plt.figure(figsize=[10,10])
	elif fileMode=='singleFrame':
		fig = plt.figure(figsize=[ 6, 6])

	xedges = np.linspace(-sLim,sLim,nBins)
	zedges = np.linspace(-sLim,sLim,nBins)
	H = np.random.rand(len(xedges),len(zedges))
	ax = plt.axes(ylim=(zedges[0], zedges[-1]), xlim=(xedges[0], xedges[-1]))

	im1 = ax.imshow(H, interpolation='nearest', origin='low', alpha=1, vmin=0, vmax=10, cmap="Reds",
				extent=[xedges[0], xedges[-1], zedges[0], zedges[-1]])

	text_time = ax.annotate("TestText",xy=(0.02,0.96),xycoords="figure fraction",
	                        horizontalalignment="left",
	                        verticalalignment="top",
	                        fontsize=20);

	plt.xlabel("r (mm)")
	plt.ylabel("z (mm)")
	fillChannel = np.ones([len(xedges)-1,len(zedges)-1])

	def animate(i):
		tsNumber = i*interval
		x = datA[:,0,tsNumber]
		z = datA[:,2,tsNumber]
		h_A, xedges2, zedges2 = np.histogram2d(z,x, bins=(xedges, zedges))
		#im1.set_array(H2);

		x = datB[:,0,tsNumber]
		z = datB[:,2,tsNumber]
		h_B, xedges2, zedges2 = np.histogram2d(z,x, bins=(xedges, zedges))

		nf_A = np.max(h_A)
		nf_B = np.max(h_B)

		rel_conc = h_A / (h_A + h_B + 0.00001)
		img_data_RGB = colormap(rel_conc)
		h_A_log = np.log10(h_A + 1)
		h_B_log = np.log10(h_B + 1)
		nf_A_log = np.max(h_A_log)
		nf_B_log = np.max(h_B_log)
		abs_dens = (h_A + h_B) / (nf_A + nf_B)
		abs_dens_log_raw = (h_A_log + h_B_log) / (nf_A_log + nf_B_log)
		abs_dens_log = abs_dens_log_raw * 0.5
		nonzero = np.nonzero(abs_dens_log > 0)
		abs_dens_log[nonzero] = abs_dens_log[nonzero] + 0.5

		if mode == "lin":
			img_data_RGB[:, :, 3] = abs_dens*alphaFactor
		elif mode== "log":
			img_data_RGB[:, :, 3] = abs_dens_log*alphaFactor

		im1.set_array(img_data_RGB)
		text_time.set_text("t="+str(times[tsNumber])+u"µs"+" "+annotateString)

		return im1

	# call the animator.  blit=True means only re-draw the parts that have changed.
	if fileMode == 'video':
		anim = animation.FuncAnimation(fig, animate, frames=nFrames, blit=False)
		return (anim)
	elif fileMode == 'singleFrame':
		animate(nFrames)
		return (fig)


def render_XZ_density_animation(projectNames,masses,resultName,nFrames=400,delay=1,annotation="",compressed=True):
	"""
	:param projectNames: simulation projects to compare (given as project basenames)
	:type projectNames: tuple of two strings
	:param masses: list of masses in the two simulation projects to compare
	:type masses: tuple of two floats
	:param resultName: basename for the rendering result
	:param nFrames: number of frames to render
	:param delay: interval in terms of time steps in the input data between the animation frames
	:type delay: int
	:param annotation: annotation string
	:type annotation: str
	:param compressed: flag if the input trajectory data is gzip compressed
	"""

	if compressed:
		file_ext =  "_trajectories.json.gz"
	else:
		file_ext = "_trajectories.json"


	tj0 = read_trajectory_file(projectNames[0]+file_ext)
	tj1 = read_trajectory_file(projectNames[1]+file_ext)
	anim = animate_simulation_z_vs_x_density([tj0,tj1],masses,nFrames,delay,sLim=7,annotateString=annotation)
	anim.save(resultName+"_densitiesComparisonXZ.mp4", fps=20, extra_args=['-vcodec', 'libx264'])
	display_animation(anim)


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


def animate_phase_space(tr, resultName, xlim=None, ylim=None, numframes=None,alpha=1.0):
	fig = plt.figure(figsize=(13, 5))
	pos = tr['positions']
	ap = tr['additional_parameters']
	masses = tr['masses']

	if not numframes:
		numframes = len(tr['times'])

	plt.subplot(1, 2, 1)
	scat1 = plt.scatter(pos[:, 0, 0], ap[:, 0, 0], s=10, alpha=alpha, c=masses)
	plt.xlabel("x position")
	plt.ylabel("x velocity")

	if ylim:
		plt.ylim(ylim[0])
	else:
		plt.ylim((np.min(ap[:, 0, :]), np.max(ap[:, 0, :])))

	if xlim:
		plt.xlim(xlim[0])
	else:
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
	                              fargs=(pos, ap, scat1, scat2))
	ani.save(resultName + "_phaseSpace.mp4", fps=20, extra_args=['-vcodec', 'libx264'])


def update_phase_space_plot(i, pos, ap, scat1, scat2):
	scat1.set_offsets(np.transpose(np.vstack([pos[:, 0, i], ap[:, 0, i]])))
	scat2.set_offsets(np.transpose(np.vstack([pos[:, 2, i], ap[:, 2, i]])))
	return scat1, scat2


def render_phase_space_animation(pname,ylim=None,xlim=None,numframes=None,alpha=1.0,compressed=True):

	if compressed:
		tr = read_trajectory_file(pname + "_trajectories.json.gz")
	else:
		tr = read_trajectory_file(pname + "_trajectories.json")
	animate_phase_space(tr, pname, ylim=ylim, xlim=xlim,alpha=alpha,numframes=numframes)


#### animation / jupyter stuff ####
##animation machinery:
#http://jakevdp.github.io/blog/2013/05/12/embedding-matplotlib-animations/

VIDEO_TAG = """<video controls>
 <source src="data:video/x-m4v;base64,{0}" type="video/mp4">
 Your browser does not support the video tag.
</video>"""

def anim_to_html(anim):
	"""
	converts an animation to html (for jupyter notebooks)
	:param anim:
	:return:
	"""
	if not hasattr(anim, '_encoded_video'):
		with NamedTemporaryFile(suffix='.mp4') as f:
			anim.save(f.name, fps=20, extra_args=['-vcodec', 'libx264'])
			video = open(f.name, "rb").read()
		anim._encoded_video = base64.b64encode(video)#.encode("base64")

	return VIDEO_TAG.format(anim._encoded_video)

from IPython.display import HTML

def display_animation(anim):
	"""
	displays an animation (in an jupyter notebook)
	"""
	plt.close(anim._fig)
	return HTML(anim_to_html(anim))
