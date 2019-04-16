# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.image import NonUniformImage
from . import trajectory as tra

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


def plot_density_z_vs_x(trajectories,timeIndex,
        xedges = np.linspace(-10,10,80),
		zedges = np.linspace(-10,10,80)
):
	"""
	Renders an density plot in a z-x projection
	:param trajectories: a trajectories vector from an imported trajectories object
	:type trajectories: trajectories vector from dict returned from readTrajectoryFile
	:param timeIndex: index of the time step to render
	"""
	x = trajectories[:,0,timeIndex]
	z = trajectories[:,2,timeIndex]
	H, xedges, yedges = np.histogram2d(x,z, bins=(xedges, zedges))
	H = H.T
	fig = plt.figure(figsize=(7, 7))

	ax = fig.add_subplot(111)
	ax.set_title('NonUniformImage: interpolated')
	im = NonUniformImage(ax, interpolation='nearest')
	xcenters = xedges[:-1] + 0.5 * (xedges[1:] - xedges[:-1])
	zcenters = zedges[:-1] + 0.5 * (zedges[1:] - zedges[:-1])
	im.set_data(xcenters, zcenters, H)
	ax.images.append(im)
	ax.set_xlim(xedges[0], xedges[-1])
	ax.set_ylim(zedges[0], zedges[-1])
	ax.set_aspect('equal')

################## High Level Simulation Project Processing Methods ######################

def animate_z_vs_x_density_plot(dat, selected, nFrames, interval,
                                select_mode='substance',
                                output_mode='video',
                                mode='lin',
                                s_lim=3, n_bins=100, basesize = 17,
                                alpha = 1, colormap = plt.cm.coolwarm,
                                annotate_string=""):
	"""
	Animate the denisties of two ion clouds in a QIT simulation in a z-x projection.

	:param dat: imported trajectories object
	:type dat: dict returned from readTrajectoryFile
	:param selected: two element list with values to select particles to be rendered
	:type selected: list
	:param nFrames: number of frames to export
	:param interval: interval in terms of time steps in the input data between the animation frames
	:param select_mode: defines the mode for selection of particles:
		"mass" for selecting by mass,
		"substance" for chemical substance / chemical id
	:param output_mode: render either a video ("video") or single frames as image files ("singleFrame")
	:param mode: scale density linearly ("lin") or logarithmically ("log")
	:param s_lim: spatial limits of the rendered spatial domain
			(given as distance from the origin of the coordinate system or explicit limits: [xlo, xhi, zlo, zhi]
	:param n_bins: number of density bins in the spatial directions or list of bin numbers ([x z])
	:type n_bins: int or list of two ints
	:param basesize: the base size of the plot
	:type basestring: float
	:param alpha: blending factor for graphical blending the densities of the two species
	:param colormap: a colormap for the density rendering (a pure species will end up on one side of the colormap)
	:param annotate_string: an optional string which is rendered into the animation as annotation
	:return: animation object or figure (depends on the file mode)
	"""


	if select_mode == 'mass':
		select_parameter = [dat[0]['masses'],dat[1]['masses']]

	elif select_mode == 'substance':
		id_column = dat[0]['additional_names'].index('chemical id')
		select_parameter = [dat[0]['additional_parameters'][:, id_column, :],
	                        dat[1]['additional_parameters'][:, id_column, :]]
	else:
		raise ValueError('Invalid select_mode')


	times = dat[0]["times"]
	times_B = dat[1]["times"]

	if len(times) != len(times_B):
		raise ValueError('Length of trajectories differ')
	if not (times == times_B).all():
		raise ValueError('The times of the trajectories differ')
	if nFrames*interval > len(times):
		raise ValueError('number of frames * interval is longer than trajectory')


	if selected[0] == "all":
		datA = dat[0]["positions"]
	else:
		datA = tra.filter_parameter(dat[0]["positions"], select_parameter[0], selected[0])

	if selected[1] == "all":
		datB = dat[1]["positions"]
	else:
		datB = tra.filter_parameter(dat[1]["positions"], select_parameter[1], selected[1])



	if output_mode== 'video':
		fig = plt.figure(figsize=[10,10])
	elif output_mode== 'singleFrame':
		fig = plt.figure(figsize=[ 6, 6])

	if not hasattr(s_lim, "__iter__"): #is not iterable
		limits = [-s_lim,s_lim,-s_lim,s_lim]
	else:
		limits = s_lim

	if not hasattr(n_bins, "__iter__"): #is not iterable
		bins = [n_bins,n_bins]
	else:
		bins = n_bins

	xedges = np.linspace(limits[0],limits[1],bins[0])
	zedges = np.linspace(limits[2],limits[3],bins[1])
	H = np.random.rand(len(xedges),len(zedges))
	fig_ratio = (limits[3]-limits[2]) / (limits[1]-limits[0])
	fig = plt.figure(figsize=(basesize,basesize*fig_ratio+basesize/10.0))
	ax = fig.add_subplot(1, 1, 1, ylim=(zedges[0], zedges[-1]), xlim=(xedges[0], xedges[-1]))

	im1 = ax.imshow(H, interpolation='nearest', origin='low', alpha=1, vmin=0, vmax=10, cmap="Reds",
				extent=[xedges[0], xedges[-1], zedges[0], zedges[-1]])

	text_time = ax.annotate("TestText",xy=(0.02,0.96),xycoords="figure fraction",
	                        horizontalalignment="left",
	                        verticalalignment="top",
	                        fontsize=20);

	plt.xlabel("x (mm)")
	plt.ylabel("z (mm)")
	fillChannel = np.ones([len(xedges)-1,len(zedges)-1])

	def animate(i):
		tsNumber = i*interval
		#if the dat objects are lists: we have filtered the particles in a way that the number of selected
		#particles change between the timesteps and we got a list of individual particle vectors
		if isinstance(datA, list):
			x = datA[tsNumber][:, 0]
			z = datA[tsNumber][:, 2]
		else:
			x = datA[:,0,tsNumber]
			z = datA[:,2,tsNumber]
		h_A, zedges2, xedges2 = np.histogram2d(z,x, bins=(zedges, xedges))

		if isinstance(datB, list):
			x = datB[tsNumber][:, 0]
			z = datB[tsNumber][:, 2]
		else:
			x = datB[:,0,tsNumber]
			z = datB[:,2,tsNumber]

		h_B, zedges2, xedges2 = np.histogram2d(z,x, bins=(zedges, xedges))

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
		abs_dens_log = abs_dens_log_raw * 0.8
		nonzero = np.nonzero(abs_dens_log > 0)
		abs_dens_log[nonzero] = abs_dens_log[nonzero] + 0.2

		if mode == "lin":
			img_data_RGB[:, :, 3] = abs_dens * alpha
		elif mode== "log":
			img_data_RGB[:, :, 3] = abs_dens_log * alpha

		im1.set_array(img_data_RGB)
		text_time.set_text("t=" + str(times[tsNumber]) +u"µs" +" " + annotate_string)

		return im1

	# call the animator.  blit=True means only re-draw the parts that have changed.
	if output_mode == 'video':
		anim = animation.FuncAnimation(fig, animate, frames=nFrames, blit=False)
		return (anim)
	elif output_mode == 'singleFrame':
		animate(nFrames)
		return (fig)


def render_XZ_density_animation(projectNames, selected, resultName, select_mode='substance', nFrames=400, interval=1,
                                s_lim=7, n_bins=50, annotation="", mode="lin", file_type='hdf5'):
	"""
	:param projectNames: simulation projects to compare (given as project basenames)
	:type projectNames: tuple of two strings
	:param selected: list of masses in the two simulation projects to compare
	:type selected: tuple of two floats
	:param resultName: basename for the rendering result
	:param select_mode: defines the mode for selection of particles:
		"mass" for selecting by mass,
		"substance" for chemical substance / chemical id
	:param nFrames: number of frames to render
	:param interval: interval in terms of time steps in the input data between the animation frames
	:type interval: int
	:param s_lim: spatial limits of the rendered spatial domain
			(given as distance from the origin of the coordinate system or explicit limits: [xlo, xhi, zlo, zhi]
	:type s_lim: float or list of two floats
	:param n_bins: number of density bins in the spatial directions or list of bin numbers ([x z])
	:type n_bins: int or list of two ints
	:param annotation: annotation string
	:type annotation: str
	:param mode: scale density linearly ("lin") or logarithmically ("log")
	:param file_type: type of the trajectory file,
		'json' for uncompressed json,
		'compressed' for compressed json
		'hdf5' for compressed hdf5
	"""

	if file_type == 'hdf5':
		file_ext = "_trajectories.hd5"
		tj0 = tra.read_hdf5_trajectory_file(projectNames[0] + file_ext)
		tj1 = tra.read_hdf5_trajectory_file(projectNames[1] + file_ext)
	elif file_type == 'compressed':
		file_ext = "_trajectories.json.gz"
		tj0 = tra.read_json_trajectory_file(projectNames[0] + file_ext)
		tj1 = tra.read_json_trajectory_file(projectNames[1] + file_ext)
	elif file_type == 'json':
		file_ext = "_trajectories.json"
		tj0 = tra.read_json_trajectory_file(projectNames[0] + file_ext)
		tj1 = tra.read_json_trajectory_file(projectNames[1] + file_ext)
	else:
		raise ValueError('illegal file type flag (not hdf5, json or compressed)')

	anim = animate_z_vs_x_density_plot([tj0, tj1], selected, nFrames, interval,
	                                   mode=mode, s_lim=s_lim, select_mode=select_mode,
	                                   annotate_string=annotation)
	anim.save(resultName + "_densitiesComparisonXZ.mp4", fps=20, extra_args=['-vcodec', 'libx264'])


def animate_scatter_plot(tr, xlim=None, ylim=None, numframes=None):
	fig = plt.figure(figsize=(13, 5))
	pos = tr['positions']
	# ap = tr['additional_parameters']
	masses = tr['masses']

	cmap = plt.cm.get_cmap('viridis')

	if not numframes:
		numframes = len(tr['times'])

	plt.subplot(1, 2, 1)
	scat1 = plt.scatter(pos[:, 0, 0], pos[:, 1, 0], s=10, alpha=0.1, c=masses, cmap=cmap)
	plt.xlabel("x position")
	plt.ylabel("y position")

	if ylim:
		plt.ylim(ylim[0])
	else:
		plt.ylim((np.min(pos[:, 1, :]), np.max(pos[:, 1, :])))

	if xlim:
		plt.xlim(xlim[0])
	else:
		plt.xlim((np.min(pos[:, 0, :]), np.max(pos[:, 0, :])))

	plt.subplot(1, 2, 2)
	scat2 = plt.scatter(pos[:, 1, 0], pos[:, 2, 0], s=10, alpha=0.1, c=masses, cmap=cmap)
	plt.xlabel("x position")
	plt.ylabel("z position")

	if ylim:
		plt.ylim(ylim[1])
	else:
		plt.ylim((np.min(pos[:, 2, :]), np.max(pos[:, 2, :])))

	if xlim:
		plt.xlim(xlim[1])
	else:
		plt.xlim((np.min(pos[:, 0, :]), np.max(pos[:, 0, :])))

	def update_scatter_plot(i, pos, scat1, scat2):

		# r_dist = np.sqrt(pos[:,0,i]**2.0 + pos[:,1,i]**2.0)
		# r_velo = np.sqrt(ap[:,0,i]**2.0 + ap[:,1,i]**2.0)
		scat1.set_offsets(np.transpose(np.vstack([pos[:, 0, i], pos[:, 1, i]])))
		scat2.set_offsets(np.transpose(np.vstack([pos[:, 0, i], pos[:, 2, i]])))
		# scat1.set_array(np.abs(sp[:,i]))
		# scat2.set_array(np.abs(sp[:,i]))
		return scat1, scat2

	ani = animation.FuncAnimation(fig, update_scatter_plot, frames=range(numframes),
	                              fargs=(pos, scat1, scat2))
	return(ani)


def render_scatter_animation(pname,result_name=None, xlim=None, ylim=None, numframes=None,compressed=True):
	if compressed:
		file_ext =  "_trajectories.json.gz"
	else:
		file_ext = "_trajectories.json"

	tr = tra.read_trajectory_file(pname + file_ext)
	ani = animate_scatter_plot(tr, xlim=xlim, ylim=ylim, numframes=numframes)

	if not result_name:
		result_name = pname

	ani.save(result_name + "_phaseSpace_spCharge.mp4", fps=20, extra_args=['-vcodec', 'libx264'])