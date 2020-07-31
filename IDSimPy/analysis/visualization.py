# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.image import NonUniformImage
from . import trajectory as tra

__all__ = (
        'plot_particles_path',
        'plot_density_xz',
        'animate_xz_density',
        'render_xz_density_animation',
		'animate_xz_density_comparison_plot',
		'render_xz_density_comparison_animation',
		'animate_scatter_plot',
		'animate_variable_scatter_plot',
		'render_scatter_animation')

# Simple Plot Methods ######################

def plot_particles_path(trajectory_data, pl_filename, p_indices, plot_mark='*-', time_range=(0, 1)):
	"""
	Plots the paths of a selection of particles in a x,z and y,z projection
	:param trajectory_data: trajectory input data
	:type trajectory_data: list of lists of trajectory dictionaries from read_trajectory_file and an according label
	:param pl_filename: the basename of the plot image files to create
	:param p_indices:
	:type p_indices: list of integers
	:param plot_mark: matplotlib plot format string which is used for the path-plots
	:type plot_mark: str
	:param time_range: range of times to plot (given as a fraction between 0 and 1)
	:type time_range: tuple of two floats between 0 and 1
	"""
	fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

	for tr in trajectory_data:
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


def plot_density_xz(
		trajectory_data, time_index,
		xedges=None, zedges=None,
		figsize=(7, 7), axis_equal=True):
	"""
	Renders an density plot in a z-x projection
	:param trajectory_data: a trajectories vector from an imported trajectories object
	:type trajectory_data: trajectories vector from dict returned from readTrajectoryFile
	:param time_index: index of the time step to render
	:type time_index: int
	:param xedges: the edges of the bins of the density plot (2d histogram bins) in x direction, if None the
	maxium extend is used with 50 bins, if a number n, the maximum extend is used with n bins
	:type xedges: iterable or list / array or int
	:param zedges: the edges of the bins of the density plot (2d histogram bins) in z direction, if None the
	maxium extend is used with 50 bins, if a number n, the maximum extend is used with n bins
	:type zedges: iterable or list / array or int
	:param figsize: the figure size
	:type figsize: tuple of two floats
	:param axis_equal: if true, the axis are rendered with equal scaling
	"""

	fig = animate_xz_density(
		trajectory_data, xedges=xedges, zedges=zedges,
		n_frames=time_index, figsize=figsize,
		axis_equal=axis_equal, output_mode='singleFrame')

	return fig

# High Level Simulation Project Processing Methods ######################


# density plots #########################################################

def animate_xz_density(
		trajectory_data,
		xedges=None, zedges=None,
		figsize=(7, 7), interval=1, n_frames=10,
		output_mode='animation', axis_equal=True):

	"""
	Animates an density plot in a z-x projection, still frames can also be rendered

	:param trajectory_data: a trajectories vector from an imported trajectories object
	:type trajectory_data: trajectories vector from dict returned from readTrajectoryFile
	:param xedges: the edges of the bins of the density plot (2d histogram bins) in x direction, if None the
	maximum extend is used with 50 bins, if a number n, the maximum extend is used with n bins
	:type xedges: iterable or list / array or int
	:param zedges: the edges of the bins of the density plot (2d histogram bins) in z direction, if None the
	maxi,um extend is used with 50 bins, if a number n, the maximum extend is used with n bins
	:type zedges: iterable or list / array or int
	:param figsize: the figure size
	:type figsize: tuple of two floats
	:param interval: interval in terms of data frames in the input data between the animation frames
	:type interval: int
	:param n_frames: number of frames to render or the frame index to render if single frame mode
	:type n_frames: int
	:param output_mode: returns animation object when 'animation', 'singleFrame' returns a single frame figure
	:type output_mode: str
	:param axis_equal: if true, the axis are rendered with equal scaling
	:type axis_equal: bool

	:return: animation or figure
	"""

	x = trajectory_data[:, 0, :]
	z = trajectory_data[:, 2, :]

	x_min = np.min(x)
	x_max = np.max(x)
	z_min = np.min(z)
	z_max = np.max(z)

	if xedges is None:
		xedges = np.linspace(x_min, x_max, 50)
	elif type(xedges) == int:
		xedges = np.linspace(x_min, x_max, xedges)

	if zedges is None:
		zedges = np.linspace(z_min, z_max, 50)
	elif type(zedges) == int:
		zedges = np.linspace(z_min, z_max, zedges)

	hist_vals, xed, zed = np.histogram2d(x[:, 0], z[:, 0], bins=(xedges, zedges))
	hist_vals = hist_vals.T
	fig = plt.figure(figsize=figsize)

	ax = fig.add_subplot(111)
	im = NonUniformImage(ax, interpolation='nearest')
	xcenters = xed[:-1] + 0.5 * (xed[1:] - xed[:-1])
	zcenters = zed[:-1] + 0.5 * (zed[1:] - zed[:-1])
	im.set_data(xcenters, zcenters, hist_vals)
	ax.images.append(im)
	im.set_extent(im.get_extent())  # workaround for minor issue in matplotlib ocurring in jupyter lab
	ax.set_xlim(xed[0], xed[-1])
	ax.set_ylim(zed[0], zed[-1])
	if axis_equal:
		ax.set_aspect('equal')

	def animate(i):
		ts_number = i*interval
		h_vals, _, _ = np.histogram2d(x[:, ts_number], z[:, ts_number], bins=(xedges, zedges))
		h_vals = h_vals.T
		im.set_data(xcenters, zcenters, h_vals)

	if output_mode == 'animation':
		anim = animation.FuncAnimation(fig, animate, frames=n_frames, blit=False)
		return anim
	elif output_mode == 'singleFrame':
		animate(n_frames)
		return fig


def render_xz_density_animation(
		project_name, result_name,
		xedges=None, zedges=None,
		figsize=(7, 7), interval=1, n_frames=None,
		axis_equal=True, file_type='hdf5'):
	"""
	Renders an animation of particle density

	:param project_name: simulation project to import and render (given as project basename)
	:type project_name: str
	:param result_name: basename for the rendering result
	:type result_name: str
	:param xedges: the edges of the bins of the density plot (2d histogram bins) in x direction, if None the
	maximum extend is used with 50 bins, if a number n, the maximum extend is used with n bins
	:type xedges: iterable or list / array or int
	:param zedges: the edges of the bins of the density plot (2d histogram bins) in z direction, if None the
	maxi,um extend is used with 50 bins, if a number n, the maximum extend is used with n bins
	:type zedges: iterable or list / array or int
	:param interval: interval in terms of data frames in the input data between the animation frames
	:type interval: int
	:param figsize: the figure size
	:type figsize: tuple of two floats
	:param interval: interval in terms of data frames in the input data between the animation frames
	:type interval: int
	:param n_frames: number of frames to render or the frame index to render if single frame mode
	:type n_frames: int
	:param axis_equal: if true, the axis are rendered with equal scaling
	:type axis_equal: bool
	:param file_type: type of the trajectory file,
		'json' for uncompressed json,
		'compressed' for compressed json
		'hdf5' for compressed hdf5
	:type file_type: str
	"""
	if file_type == 'hdf5':
		file_ext = "_trajectories.hd5"
		tr = tra.read_hdf5_trajectory_file(project_name + file_ext)
	elif file_type == 'legacy_hdf5':
		file_ext = "_trajectories.hd5"
		tr = tra.read_legacy_hdf5_trajectory_file(project_name + file_ext)
	elif file_type == 'compressed':
		file_ext = "_trajectories.json.gz"
		tr = tra.read_json_trajectory_file(project_name + file_ext)
	elif file_type == 'json':
		file_ext = "_trajectories.json"
		tr = tra.read_json_trajectory_file(project_name + file_ext)
	else:
		raise ValueError('illegal file type flag (not hdf5, json or compressed)')

	if not n_frames:
		n_frames = len(tr['times'])

	ani = animate_xz_density(
		tr['positions'], xedges=xedges, zedges=zedges, n_frames=n_frames, figsize=figsize,
		axis_equal=axis_equal, interval=interval, output_mode='animation')

	ani.save(result_name + "_densityXZ.mp4", fps=20, extra_args=['-vcodec', 'libx264'])


def animate_xz_density_comparison_plot(
		trajectory_data, selected, n_frames, interval,
		select_mode='substance', output_mode='video', mode='lin',
		s_lim=3, n_bins=100, basesize=17, alpha=1, colormap=plt.cm.coolwarm,
		annotate_string=""):
	"""
	Animate the densities of two mostly symmetric ion clouds (probably from a QIT simulation) in a z-x projection.

	:param trajectory_data: imported trajectories object
	:type trajectory_data: dict returned from the readTrajectoryFile methods
	:param selected: two element list with values to select particles to be rendered
	:type selected: list
	:param n_frames: number of frames to export
	:param interval: interval in terms of data frames in the input data between the animation frames
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
	:type basesize: float
	:param alpha: blending factor for graphical blending the densities of the two species
	:param colormap: a colormap for the density rendering (a pure species will end up on one side of the colormap)
	:param annotate_string: an optional string which is rendered into the animation as annotation
	:return: animation object or figure (depends on the file mode)
	"""

	if select_mode == 'mass':
		select_parameter = [trajectory_data[0]['masses'], trajectory_data[1]['masses']]

	elif select_mode == 'substance':
		id_column = trajectory_data[0]['additional_names'].index('chemical id')
		select_parameter = [
			trajectory_data[0]['additional_parameters'][:, id_column, :],
			trajectory_data[1]['additional_parameters'][:, id_column, :]]
	else:
		raise ValueError('Invalid select_mode')

	times_a = trajectory_data[0]["times"]
	times_b = trajectory_data[1]["times"]

	if len(times_a) != len(times_b):
		raise ValueError('Length of trajectories differ')
	if not (times_a == times_b).all():
		raise ValueError('The times of the trajectories differ')
	if n_frames*interval > len(times_a):
		raise ValueError(
			'number of frames * interval (' + str(n_frames * interval) +
			') is longer than trajectory (' + str(len(times_a)) + ')')

	if selected[0] == "all":
		dat_a = trajectory_data[0]["positions"]
	else:
		dat_a = tra.filter_parameter(trajectory_data[0]["positions"], select_parameter[0], selected[0])

	if selected[1] == "all":
		dat_b = trajectory_data[1]["positions"]
	else:
		dat_b = tra.filter_parameter(trajectory_data[1]["positions"], select_parameter[1], selected[1])

	if output_mode == 'video':
		plt.figure(figsize=[10, 10])
	elif output_mode == 'singleFrame':
		plt.figure(figsize=[6, 6])

	if not hasattr(s_lim, "__iter__"):  # is not iterable
		limits = [-s_lim, s_lim, -s_lim, s_lim]
	else:
		limits = s_lim

	if not hasattr(n_bins, "__iter__"):  # is not iterable
		bins = [n_bins, n_bins]
	else:
		bins = n_bins

	xedges = np.linspace(limits[0], limits[1], bins[0])
	zedges = np.linspace(limits[2], limits[3], bins[1])
	h_vals = np.random.rand(len(xedges), len(zedges))
	fig_ratio = (limits[3]-limits[2]) / (limits[1]-limits[0])
	fig = plt.figure(figsize=(basesize, basesize*fig_ratio+basesize/10.0))
	ax = fig.add_subplot(1, 1, 1, ylim=(zedges[0], zedges[-1]), xlim=(xedges[0], xedges[-1]))

	im1 = ax.imshow(
		h_vals, interpolation='nearest', origin='low', alpha=1, vmin=0, vmax=10, cmap="Reds",
		extent=[xedges[0], xedges[-1], zedges[0], zedges[-1]])

	text_time = ax.annotate(
		"TestText", xy=(0.02, 0.96), xycoords="figure fraction",
		horizontalalignment="left", verticalalignment="top",
		fontsize=20);

	plt.xlabel("x (mm)")
	plt.ylabel("z (mm)")

	def animate(i):
		ts_number = i*interval
		# if the dat objects are lists: we have filtered the particles in a way that the number of selected
		# particles change between the timesteps and we got a list of individual particle vectors
		if isinstance(dat_a, list):
			x = dat_a[ts_number][:, 0]
			z = dat_a[ts_number][:, 2]
		else:
			x = dat_a[:, 0, ts_number]
			z = dat_a[:, 2, ts_number]
		h_a, zedges2, xedges2 = np.histogram2d(z, x, bins=(zedges, xedges))

		if isinstance(dat_b, list):
			x = dat_b[ts_number][:, 0]
			z = dat_b[ts_number][:, 2]
		else:
			x = dat_b[:, 0, ts_number]
			z = dat_b[:, 2, ts_number]

		h_b, zedges2, xedges2 = np.histogram2d(z, x, bins=(zedges, xedges))

		nf_a = np.max(h_a)
		nf_b = np.max(h_b)

		rel_conc = h_a / (h_a + h_b + 0.00001)
		img_data_rgb = colormap(rel_conc)
		h_a_log = np.log10(h_a + 1)
		h_b_log = np.log10(h_b + 1)
		nf_a_log = np.max(h_a_log)
		nf_b_log = np.max(h_b_log)
		abs_dens = (h_a + h_b) / (nf_a + nf_b)
		abs_dens_log_raw = (h_a_log + h_b_log) / (nf_a_log + nf_b_log)
		abs_dens_log = abs_dens_log_raw * 0.8
		nonzero = np.nonzero(abs_dens_log > 0)
		abs_dens_log[nonzero] = abs_dens_log[nonzero] + 0.2

		if mode == "lin":
			img_data_rgb[:, :, 3] = abs_dens * alpha
		elif mode == "log":
			img_data_rgb[:, :, 3] = abs_dens_log * alpha

		im1.set_array(img_data_rgb)
		text_time.set_text("t=" + str(times_a[ts_number]) + u"µs" + " " + annotate_string)

		return im1

	# call the animator.  blit=True means only re-draw the parts that have changed.
	if output_mode == 'video':
		anim = animation.FuncAnimation(fig, animate, frames=n_frames, blit=False)
		return anim
	elif output_mode == 'singleFrame':
		animate(n_frames)
		return fig


def render_xz_density_comparison_animation(
		project_names, selected, result_name,
		select_mode='substance', n_frames=400, interval=1,
		s_lim=7, n_bins=50, base_size=12,
		annotation="", mode="lin", file_type='legacy_hdf5'):
	"""
	Reads two trajectories, renders XZ density projection of two ion clouds in the trajectories and writes
	a video file with the result.

	:param project_names: simulation projects to compare (given as project basenames)
	:type project_names: tuple of two strings
	:param selected: list of masses in the two simulation projects to compare
	:type selected: tuple of two floats
	:param result_name: basename for the rendering result
	:param select_mode: defines the mode for selection of particles:
		"mass" for selecting by mass,
		"substance" for chemical substance / chemical id
	:param n_frames: number of frames to render
	:param interval: interval in terms of data frames in the input data between the animation frames
	:type interval: int
	:param s_lim: spatial limits of the rendered spatial domain
			(given as distance from the origin of the coordinate system or explicit limits: [xlo, xhi, zlo, zhi]
	:type s_lim: float or list of two floats
	:param n_bins: number of density bins in the spatial directions or list of bin numbers ([x z])
	:type n_bins: int or list of two ints
	:param base_size: the base size of the plot
	:type base_size: float
	:param annotation: annotation string
	:type annotation: str
	:param mode: scale density linearly ("lin") or logarithmically ("log")
	:param file_type: type of the trajectory file,
		'json' for uncompressed json,
		'compressed' for compressed json
		'hdf5' for compressed hdf5
	"""

	if file_type == 'legacy_hdf5':
		file_ext = "_trajectories.hd5"
		tj0 = tra.read_legacy_hdf5_trajectory_file(project_names[0] + file_ext)
		tj1 = tra.read_legacy_hdf5_trajectory_file(project_names[1] + file_ext)
	elif file_type == 'compressed':
		file_ext = "_trajectories.json.gz"
		tj0 = tra.read_json_trajectory_file(project_names[0] + file_ext)
		tj1 = tra.read_json_trajectory_file(project_names[1] + file_ext)
	elif file_type == 'json':
		file_ext = "_trajectories.json"
		tj0 = tra.read_json_trajectory_file(project_names[0] + file_ext)
		tj1 = tra.read_json_trajectory_file(project_names[1] + file_ext)
	else:
		raise ValueError('illegal file type flag (not hdf5, json or compressed)')

	anim = animate_xz_density_comparison_plot(
		[tj0, tj1], selected, n_frames, interval,
		mode=mode, s_lim=s_lim, select_mode=select_mode, n_bins=n_bins,
		basesize=base_size, annotate_string=annotation)
	anim.save(result_name + "_densitiesComparisonXZ.mp4", fps=20, extra_args=['-vcodec', 'libx264'])


# scatter plots #############################################################
def animate_scatter_plot(
		trajectory_data, xlim=None, ylim=None, zlim=None,
		n_frames=None, interval=1,
		color_parameter=None, crange=None, cmap=plt.cm.get_cmap('viridis'),
		alpha=0.1, figsize=(13, 5)):
	"""
	Generates a scatter animation of the particles in an ion trajectory

	:param trajectory_data: a particle trajectory
	:type trajectory_data: dict returned from the readTrajectoryFile methods
	:param xlim: limits of the plot in x direction (if None, the maximum of the x position range is used)
	:type xlim: tuple of two floats
	:param ylim: limits of the plot in y direction (if None, the maximum of the y position range is used)
	:type ylim: tuple of two floats
	:param zlim: limits of the plot in z direction (if None, the maximum of the z position range is used)
	:type zlim: tuple of two floats
	:param n_frames: number of rendered frames, (if None the maximum number of frames is rendered)
	:type n_frames: int
	:param interval: interval in terms of data frames in the input data between the animation frames
	:type interval: int
	:param color_parameter: name of an additional parameter of the trajectory used for coloring or a vector of manual
	values for coloring
	:type color_parameter: str or iterable (ndarray, list, tuple)
	:param crange: manual color range, given as tuple. Colormap spans from c_range[0] to c_range[1]
	:type crange: two element tuple of numeric
	:param cmap: a matplotlib colormap for colorization of the scatter plot
	:type cmap: matplotlib.colors.Colormap
	:param alpha: an alpha value for the plots
	:type alpha: float
	:return: an animation object with the animation
	:param figsize: size of the figure of the plot
	:type figsize: tuple of two numbers
	"""
	fig = plt.figure(figsize=figsize)
	positions = trajectory_data['positions']
	n_timesteps = trajectory_data['n_timesteps']

	c_param = None
	if not (color_parameter is None):
		ap = trajectory_data['additional_parameters']
		ap_names = trajectory_data['additional_names']

		if type(color_parameter) is str:
			cp_index = ap_names.index(color_parameter)
			c_param = ap[:, cp_index, :]
		elif hasattr(color_parameter, "__iter__"):  # is iterable
			c_param = np.tile(color_parameter, (n_timesteps, 1)).T

	if not n_frames:
		n_frames = int(np.floor(n_timesteps/interval))

	if n_frames*interval > n_timesteps:
		raise ValueError(
			'number of frames * interval (' + str(n_frames * interval) +
			') is longer than trajectory (' + str(n_timesteps) + ')')

	def create_plot(xindex, yindex, x_li, y_li):
		if color_parameter is None:
			scatterplot = plt.scatter(positions[:, xindex, 0], positions[:, yindex, 0], s=10, alpha=alpha)
		else:
			if crange is None:
				scatterplot = plt.scatter(
					positions[:, xindex, 0], positions[:, yindex, 0], s=10,
					alpha=alpha, c=c_param[:, 0], cmap=cmap)
			else:
				scatterplot = plt.scatter(
					positions[:, xindex, 0], positions[:, yindex, 0], s=10,
					alpha=alpha, c=c_param[:, 0], vmin=crange[0], vmax=crange[1], cmap=cmap)

		if y_li:
			plt.ylim(y_li)
		else:
			plt.ylim((np.min(positions[:, yindex, :]), np.max(positions[:, yindex, :])))

		if x_li:
			plt.xlim(x_li)
		else:
			plt.xlim((np.min(positions[:, xindex, :]), np.max(positions[:, xindex, :])))

		return scatterplot

	plt.subplot(1, 2, 1)
	scat_xy = create_plot(0, 1, xlim, ylim)
	plt.xlabel("x position")
	plt.ylabel("y position")

	plt.subplot(1, 2, 2)
	scat_xz = create_plot(0, 2, xlim, zlim)
	plt.xlabel("x position")
	plt.ylabel("z position")

	def update_scatter_plot(i, pos, scat1, scat2):
		ts = i * interval
		scat1.set_offsets(np.transpose(np.vstack([pos[:, 0, ts], pos[:, 1, ts]])))
		scat2.set_offsets(np.transpose(np.vstack([pos[:, 0, ts], pos[:, 2, ts]])))

		if not (c_param is None):
			scat1.set_array(c_param[:, ts])
			scat2.set_array(c_param[:, ts])

		return scat1, scat2

	ani = animation.FuncAnimation(
		fig, update_scatter_plot, frames=range(n_frames),
		fargs=(positions, scat_xy, scat_xz))
	return ani


def animate_variable_scatter_plot(
		trajectory_data, xlim=None, ylim=None, zlim=None, n_frames=None, interval=1,
		color_parameter=None, cmap=plt.cm.get_cmap('viridis'), alpha=0.1, figsize=(13, 5)):
	"""
	TODO:/ FIXME: Usage of color parameter is not yet implemented
	(only color parameter as aux parameter here)

	Generates a scatter animation of the particles in an ion trajectory
	with varying particle number in the simulation frames

	:param trajectory_data: a particle trajectory
	:type trajectory_data: dict returned from the readTrajectoryFile methods
	:param xlim: limits of the plot in x direction (if None, the maximum of the x position range is used)
	:type xlim: tuple of two floats
	:param ylim: limits of the plot in y direction (if None, the maximum of the y position range is used)
	:type ylim: tuple of two floats
	:param zlim: limits of the plot in z direction (if None, the maximum of the z position range is used)
	:type zlim: tuple of two floats
	:param n_frames: number of rendered frames, (if None the maximum number of frames is rendered)
	:type n_frames: int
	:param interval: interval in terms of data frames in the input data between the animation frames
	:type interval: int
	:param color_parameter: name of an additional parameter of the trajectory used for coloring or a vector of manual
	values for coloring
	:type color_parameter: str or iterable (ndarray, list, tuple)
	:param cmap: a matplotlib colormap for colorization of the scatter plot
	:type cmap: matplotlib.colors.Colormap
	:param alpha: an alpha value for the plots
	:type alpha: float
	:return: an animation object with the animation
	:param figsize: size of the figure of the plot
	:type figsize: tuple of two numbers
	"""
	fig = plt.figure(figsize=figsize)
	n_timesteps = trajectory_data['n_timesteps']
	pos = trajectory_data['positions']

	if 'additional_parameters' in trajectory_data.keys():
		ap = trajectory_data['additional_parameters']
		ap_names = trajectory_data['additional_names']

	c_param = None
	if not (color_parameter is None):

		if type(color_parameter) is str:
			cp_index = ap_names.index(color_parameter)
			c_param = [ts_ap[:, cp_index] for ts_ap in ap]
		elif hasattr(color_parameter, "__iter__"):  # is iterable
			c_param = np.tile(color_parameter,(n_timesteps,1)).T

	if not n_frames:
		n_frames = int(np.floor(n_timesteps/interval))

	if n_frames*interval > n_timesteps:
		raise ValueError(
			'number of frames * interval (' + str(n_frames * interval) +
			') is longer than trajectory (' + str(n_timesteps) + ')')

	def render_scatter_plot(i):
		plt.clf()  # clear figure for a fresh plot

		ts_pos = pos[i]
		# ts_ap = ap[i]

		plt.subplot(1, 2, 1)
		if color_parameter is None:
			plt.scatter(ts_pos[:, 0], ts_pos[:, 1], s=10, alpha=alpha)
		else:
			# ts_cp = c_param[i]
			plt.scatter(ts_pos[:, 0], ts_pos[:, 1], s=10, alpha=alpha)  # c=ts_cp[0], cmap=cmap)

		plt.xlabel("x position")
		plt.ylabel("y position")

		if ylim:
			plt.ylim(ylim)
		else:
			plt.ylim((np.min(ts_pos[:, 1]), np.max(ts_pos[:, 1])))

		if xlim:
			plt.xlim(xlim)
		else:
			plt.xlim((np.min(ts_pos[:, 0]), np.max(ts_pos[:, 0])))

		plt.subplot(1, 2, 2)
		if color_parameter is None:
			plt.scatter(ts_pos[:, 0], ts_pos[:, 2], s=10, alpha=alpha)
		else:
			plt.scatter(ts_pos[:, 0], ts_pos[:, 2], s=10, alpha=alpha)  # , c=c_param[:,0], cmap=cmap)
		plt.xlabel("x position")
		plt.ylabel("z position")

		if zlim:
			plt.ylim(zlim)
		else:
			plt.ylim((np.min(ts_pos[:, 2]), np.max(ts_pos[:, 2])))

		if xlim:
			plt.xlim(xlim)
		else:
			plt.xlim((np.min(ts_pos[:, 0]), np.max(ts_pos[:, 0])))

	ani = animation.FuncAnimation(fig, render_scatter_plot, frames=range(n_frames))
	return ani


def render_scatter_animation(
		project_name, result_name, xlim=None, ylim=None, zlim=None, n_frames=None, interval=1,
		color_parameter=None, cmap=plt.cm.get_cmap('viridis'), alpha=0.1, fps=20,
		figsize=(13, 5), file_type='hdf5'):
	"""
	Reads an ion trajectory file, generates a scatter animation of the particles in an ion trajectory and
	writes a video file with the animation

	:param project_name: simulation project to read and animate (given as basename)
	:type project_name: str
	:param result_name: name of the result video file
	:type result_name: str
	:param xlim: limits of the plot in x direction (if None, the maximum of the x position range is used)
	:type xlim: tuple of two floats
	:param ylim: limits of the plot in y direction (if None, the maximum of the y position range is used)
	:type ylim: tuple of two floats
	:param zlim: limits of the plot in z direction (if None, the maximum of the z position range is used)
	:type zlim: tuple of two floats
	:param n_frames: number of rendered frames, (if None the maximum number of frames is rendered)
	:type n_frames: int
	:param interval: interval in terms of data frames in the input data between the animation frames
	:type interval: int
	:param color_parameter: name of an additional parameter of the trajectory used for coloring or a vector of manual
	values for coloring
	:type color_parameter: str or iterable (ndarray, list, tuple)
	:param cmap: a matplotlib colormap for colorization of the scatter plot
	:type cmap: matplotlib.colors.Colormap
	:param alpha: an alpha value for the plots
	:type alpha: float
	:param fps: frames per second in the rendered video
	:type fps: int
	:param figsize: size of the figure of the plot
	:type figsize: tuple of two numbers
	:param file_type: type of the trajectory file,
		'json' for uncompressed json,
		'compressed' for compressed json
		'hdf5' for compressed hdf5
	:type file_type: str
	"""
	if file_type == 'hdf5':
		file_ext = "_trajectories.hd5"
		tr = tra.read_hdf5_trajectory_file(project_name + file_ext)
	elif file_type == 'legacy_hdf5':
		file_ext = "_trajectories.hd5"
		tr = tra.read_legacy_hdf5_trajectory_file(project_name + file_ext)
	elif file_type == 'compressed':
		file_ext = "_trajectories.json.gz"
		tr = tra.read_json_trajectory_file(project_name + file_ext)
	elif file_type == 'json':
		file_ext = "_trajectories.json"
		tr = tra.read_json_trajectory_file(project_name + file_ext)
	else:
		raise ValueError('illegal file type flag (not legacy_hdf5, hdf5, json or compressed)')

	if tr['static_trajectory']:
		plot_fct = animate_scatter_plot
	else:
		plot_fct = animate_variable_scatter_plot

	ani = plot_fct(
		tr, xlim=xlim, ylim=ylim, zlim=zlim, n_frames=n_frames, interval=interval,
		color_parameter=color_parameter, cmap=cmap, alpha=alpha, figsize=figsize)

	ani.save(result_name + "_scatter.mp4", fps=fps, extra_args=['-vcodec', 'libx264'])
