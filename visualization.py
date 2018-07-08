# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pylab as plt
from matplotlib import animation
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

################## High Level Simulation Project Processing Methods ######################

def animate_z_vs_x_density_plot(dat,masses,nFrames,interval,
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
	datA = tra.filter_mass(dat[0]["positions"],dat[0]["masses"],masses[0])
	datB = tra.filter_mass(dat[1]["positions"],dat[1]["masses"],masses[1])

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

	tj0 = tra.read_trajectory_file(projectNames[0]+file_ext)
	tj1 = tra.read_trajectory_file(projectNames[1]+file_ext)
	anim = animate_z_vs_x_density_plot([tj0,tj1],masses,nFrames,delay,sLim=7,annotateString=annotation)
	anim.save(resultName+"_densitiesComparisonXZ.mp4", fps=20, extra_args=['-vcodec', 'libx264'])


def animate_z_vs_x_scatter_plot(tr, xlim=None, ylim=None, numframes=None):
	fig = plt.figure(figsize=(13, 5))
	pos = tr['positions']
	# ap = tr['additional_parameters']
	masses = tr['masses']

	cmap = plt.cm.get_cmap('viridis')

	if not numframes:
		numframes = len(tr['times'])

	plt.subplot(1, 2, 1)
	scat1 = plt.scatter(pos[:, 0, 0], pos[:, 1, 0], s=10, alpha=0.1, c=masses, cmap=cmap)
	# plt.xlabel("x position")
	# plt.ylabel("x velocity")

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
	# plt.xlabel("z position")
	# plt.ylabel("z velocity")

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


def render_XZ_scatter_animation(pname,result_name=None, xlim=None, ylim=None, numframes=None,compressed=True):
	if compressed:
		file_ext =  "_trajectories.json.gz"
	else:
		file_ext = "_trajectories.json"

	tr = tra.read_trajectory_file(pname + file_ext)
	ani = animate_z_vs_x_scatter_plot(tr, xlim=xlim,ylim=ylim,numframes=numframes)

	if not result_name:
		result_name = pname

	ani.save(result_name + "_phaseSpace_spCharge.mp4", fps=20, extra_args=['-vcodec', 'libx264'])