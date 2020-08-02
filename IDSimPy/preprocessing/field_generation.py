# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import vtk


def transform_2d_axial_to_3d(R, Z, V, radial_component=False):
	"""
	Transforms 2d axial symmetric data into 3d cartesian data
	"""
	grid_r = R[0, :]
	len_r = len(grid_r)
	grid_z = Z[:, 0]

	X, Y, Z = np.meshgrid(grid_z, grid_r, grid_r)

	if not radial_component:
		result = np.zeros(np.shape(X))
	else:
		# prepare result for 2 component radial vector values:
		result = np.zeros(np.shape(X)+(2,))

	#iterate through entire 3d cartesian space domain:
	for i in range(len(grid_z)):
		for j in range(len(grid_r)):
			for k in range(len(grid_r)):
				x = X[j, i, k]
				y = Y[j, i, k]
				z = Z[j, i, k]

				r = np.sqrt(y*y + z*z) #radial distance
				r_ic = np.searchsorted(grid_r,r) #upper cell index in r direction were the current distance fits
				r_if = r_ic -1 #lower cell index

				# simple linear interpolation:
				if r_ic < len_r:

					# distance to the lower grid point:
					d_f = r - grid_r[r_if]

					# normalize distances:
					dn_f = d_f / (grid_r[r_ic]-grid_r[r_if])
					dn_c = 1 - dn_f

					# calculate interpolated value (distances are switched because low distance mean high weight of the value):
					res = dn_c * V[i, r_if] + dn_f * V[i, r_ic]

					if radial_component == True:
						# rotate radial component:
						phi = np.arctan2(y,z)
						result[j, i, k, 0] = np.sin(phi)*res
						result[j, i, k, 1] = np.cos(phi)*res
					else:
						result[j, i, k] = res

				else:
					if radial_component == True:
						result[j, i, k, :] = [0,0]
					else:
						result[j, i, k] = 0

	#flip LR:
	X_f = np.concatenate([X[:, :, ::-1], X], axis=2)
	Y_f = np.concatenate([Y[:, :, ::-1], Y], axis=2)
	Z_f = np.concatenate([Z[:, :, ::-1] * -1.0, Z], axis=2)

	if radial_component == True:
		flipped = np.copy(result[:, :, ::-1])
		flipped[:,:,:,1] = flipped[:,:,:,1]*-1
		result_f = np.concatenate([flipped, result], axis=2)
	else:
		result_f = np.concatenate([result[:, :, ::-1], result], axis=2)

	# flip UD:
	X_c = np.concatenate([X_f[::-1,: , :], X_f], axis=0)
	Y_c = np.concatenate([Y_f[::-1,: , :]* -1.0, Y_f], axis=0)
	Z_c = np.concatenate([Z_f[::-1,: , :], Z_f], axis=0)

	if radial_component == True:
		flipped = np.copy(result_f[::-1, :, :])
		flipped[:,:,:,0] = flipped[:,:,:,0]*-1
		result_c = np.concatenate([flipped, result_f], axis=0)
	else:
		result_c = np.concatenate([result_f[::-1, :, :], result_f], axis=0)

	return (X_c,Y_c,Z_c,result_c)



def write_3d_vector_fields_as_vtk_point_data(dat,result_filename,scale_factor=1.0):
	vtk_p = vtk.vtkPoints()
	x_vec, y_vec, z_vec = [np.array(x)*scale_factor for x in dat["grid_points"]]
	fields_dat = dat["fields"]

	xlen = len(x_vec)
	ylen = len(y_vec)
	zlen = len(z_vec)

	vtk_fields = []
	for fi in fields_dat:
		vfi = vtk.vtkDoubleArray()
		vfi.SetNumberOfComponents(3)
		vfi.SetName(fi['name'])
		vtk_fields.append(vfi)

	n_fields = len(vtk_fields)

	#x_dat = fields_dat[component_indices[0]]['data']
	#y_dat = fields_dat[component_indices[1]]['data']
	#z_dat = fields_dat[component_indices[2]]['data']

	for zi in range(zlen):
		for yi in range(ylen):
			for xi in range(xlen):
				vtk_p.InsertNextPoint([x_vec[xi], y_vec[yi], z_vec[zi]])
				for i in range(n_fields):
					vtk_fields[i].InsertNextTuple([
						fields_dat[i]['data'][0][yi, xi, zi],
						fields_dat[i]['data'][1][yi, xi, zi],
						fields_dat[i]['data'][2][yi, xi, zi]
					])

	vtk_grid = vtk.vtkStructuredGrid()
	vtk_grid.SetDimensions(xlen,ylen,zlen)
	vtk_grid.SetPoints(vtk_p)
	for i in range(n_fields):
		vtk_grid.GetPointData().AddArray(vtk_fields[i])

	#print(vtk_grid)
	writer = vtk.vtkXMLStructuredGridWriter()
	writer.SetFileName(result_filename)
	writer.SetInputData(vtk_grid)
	writer.Write()

def write_3d_scalar_fields_as_vtk_point_data(dat, result_filename, scale_factor=1.0):
	vtk_p = vtk.vtkPoints()
	x_vec,y_vec,z_vec = [np.array(x)*scale_factor for x in dat["grid_points"]]
	fields_dat = dat["fields"]

	xlen = len(x_vec)
	ylen = len(y_vec)
	zlen = len(z_vec)

	vtk_fields = []
	for fi in fields_dat:
		vfi = vtk.vtkDoubleArray()
		vfi.SetName(fi["name"])
		vtk_fields.append(vfi)

	n_fields = len(vtk_fields)

	for zi in range(zlen):
		for yi in range(ylen):
			for xi in range(xlen):
				vtk_p.InsertNextPoint([x_vec[xi],y_vec[yi],z_vec[zi]])
				for i in range(n_fields):
					vtk_fields[i].InsertNextValue(fields_dat[i]["data"][yi,xi,zi])

	vtk_grid = vtk.vtkStructuredGrid()
	vtk_grid.SetDimensions(xlen,ylen,zlen)
	vtk_grid.SetPoints(vtk_p)
	for i in range(n_fields):
		vtk_grid.GetPointData().AddArray(vtk_fields[i])

	#print(vtk_grid)
	writer = vtk.vtkXMLStructuredGridWriter()
	writer.SetFileName(result_filename)
	writer.SetInputData(vtk_grid)
	writer.Write()

def plot_3d_grid(meshgrid,field_dat,Xi):
	'''
	Plots a field imported from a comsol 3d csv file
	:param meshgrid: the meshgrid (as defined by numpy meshgrid) for the 3d field
	:param field_dat: the scalar field to plot
	:param Xi: the index of the slice in x direction to plot
	'''
	X,Y,Z = meshgrid
	P = field_dat
	X_ = X[:,Xi,  :]
	Y_ = Y[:,Xi,  :]
	Z_ = Z[:,Xi,  :]
	P_ = P[:,Xi,  :]
	fig = plt.figure()
	#ax = fig.gca(projection='3d')
	#surf = ax.plot_surface(Z_,Y_,P_,linewidth=0, antialiased=False,cmap=cm.coolwarm)
	#ax.set_zlim(-10, 10)
	#fig.colorbar(surf, shrink=0.5, aspect=5)
	plt.contourf(Z_,Y_,P_,20)
	plt.colorbar()
	plt.show()
