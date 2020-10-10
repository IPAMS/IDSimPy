.. _usersguide-visualization:

========================================
Visualization of IDSimF ion trajectories
========================================

IDSimPy provides functionality to plot and animate particle trajectory data from IDSimF results. Visualizing results is often key for the understanding of simulation results, and is thus an important aspect of IDSimPy. 


Custom plotting with matplotlib
===============================

Since the data structures in :py:class:`.Trajectory` objects are comparably simple, it is easy to do custom plotting of particle simulation result data with visualization libraries, e.g. matplotlib. For example, particle positions can be easily depicted in a scatter plot with matplotlib: 

.. code-block:: python 

    import os
    import numpy as np
    import matplotlib.pyplot as plt
    import IDSimPy.analysis.trajectory as tr


    # open a (in this case legacy) hdf5 trajectory file:
    hdf5_file = os.path.join('test','analysis','data', 'qitSim_2019_04_scanningTrapTest',
                                        'qitSim_2019_04_10_001_trajectories.hd5')

    tra = tr.read_legacy_hdf5_trajectory_file(hdf5_file)


    # extract positions of first recorded time step: 
    ts = traj_hdf5.get_positions(0)

    # create a 3d scatter plot of the particle positions in the first time step: 
    x,y,z = ts[:,0],ts[:,1],ts[:,2]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x,y,z)

This yields something like 

.. image:: images/user_guide_visualization_custom_scatter.svg
    :width: 400
    :alt: custom 3d scatter plot with matplotlib

Particle trace plots
====================

The traces of individual particles can be plotted in a flexible way with :py:func:`.plot_particle_traces`. This function can plot individual particle traces from different trajectories in one figure, which allows the direct comparison of different particles. 

The function renders the plot as PDF and takes the name of the result PDF file as first argument. The second argument, ``particle_definitions``, defines the particles to plot. It is a tuple of tuples, which each define the particles to plot for a :py:class:`.Trajectory` object. Each of those configuration lines consist of a :py:class:`.Trajectory`, a list of particle indices to plot, and a legend label for the series of particles defined by the line. 

For example 

.. code-block:: python 

    particle_definition = [
        (tra_1, (1, 2), "10_001"),
        (tra_2, 20, "10_002"),
    ]

in the following example, defines the particles 1 and 2 from trajectory :py:data:`tra_1` with the label ``10_001`` and particle 20 from trajectory :py:data:`tra_2` with the label ``10_002``:

.. code-block:: python 

    import os
    import IDSimPy.analysis.trajectory as tr
    import IDSimPy.analysis.visualization as vis

    # Read two (legacy) HDF5 trajectory files from the test files
    dat_path = os.path.join('..','test','analysis','data','qitSim_2019_04_scanningTrapTest')
    tra_1 = tr.read_legacy_hdf5_trajectory_file(os.path.join(dat_path,'qitSim_2019_04_10_001_trajectories.hd5'))
    tra_2 = tr.read_legacy_hdf5_trajectory_file(os.path.join(dat_path,'qitSim_2019_04_10_002_trajectories.hd5'))

    # Define parameters for plot
    result_name = 'test_particle_plotting_01'
    particle_definition = [
        (tra_1, (1, 2), "10_001"),
        (tra_2, 20, "10_002"),
    ]

    # Plot
    vis.plot_particle_traces(result_name, particle_definition)


The example yields something like 

.. image:: images/user_guide_visualization_particle_traces.svg
    :alt: Plot of particle traces rendered with IDSimPy

Particle density plots and animations
=====================================

Particle scatter plots and animations
=====================================
