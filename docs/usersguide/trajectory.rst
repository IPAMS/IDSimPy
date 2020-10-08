.. _usersguide-trajectory:

===========================================================
Reading, filtering and analyzing IDSimF ion trajectory data
===========================================================

IDSimF applications typically produce result files which records the positions and additional attributes  of an simulated particle ensemble. Often, the combined dynamics of the simulated particle ensemble, recorded in an IDSimF result file, is called a *simulation trajectory*. 

One primary task of IDSimPy is to make IDSimF simulation trajectories available for analysis. Thus, IDSimPy aims to provide a simple to use but capable interface for reading IDSimF data into convenient data objects and analyze the imported data. 


The Trajectory class
====================

IDSimF particle simulation trajectory data is read into :py:class:`.Trajectory` objects which bundles the different parts of a simulation trajectory. 

-----------------------------
Positions and simulated times
-----------------------------

A minimal simulation trajectory holds at least the positions of the simulated particles at the recorded time steps of the simulation and the times of the time steps. The :py:attr:`.positions` and :py:attr:`.times` attributes of the Trajectory class can be directly accessed: 

.. code-block:: python

    # tra is an instance of Trajectory
    print(tra.times)

yields the raw times vector, for example: 

.. code-block:: none

    [0.0e+00 2.0e-06 4.0e-06 6.0e-06 8.0e-06 1.0e-05 1.2e-05 1.4e-05 1.6e-05
 1.8e-05 2.0e-05 2.2e-05 2.4e-05 2.6e-05 2.8e-05 3.0e-05 3.2e-05 3.4e-05
 3.6e-05 3.8e-05 4.0e-05 4.2e-05 4.4e-05 4.6e-05 4.8e-05 5.0e-05 5.2e-05
 5.4e-05 5.6e-05 5.8e-05 6.0e-05 6.2e-05 6.4e-05 6.6e-05 6.8e-05 7.0e-05
 7.2e-05 7.4e-05 7.6e-05 7.8e-05 8.0e-05 8.2e-05 8.4e-05 8.6e-05 8.8e-05
 9.0e-05 9.2e-05 9.4e-05 9.6e-05 9.8e-05 1.0e-04]

Similarly, the :py:attr:`positions` are directly accessible: 

.. code-block:: python

    # tra is an instance of Trajectory
    print(tra.positions)

yields the raw positions table, for example: 

.. code-block:: console

    [[[ 3.02714150e-04  1.07762564e-04 -2.31972357e-04 ...  2.97682673e-05
        2.97682673e-05  2.97682673e-05]
    [-2.14958651e-04 -7.65226723e-05  1.64724581e-04 ... -2.11385777e-05
    -2.11385777e-05 -2.11385777e-05]
    [ 4.15690971e-04 -3.76640237e-04  2.33336046e-04 ... -5.01252245e-03
    -5.01252245e-03 -5.01252245e-03]]

    ...

    [[ 4.52446606e-04  2.13949286e-04 -2.61958077e-04 ... -4.44824836e-04
    -2.72496836e-04  1.99392889e-04]
    [-3.64586303e-04 -1.72402622e-04  2.11088627e-04 ...  3.58444609e-04
        2.19580863e-04 -1.60672920e-04]
    [ 4.42521385e-04 -3.25540517e-04 -2.27704595e-06 ... -1.75142908e-04
        4.18572658e-04 -3.89383291e-04]]]

The exact type and shape of :py:attr:`positions` depends on the type of the Trajectory.

--------------------------------
Static vs. variable Trajectories
--------------------------------

Trajectories can be ``static`` or ``variable`` with respect to the number of particles: The number of particles in the trajectory do not change across the time steps in a ``static`` trajectory while the number changes between time steps in a ``variable`` trajectory. 

If a trajectory is static or variable can be determined by the :py:attr:`is_static_trajectory` flag attribute: 

.. code-block:: python

    # tra is a static instance of Trajectory
    print(tra.is_static_trajectory)
    # yields: True
    

How the positions are stored depends if it is a static or a variable trajectory: 

For **static** trajectories, the positions are stored in a three dimensional Numpy array. The particle index is the first dimension, the spatial dimension (``x``, ``y``, ``z``) is the second and the time steps are the third dimension. Thus, for example a static trajectory with an ensemble of 6 particles with 20 time steps would have a positions array with the shape ``(6, 3, 20)``.

For **variable** trajectories, the positions are stored as ``list`` of individual Numpy arrays, one per time step, which store the positions of the particles in the individual time step. The dimensions in the time step specific arrays are the particle index as first, and the spatial dimension as second dimension. Thus, a simulation with 3 time steps with 2, 5 and 9 particles in the first, second and third time step would have the shape: 

.. code-block:: python 

    [(2, 3), 
     (5, 3), 
     (9, 3)]

-------------------
Particle attributes
-------------------

IDSimF simulation applications can store an arbitrary number of additional attributes for the individual simulated particles in the simulation result files. Typical examples of particle attributes are the components of the velocity vector, the temperature or the chemical identity of the simulated particles. 

Particle attributes are stored in the :py:attr:`particle_attributes` attribute of the Trajectory object. They are stored in a data structure which is similar to :py:attr:`positions`. A static trajectory stores the particle attributes in a three dimensional Numpy Array with the dimensions ``[particle, attribute, time step]``. A variable trajectory stores the particle attributes as ``list`` of two dimensional Numpy arrays, one array per time step. The arrays have the dimensions ``[particle, attribute]``. 

The names of the particle attribute columns accessible in the :py:attr:`particle_attribute_names` attribute of the Trajectory object: 

.. code-block:: python

    print(trj.particle_attribute_names)

with a Trajectory ``trj`` yields for example 

.. code-block:: none

    ['velocity x', 'velocity y', 'velocity z', 'chemical id']


------------------------------
Trajectory data access methods
------------------------------

Unified access methods for positions and particle attributes
------------------------------------------------------------

The Trajectory class provides with :py:meth:`.Trajectory.get_positions` and with :py:meth:`.Trajectory.get_particle_attributes` unified access methods to the positions and particle attributes, which is independent from if the trajectory is static: 

.. code-block:: python

    # get particle positions in third time step from Trajectory tra:
    time_step = tra.get_positions(2)

    # get particle attributes in third time step from Trajectory tra:
    attributes = tra.get_particle_attributes(2)

The resulting positions array for a time step is always an array with the dimensions ``[particle, spatial dimension]``, the resulting particle attributes array for a time step is always an array with the dimensions ``[particle, attribute]``.

Single particle access
----------------------

Access to the position and attributes of a single particle at a specific time step in a trajectory is possible with the :py:meth:`.Trajectory.get_particle` method. It takes the particle and a time step index and returns the position and the particle attributes of the specified particle: 

.. code-block:: python 

    particle_index = 2
    time_step_index = 4
    position, attributes = traj.get_particle(particle_index, time_step_index)


Particle number
---------------

The number of particles in a trajectory is accessible with :py:meth:`.Trajectory.get_n_particles`. Static trajectories have a time step independent number of particles: 

.. code-block:: python 

    number_of_particles = tra.get_n_particles()

The number of particles vary between time steps in variable trajectories. Thus, the time step has to be specified for a variable trajectory: 

.. code-block:: python 

    time_step_index = 20
    number_of_particles = variable_tra.get_n_particles(time_step_index)

------------------------------
Optional trajectory attributes
------------------------------

Trajectory objects can have an arbitrary set of optional attributes, which are not commonly set by all IDSimF simulation applications. Typical examples are the masses of simulated particles or the charges of simulated particles. The optional trajectory attributes are technically stored as key-value pairs in a ``dict`` which can be accessed with the :py:attr:`optional_attributes` attribute of the Trajectory class. 

To allow structured access to the optional trajectory attributes, an extensible set of semantic keys is provided by the :py:class:`.OptionalAttribute` enum class. 

For example, the retrieval of the particle masses from a trajectory ``tra`` is done by

.. code-block:: python 
    
    import IDSimPy.analysis as ia

    particle_masses = tra.optional_attributes[ia.OptionalAttribute.PARTICLE_MASSES]

:py:class:`.OptionalAttribute` has currently two optional trajectory attribute keys: 

* :py:attr:`.OptionalAttribute.PARTICLE_MASSES` masses of the simulated particles
* :py:attr:`.OptionalAttribute.PARTICLE_CHARGES` charges of the simulated particles



Reading trajectory data files
=============================

IDSimPy provides file reader functions which read IDSimF trajectory files and construct :py:class:`.Trajectory` objects from the read data. 

The primary IDSimF trajectory file format is HDF5 which can be opened with :py:func:`.read_hdf5_trajectory_file`. The file reading function takes the name of the HDF5 trajectory file to open as argument and returns a :py:class:`.Trajectory` object:

.. code-block:: python 

    import IDSimPy.analysis.trajectory as tra

    hdf5_file_name = os.path.join('data', 'simulation_trajectory.hd5')
    traj = tra.read_hdf5_trajectory_file(hdf5_file_name)


There are two legacy file formats which are used by some legacy IDSimF applicatiions: JSON trajectories and legacy HDF5 files. They can be opened in a similar way by their specific reading functions :py:func:`.read_json_trajectory_file` and :py:func:`.read_legacy_hdf5_trajectory_file`.

Filtering trajectory data and selecting particles
=================================================

The analysis of trajectory data often requires the selection of individual groups of particles from trajectory data based on some characteristics or conditions of the particles, e.g. particle attributes. The selection of particles is done with *filtering* functions, which take a :py:class:`.Trajectory`, apply a selection and construct a new Trajectory object with the filtered particle ensemble. 

A simple selection method is to select particles by a given value of a specific particle attribute. This is done with :py:func:`.filter_attribute`, which takes a :py:class:`.Trajectory` to be filtered, the name of the particle attribute which should be used for filtering and the value which is filtered for. For example, selection of all particles with a ``chemical_id`` of 2 from a trajectory object ``tra``:

.. code-block:: python 

    import IDSimPy.analysis as ia

    tra_filtered = ia.filter_attribute(tra, 'chemical id', 2)

If simple selection based on a single particle attribute is not sufficient, :py:func:`.select` provides a more flexible mechanism to select particles from a :py:class:`.Trajectory` based on custom conditions. This function also takes a :py:class:`.Trajectory` object with the data to filter. The second argument to the function is a custom derived or constructed particle attribute which should be used for filtering ("selector_data"). The third argument is the value to filter for. 

For example, selection of all particles with a position outside a radius of 5.0e-4 around the coordinate system origin: 

.. code-block:: python 

    
    import numpy as np
    import IDSimPy.analysis.trajectory as tr

    # `tra` is an imported trajectory object 

    # push positions of individual time steps into a vector for processing: 
    positions = [ tra.get_positions(i) for i in range(tra.n_timesteps)]

    # calculate length of position vector of the particles and check if longer than 5.0e-4: 
    condition = [ np.linalg.norm(pos, axis=1) > 5.0e-4 for pos in positions]

    # filter trajectory with custom condition:
    tra_filtered = tr.select(tra, condition, True)

If selector data is a one dimensional vector, the same filtering is applied to all time steps. If selector data is a ``list`` of selector data vectors, one per time step, an individual filtering for every time step is applied. 

Analyzing trajectory data
=========================

It is planned to provide a set of functions with IDSimPy to analyze IDSimF trajectory data. Currently, only one general analysis function is part of IDSimPy: :py:func:`.center_of_charge` takes a :py:class:`.Trajectory` object and returns the position of the center of charge for every time step. 

