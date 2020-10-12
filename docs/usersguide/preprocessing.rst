.. _usersguide-preprocessing:

==================================================
Preprocessing and generating input data for IDSimF 
==================================================

IDSimPy provides also functionality to generate some types of input data for IDSimF simulations. There are currently two main types of input data for IDSimF simulations: 

* **Ion cloud files**, which define a custom set of particles. They are primarily used for initialization of the simulated particle ensemble at the begin of the IDSimF simulation run. 
* **Field data files**, which define scalar or vector fields on a regular spatial grid. They are commonly used to transfer spatially resolved input data, e.g. electric fields or gas flow data from other numerical solvers to IDSimF simulations. 

Both file types can be generated with IDSimPy. 

Particle ensemble (ion cloud) files
===================================

---------------------
Ion cloud file format
---------------------

Ion cloud files are simple text files with data columns separated by the semicolon character as delimiter (csv files). Every line defines an individual particle. The lines have 9 columns and the first character can be ``#`` to mark a line as comment. 

For example:

.. code-block:: none

    #pos x; pos y; pos z; vx; vy; vz; charge; mass_amu; time of birth
    1.00;1.00;1.00;1.00;1.00;1.00;1.00;100.0;0
    1.00;2.00;1.00;10.00;10.00;10.00;-1;200.0;0
    -10.00;-20.00;-10.00;-10.00;10.00;-10.00;2.0;300.0;1e-5
    1.00;2.00;1.00;10.00;10.00;10.00;-10.5;200.0;3e-5

As indicated by the comment in the example, the data columns of the file are: 

* position in x, y, z direction
* velocity in x, y, z direction (vx, vy, vz)
* the charge part in elementary charges
* the particle mass in u (atomic mass unit) 
* the time when the particle should come into existance in the simulation (time of birth - tob)

---------------------
Generating ion clouds
---------------------

The module :py:mod:`.preprocessing.ion_cloud_generation` provides functionality to define ion ensembles and write them to ion cloud files. 

The module provides functions to define individual groups of particles in defined geometric shapes (e.g. cylinders or spheres) and functions to modify characteristics of those particle groups. The complete ion cloud is then built by combining the subgroups of particles. The combined ion cloud is then written to a ion cloud file. 

The following example shows this basic principle: Two groups of particles with random positions within a cylinder in x direction are defined. The first particle group has particles with charge 1 and mass 1, the second group has particles with charge 2 and mass 10. The groups are combined and written to an ion cloud file:

.. code-block:: python 

    import numpy as np
    import IDSimPy.preprocessing.ion_cloud_generation as cl

    # define geometric parameters of cylinder: 
    cyl_r = 0.5
    cyl_z = 5.0

    # define cylindric random ion clouds for two particle types:
    cloud_p1 = cl.define_cylinder_x_dir(100, cyl_r, cyl_z, 1, 1)  # 100 particles, charge 1, mass 1
    cloud_p2 = cl.define_cylinder_x_dir(150, cyl_r, cyl_z, 2, 10) # 150 particles, charge 2, mass 10

    # combine sub-clouds and write ion cloud to file: 
    cloud = np.vstack((cloud_p1, cloud_p2))
    cl.write_cloud_file(cloud, 'test_cloud.csv')


-----------------------------------
Modifying ion clouds and ion groups
-----------------------------------

The functions defining particle groups return the defined particles as array, which allows the direct modification / manipulation of the ion group. For example, a translation of the ion positions can be achieved by 

.. code-block:: python 

    import IDSimPy.preprocessing.ion_cloud_generation as cl

    cloud = cl.define_cylinder_x_dir(100, cyl_r, cyl_z, 1, 1)  # 100 particles, charge 1, mass 1
    cloud[:,0] = cloud[:,0] + 2.0 # shift particle group +2.0 in x direction

There are some functions which modifies an ion cloud in more complex ways. For example, :py:func:`.add_thermalized_kinetic_energy` adds a random thermalized velocity component to the particles in an ion cloud. 


