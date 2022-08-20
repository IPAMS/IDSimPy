.. _usersguide-preprocessing:

==================================================
Preprocessing and generating input data for IDSimF 
==================================================

IDSimPy provides functionality to generate simulation run configurations and other input data for IDSimF simulations. There are currently three main types of input and run configuration data for IDSimF simulations: 

+ **Simulation run configuration files**, which define the parameters / configuration of a simulation run. 
* **Ion cloud files**, which define a custom set of particles. They are primarily used for initialization of the simulated particle ensemble at the begin of the IDSimF simulation run. 
* **Field data files**, which define scalar or vector fields on a regular spatial grid. They are commonly used to transfer spatially resolved input data, e.g. electric fields or gas flow data from other numerical solvers to IDSimF simulations. 

All three file types can be generated with IDSimPy. 

Simulation run configurations
=============================

Simulation run configurations are JSON files which can also contain C-style comments. The details of the individual configuration options and the detailed format are described in the IDSimF documentation. 

----------------------------------------------------------
Generation of simulation run configurations from templates
----------------------------------------------------------

Since it is common to variate individual simulation parameters in series of simulation runs, IDSimPy allows to generate series of simulation run configurations from template files and a set of parameter values with :py:func:`.generate_run_configurations_from_template`. This process replaces indexed place holders in the template files with parameter values to generate the run configuration files. 

.. note::
    The place holder replacement is done purely on text level, currently there is no semantic checking of the templates or the results


Templates are text files with replacement tags / place holders. The replacement tags have the form 

.. code-block:: none

    %%-i-%%

with an index number ``i``. Therefore the place holder for the first replacement location would be 

.. code-block:: none

    %%-0-%%

while the second replacement location would be indicated by 

.. code-block:: none

    %%-1-%%

A simple example template with three parameters to set would be 

.. code-block:: none

    {
        "sim_time_steps":40000,
        "cv_phase_shift": %%-0-%%,
        "simulation_mode": %%-1-%%,
        "sv":%%-2-%%
    }

The parameter values to set are providet to :py:func:`.generate_run_configurations_from_template` as two dimensional list / tuple structure. For example, if the template from above is available in a file with the name ``configuration_template.tmpl``, the generation of three simulation run configuration files would be 

.. code-block:: python

        import os
        import IDSimPy.preprocessing as ip


    	parameters = (
			(0.5, 'square', 1000),
			(0.1, 'sin', 2500),
			(0.2, 'bisin', 4500)
		)

		template_file = 'configuration_template.tmpl')
		result_basename = 'sim_run_')

		ip.generate_run_configurations_from_template(template_file, parameters, result_basename)

This generates three simulation run configuration files (`sim_run_00.json`, `sim_run_01.json`, `sim_run_02.json`), one per row of the provided `parameters``. An individual row defines the parameter values to be set in one individual result file. For example `sim_run_01.json` of the example would be

.. code-block:: none

    {
        "sim_time_steps":40000,
        "cv_phase_shift": 0.1,
        "simulation_mode": "sin",
        "sv": 2500
    }

.. note::
    Since there is no semantic interpretation of the template files, the simple replacement mechanism of :py:func:`.generate_run_configurations_from_template` is applicable for other input file types, e.g. RS chemical configuration files, too.



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

The module provides functions to define individual groups of particles in defined geometric shapes (e.g. cylinders or spheres) and functions to modify characteristics of those particle groups. The complete ion cloud is then built by combining the subgroups of particles. The combined ion cloud is then written to an ion cloud file. 

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


.. _usersguide-preprocessing-field-generation:

Generating scalar and vector field input data for IDSimF
========================================================

IDSimF can import fields of scalar and vector values on a regular grid. Those fields are imported by IDSimF from HDF5 files with a defined structure. The module :py:mod:`.preprocessing.field_generation` provides an interface to write such field files from a compact structured data representation. 

The primary functions in the module are :py:func:`write_3d_scalar_fields_to_hdf5` which writes a set of scalar fields to a HDF5 file and :py:func:`write_3d_vector_fields_to_hdf5` which writes a vector field to a HDF5 file. 

--------------------------
Basic field representation
--------------------------

The field export functions expect the data to export in a defined compact structure. Data objects are dictionaries (:py:obj:`dict`) with two primary entries: :py:data:`grid_points` and :py:data:`fields`. 

Fields represent data on a regular spatial grid, which is defined by the positions grid nodes on the spatial axes. The entry :py:data:`grid_points` is a :py:obj:`list` which consists of three lists of grid positions, one for every spatial dimension. A valid :py:data:`grid_points` entry would thus be for example 

.. code-block:: python 

    [[0, 2, 5, 15], [0, 2, 10], [0, 2, 5, 7, 10]]

As this example shows, the grid points do not have to be equidistant and can differ between the spatial dimensions.


The :py:data:`fields` entry contains the actual field data. Since field files can contain multiple individual data fields on the same spatial grid, :py:data:`fields` is a :py:obj:`list` of dictionaries (:py:obj:`dict`), each defining one individual data field. Such an individual field entry has two entries: :py:data:`name` which is a name / identifier of the individual data field, and :py:data:`data` which contain the actual data. The data is given as three dimensional numpy array for scalar data fields and as four dimensional numpy array for vector data fields.  A valid :py:data:`fields` entry with two data fields would therefore be

.. code-block:: python 

    # dt_a and dt_b would be the prepared field data arrays with the actual field data: 
    fields = [ {'name': 'test_field_a', 'data': dt_a}, {'name': 'test_field_b', 'data': dt_b}]

with the numpy arrays :py:data:`dt_a` and :py:data:`dt_b`. 

-------------------
Scalar field export 
-------------------

Scalar fields are written to HDF5 files with :py:func:`.write_3d_scalar_fields_to_hdf5`. The data arrays in the :py:data:`fields` entry of the data to export are expected to have three dimensions and a shape compatible with :py:data:`grid_points`. 

The following example shows how to define a linear field with increasing values in x,y,z direction and how to write this field to a HDF5 file for IDSimF: 

.. code-block:: python 

    import numpy as np
    import IDSimPy.preprocessing.field_generation as fg

    # define simple linear scalar field:
    grid_points = [[0, 2, 5, 15], [0, 2, 10], [0, 2, 5, 7, 10]]
    x_g, y_g, z_g = np.meshgrid(grid_points[0], grid_points[1], grid_points[2], indexing='ij')
    linear_field = x_g + y_g + z_g

    # define data to export: 
    fields = [{'name': 'test_field', 'data': linear_field}]
    dat = {"grid_points": grid_points, "fields": fields}

    fg.write_3d_scalar_fields_to_hdf5(dat, 'test_linear_scalar_field.h5')

-------------------
Vector field export 
-------------------

Vector fields are written to HDF5 files with :py:func:`.write_3d_vector_fields_to_hdf5`. The data arrays in the :py:data:`fields` entry of the data to export are expected to have four dimensions and a shape compatible with :py:data:`grid_points`. 

The following example shows how to define two vector fields with simple increasing components in x,y,z direction and how to write those fields to a HDF5 file for IDSimF: 

.. code-block:: python

    import numpy as np
    import IDSimPy.preprocessing.field_generation as fg

    # define two simple linear vector fields:
    grid_points = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20], [-10, 0, 10], [0, 10]]
    x_g, y_g, z_g = np.meshgrid(grid_points[0], grid_points[1], grid_points[2], indexing='ij')
    v_zero = np.zeros(np.shape(x_g))

    # linear increasing components for the vector fields:
    v_x1 = x_g
    v_y1 = y_g
    v_z1 = z_g

    v_x2 = x_g * 2.0
    v_y2 = y_g * 4.0
    v_z2 = z_g * 6.0

    # prepare data to export, vector components are given as list of individual arrays: 
    fields = [
        {'name': 'test_vectorfield_1', 'data': [v_x1, v_y1, v_z1]},
        {'name': 'test_vectorfield_2', 'data': [v_x2, v_y2, v_z2]}
    ]

    # export data
    dat = {"grid_points": grid_points, "fields": fields}
    fg.write_3d_vector_fields_to_hdf5(dat, 'test_linear_vector_field.h5')

