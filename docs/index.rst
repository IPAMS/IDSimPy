==================
Welcome to IDSimPy 
==================

IDSimPy is an pre- and postprocessing package for `IDSimF <https://github.com/IPAMS/IDSimF>`_. IDSimF, the *Ion Dynamics Simulation Framework*, is an open source framework for the simulation of non relativistic dynamics of molecular ions. The primary application of IDSimF is in the domain of mass spectrometry and ion mobility spectrometry.


Introduction and Overview
=========================

IDSimPy is an Python companion package to IDSimF. IDSimPy bundles functionality for the pre- and postprocessing of input data for IDSimF simulations and of IDSimF result data. IDSimPy aims to provide a simple and productive interface to IDSimF. 

--------
Features
--------

The current main features are: 

    * Reading of IDSimF trajectory data
    * Filtering / statistical analysis of trajectroy data
    * Visualization of IDSimF results (plotting and animation rendering)
    * Preparation of ion cloud files
    * Transformation of field data (electric fields / flow fields) from other codes (Comsol / OpenFOAM) to IDSimF input files

.. toctree::
    :maxdepth: 1
    :caption: User Guide:

    usersguide/trajectory
    usersguide/visualization
    usersguide/comsol_import

.. toctree::
    :maxdepth: 1
    :caption: Module Documentation:

    modules/modules_documentation


==================
Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`