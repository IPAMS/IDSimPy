==================
Welcome to IDSimPy 
==================

IDSimPy is a pre- and postprocessing package for `IDSimF <https://github.com/IPAMS/IDSimF>`_. IDSimF, the *Ion Dynamics Simulation Framework*, is an open source framework for the simulation of non-relativistic dynamics of molecular ions. The primary application of IDSimF is in the domain of mass spectrometry and ion mobility spectrometry.


Introduction and Overview
=========================

IDSimPy is a Python companion package to IDSimF. IDSimPy bundles functionality for the pre- and postprocessing of input data for IDSimF simulations and of IDSimF result data. IDSimPy aims to provide a simple and productive interface to IDSimF. 

--------
Features
--------

The current main features are: 

    * Reading of IDSimF trajectory and chemistry data
    * Filtering / statistical analysis of trajectory data
    * Visualization of IDSimF results (plotting and animation rendering)
    * Preparation of ion cloud files
    * Transformation of field data (electric fields / flow fields) from other codes (Comsol / OpenFOAM) to IDSimF input files

.. toctree::
    :maxdepth: 1
    :caption: Installation:

    installation/installation

.. toctree::
    :maxdepth: 1
    :caption: User Guide:

    usersguide/trajectory
    usersguide/chemistry
    usersguide/visualization
    usersguide/preprocessing    
    usersguide/comsol_import

.. toctree::
    :maxdepth: 1
    :caption: Package / Module Documentation:

    modules/package_analysis
    modules/package_preprocessing


==================
Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`