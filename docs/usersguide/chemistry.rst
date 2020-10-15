.. _usersguide-chemistry:

==============================================
Reading and analyzing IDSimF chemistry results
==============================================

IDSimF is primarily a charged particle trajectory simulation code, but it also includes particle based chemical kinetics. The simulation of chemical kinetics is typically part of simulation of the trajectories of reactive charged particles, but there are also solvers which only simulate kinetics, e.g. with the assumption of an ideally mixed reactor. 

The results of chemical kinetics simulations in IDSimF are often intrinsically also part of a trajectory result file, since solvers often write the chemical identifier (chemical ID) of the simulated particles to the trajectory. However, the primary result of chemical simulations in IDSimF is s separate file, the *reaction simulation result*. 

Currently, this file is a simple delimiter separated text file, which contains two header lines and data columns with the simulation time step, the simulated time and the number of particles of the individual chemical species, for example:

.. code-block:: none 

    RS C++ result
    Timestep ; Time ;  Cl_1 ; Cl_2 ; Cl_3 ;
    0 ; 0 ;  200 ; 200 ; 200 ;
    200 ; 2e-06 ;  190 ; 210 ; 200 ;
    400 ; 4e-06 ;  180 ; 220 ; 200 ;
    600 ; 6e-06 ;  170 ; 230 ; 200 ;
    800 ; 8e-06 ;  166 ; 234 ; 200 ;
    1000 ; 1e-05 ;  158 ; 242 ; 200 ;
    1200 ; 1.2e-05 ;  149 ; 251 ; 200 ;
    1400 ; 1.4e-05 ;  146 ; 254 ; 200 ;

Here, three ``discrete`` chemical species were simulated and recorded: `Cl_1`, `Cl_2` and `Cl_3`. Discrete species are the actively simulated chemical species, which are described by individual particles by the reaction simulation (RS) module of IDSimF. There are other chemical species in RS, which are not written to chemistry result files. See the IDSimF documentation for details. 

Reading of concentration files
==============================

Concentration files can be read with :py:func:`.analysis.chemistry.read_concentration_file`, which returns a Pandas data frame with the tabular data from the concentration file, for example: 

.. code-block:: python 

    import IDSimPy.analysis.chemistry as ch

    chemistry_file_name = os.path.join('testdata','reaction_simulation.txt')
    conc = ch.read_concentration_file(chemistry_file_name)

Simple plotting of concentration files
======================================

:py:func:`.analysis.chemistry.plot_concentration_file` provides a simple plot function to visualize a concentration file. 

