# -*- coding: utf-8 -*-

"""
Analysis and helper module for simulations of ion dynamics with BTree code by PTC

Modules:
  * constants: Definition of some physical and simulation constants
  * trajectory: Input / output, transformations and format translations of ion trajectory data
  * basic_visualization: General / basic visualization methods for trajectory data
  * ion_cloud_generation: Generation of ion cloud initialization files
  * qitsim_analysis: Analysis of QIT and FT-QIT simulations with BTree / Ion Dynamics Simulation Framework code
  * spacecharge_analysis: Detailed analysis of space charge dynamics in simulations with
  BTree / Ion Dynamics Simulation Framework Code
"""

from . import constants
from .constants import *
from . import trajectory
from .trajectory import *
from . import visualization
from .visualization import *
from . import ion_cloud_generation
from .ion_cloud_generation import *
from . import qitsim_analysis
from .qitsim_analysis import *
from . import spacecharge_analysis
from .spacecharge_analysis import *

__all__ = ['constants', 'trajectory','visualization', 'ion_cloud_generation','qitsim_analysis','spacecharge_analysis']