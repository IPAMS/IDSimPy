# -*- coding: utf-8 -*-

"""
Analysis module for results of ion dynamics simulations with IDSimF

Modules:
  * constants: Definition of some physical and simulation constants
  * trajectory: Reading, analysis, transformations and format translations of ion trajectory data
  * visualization: General / basic visualization for ion trajectory data
  * chemistry: Rading and basic visualization of chemistry (RS) simulations with IDsimF
  * qitsim_analysis: Analysis of QIT and FT-QIT simulations with IDsimF
  * spacecharge_analysis: Detailed analysis of space charge dynamics in simulations with IDSimF
"""
from .trajectory import *
from .visualization import *

from . import visualization
from . import chemistry
from . import qitsim_analysis
from . import spacecharge_analysis