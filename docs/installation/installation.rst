.. _installation:

==============================
Prerequisites and Installation
==============================

Prerequisites
=============

IDSimPy is a Python 3 package. It requires a working and up to date Python 3 environment. IDSimPy was tested with Python 3.6. All other dependencies (Numpy, Matplotlib, Pandas, h5py and vtk) will be installed by the setup script. 

Installation
============

IDSimPy is currently not yet listed on PyPi `PyPi <https://pypi.org>`_.  Therefore, the public `Git repository <https://github.com/IPAMS/IDSimPy>`_  has to be cloned and the package has to be installed from source. 

With git installed clone the repository to a local directory: 

.. code-block:: console

    git clone https://github.com/IPAMS/IDSimPy.git

Then change into the cloned directory and install IDSimPy from source by invoking the setup script:

.. code-block:: console

    cd IDSimPy
    python setup.py install


Alternatively, IDSimPy can also installed from the cloned repository with pip:

.. code-block:: console

    cd IDSimPy
    pip install .
