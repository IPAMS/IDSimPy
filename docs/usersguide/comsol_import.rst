.. _usersguide-comsol_import:

=====================================================
Import of Comsol Multiphysics result data into IDSimF
=====================================================

`Comosol Multiphysics <https://www.comsol.com/>`_ is a commercial simulation package for the simulation of complex multiphysical models. It is often used in science and engineering for the simulation of e.g. electrostatic fields, gas flows, heat distribution etc. 

The IDSimPy function :py:func:`.import_comsol_3d_csv_grid` can import field data which was exported from Comsol Multiphysics as plain CSV text file on a regular grid in the `Grid` format. This allows to translate such data into IDSimF input. 
The CSV field files exported from Comsol can contain multiple scalar fields. Vector fields can be exported from Comsol by exporting the individual vector dimensions separately as scalar fields. 

Reading a Comsol CSV file is straight forward, :py:func:`.import_comsol_3d_csv_grid` returns the data in a `dict` structure similar to the structure expected by :py:func:`.write_3d_scalar_fields_to_hdf5` and :py:func:`.write_3d_vector_fields_to_hdf5` (see :ref:`usersguide-preprocessing-field-generation`): 

.. code-block:: python 

    import os
    import IDSimPy.preprocessing.comsol_import as ci

    comsol_file_path = os.path.join('test', 'testfiles', 'transfer_quad_rf_field.csv.gz')
    dat = ci.import_comsol_3d_csv_grid(comsol_file_path)

    print(dat.keys()) # prints "dict_keys(['grid_points', 'meshgrid', 'fields'])"

    # access to individual data fields: 
    V_field = dat['fields'][0]
    print(V_field['name']) # prints "% V (V)"
    print(V_field['data'].shape) # prints "(160, 60, 60)"

Note that the CSV file can also be compressed with gzip as shown in the example. 

Since the structure of the data imported from Comsol files is basically the same as used for field export, translation of field data is also straight forward:

.. code-block:: python 

    import os
    import IDSimPy.preprocessing.comsol_import as ci
    import IDSimPy.preprocessing.field_generation as fg

    # import comsol data
    comsol_file_path = os.path.join('test', 'testfiles', 'transfer_quad_rf_field.csv.gz')
    dat = ci.import_comsol_3d_csv_grid(comsol_file_path)

    # create electric vector field from individual components:
    e_x = dat['fields'][1]['data']
    e_y = dat['fields'][2]['data']
    e_z = dat['fields'][3]['data']
    e_field = (e_x, e_y, e_z)

    # create a new dat object with the vector field data
    dat_v_fields = [{'name': 'electric field', 'data':e_field}]
    dat_v = {'grid_points': dat['grid_points'], 'meshgrid': dat['meshgrid'], 'fields': dat_v_fields}

    # write vector field data to HDF5 file:
    fg.write_3d_vector_fields_to_hdf5(dat_v, 'transfer_quad_rf_field.h5')

    # create n new dat object with the electric potential:
    dat_u_fields = [dat['fields'][0]]
    dat_u = {'grid_points': dat['grid_points'], 'meshgrid': dat['meshgrid'], 'fields': dat_u_fields}

    # write potential scalar field
    fg.write_3d_scalar_fields_to_hdf5(dat_u, 'transfer_quad_rf_potential.h5')