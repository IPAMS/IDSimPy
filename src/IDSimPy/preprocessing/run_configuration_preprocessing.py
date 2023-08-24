
def generate_run_configurations_from_template(template_filename, parameter_values, result_basename):
    """
    Generates simulation input files from a simple template file,
    a vector of parameter values and a basename for the resulting input files

    A template file is a simple text file, which contains replacement tokens of the form
    `%%-0-%%` with a running index

    Parameter values are a list of parameter value lists, with one parameter value vector per resulting file

    Example:

    .. code-block:: python

        parameters = (
            (result_file_1_parameter_1, result_file_1_parameter_2),
            (result_file_2_parameter_1, result_file_2_parameter_2),
            (result_file_2_parameter_1, result_file_2_parameter_2)
        )

    :param template_filename: Name of the template file
    :param parameter_values: list List of parameter values
    :param result_basename: Basename of the resulting generated input files
    """
    with open(template_filename) as template_file:
        template = template_file.read()

        for i, pv in enumerate(parameter_values):
            if not isinstance(pv, (list, tuple)):
                pv = (pv,)

            buf = template
            for vi, value in enumerate(pv):
                if type(value) is str:
                    value_string = '\"'+str(value)+'\"'
                else:
                    value_string = str(value)

                buf = buf.replace('%%-{:d}-%%'.format(vi), value_string)

            result_str = buf
            result_filename = result_basename + '{:02d}.json'.format(i)
            with open(result_filename, 'w') as result_file:
                result_file.write(result_str)
