from __future__ import (absolute_import, division, print_function)

import re
import os
import h5py
import logging


def search_attribute(handle, name, ignore_case=False):
    r"""Find HDF5 entries containing a particular attribute

    Parameters
    ----------
    handle :
        Root entry from which to start the search

    name : str
        Regular expression pattern to search in attributes' names

    Returns
    -------
    list
        All entries with a matching attribute name. Each entry of the form
        (HDF5-instance, (attribute-key, attribute-vale))
    """
    ic = re.IGNORECASE if ignore_case else 0
    matches = list()
    for k in handle.attrs.keys():
        if re.search(name, k, flags=ignore_case):
            matches.append((handle, (k, handle.attrs[k])))
    if isinstance(handle, h5py.Group):
        for i in handle:
            matches.extend(search_attribute(handle[i], name, ignore_case=ic))
    return matches


def histogram_to_point_data(values):
    r"""Transform histogram(s) to point data

    Parameters
    ----------
    values : :class:`~numpy:numpy.ndarray`
        Array with histogram data

    Returns
    -------
    :class:`~numpy:numpy.ndarray`
        Array with point data
    """
    if values.ndim == 1:
        return (values[1:] + values[:-1]) / 2.0
    else:
        return (values[::, 1:] + values[::, :-1]) / 2.0


def load_nexus_processed(file_name):
    r"""Load data from a Mantid Nexus processed file

    Parameters
    ----------
    file_name : str
        Path to file

    Returns
    -------
    dict
        keys are q(momentum transfer), x(energy or time), y(intensities), and
        errors(e)
    """
    with h5py.File(file_name) as f:
        data = f['mantid_workspace_1']
        w = data['workspace']
        x = w['axis1'].value  # energy or time values
        y = w['values'].value  # intensities
        e = w['errors'].value  # indeterminacies in the intensities
        # Transform to point data
        if len(x) == 1 + len(y[0]):
            x = histogram_to_point_data(x)
        # Obtain the momentum transfer values
        q = w['axis2'].value
        if w['axis2'].attrs['units'] != 'MomentumTransfer':
            logging.warning('Units of vertical axis is not MomentumTransfer')
        # Transform Q values to point data
        if len(q) == 1 + len(y):
            q = histogram_to_point_data(q)
        return dict(q=q, x=x, y=y, e=e)


def load_nexus(file_name):
    r"""

    Parameters
    ----------
    file_name : str
        Absolute path to file

    Returns
    -------

    """
    data = None
    # Validate extension
    _, extension = os.path.splitext(file_name)
    if extension != '.nxs':
        raise IOError('File extension is not .nxs')
    # Validate content
    with h5py.File(file_name) as f:
        if 'mantid_workspace_1' in f.keys():
            data = load_nexus_processed(file_name)
        else:
            raise IOError('No reader found for this HDF5 file')
    return data
