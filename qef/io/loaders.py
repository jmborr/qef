from __future__ import (absolute_import, division, print_function)

import re
import os
import numpy as np
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
        e = w['errors'].value  # undeterminacies in the intensities
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


def load_dave(file_name, to_meV=True):
    r"""

    Parameters
    ----------
    file_name : str
        Path to file
    to_meV : bool
        Convert energies from micro-eV to mili-eV

    Returns
    -------
    dict
        keys are q(momentum transfer), x(energy or time), y(intensities), and
        errors(e)
    """
    comment_symbol = '#'
    entries = dict(n_e='Number of Energy transfer values',
                   n_q='Number of q values',
                   e='Energy transfer (micro eV) values',
                   q='q (1/Angstroms) values',
                   g='Group')
    with open(file_name) as f:
        all = f.read()

    # Check all entries are present in the file
    for entry in entries.values():
        if all.find('{} {}'.format(comment_symbol, entry)) < 0:
            raise IOError('{} not found in {}'.format(entry, file_name))

    # Load number of energies and Q values
    def load_number_of_items(key):
        pattern = re.compile(r'(\d+)')
        # starting position for search
        start = all.find(entries[key]) + len(entries[key])
        return int(pattern.search(all[start:]).group(1))

    n_e = load_number_of_items('n_e')  # number of energy values
    n_q = load_number_of_items('n_q')  # number of Q values

    # Load energies and Q values
    fexpr = r'(\-*\d+\.*\d*e*\-*\d*)'  # float entry (also scientific mode)

    def load_items(n, key):
        pattern = re.compile(fexpr)
        start = all.find(entries[key]) + len(entries[key])
        matches = pattern.finditer(all[start:])
        return np.asarray([float(m.group(1)) for m in matches][: n])

    x = load_items(n_e, 'e')  # energy values
    if to_meV:
        x *= 1.0E-03  # assumed data is in micro-eV
    q = load_items(n_q, 'q')  # Q values

    # Load intensities
    y = list()
    e = list()
    pattern = re.compile(fexpr)
    for i in range(n_q):
        entry = entries['g'] + ' {}'.format(i)
        # starting position to search Group "i"
        start = all.find(entry) + len(entry)
        matches = pattern.finditer(all[start:])
        # ye contains both intensities and errors
        ye = np.asarray([float(m.group(1)) for m in matches][: 2 * n_e])
        y.append(list(ye[::2]))  # every other item in the list
        e.append(list(ye[1::2]))  # shift one, and then every other item

    return dict(q=q, x=x, y=np.asarray(y), e=np.asarray(e))
