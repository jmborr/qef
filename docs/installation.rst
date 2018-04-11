.. highlight:: shell

============
Installation
============


Stable release
--------------

To install qef, run this command in your terminal:

.. code-block:: console

    $ pip install qef

This is the preferred method to install qef, as it will always install the most recent stable release.

If you don't have `pip`_ installed, this `Python installation guide`_ can guide
you through the process.

.. _pip: https://pip.pypa.io
.. _Python installation guide: http://docs.python-guide.org/en/latest/starting/installation/


From sources
------------

The sources for qef can be downloaded from the `Github repo`_.

You can either clone the public repository:

.. code-block:: console

    $ git clone git://github.com/jmborr/qef

Or download the `tarball`_:

.. code-block:: console

    $ curl  -OL https://github.com/jmborr/qef/tarball/master

Once you have a copy of the source, you can install it with:

.. code-block:: console

    $ python setup.py install


.. _Github repo: https://github.com/jmborr/qef
.. _tarball: https://github.com/jmborr/qef/tarball/master


Testing & Tutorials Data
------------------------

The external repository `qef_data <https://github.com/jmborr/qef_data>`
contains all data files used in testing, examples, and tutorials.
There are several ways to obtain this dataset:

1. Clone the repository with a git command in a terminal:

.. code :: bash

    cd some/directory/
    git clone https://github.com/jmborr/qef_data.git

2. Download all data files as a zip file using GitHub's web interface:

.. image:: images/data_download_zip.png
    :width: 800px
    :align: center
    :alt: download dataset as zipped file

3. Download individual files using GitHub's web interface by browsing to the file, then click in Download button

.. image:: images/data_download_file.png
    :width: 800px
    :align: center
    :alt: download a single file
