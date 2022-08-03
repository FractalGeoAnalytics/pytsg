Quick Start
===========

Here is what you need to get going, but you may also take a look at the :ref:`examples<Example Gallery>`.

Data Structure
--------------

If using the top level importer the data is assumed to follow this structure.

.. code-block:: console

   \HOLENAME
         \HOLEMAME_tsg.bip
         \HOLENAME_tsg.tsg
         \HOLENAME_tsg_tir.bip
         \HOLENAME_tsg_tir.tsg
         \HOLENAME_tsg_hires.dat
         \HOLENAME_tsg_cras.bip

Example Usage
-------------

.. code-block:: python

   from matplotlib import pyplot as plt
   from pytsg import parse_tsg
   data = parse_tsg.read_package('/data/ETG0187')

   plt.plot(data.nir.wavelength, data.nir.spectra[0,0:10,:].T)
   plt.plot(data.tir.wavelength, data.tir.spectra[0,0:10,:].T)
   plt.xlabel('Wavelength nm')
   plt.ylabel('Reflectance')
   plt.title('pytsg reads tsg files')
   plt.show()


If you would prefer to have full control over importing individual files the following syntax is what you need

.. code-block:: python

   # bip files
   nir = parse_tsg.read_tsg_bip_pair('ETG0187_tsg.tsg','ETG0187_tsg.bip','nir')
   tir = parse_tsg.read_tsg_bip_pair('ETG0187_tsg_tir.tsg','ETG0187_tsg_tir.bip','tir')

   # cras file
   cras = parse_tsg.read_cras('ETG0187_tsg_cras.bip')

   # hires dat file
   lidar = parse_tsg.read_cras('ETG0187_tsg_hires.dat')



.. toctree::
   :maxdepth: 2
