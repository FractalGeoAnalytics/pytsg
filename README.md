# pytsg
## Rationale
The spectral geologist (TSG) is an industry standard software for hyperspectral data analysis
https://research.csiro.au/thespectralgeologist/

pytsg is an open source one function utility that imports the spectral geologist file package into a simple object.

## Installation
Installation is simple
```pip install pytsg```

## Usage

If using the top level importer the data is assumed to follow this structure
```
\HOLENAME
         \HOLEMAME_tsg.bip
         \HOLENAME_tsg.tsg
         \HOLENAME_tsg_tir.bip
         \HOLENAME_tsg_tir.tsg
         \HOLENAME_tsg_hires.dat
         \HOLENAME_tsg_cras.bip

```

```python
from matplotlib import pyplot as plt
from pytsg import parse_tsg

data = parse_tsg.read_package('example_data/ETG0187')

plt.plot(data.nir.wavelength, data.nir.spectra[0, 0:10, :].T)
plt.plot(data.tir.wavelength, data.tir.spectra[0, 0:10, :].T)
plt.xlabel('Wavelength nm')
plt.ylabel('Reflectance')
plt.title('pytsg reads tsg files')
plt.show()

```

If you would prefer to have full control over importing individual files the following syntax is what you need

```python

# bip files
nir = parse_tsg.read_tsg_bip_pair('ETG0187_tsg.tsg','ETG0187_tsg.bip','nir')
tir = parse_tsg.read_tsg_bip_pair('ETG0187_tsg_tir.tsg','ETG0187_tsg_tir.bip','tir')

# cras file
cras = parse_tsg.read_cras('ETG0187_tsg_cras.bip')

# hires dat file
lidar = parse_tsg.read_lidar('ETG0187_tsg_hires.dat')


```
ASD File reading support has been added with the addition of the asdreader from https://github.com/ajtag/asdreader.
pytsg includes the asdreader from ajtag as is. The intention of adding the asdreader is to allow for a simpler user workflow by allowing pip installation

ASD files can be read like so.


```python
from pytsg.asdreader import asdreader
asd = asdreader.reader("pathtoasd.asd")
```

## Thanks
Thanks to CSIRO and in particular Andrew Rodger for his assistance in decoding the file structures.