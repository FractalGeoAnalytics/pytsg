# pyrexia
## Reasoning
pyrexia is a one function utility that imports the spectral geologist file package into a simple object.

https://research.csiro.au/thespectralgeologist/

if your data is in the below structure then you would pass
only the folder name to the function

## Installation
Installation is simple 
```pip install https://github.com/FractalGeoAnalytics/pyrexia```

## Usage

```python

import pyrexia

pyrexia.read_package('\HOLENAME')

```

the data format is assumed to follow this structure
```
\HOLENAME
         \HOLEMAME_tsg.bip
         \HOLENAME_tsg.tsg
         \HOLENAME_tsg_tir.bip
         \HOLENAME_tsg_tir.tsg
         \HOLENAME_tsg_hires.dat
         \HOLENAME_tsg_cras.bip

```
## 
## Thanks
Thanks to CSIRO for all their assistance in decoding the file structure