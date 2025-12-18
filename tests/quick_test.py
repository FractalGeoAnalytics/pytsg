from src.pytsg.parse_tsg import read_package,read_tsg_bip_pair
from matplotlib import pyplot as plt
hl4 = read_package(r'data/Case_study_data_PDD174/L1')
hl3 = read_package(r'data/Case_study_data_PDD174/PDD174_HL3')

plt.plot(hl3.nir.wavelength, hl3.nir.spectra[:10,:].T,'tab:red')
plt.plot(hl3.tir.wavelength,hl3.tir.spectra[:10,:].T,'tab:blue')

plt.plot(hl4.nir.wavelength,hl4.nir.spectra[:10,:].T,'r')
plt.plot(hl4.mir.wavelength,hl4.mir.spectra[:10,:].T,'g')
plt.plot(hl4.tir.wavelength,hl4.tir.spectra[:10,:].T,'b')
plt.show()
hl4.mir

read_tsg_bip_pair(folder_name=)
b = read_package(r'example_data/GSNSW_testrocks',read_cras_file=True)
plt.imshow(b.cras.image)
plt.show()
plt.plot(b.nir.wavelength,b.nir.spectra.T,'r')
plt.plot(b.mir.wavelength,b.mir.spectra.T,'g')
plt.plot(b.tir.wavelength,b.tir.spectra.T,'b')
plt.show()

a = read_package(foldername = r'example_data/HL4_testdata/Coretray',read_cras_file=True)
plt.plot(a.lidar)
plt.show()
plt.imshow(a.cras.image)
plt.show()
dir(a)
dir()
a.cra
plt.plot(a.nir.wavelength,a.mir.spectra.T)
dir(a.mir)

read_tsg_bip_pair(r'example_data/HL4_testdata/Coretray/PE257D_0001_tsgtray_tir.tsg',r'example_data/HL4_testdata/Coretray/PE257D_0001_tsgtray_tir.bip','mir')
import numpy as np
from matplotlib import pyplot as plt

with open(r'example_data/PE257D/GSNSW_PE257D_0001_20250227163455_00.rawProf3d.bin','rb') as file:
    bin = file.read()

import struct
nbin = len(bin)
(nbin-1)/5
(nbin-22)/1
bin[0:22]
b1 = bin[22:]
len(bin)/1280
len(bin)-53429*1280
2000*20000
len(bin)-966
53429*0.05

np.finfo(np.float16)

aa = np.frombuffer(b1,dtype=np.float32)
(aa.shape[0]/3)
plt.plot(aa[:40000],'.')
plt.show()
plt.imshow(aa[0:100])
plt.show()
aa[0:100]
X, Y, Z
np.float(31)
bin[23:23+16]

seq = np.frombuffer(bin[22:],dtype=np.uint8)
len(seq)/2/2/2/2/3/3
2*2*2*2*3*3
48+48+ 48
1280
plt.plot(seq,'-')
plt.show()
