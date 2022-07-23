# reading .tsg files
import os
from typing import Any
from matplotlib import pyplot as plt
import numpy as np
import re

def read_tsg_file(filename:str) ->"list[str]":
    with open(filename) as file:
        lines:list[str] = file.readlines()
    return lines


def section_splitter(tsg_file:"list[str]")->None:
    reg_exp:str = r'\[[a-zA-z0-9\ _]+\]'
    re_section:re = re.compile(reg_exp)
    n_lines:int = len(tsg_file)
    i:int
    tmp_line:str
    tmp:list[Any]
    current_section:str
    for i in range(n_lines):
        tmp_line = tsg_file[i]
        if len(tmp_line) > 0:
            tmp = re_section.findall(tsg_file[i])
            if len(tmp) >0:
                current_section = tmp[0]
            print(f'{current_section}:{i}')

def read_tsg_bip(filename):
    '''
    the tsg.bip file is generally in float32 
    you need to reshape by the following method
    2 x samples x channels
    '''

def ReadTSG(filename):
    '''tsg has a header file that outlines the resolution for the scanner the number of samples
     and the names of the
    '''


filename = '/home/ben/accelerated_geoscience/ETG0187/ETG0187_tsg.tsg'

tsg_file = read_tsg_file(filename)
section_splitter(tsg_file)
iles  = os.listdir('ETG0187')

import numpy as np


a = np.fromfile('ETG0187/ETG0187_tsg.bip',dtype=np.float32)
# the hires .dat file is f32 and the actual data starts at pos 640
a = np.fromfile('ETG0187/ETG0187_tsg_hires.dat',dtype=np.float32)
ns = 2835
plt.plot(np.reshape(a[0:(531*ns)],(ns,531)).T)
plt.show()
b = a.reshape((2, ns, 531))
plt.plot(b[1,:,79])
plt.show()
plt.imshow(b[1,:,117],aspect ='auto')
plt.show(block=False)
na = len(a)
samps= 2835*2
chans = 531
b = a.reshape(chans,samps)
c = a.reshape(samps, chans)
plt.figure()
plt.imshow(b)
plt.figure()
plt.imshow(c)
plt.show()

plt.plot(b[0:200,1])
plt.show()
steps = np.arange(0,samps*chans,samps)
ss = np.zeros((samps, chans))
plt.plot(steps)
plt.show()
isq = np.repeat(np.arange(samps),chans)
jsq = np.tile(np.arange(chans),samps)
it = np.arange(na)
out = np.zeros((samps, chans))
b = steps.reshape((chans,samps))
plt.imshow(b,aspect='auto')
plt.show()
for i in range(na):
    out[isq[i],jsq[i]] = a[i]
plt.imshow(out,aspect='auto')
plt.show()

plt.plot(out[:,11])
plt.ylim(0,1)

b = a.reshape((-1,531))
b[b<0]=0
b[b>.001]=.001
np.isfinite(b).sum(0)
b[0:1417].max()
plt.imshow(b,aspect='auto')
plt.plot(np.sum(b))
plt.ylim([0,1])
plt.show()
531/2
2835/2
1417*2