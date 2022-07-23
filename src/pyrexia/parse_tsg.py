import re
from pathlib import Path
from typing import Union
import numpy as np
from matplotlib import pyplot as plt

def read_tsg_file(filename: str) -> "list[str]":
    """Reads the files with the .tsg extension which are almost a toml file
    but not quite so the standard parser doesn't work
    
    Quite simply this function reads the file and strips the newlines at the end
    to simplify processing later on
    
    """
    lines: list[str] = []
    tmp_line: str
    with open(filename) as file:
        for line in file:
            tmp_line = line.rstrip()
            lines.append(tmp_line)
    return lines


def find_header_sections(fstr: "list[str]"):
    """Finds the header sections of the .tsg file
    """
    re_strip: re.Pattern = re.compile("^\\[[a-zA-Z0-9\\s ]+\\]")
    positions: list[int] = []
    for i, s in enumerate(fstr):
        if len(re_strip.findall(s)) > 0:
            positions.append(i)
    n_headers: int = len(positions)
    sections: dict[str, tuple[int]] = {}
    tmp_section: tuple[int]
    tmp_name: str
    for i in range(n_headers - 1):
        tmp_section = (positions[i] + 1, positions[i + 1] - 1)
        tmp_name = fstr[positions[i]].strip("[]")
        sections.update({tmp_name: tmp_section})
    return sections


def extract_section(fstr: str,header_sections:"dict[str, tuple[int]]") -> "list[int]":
    return [1]

def parse_tsg(fstr:str,headers: "dict[str, str]") -> "dict[str, Union[str, pd.DataFrame]]":
    d_info = {}
    final_sample = []
    for k in headers.keys():
        if k == 'sample headers':
            for i in fstr[headers[k][0]:headers[k][1]]:
                kk = parse_kvp(i,':')            
                k0 = list(kk.keys())
                tmp_sample = {}
                tmp_sample.update({'sample':k0[0]})
                for j in kk[k0[0]].split():
                    tmp_keys = parse_kvp(j)
                    if not tmp_keys is None:
                        tmp_sample.update(tmp_keys)            
                final_sample.append(tmp_sample)
            d_info.update({k:pd.DataFrame(final_sample)})
        else:
            tmp_out = {}
            for i in fstr[headers[k][0]:headers[k][1]]:
                tmp = parse_kvp(i)
                if not tmp is None:
                    tmp_out.update(tmp)
            d_info.update({k:tmp_out})


    return d_info

def parse_kvp(line: str,split:str='=') -> "Union[dict[str, str],None]":
    if line.find(split)>=0:
        split_line = line.split(split)
        key = split_line[0].strip()
        value = split_line[1].strip()
        kvp = {key:value}
    else:
        kvp = None
    return kvp 


filename = "/home/ben/pyrexia/data/ETG0187/ETG0187_tsg.tsg"
filename = '/home/ben/pyrexia/data/jmdh001/JMDH001_tsg_tir.tsg'
fstr = read_tsg_file(filename)
headers = find_header_sections(fstr)
d_info = parse_tsg(fstr, headers)


filename ='/home/ben/pyrexia/data/jmdh001/JMDH001_tsg_tir.bip'
 
def read_bip(filename:Union[str, Path], coordinates:"dict[str, str]")->np.ndarray:
    # load array in 1d
    tmp_array:np.ndarray = np.fromfile(filename,dtype=np.float32)
    # extract information on array shape
    n_bands:int = int(coordinates['lastband'])
    n_samples:int = int(coordinates['lastsample'])

    ra = np.reshape(tmp_array,(2, n_samples,n_bands))
    return ra


tir = read_bip(filename, d_info['coordinates'])
nir = read_bip(filename, d_info['coordinates'])
