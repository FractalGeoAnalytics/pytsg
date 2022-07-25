import re
from pathlib import Path
from typing import Union
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

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


def find_header_sections(tsg_str: "list[str]"):
    """Finds the header sections of the .tsg file"""
    re_strip: re.Pattern = re.compile("^\\[[a-zA-Z0-9 ]+\\]")
    positions: "list[int]" = []
    for i, s in enumerate(tsg_str):
        if len(re_strip.findall(s)) > 0:
            positions.append(i)
    positions.append(len(tsg_str))
    n_headers: int = len(positions)
    sections: "dict[str, tuple[int,int]]" = {}
    tmp_section: "tuple[int,int]"
    tmp_name: str
    for i in range(n_headers - 1):
        tmp_section = (positions[i] + 1, positions[i + 1] - 1)
        tmp_name = tsg_str[positions[i]].strip("[]")
        sections.update({tmp_name: tmp_section})
    return sections


def extract_section(fstr: str, header_sections: "dict[str, tuple[int]]") -> "list[int]":
    return [1]


def parse_tsg(
    fstr: str, headers: "dict[str, str]"
) -> "dict[str, Union[str, pd.DataFrame]]":
    d_info = {}
    final_sample = []
    start: int
    end: int
    for k in headers.keys():
        start = headers[k][0]
        end = headers[k][1]
        if k == "sample headers":
            for i in fstr[start:end]:
                kk = parse_kvp(i, ":")
                k0 = list(kk.keys())
                tmp_sample = {}
                tmp_sample.update({"sample": k0[0]})
                for j in kk[k0[0]].split():
                    tmp_keys = parse_kvp(j)
                    if not tmp_keys is None:
                        tmp_sample.update(tmp_keys)
                final_sample.append(tmp_sample)
            d_info.update({k: pd.DataFrame(final_sample)})
        if k == "wavelength specs":
            split_wavelength = fstr[start:end][0].split()
            tmp_wave = {
                "start": float(split_wavelength[0]),
                "end": float(split_wavelength[1]),
                "unit": split_wavelength[-1],
            }

            d_info.update({k: tmp_wave})

        else:
            tmp_out = {}
            for i in fstr[start:end]:
                tmp = parse_kvp(i)
                if not tmp is None:
                    tmp_out.update(tmp)
            d_info.update({k: tmp_out})

    return d_info


def parse_kvp(line: str, split: str = "=") -> "Union[dict[str, str],None]":
    if line.find(split) >= 0:
        split_line = line.split(split)
        key = split_line[0].strip()
        value = split_line[1].strip()
        kvp = {key: value}
    else:
        kvp = None
    return kvp


def read_bip(filename: Union[str, Path], coordinates: "dict[str, str]") -> np.ndarray:
    # load array in 1d
    tmp_array: np.ndarray = np.fromfile(filename, dtype=np.float32)
    # extract information on array shape
    n_bands: int = int(coordinates["lastband"])
    n_samples: int = int(coordinates["lastsample"])

    ra = np.reshape(tmp_array, (2, n_samples, n_bands))
    return ra

def calculate_wavelengths(
    wavelength_specs: "dict[str,float]", coordinates: "dict[str, str]"
) -> np.ndarray:
    wavelength_range: float = wavelength_specs["end"] - wavelength_specs["start"]
    resolution: float = wavelength_range / (int(coordinates["lastband"]) - 1)

    return np.arange(wavelength_specs["start"], wavelength_specs["end"]+resolution, resolution)

def read_hires_dat(filename:Union[str, Path])->np.ndarray:
    # the hires .dat file is f32 and the actual data starts at pos 640
    lidar = np.fromfile(filename,dtype=np.float32,offset=640)
    return lidar



def parse_package(foldername:Union[str,Path]):
    # convert string to Path because we are wanting to use Pathlib objects to manage the folder structure
    if isinstance(foldername,str):
        foldername = Path(foldername)
    files = foldername.glob('*.*')    
    # we are parsing the folder structure here and checking that 
    # pairs of files exist in this case we are making sure
    # that there are .tsg files with corresponding .bip files
    # we will parse the lidar height data because we can
    # _cras.bip files are still a mystery
    for f in files:
        f
    # once we have paired the .tsg and .bip files run the reader
    # for the nir/swir and then tir
    # read nir/swir
    filename = "/home/ben/pyrexia/data/jmdh001/JMDH001_tsg.tsg"
    fstr = read_tsg_file(filename)
    headers = find_header_sections(fstr)
    t_info = parse_tsg(fstr, headers)
    filename = "/home/ben/pyrexia/data/jmdh001/JMDH001_tsg_tir.bip"
    tir = read_bip(filename, t_info["coordinates"])

    # read tir
    fstr = read_tsg_file(filename)
    headers = find_header_sections(fstr)
    n_info = parse_tsg(fstr, headers)

    nir_wv = calculate_wavelengths(n_info["wavelength specs"],n_info["coordinates"])
    tir_wv = calculate_wavelengths(t_info["wavelength specs"],t_info["coordinates"])

foldername  = r'C:\Users\ben\pyrexia\data\ETG0187\ETG0187'
filename = "/home/ben/pyrexia/data/jmdh001/JMDH001_tsg_tir.tsg"


nir = read_bip(filename, n_info["coordinates"])
plt.plot(nir_wv, nir[0, 1, :])
plt.plot(tir_wv, tir[0, 1, :])
plt.show()