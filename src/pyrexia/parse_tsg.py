from pathlib import Path

import toml
def read_tsg_file(filename:str) ->"list[str]":
    """Reads the files with the .tsg extension which are effectively .toml
    flies 
    """
    with open(filename) as file:
        lines:list[str] = file.readlines()
    return lines



read_tsg_file('/home/ben/pyrexia/data/ETG0187/ETG0187_tsg.tsg')