from concurrent.futures import process
import io
from pickletools import int4
import re
import struct
from dataclasses import dataclass
from pathlib import Path
from turtle import pos
from typing import Any, NamedTuple, Tuple, Union

import numpy as np
import pandas as pd
from numpy.core._exceptions import _ArrayMemoryError
from numpy.typing import NDArray
from PIL import Image

class CrasHeader(NamedTuple):
    id: str  # starts with "CoreLog Linescan ".   If it starts with "CoreLog Linescan 1." then it supports compression (otherwise ignore ctype).
    ns: int  # image width in pixels
    nl: int  # image height in lines
    nb: int  # number of bands (1 or 3  but always 3 for HyLogger 1 / 2 / 3)
    org: int  # interleave (1=BIL  2=BIP  and compressed rasters are always BIP while uncompressed ones are always BIL)
    dtype: int  # datatype (unused  always byte)
    specny: int  # number of linescan lines per dataset sample
    specnx: int  # unused
    specpx: int  # unused  intended to be the linescan column that relates to the across-scan position of the (1D) spectral dataset
    ctype: int  # compression type (0=uncompressed  1=jpeg chunks)
    chunksize: int  # number of image lines per chunk for jpeg-compressed rasters
    nchunks: int  # number of compressed image chunks (jpeg compression)
    csize32_obs: int  # size in bytes of comressed image data (OBSOLETE - not used anywhere any more.   However it will be set in old linescan rasters so I cant easily recycle it.   Also  there are some compressed rasters out there that are >4GB in size)
    ntrays: int  # number of trays (number of tray-table records after the image data)
    nsections: int  # number of sections (number of section-table records after the image data)
    finerep: int  # chip-mode datasets - number of spectral measurements per chip bucket (and theres one image frame per bucket)
    jpqual: int  # jpeg quality factor  0..100 (jpeg compression)


class TrayInfo(NamedTuple):
    utlengthmm: float  # "untrimmed" length of tray imagery in mm
    baseheightmm: float  # height of bottom of tray above table
    coreheightmm: float  # height of (top of) core above ..something or other (I don't actually use it for the linescan raster)
    nsections: int  # number of core sections
    nlines: int  # number of image lines in this tray


class SectionInfo(NamedTuple):
    utlengthmm: float  # untrimmed length of imagery in mm (could be less than the tray's)
    startmm: float  # start position (along scan) in mm
    endmm: float  # end position in mm
    trimwidthmm: float  # active (section-overlap-corrected) image width in mm
    startcol: int  # number of image lines in this tray
    endcol: int  # end pixel across for active (section-overlap-corrected) imagery
    nlines: int  # number of image lines in this section


class BandHeaders(NamedTuple):
    band: int
    name: str
    class_number: int
    flag: int  # I'm not sure what this does but 2 indicates that it is a mappable class


@dataclass
class Cras:
    image: NDArray
    tray: "list[TrayInfo]"
    section: "list[SectionInfo]"


@dataclass
class Spectra:
    spectrum_name: str
    spectra: NDArray
    wavelength: NDArray
    sampleheaders: pd.DataFrame
    classes: "list[dict[str, Any]]"
    bandheaders: "list[BandHeaders]"
    scalars: pd.DataFrame

@dataclass
class TSG:
    nir: Spectra
    tir: Spectra
    cras: Cras
    lidar: Union[NDArray, None]

    def __repr__(self) -> str:
        tsg_info: str = "This is a TSG file"
        return tsg_info



section_info_format: str = "4f3i"
tray_info_format: str = "3f2i"
head_format: str = "20s2I8h4I2h"
# using memory mapping
filename = 'example_data/SWMB007d/SWMB007d_chips_tsg_cras.bip'
filename = 'data/ETG0187/ETG0187_tsg_cras.bip'
filename= '/home/ben/pyrexia/data/RC_hyperspectral_geochem/8/RC_data8_cras.bip'
file.close()
# file = mmap.mmap(fopen.fileno(), 0)
file = open(filename, "rb")
# read the 64 byte header from the .cras file
bytes = file.read(64)
# create the header information
header = CrasHeader(*struct.unpack(head_format, bytes))

# Create the chunk_offset_array
# which determines which point of the file to enter to read the .jpg image

file.seek(64)
b = file.read(4 * (header.nchunks + 1))
chunk_offset_array = np.ndarray((header.nchunks + 1), np.uint32, b)

# check for the existance of sections or  trays
# if they exist we are going to skip ahead and import them first
# as we are going to use them to section the images to a per spectrum basis
# so the cras file uses compressed jpg chunks of approximately fixed dimension
# so we need to use the section and tray information to calculate the correct
# image size that matches the spectra 
# we will set up an array that we will use to accumulate the images into
# then as each image is accumulated we dump it to disk and call name it the sample name
# it is likely that we need two accumulation arrays the first as a bin to hold the images
# as they are read to disk and the second to hold the image that we are going to export
if header.nsections>0 or  header.ntrays>0:
    # the tray info section if it exists should start after the last image
    # the section info and if there is a tray info section then it should be after the tray info section
    info_table_start = (
        64
        + (header.nchunks + 1) * 4
        + chunk_offset_array[header.nchunks]
        - chunk_offset_array[0]
    )
    file.seek(info_table_start)

    tray: "list[TrayInfo]" = []
    for i in range(header.ntrays):
        bytes = file.read(20)
        tray.append(TrayInfo(*struct.unpack(tray_info_format, bytes)))

    section: "list[SectionInfo]" = []
    for i in range(header.nsections):
        bytes = file.read(28)
        section.append(SectionInfo(*struct.unpack(section_info_format, bytes)))

# it seems to be best to allocate memory for each of the sections if there are multiple sections we 
# empty the array and create a new one of the correct dimension
# on third thoughts we will precalculate which chunks are going to which section because we know that
# then loop over sets of chunks dumping to disk incrementally.
# at this stage I'm not sure it will work on drill core
# no the header contains the chunk dimensions
# loop over the section
# it seems that you need to have the sample header information from the
# nir/tir spectra we use nir because it should always be there
# once we have that information we are going to caculate the number of pixels required
# in the y direction that represent a single spectrum and the option will also be to dump 
# all the spectra to disk named as H_SAMPLE in a subfolder which will take an impressive amount of space
# but such are the vagaries of ML
data= parse_tsg.read_package('example_data/SWMB007d')
nir= parse_tsg.read_tsg_bip_pair('/home/ben/pyrexia/data/RC_hyperspectral_geochem/8/RC_data8.tsg','/home/ben/pyrexia/data/RC_hyperspectral_geochem/8/RC_data8.bip','nir')
# here is the calculation of the number of pixels per spectra
nir.sampleheaders.D.map(lambda x: float(x))
total_len = []
for i, sec in enumerate(section):
    total_len.append(sec.utlengthmm)
    yres = sec.utlengthmm/sec.nlines
yres*header.chunksize
nir.scalars['SecDist (mm)']
nir.scalars['SecSamp']
nir.scalars['Tray']
nir.sampleheaders
tray[0]
nir.scalars['avg_colour plain']
.columns.to_list()
# we currently are reading the entire cras.bip file
# which can cause issues due to memory allocation  well handle that case
# with some error handling here where we quit while we are ahead
try:
    cras = np.zeros((header.nl, header.ns, header.nb), dtype=np.uint8)
    array_ok = True
except _ArrayMemoryError:
    array_ok = False
    cras = np.zeros(1, dtype=np.uint8)
# if the array fits into memory then proceed to decode the .jpgs
# using the chunk_offset_array to correctly index to the right location
# TODO: it might be worthwhile to modify this code to manage the case
# when you might like to have images saved if the file is too big to fit
# into ram

# this is the mode where we calculate the size of each of the images
# let's allocate an array of 2x chunk size to be sure
# then loop over each chunk
curpos: int = 0
nr: int
curmm:float = 0.0
cursample:int = 0 
sample_length = nir.scalars['SecDist (mm)'].diff()
# this is only na for the first sample
idx_sample_na = (sample_length.isna()) | (sample_length<0)
sample_length[idx_sample_na] = nir.scalars['SecDist (mm)'][idx_sample_na]
# pd is slow for lots of accesses
sample_array:NDArray = sample_length.values 
sec = section[0]

file.close()
cbands = [620,552,460]
np.where(nir.wavelength == 460)
fm = nir.scalars['Final Mask'] == 1
nir.spectra.shape

rgb = nir.spectra[:,[60, 43, 20]]
rgb = rgb/rgb.sum(1).reshape(-1,1)
tirgb = []
for i in Path('/home/ben/pyrexia/data/RC_hyperspectral_geochem/8/IMGS').glob('*.jpg'):
    with open(i,'rb') as f:
        img = decode_jpeg(f.read(),colorspace ='RGB')
    tirgb.append(np.mean(np.mean(img[:,460:1000,:],1),0))
plt.imshow(img[:,460:1000,:])
plt.show()
tirgb = np.stack(tirgb)
irgb = tirgb/tirgb.sum(1).reshape(-1,1)
plt.scatter(rgb[:,0],rgb[:,1],c=rgb,marker='+')
plt.scatter(irgb[:,0],irgb[:,1],c=irgb,marker='.')
plt.show()


#################################################################
# start of working section
################################################################
plt.plot(nir.spectra[0])
plt.show()
outfolder:Path = Path('/home/ben/pyrexia/data/RC_hyperspectral_geochem/8/IMGS')
if not outfolder.exists():
    outfolder.mkdir()


section_array = nir.sampleheaders['L'].astype(int).values - 1
curchunk:int = 0
curline:int = 0
cursample:int = 0
i =0
sec = section[i]
processed_lines:int = 0
leading_bin = np.zeros((header.chunksize, header.ns,header.nb),dtype='uint8')
for i,sec in enumerate(section):
    # pixel resolution
    yres = sec.utlengthmm/sec.nlines
    # allocate the section
    working:NDArray = np.zeros((sec.nlines, header.ns,header.nb),dtype='uint8')
    curpos: int = 0
    nr: int
    # if we've dropped any information into the leading bin
    # dump it out into for into the working array
    if np.any(leading_bin):
        idx_bin_fill = np.all(np.any(leading_bin,1),1)
        pos_fill = np.where(idx_bin_fill)[0]
        working[pos_fill] = leading_bin[pos_fill]
        curpos = pos_fill[-1]+1
        # empty the leading bin
        leading_bin = leading_bin*0
    # you need to monitor the processed lines to maintain this loop
    while (curchunk*header.chunksize-processed_lines) < sec.nlines:
        total_offset = chunk_offset_array[curchunk] + 4 * (header.nchunks + 1) + 64
        chunksize_in_bytes = chunk_offset_array[curchunk + 1] - chunk_offset_array[curchunk]
        file.seek(total_offset)
        chunk = file.read(chunksize_in_bytes)
        img = decode_jpeg(chunk,colorspace ='BGR')
        np_image = np.flipud(img)
        nr = np_image.shape[0]
        end_pos = (curpos + nr)
        
        if end_pos <=  sec.nlines:
            working[curpos : end_pos, :, :] = np_image
        elif end_pos > sec.nlines:
            nextra = end_pos-sec.nlines
            end_pos = sec.nlines
            end_np = nr-nextra
            working[curpos : end_pos, :, :] = np_image[0:end_np]
            # put the remaining information into leading bin
            leading_bin[0:nextra,:,:] = np_image[end_np:nr]

        curpos = curpos + nr
        # increment the chunk
        curchunk+=1

    # book keeping the processed lines
    processed_lines += sec.nlines

    idx_section = section_array == i 
    im_cuts = np.floor(sample_array[idx_section]/yres).astype(int)
    cut_array = np.concatenate([[0],np.cumsum(im_cuts).ravel()])
     
    n_cuts = len(cut_array)
    for j in range(n_cuts-1):
        current_image = working[cut_array[j]:cut_array[j+1]]
        tmp_file = '{}.jpg'.format(cursample)
        outfile = outfolder.joinpath(tmp_file)
        outjpg = encode_jpeg(current_image)
        with open(outfile,'wb') as tmpf:
            tmpf.write(outjpg)
        cursample+=1

#################################################################
# endof working section
################################################################3
plt.imshow(working)
plt.show()
from simplejpeg import decode_jpeg,encode_jpeg
sec = section[0]
for i in range(10):#range(header.nchunks):
    total_offset = chunk_offset_array[i] + 4 * (header.nchunks + 1) + 64
    chunksize_in_bytes = chunk_offset_array[i + 1] - chunk_offset_array[i]
    file.seek(total_offset)
    chunk = file.read(chunksize_in_bytes)
    from datetime import datetime

    start = datetime.now()
    for i in range(100):
        img = Image.open(io.BytesIO(chunk))
        np_image = np.flip(np.array(img), -1)
    end = datetime.now()
    print(end-start)

    start = datetime.now()
    for i in range(100):
        k =decode_jpeg(chunk,colorspace ='BGR')
    end = datetime.now()
    print(end-start)
    # reverse the channels
    # and flip the image upsidedown
    #np_image = np.flipud(np.flip(np.array(img), -1))
    np_image = np.flip(np.array(img), -1)
    nr = np_image.shape[0]
    slice_iter = 0
    imits = 0
    working[0:nr,:,:] = np_image
    while slice_iter <= nr:

        cur_length = sample_array[cursample]
        # calc the number of pixels in the sample
        cur_pix = int(np.ceil(cur_length/yres))
        plt.figure()
        plt.imshow(working[slice_iter:cur_pix+slice_iter])
        slice_iter += cur_pix
        imits += 1
    plt.show()
    plt.imshow(working)

    section[0]
    curpos = curpos + nr
    curmm = curpos*yres
i= 0
plt.plot(sample_length)
plt.show()
plt.plot(nir.scalars['SecDist (mm)'].diff())
plt.show()
25*7519
    .sum()
header.nl*yres
file.close()

# core is easy
8/yres
# chips calc is different
# need to use a virtual section which I hope is included int he .tsg file
(sum(total_len)/yres)/361
data.nir.scalars.columns.to_list()
plt.plot(data.nir.scalars['sec_end_mask'])
plt.plot(np.ceil(data.nir.scalars["SecDist (mm)"]/yres))
plt.show()

# 361 is the answer for chips

plt.imshow(np_image[0:414])
plt.plot(np.diff(data.nir.sampleheaders.D.map(lambda x: float(x))),'.')
plt.show()
i.nlines/512

# we currently are reading the entire cras.bip file
# which can cause issues due to memory allocation  well handle that case
# with some error handling here where we quit while we are ahead
try:
    cras = np.zeros((header.nl, header.ns, header.nb), dtype=np.uint8)
    array_ok = True
except _ArrayMemoryError:
    array_ok = False
    cras = np.zeros(1, dtype=np.uint8)
# if the array fits into memory then proceed to decode the .jpgs
# using the chunk_offset_array to correctly index to the right location
# TODO: it might be worthwhile to modify this code to manage the case
# when you might like to have images saved if the file is too big to fit
# into ram
i = 0
if array_ok:
    curpos: int = 0
    nr: int
    for i in range(header.nchunks):
        total_offset = chunk_offset_array[i] + 4 * (header.nchunks + 1) + 64
        chunksize_in_bytes = chunk_offset_array[i + 1] - chunk_offset_array[i]
        file.seek(total_offset)
        chunk = file.read(chunksize_in_bytes)
        img = Image.open(io.BytesIO(chunk))
        # reverse the channels
        # and flip the image upsidedown
        np_image = np.flipud(np.flip(np.array(img), -1))
        plt.imshow(np_image)
        plt.show()

        nr = np_image.shape[0]
        cras[curpos : (curpos + nr), :, :] = np_image
        curpos = curpos + nr

file.close()

i