# using memory mapping
filename = r'C:\Users\ben\pyrexia\example_data\SWMB007s\SWMB007s_chips_tsg_cras.bip'

# file = mmap.mmap(fopen.fileno(), 0)
file = open(filename, "rb")

# read the 64 byte header from the .cras file
bytes = file.read(64)
# create the header information
header = CrasHeader(*struct.unpack(head_format, bytes))
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

    tray: list[TrayInfo] = []
    for i in range(header.ntrays):
        bytes = file.read(20)
        tray.append(TrayInfo(*struct.unpack(tray_info_format, bytes)))

    section: list[SectionInfo] = []
    for i in range(header.nsections):
        bytes = file.read(28)
        section.append(SectionInfo(*struct.unpack(section_info_format, bytes)))

# it seems to be best to allocate memory for each of the sections if there are multiple sections we 
# empty the array and create a new one of the correct dimension
# on third thoughts we will precalculate which chunks are going to which section because we know that
# then loop over sets of chunks dumping to disk incrementally.
# at this stage I'm not sure it will work on drill core
# no the header contains the chunk dimensions
# no loop over the tray
for i in tray:
    yres = i.utlengthmm/i.nlines
    npix = int(np.ceil(i.baseheightmm/yres))

# which determines which point of the file to enter to read the .jpg image
plt.imshow(data.cras.image[7140:8240])
plt.show()
file.seek(64)
b = file.read(4 * (header.nchunks + 1))
chunk_offset_array = np.ndarray((header.nchunks + 1), np.uint32, b)

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
        nr = np_image.shape[0]
        cras[curpos : (curpos + nr), :, :] = np_image
        curpos = curpos + nr

file.close()