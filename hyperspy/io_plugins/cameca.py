# -*- coding: utf-8 -*-
# Copyright 2007-2016 The HyperSpy developers
#
# This file is part of  HyperSpy.
#
#  HyperSpy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
#  HyperSpy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with  HyperSpy.  If not, see <http://www.gnu.org/licenses/>.


import os
import logging
import warnings

import numpy as np

from hyperspy.misc.array_tools import sarray2dict

_logger = logging.getLogger(__name__)

im_extensions = ('im', 'IM')
im_info_extensions = ('im_chk', 'IM_CHK')
# Plugin characteristics
# ----------------------
format_name = 'CAMECA'
description = 'Format used for the Cameca NanoSIMS'
full_support = False
# Recognised file extension
file_extensions = im_extensions
default_extension = 0

# Writing capabilities
writes = False


# ----------------------
# Header im_chk file
def find_nanosims_masses_startingpoint(text):
    '''Finds the starting point in the im_chk file for the table of labels, masses and calibrations'''
    mass_start = None
    lines = text.split('\n')
    for i in range(len(lines)):
        if lines[i].startswith('Mass#'):
            mass_start = i + 1
    return mass_start, lines


def chk_im_reader(filename, header, **kwds):
    '''Returns info found in the im_chk file'''
    import os
    try:
        kwds["chk_im_path"]
    except KeyError:
        pass
    else:
        chk_im_path = kwds["chk_im_path"]

    #backuppath = "/Users/thomas/Dropbox/0_DPhil/0_Data/0_NanoSIMS/Header chk Files/"

    path, tail = os.path.split(filename)
    path += "/"
    try:
        f = open(path + tail.replace("im", "chk_im"))
    except FileNotFoundError:
        print("Info file " + str(tail) + " not found locally: " + str(path))

        try :
            chk_im_path
        except NameError:
            print("If .chk_im file is available in a different folder, use hs.load(..., chk_im_path=path) to specify"
                  " containing-folder or file")
            return
        else:
            if os.path.isfile(chk_im_path):
                try:
                    f = open(chk_im_path)

                except FileNotFoundError:
                    print("Info file " + str(tail) + " NOT FOUND at: " + chk_im_path )
                    print("Continuing to load file without extra info...")
                    return
            elif os.path.isdir(chk_im_path):
                if chk_im_path[-1] != "/":
                    chk_im_path += "/"
                try:
                    f = open(chk_im_path + tail.replace("im", "chk_im"))

                except FileNotFoundError:
                    print("Info file NOT FOUND in folder: " + chk_im_path)
                    print("Continuing to load file without extra info...")
                    return
            else:
                print("Something went wrong with readin the INFO file")
                print("Continuing to load file without extra info...")
                return
    text = ""
    try:
        text = f.read()

    except UnicodeDecodeError as U:
        print(U)
        print("This error is likely in association with an erronous chk_im file")
        return
    else:
        (mass_start, lines) = find_nanosims_masses_startingpoint(text)
        species = []
        mass = []
        detector = []
        tc = []
        bfield = []
        radius = []

        for i in range(header['number_of_masses']):
            species.append(lines[mass_start + i][6:18].strip(' '))
            if species == '':
                species = 'SE '

            mass.append(lines[mass_start + i][18:].split()[0])
            detector.append(lines[mass_start + i][18:].split()[1])
            tc.append(lines[mass_start + i][18:].split()[2])
            bfield.append(lines[mass_start + i][18:].split()[3])
            radius.append(lines[mass_start + i][18:].split()[4])

        compounds = {
            'species': species,
            'mass': mass,
            'detector': detector,
            'tc': tc,
            'bfield': bfield,
            'radius': radius,
        }
        return compounds


# Image im file
def get_endian(file):
    '''
    Check endian by seeing how large the value in bytes 8:12 are

    Parameters
    ----------
    file: file object

    Returns
    -------
    endian: string, either '>' (big-endian) or '<' (small-endian) depending on OS that saved the file

    '''
    file.seek(8)
    header_size = np.fromfile(file,
                              dtype=np.dtype('>u4'),
                              count=1)
    if header_size < 2e6:
        endian = '>'  # (big-endian)
    else:
        endian = '<'  # (small-endian)
    return endian


def get_header_dtype_list(file, endian):
    '''Parse header info from file

    Parameters
    ----------
    file: file object
    endian: string, either '>' (big-endian) or '<' (small-endian) depending on OS that saved the file
    Returns
    -------
    header: np.ndarray, dictionary-like object of image properties

    '''

    # Read the first part of the header
    header_list1 = [
        ('release', endian + 'u4'),
        ('analysis_type', endian + 'u4'),
        ('header_size', endian + 'u4'),
        ('sample_type', endian + 'u4'),
        ('data_present', endian + 'u4'),
        ('stage_position_x', endian + 'i4'),
        ('stage_position_y', endian + 'i4'),
        ('analysis_name', endian + 'S32'),
        ('username', endian + 'S16'),
        ('samplename', endian + 'S16'),
        ('date', endian + 'S16'),
        ('time', endian + 'S16'),
        ('filename', endian + 'S16'),
        ('analysis_duration', endian + 'u4'),
        ('cycle_number', endian + 'u4'),
        ('scantype', endian + 'u4'),
        ('magnification', endian + 'u2'),
        ('sizetype', endian + 'u2'),
        ('size_detector', endian + 'u2'),
        ('no_used', endian + 'u2'),
        ('beam_blanking', endian + 'u4'),
        ('pulverisation', endian + 'u4'),
        ('pulve_duration', endian + 'u4'),
        ('auto_cal_in_anal', endian + 'u4'),
        ('autocal', endian + 'S72'),
        ('sig_reference', endian + 'u4'),
        ('sigref', endian + 'S156'),
        ('number_of_masses', endian + 'u4'),
    ]
    file.seek(0)
    header1 = np.fromfile(file,
                          dtype=np.dtype(header_list1),
                          count=1)

    # Once we know what it tells us the header_size is, we can get the next
    # set of info
    file.seek(header1['header_size'][0] - 78)

    header_list2 = [
        ('width_pixels', endian + 'u2'),
        ('height_pixels', endian + 'u2'),
        ('pixel_size', endian + 'u2'),
        ('number_of_images', endian + 'u2'),
        ('number_of_planes', endian + 'u2'),
        ('raster', endian + 'u4'),
        ('nickname', endian + 'S64'),
    ]

    header2 = np.fromfile(file,
                          dtype=np.dtype(header_list2),
                          count=1)
    # finally, the element names are at positions offset depending on endianness.
    # the mass is in the first 64 bytes every 192 bytes. I strip off the whitespace at the end.
    header_list3 = [
        ('mass_name', endian + 'S64'),
        ('placeholder2', endian + 'S128'),
    ]

    if endian == '>':  # big-endian
        offset = 452 + 8
    else:  # endian == '<':  # small-endian
        offset = 412 + 192 + 56

    mass_names = []
    file.seek(offset + 56)
    for i in range(header1['number_of_masses'][0]):
        header3 = np.fromfile(file,
                              dtype=np.dtype(header_list3),
                              count=1)
        mass_name = header3['mass_name'][0].decode()
        mass_name = ''.join(mass_name.split('\x00'))

        if mass_name == '':
            mass_name = 'SE '

        mass_names.append(mass_name)

    header1 = sarray2dict(header1)
    header2 = sarray2dict(header2)

    header = {}

    for key in header1:
        header[key] = header1[key]

    for key in header2:
        header[key] = header2[key]

    header['mass_names'] = mass_names

    header['raster'] /= 1000 # It reports raster in nm, but NanoSIMS is almost always in um

    file.seek(0)
    return header


def file_reader(filename, *args, **kwds):
    ext = os.path.splitext(filename)[1][1:]
    if ext in im_extensions:
        return im_reader(filename, *args, **kwds)
        # elif ext in emi_extensions:
        # return emi_reader(filename, *args, **kwds)


def im_reader(filename, *args, **kwds):
    '''Reads the information from the file and returns it in the HyperSpy
    required format.

    '''
    header, data = load_im_file(filename)

    chk_labels = chk_im_reader(filename, header, **kwds)
    if chk_labels != None:
        header = {**header, **chk_labels} # unpacks header and chk_labels into new header

    # Image mode

    axes = []
    array_shape = []
    chk_exists = False
    if chk_exists is True:
        # set units based on that info
        units = 'unitsfromchkfile'
    else:
        units = 'um'

    # Z axis
    axes.append({
        'name': 'z',
        'index_in_array': 0,
        'offset': 0,
        'scale': 1,
        'units': '',
        'size': header['number_of_planes'],
    })
    array_shape.append(header['number_of_planes'])

    # Y axis
    axes.append({
        'name': 'y',
        'index_in_array': 0,
        'offset': 0,
        'scale': header['raster'] / header['height_pixels'],
        'units': units,
        'size': header['height_pixels'],
    })
    array_shape.append(header['height_pixels'])

    # X axis
    axes.append({
        'name': 'x',
        'index_in_array': 0,
        'offset': 0,
        'scale': header['raster'] / header['width_pixels'],
        'units': units,
        'size': header['width_pixels'],
    })

    array_shape.append(header['width_pixels'])

    # If the acquisition stops before finishing the job, the stored file will
    # contain only zeroes in all remaining slices. Better remove them.
    metadata_titles = []
    for i in range(header['number_of_masses']):
        try:
            header['mass']
        except KeyError:
            metadata_titles.append(header['mass_names'][i])
        else:
            metadata_titles.append(header['mass_names'][i] + '- ' + header['mass'][i])

    dictionary_list = []
    for i in range(header['number_of_masses']):
        dc = data[i]
        # Set? original_metadata = {}

        dictionary = {
            'data': dc,
            'metadata': {
                'General': {
                    'title': metadata_titles[i],
                    'original_filename': header['filename'],
                },
                'Signal': {
                    'record_by': 'image',
                    'signal_type': '',
                },
            },
            'axes': axes,
        }
        dictionary_list.append(dictionary)
    # Return a list of dictionaries
    return dictionary_list


def load_im_file(filename):
    _logger.info('Opening the file: %s', filename)
    with open(filename, 'rb') as f:
        # Check endian of bytes, as it depends on the OS that saved the file
        endian = get_endian(f)
        header = get_header_dtype_list(f, endian=endian)

        for key in header:
            if type(header[key]) == np.bytes_:
                try:
                    header[key] = header[key].decode()
                except UnicodeDecodeError:
                    pass

        # Read the first element of data offsets
        f.seek(header['header_size'])
        # Data can either be of data type uint16 or uint32 - maybe even uint64

        datadtype = endian + 'u' + str(header['pixel_size'])

        data = np.fromfile(f,
                           dtype=datadtype,
                           count=header['number_of_masses'] * header['number_of_planes'] * header['width_pixels'] *
                                 header['height_pixels'])
        #data = data.astype("uint32") # Apparently Hyperspy checks if the dtype needs changing when binning data

        # Reshape into shape (images*planes, width, height)
        #print(data.shape)
        #print(header['number_of_masses'], header['number_of_planes'], header['width_pixels'], header['height_pixels'])
        #print(header['number_of_masses'] * header['number_of_planes'] * header['width_pixels'] * header['height_pixels'])
        #print(data.shape)
        #print(header['number_of_masses'], header['number_of_planes'], header['width_pixels'], header['height_pixels'])
        actual_number_of_planes_rounded = int(len(data)/(header['number_of_masses'] * header['width_pixels'] * header['height_pixels']))
        length_of_real_data = header['number_of_masses'] * header['width_pixels'] * header['height_pixels'] * actual_number_of_planes_rounded
        data = data[:length_of_real_data]
        if actual_number_of_planes_rounded != header['number_of_planes']:
            warnings.warn("The file contains fewer planes than expected.")
        data = data.reshape(header['number_of_masses'] * actual_number_of_planes_rounded, header['width_pixels'],
                            header['height_pixels'])

        data = np.array([data[i::header['number_of_masses']] for i in range(header['number_of_masses'])])

    return header, data
