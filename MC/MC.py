from __future__ import absolute_import, division, print_function
import numpy as np
import os

__all__ = ["import_psyphysdata_mc",
           "importfromstandardcsv_moz",
           "combineconditions_mc",
           "array2tupleofvectors",
           "findfilenameindirectory",
           "import_subjectdatainfo_mc"]




def import_psyphysdata_mc(f_name, path):
    """
    Feb 22, 2019
    Mert Ozkan
    Dartmouth College
    Motion Coherence

    Imports datasets from log files for the Motion Coherence experiment.
    Usage: trl_no, dxn, coh, isOK, key, rt = import_psyphysdata_mc(f_name,path)

    Invalid reactions are encoded as key = 0
    """

    dat_path = '/'.join([path, f_name])
    f = open(dat_path, 'r')

    # Data log starts 2 lines after the data pointer
    dat_ptr = '%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%'
    read_l = f.readlines()
    for idx in range(len(read_l)):
        if dat_ptr in read_l[idx]:
            dat_start_idx = idx + 2
            break
    trl_no = []
    dxn = []
    coh = []
    isOK = []
    key = []
    rt = []
    for trl in read_l[dat_start_idx:]:
        trl_dat = trl.split()
        trl_no.append(int(trl_dat[0]))
        dxn.append(int(trl_dat[1]))
        coh.append(float(trl_dat[2]))
        isOK.append(int(trl_dat[3]))
        if trl_dat[4] == 'space' or trl_dat[4] == '-':
            key.append(0)
        else:
            key.append(int(trl_dat[4]))

        rt.append(float(trl_dat[5]))

    trl_no = np.array(trl_no)
    dxn = np.array(dxn)
    coh = np.array(coh)
    isOK = np.array(isOK)
    key = np.array(key)
    rt = np.array(rt)
    return trl_no, dxn, coh, isOK, key, rt


def import_subjectdatainfo_mc(path):
    """
    Feb 22, 2019
    Mert Ozkan
    Dartmouth College
    Motion Coherence

    Import subject data inventory from data_inventory_mc.txt
    Usage: sub_no, sub_name, sxn_sq, bhv_ptr, eeg_ptr, trl_rej_1, trl_rej_2 = import_subjectdatainfo_mc(path)
    """

    f = open(path, 'r')
    read_l = f.readlines()

    sub_no = []
    sub_name = []
    sxn_sq = []
    bhv_ptr = []
    eeg_ptr = []
    trl_rej_1 = []
    trl_rej_2 = []
    for idx in range(len(read_l)):
        if '*' in read_l[idx]:
            inv_st = read_l[idx + 1].split('; ')
            sub_no.append(int(inv_st[0]))
            sub_name.append(inv_st[1])
            sxn_sq.append(
                np.array(inv_st[2].split(', ')).astype(int))
            bhv_ptr.append(inv_st[3])
            eeg_ptr.append(inv_st[4])
            trl_rej_1.append(
                np.array(inv_st[5].split(', ')).astype(int))
            if inv_st[6][-1:] == '\n':
                inv_st[6] = inv_st[6][:-1]
            if inv_st[6] != '':
                trl_rej_2.append(
                    np.array(inv_st[6].split(', ')).astype(int))
            else:
                trl_rej_2.append(
                    np.array([]).astype(int))
    return sub_no, sub_name, sxn_sq, bhv_ptr, eeg_ptr, trl_rej_1, trl_rej_2


def findfilenameindirectory(path, kw, numberoffiles=None, maxnumberoffiles=None, minnumberoffiles=None):

    found = 0
    contents = os.listdir(path)
    subset_contents = contents
    for whKW in kw:
        contents = subset_contents
        subset_contents = []
        for whContent in contents:
            if whKW in whContent:
                found += 1
                subset_contents.append(whContent)

    assert not((numberoffiles is not None and numberoffiles != len(subset_contents)) or \
           (maxnumberoffiles is not None and maxnumberoffiles < len(subset_contents)) or \
           (minnumberoffiles is not None and minnumberoffiles > len(subset_contents))),\
        f'''
        File Quantity Mismatch! in findfilenameindirectory_mc()
        Expected number of files: {numberoffiles}
        Expected maximum number of files: {maxnumberoffiles}
        Expected minimum number of files: {minnumberoffiles}
        Number of files found: {len(subset_contents)}
        Keywords: {kw}
        Filenames: {subset_contents}
        Path = {path}'''
    return subset_contents


def combineconditions_mc(whSub, sq, path):

    conds_id = np.array(
        ['translational.log', 'rotational.log', 'radial.log'])  # The index corresponding to the condition number
    conds_sq = conds_id[sq]

    prev_sxn_trl_no = 0
    prev_trl_no = np.array([])
    prev_dxn = np.array([])
    prev_coh = np.array([])
    prev_isOK = np.array([])
    prev_key = np.array([])
    prev_rt = np.array([])
    prev_cond_arr = np.array([])

    for cond in conds_sq:
        f_name = findfilenameindirectory(path, [whSub, cond], numberoffiles=1)
        assert len(f_name) == 1, f'''
            No unique file for '{whSub}' and {cond}
            <combineconditions_mc>
            '''
        trl_no, dxn, coh, isOK, key, rt = import_psyphysdata_mc(f_name[0], path)

        cond_arr = np.array(
            [np.where(
                conds_id == cond
            )] * len(trl_no)
        )
        trl_no += prev_sxn_trl_no
        prev_sxn_trl_no = trl_no[-1]

        prev_trl_no = np.concatenate((prev_trl_no, trl_no), axis=None)
        prev_cond_arr = np.concatenate((prev_cond_arr, cond_arr), axis=None)
        prev_dxn = np.concatenate((prev_dxn, dxn), axis=None)
        prev_coh = np.concatenate((prev_coh, coh), axis=None)
        prev_isOK = np.concatenate((prev_isOK, isOK), axis=None)
        prev_key = np.concatenate((prev_key, key), axis=None)
        prev_rt = np.concatenate((prev_rt, rt), axis=None)

    return prev_trl_no, prev_cond_arr, prev_dxn, prev_coh, prev_isOK, prev_key, prev_rt


def importfromstandardcsv_moz(path, whType='str'):
    """
    Imports data from .csv files in the following format.
    The format of the file should follow the rules below:
        1. Lines starting with '#' are ignored. A description of the file should be given here.
        2. If data contains a list:
            a. Elements should be separated with space.
            b. The function will return the list as a string without characters '[' or ']'.
    """

    if whType not in ['str', 'float', 'int', 'bool']:
        print('''whType should be equal to one of the following: 'str', 'float', 'int','bool''')

    f = open(path, 'r')

    f_l = f.readlines()

    dat = []
    for l in f_l:
        if l[0] != '#':
            trl = l.split(', ')
            for idx in range(len(trl)):
                for spcl_char in ['\n', '[', ']']:
                    if spcl_char in trl[idx]:
                        if spcl_char == '[' and whType != 'str':
                            whType = 'str'
                            print(f'''
                            The file contains a list. Therefore, the output will be returned as type string!
                            filename: {path}''')
                        trl[idx] = trl[idx].replace('\n', '')
            dat.append(trl)
    dat = np.array(dat).astype(whType)
    return dat


def array2tupleofvectors(arr):
    col = []
    for whCol in range(arr.shape[1]):
        col.append(arr[:, whCol])
    return tuple(col)

