#!/usr/bin/env python

"""
New thing to try out, doing analysis with numpy, converting back to ROOT to do histogramming
Creator: Erich Schmitz
Date: Feb 22, 2019
"""

import ROOT as rt
import numpy as np
import root_numpy as rnp
import numpy.lib.recfunctions as rfc
import os
from collections import OrderedDict

rt.gROOT.SetBatch()
rt.TH1.AddDirectory(rt.kFALSE)

def get_tree_info_singular(sample_, file_name_, tree_names_, variable_list_, cuts_to_apply_=None):
    """
    Same as get_tree_info_plural, but runs over a single file
    returns structured array containing the list of variables
    """
    tmp_array = {}
    tmp_array[sample_] = OrderedDict()
    for tree in tree_names_:
        tmp_f = rt.TFile(file_name_, 'r')
        tmp_t = tmp_f.Get(tree)
        if bool(tmp_t) and tmp_t.InheritsFrom(rt.TTree.Class()): 
            tmp_array[sample_][tree] = rnp.tree2array(tmp_t, branches=variable_list_, selection=cuts_to_apply_)
        else:
            print 'tree: ' + tree + ' is not a tree, skipping'
    
    return tmp_array

def get_tree_info_plural(file_list_, tree_list_, variable_list_, cuts_to_apply_=None):
    """
    Get the variables listed as variable list, turn into numpy arrays, used over a list of multiple files
    """
    event_dict = OrderedDict()
    

    for tree_name in tree_list_:
        event_dict[tree_name] = []
    n_files = len(file_list_)

    for ifile, file_name in enumerate(file_list_):
        print 'reading file:', ifile+1, '/', n_files

        tmp_f = rt.TFile(file_name, 'r')
        print file_name
        for tree_name in tree_list_:
            tmp_t = tmp_f.Get(tree_name)
            if bool(tmp_t) and tmp_t.InheritsFrom(rt.TTree.Class()): 
                tmp_array = rnp.tree2array(tmp_t, branches=variable_list_, selection=cuts_to_apply_)
            else:
                print 'tree: ' + tree_name + ' is not a tree, skipping'

            event_dict[tree_name].append(tmp_array)
    print 'done reading'
    print 'restructuring arrays...'

    
    for tree in event_dict:
        event_dict[tree] = np.concatenate([struc for struc in event_dict[tree]])

    print 'done restructuring'
    return event_dict 


def reduce_and_condense(file_list_of_file_lists, variable_list):
    print 'for posterity'

    tree_chain = rt.TChain('deepntuplizer/tree')
    for file_list in file_list_of_file_lists:
        files = open(file_list).readlines()
        files = [f.replace('\n','') for f in files]
        for file_name in files:
            tree_chain.Add(file_name)
        branches = [b.GetName() for b in tree_chain.GetListOfBranches()]
    for branch in branches:
        if branch not in variable_list:
            tree_chain.SetBranchStatus(branch, 0)
    file_out = rt.TFile('output_condensed.root', 'recreate')
    reduced_tree = tree_chain.CloneTree()
    reduced_tree.Write()
    file_out.Close()


def reduce_singular(in_file_name_, out_file_name_, variable_list_):
    tmp_f = rt.TFile(int_file_name_, 'r')
    tmp_t = tmp_f.Get('deepntuplizer/tree')

    branches = [b.GetName() for b in tmp_t.GetListOfBranches()]        
    for branch in branches:
        if branch not in variable_list:
            tmp_t.SetBranchStatus(branch, 0)

    out_file = rt.TFile(out_file_name_, 'recreate')
    reduced_tree = tmp_t.CloneTree()
    reduced_tree.Write()
    file_out.Close()

##### old version #####
# def process_the_samples(input_sample_list_, variable_list_, cut_list_, truncate_file_ = None, tree_in_dir_ = None):
#     array_list = OrderedDict()
# 
#     for sample, folder in input_sample_list_.items():
#         print sample, folder
#         file_list = [os.path.join(folder, f) for f in os.listdir(folder) if (os.path.isfile(os.path.join(folder, f)) and ('.root' in f))]
#         # Get file structure, in case there is a grid of mass points
#         f_struct_tmp = rt.TFile(file_list[0], 'r')
#         tree_list = []
#         if tree_in_dir_ is not None:
#             tree_list = [directory.GetName()+'/'+tree_in_dir_ for directory in f_struct_tmp.GetListOfKeys()]
#         else:
#             tree_list = [tree.GetName() for tree in f_struct_tmp.GetListOfKeys()]
# 
#         if truncate_file_ is not None:
#             file_list = file_list[:truncate_file_]
#         if 'SMS' in sample:
#             tree_name_mass = [(int(mass.split('_')[0]), int(mass.split('_')[1])) for mass in tree_list]
#             tree_name_mass.sort(key=lambda x: int(x[0]))
#             tree_list = [str(mom) + '_' + str(child) for mom, child in tree_name_mass]
# 
#         if variable_list_ is None:
#             variable_list_ = [branch.GetName() for branch in f_struct_tmp.Get(tree_list[0]).GetListOfBranches()]
#         f_struct_tmp.Close()
#         variable_list_ = list(OrderedDict.fromkeys(variable_list_))
#         array_list[sample] = get_tree_info_plural(file_list, tree_list, variable_list_)     
# 
#     file_list = None
# 
#     return array_list
##########################


def process_the_samples(input_sample_list_, truncate_file_ = None, tree_in_dir_ = None):
    list_of_files = OrderedDict()

    for sample, folder in input_sample_list_.items():
        print sample, folder
        file_list = [os.path.join(folder, f) for f in os.listdir(folder) if (os.path.isfile(os.path.join(folder, f)) and ('.root' in f))]
        # Get file structure, in case there is a grid of mass points
        f_struct_tmp = rt.TFile(file_list[0], 'r')
        tree_list = []
        if tree_in_dir_ is not None:
            tree_list = [directory.GetName()+'/'+tree_in_dir_ for directory in f_struct_tmp.GetListOfKeys()]
        else:
            tree_list = [tree.GetName() for tree in f_struct_tmp.GetListOfKeys()]

        if truncate_file_ is not None:
            file_list = file_list[:truncate_file_]
        if 'SMS' in sample:
            trees_to_keep = ['500_100', '500_325', '775_600']
            tree_name_mass = [(int(mass.split('_')[0]), int(mass.split('_')[1])) for mass in tree_list]
            tree_name_mass.sort(key=lambda x: int(x[0]))
            if trees_to_keep is not None:
                tree_list = trees_to_keep
            else:
                tree_list = [str(mom) + '_' + str(child) for mom, child in tree_name_mass]

        f_struct_tmp.Close()
        list_of_files[sample] = OrderedDict([('files', file_list), ('trees', tree_list)])
    return list_of_files



def write_hists_to_file(hists_, out_file_name_):

    out_file = rt.TFile.Open(out_file_name_, "recreate")
    print 'writing histograms to: ', out_file.GetName()
    out_file.cd()
    for sample in hists_:
        sample_dir = out_file.mkdir(sample)
        for tree in hists_[sample]:
            sample_dir.cd()
            tmp_dir = sample_dir.mkdir(tree)
            tmp_dir.cd()
            for hist in hists_[sample][tree].values():
                hist.Write()
    out_file.Close()
    print 'finished writing'


def write_arrays_to_trees(arrays_, out_file_name_):
    out_file = rt.TFile.Open(out_file_name_, "recreate")
    print 'converting arrays and writing to: ', out_file.GetName()
    for sample in arrays_:
        out_file.cd()
        sample_dir = out_file.mkdir(sample)
        sample_dir.cd()
        for tree in arrays_[sample]:
            tmp_tree = rnp.array2tree(arrays_[sample][tree])
            tmp_tree.Write()
    out_file.Close()
    print 'finished writing'

def write_arrays_to_file(arrays_, out_file_name_):
    """
    save ndarray to .npy file
    """
    if '.npy' in out_file_name_:
        np.save(out_file_name_, arrays_)
    else:
        raise ValueError(out_file_name_.split('.')[-1] + ' is the wrong file type, please use npy')
    
