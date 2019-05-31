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
from get_those_tree_objects_with_numpy import *
from collections import OrderedDict
import time
rt.gROOT.SetBatch()
rt.TH1.AddDirectory(rt.kFALSE)

########## get_histograms function template ############

########################################################


def get_histograms(list_of_files_, variable_list_, cuts_to_apply_=None):
    
    hist = OrderedDict()
    counts = OrderedDict()
    for sample in list_of_files_:
        hist[sample] = OrderedDict()
        counts[sample] = OrderedDict()
        for tree_name in list_of_files_[sample]['trees']:
            print '\nReserving Histograms for:', sample, tree_name 
            hist[sample][tree_name] = OrderedDict()
            counts[sample][tree_name] = OrderedDict()
            # Reserve histograms
            hist[sample][tree_name]['MET'] = rt.TH1D('MET_'+sample+'_'+tree_name, 'E_{T}^{miss} [GeV]', 500, 0, 1000)
            hist[sample][tree_name]['S_Flavor_jet'] = rt.TH1D('S_Flavor_jet_'+sample+'_'+tree_name, 'Flavor S jets', 20, 0, 21)
            hist[sample][tree_name]['ISR_Flavor_jet'] = rt.TH1D('ISR_Flavor_jet_'+sample+'_'+tree_name, 'Flavor S jets', 20, 0, 21)
            hist[sample][tree_name]['S_Flavor_lep'] = rt.TH1D('S_Flavor_lep_'+sample+'_'+tree_name, 'Flavor S leps', 20, 0, 21)
            hist[sample][tree_name]['ISR_Flavor_lep'] = rt.TH1D('ISR_Flavor_lep_'+sample+'_'+tree_name, 'Flavor S leps', 20, 0, 21)

            hist[sample][tree_name]['Lep_to_Charge'] = rt.TH1D('Lep_to_Charge_'+sample+'_'+tree_name, '', 2, 0, 2, 2, 0, 2)
            hist[sample][tree_name]['Lep_to_Lep'] = rt.TH1D('Lep_to_Lep_'+sample+'_'+tree_name, '', 2, 0, 2, 2, 0, 2)

            hist[sample][tree_name]['RISR'] = rt.TH1D('risr_'+sample+'_'+tree_name, 'RISR', 500, 0, 2)
            hist[sample][tree_name]['PTISR'] = rt.TH1D('ptisr_'+sample+'_'+tree_name, 'p_{T} ISR [GeV]', 500, 0, 1000)
            hist[sample][tree_name]['PTCM'] = rt.TH1D('ptcm_'+sample+'_'+tree_name, 'p_{T} CM [GeV]', 500, 0, 1000)

            hist[sample][tree_name]['RISR_PTISR'] = rt.TH2D('RISR_PTISR_'+sample+'_'+tree_name, 'RISR_PTISR', 500, 0, 2, 500, 0, 1000)

            hist[sample][tree_name]['RISR_PTCM'] = rt.TH2D('RISR_PTCM_'+sample+'_'+tree_name, 'RISR_PTCM', 500, 0, 2, 500, 0, 1000)

            hist[sample][tree_name]['PTCM_div_PTISR'] = rt.TH1D('PTCM_div_PTISR_'+sample+'_'+tree_name, 'dphi_PTCM_div_PTISR', 500, 0, 1)
            hist[sample][tree_name]['dphi_PTCM_div_PTISR'] = rt.TH2D('dphi_PTCM_div_PTISR_'+sample+'_'+tree_name, 'dphi_PTCM_div_PTISR', 500, 0, np.pi, 500, 0, 1)
            hist[sample][tree_name]['dphi_PTCM'] = rt.TH2D('dphi_PTCM_'+sample+'_'+tree_name, 'dphi_PTCM', 500, 0, np.pi, 500, 0, 1000)

            hist[sample][tree_name]['PTISR_PTCM'] = rt.TH2D('PTISR_PTCM_'+sample+'_'+tree_name, 'PTISR_PTCM', 500, 0, 1000, 500, 0, 1000)

            hist[sample][tree_name]['S_ISR_N_jet'] = rt.TH2D('S_ISR_PT_jet_'+sample+'_'+tree_name, 'Eta_PT_jet', 50, 0, 50, 50, 0, 50)
            hist[sample][tree_name]['S_ISR_N_lep'] = rt.TH2D('S_ISR_PT_lep_'+sample+'_'+tree_name, 'Eta_PT_lep', 50, 0, 50, 50, 0, 50)

            hist[sample][tree_name]['S_ISR_N_loose_jet'] = rt.TH2D('S_ISR_PT_jet_'+sample+'_'+tree_name, 'Eta_PT_jet', 50, 0, 50, 50, 0, 50)
            hist[sample][tree_name]['S_ISR_N_medium_jet'] = rt.TH2D('S_ISR_PT_jet_'+sample+'_'+tree_name, 'Eta_PT_jet', 50, 0, 50, 50, 0, 50)
            hist[sample][tree_name]['S_ISR_N_tight_jet'] = rt.TH2D('S_ISR_PT_jet_'+sample+'_'+tree_name, 'Eta_PT_jet', 50, 0, 50, 50, 0, 50)

            hist[sample][tree_name]['RISR_N_jet'] = rt.TH2D('RISR_N_jet_'+sample+'_'+tree_name, 'RISR_PTISR', 500, 0, 2, 500, 0, 1000)
            hist[sample][tree_name]['RISR_N_lep'] = rt.TH2D('RISR_N_lep_'+sample+'_'+tree_name, 'RISR_PTISR', 500, 0, 2, 500, 0, 1000)

            hist[sample][tree_name]['RISR_N_S_jet'] = rt.TH2D('RISR_N_S_jet_'+sample+'_'+tree_name, 'RISR_PTISR', 500, 0, 2, 500, 0, 1000)
            hist[sample][tree_name]['RISR_N_S_lep'] = rt.TH2D('RISR_N_S_lep_'+sample+'_'+tree_name, 'RISR_PTISR', 500, 0, 2, 500, 0, 1000)

            hist[sample][tree_name]['RISR_N_ISR_jet'] = rt.TH2D('RISR_N_ISR_jet_'+sample+'_'+tree_name, 'RISR_PTISR', 500, 0, 2, 500, 0, 1000)
            hist[sample][tree_name]['RISR_N_ISR_lep'] = rt.TH2D('RISR_N_ISR_lep_'+sample+'_'+tree_name, 'RISR_PTISR', 500, 0, 2, 500, 0, 1000)

            hist[sample][tree_name]['PTISR_N_jet'] = rt.TH2D('PTISR_N_jet_'+sample+'_'+tree_name, 'PTISR_PTISR', 500, 0, 2, 500, 0, 1000)
            hist[sample][tree_name]['PTISR_N_lep'] = rt.TH2D('PTISR_N_lep_'+sample+'_'+tree_name, 'PTISR_PTISR', 500, 0, 2, 500, 0, 1000)

            hist[sample][tree_name]['PTISR_N_S_jet'] = rt.TH2D('PTISR_N_S_jet_'+sample+'_'+tree_name, 'PTISR_PTISR', 500, 0, 2, 500, 0, 1000)
            hist[sample][tree_name]['PTISR_N_S_lep'] = rt.TH2D('PTISR_N_S_lep_'+sample+'_'+tree_name, 'PTISR_PTISR', 500, 0, 2, 500, 0, 1000)

            hist[sample][tree_name]['PTISR_N_ISR_jet'] = rt.TH2D('PTISR_N_ISR_jet_'+sample+'_'+tree_name, 'PTISR_PTISR', 500, 0, 2, 500, 0, 1000)
            hist[sample][tree_name]['PTISR_N_ISR_lep'] = rt.TH2D('PTISR_N_ISR_lep_'+sample+'_'+tree_name, 'PTISR_PTISR', 500, 0, 2, 500, 0, 1000)

            hist[sample][tree_name]['PTCM_N_jet'] = rt.TH2D('PTCM_N_jet_'+sample+'_'+tree_name, 'PTCM_PTISR', 500, 0, 2, 500, 0, 1000)
            hist[sample][tree_name]['PTCM_N_lep'] = rt.TH2D('PTCM_N_lep_'+sample+'_'+tree_name, 'PTCM_PTISR', 500, 0, 2, 500, 0, 1000)

            hist[sample][tree_name]['PTCM_N_S_jet'] = rt.TH2D('PTCM_N_S_jet_'+sample+'_'+tree_name, 'PTCM_PTISR', 500, 0, 2, 500, 0, 1000)
            hist[sample][tree_name]['PTCM_N_S_lep'] = rt.TH2D('PTCM_N_S_lep_'+sample+'_'+tree_name, 'PTCM_PTISR', 500, 0, 2, 500, 0, 1000)

            hist[sample][tree_name]['PTCM_N_ISR_jet'] = rt.TH2D('PTCM_N_ISR_jet_'+sample+'_'+tree_name, 'PTCM_PTISR', 500, 0, 2, 500, 0, 1000)
            hist[sample][tree_name]['PTCM_N_ISR_lep'] = rt.TH2D('PTCM_N_ISR_lep_'+sample+'_'+tree_name, 'PTCM_PTISR', 500, 0, 2, 500, 0, 1000)

            
        for ifile, in_file in enumerate(list_of_files_[sample]['files']):
            sample_array = get_tree_info_singular(sample, in_file, list_of_files_[sample]['trees'], variable_list_, cuts_to_apply_)
            for tree_name in sample_array[sample]:
                print '\nGetting Histograms for:', sample, tree_name, in_file
                print 'file: ', ifile+1, ' / ', len(list_of_files_[sample]['files'])

                pt_jet = np.array(sample_array[sample][tree_name]['PT_jet'])
                flavor_jet = np.array(sample_array[sample][tree_name]['Flavor_jet'])
                base_isr_index_jet = np.array(sample_array[sample][tree_name]['index_jet_ISR'])
                base_s_index_jet = np.array(sample_array[sample][tree_name]['index_jet_S'])
                bjet_tag = np.array(sample_array[sample][tree_name]['Btag_ID_jet'])

                pt_lep = np.array(sample_array[sample][tree_name]['PT_lep'])
                charge_lep = np.array(sample_array[sample][tree_name]['Charge_lep'])
                id_lep = np.array(sample_array[sample][tree_name]['ID_lep'])
                pdgid_lep = np.array(sample_array[sample][tree_name]['PDGID_lep'])
                base_isr_index_lep = np.array(sample_array[sample][tree_name]['index_lep_ISR'])
                base_s_index_lep = np.array(sample_array[sample][tree_name]['index_lep_S'])

                met = np.array(sample_array[sample][tree_name]['MET'])
                base_risr = np.array(sample_array[sample][tree_name]['RISR'])
                base_ptisr = np.array(sample_array[sample][tree_name]['PTISR'])
                base_ptcm = np.array(sample_array[sample][tree_name]['PTCM'])
                base_dphi = np.array(sample_array[sample][tree_name]['dphiCMI'])
                weight = np.array(sample_array[sample][tree_name]['weight'])

                jet_len = [len(jets) for jets in pt_jet]
                max_n_jets = np.amax(jet_len)

                pt_jet = np.array([np.pad(jets, (0, max_n_jets - len(jets)), 'constant', constant_values=np.nan) for jets in pt_jet]) 
                flavor_jet = np.array([np.pad(jets, (0, max_n_jets - len(jets)), 'constant', constant_values=np.nan) for jets in flavor_jet]) 
                bjet_tag = np.array([np.pad(jets, (0, max_n_jets - len(jets)), 'constant', constant_values=np.nan) for jets in bjet_tag])

                lep_len = [len(leps) for leps in pt_lep]
                max_n_leps = np.amax(lep_len)

                pt_lep = np.array([np.pad(leps, (0, max_n_leps - len(leps)), 'constant', constant_values=np.nan) for leps in pt_lep]) 
                charge_lep = np.array([np.pad(leps, (0, max_n_leps - len(leps)), 'constant', constant_values=np.nan) for leps in charge_lep]) 
                pdgid_lep = np.array([np.pad(leps, (0, max_n_leps - len(leps)), 'constant', constant_values=np.nan) for leps in pdgid_lep]) 
                only_2_leps = [True for lep in lep_len if len(lep)==2]
                only_2_opp_leps = [True for lep, charge in zip(only_2_leps, ch_lep) if lep and len(charge[charge>0])>0 and len(charge[charge<0])>0]

                risr = np.array([entry[:3] for entry in base_risr])
                ptisr = np.array([entry[:3] for entry in base_ptisr])
                ptcm = np.array([entry[:3] for entry in base_ptcm])
                dphi = np.array([entry[:3] for entry in base_dphi])
                isr_index_jet = np.array([entry[:3] for entry in base_isr_index_jet])
                s_index_jet = np.array([entry[:3] for entry in base_s_index_jet])
                isr_index_lep = np.array([entry[:3] for entry in base_isr_index_lep])
                s_index_lep = np.array([entry[:3] for entry in base_s_index_lep])

                # test_loose = [1,2,3]
                # test_medium = [2,3]
                # test_tight = [3]
                test_loose = [2,3,4]
                test_medium = [3,4]
                test_tight = [4]

                risr = risr[:, 2]
                ptisr = ptisr[:, 2]
                ptcm = ptcm[:, 2]
                isr_index_jet = isr_index_jet[:, 2]
                s_index_jet = s_index_jet[:, 2]
                isr_index_lep = isr_index_lep[:, 2]
                s_index_lep = s_index_lep[:, 2]
                dphi = dphi[:, 0]

                # risr_lepV_jetI = risr[:,0]
                # risr_lepV_jetA = risr[:,1]
                # risr_lepA_jetA = risr[:,2]
                
                print '\ncreating masks and weights'
                print '-> bjet masks'
                loose_mask = np.isin(bjet_tag, test_loose)
                medium_mask = np.isin(bjet_tag, test_medium)
                tight_mask = np.isin(bjet_tag, test_tight)

                has_2_loose = [True for mask in loose_mask if len(mask[mask]) >= 2]
                has_2_medium = [True for mask in medium_mask if len(mask[mask]) >= 2]
                has_2_medium = [True for mask in medium_mask if len(mask[mask]) >= 2]

                print '-> S bjet masks'
                loose_s_mask = np.array([mask[index] for mask, index in zip(loose_mask, s_index_jet)])
                medium_s_mask = np.array([mask[index] for mask, index in zip(medium_mask, s_index_jet)])
                tight_s_mask = np.array([mask[index] for mask, index in zip(tight_mask, s_index_jet)])

                print '-> ISR bjet masks'
                loose_isr_mask = np.array([mask[index] for mask, index in zip(loose_mask, isr_index_jet)])
                medium_isr_mask = np.array([mask[index] for mask, index in zip(medium_mask, isr_index_jet)])
                tight_isr_mask = np.array([mask[index] for mask, index in zip(tight_mask, isr_index_jet)])

                print '-> event bjet masks'
                is_loose = np.array([np.any(event) for event in loose_mask])
                is_medium = np.array([np.any(event) for event in medium_mask])
                is_tight = np.array([np.any(event) for event in tight_mask])

                print '-> jet weights'
                jet_weight = np.array([np.array([np.float64(event)]*len(jets[~np.isnan(jets)])) for jets, event in zip(pt_jet, weight)]) 
                jet_weight = np.array([np.pad(w, (0, max_n_jets - len(w)), 'constant', constant_values=np.nan) for w in jet_weight]) 

                s_jet_weight = np.array([jets[index] for jets, index in zip(jet_weight, s_index_jet)])
                isr_jet_weight = np.array([jets[index] for jets, index in zip(jet_weight, isr_index_jet)])

                pt_s_jet = np.array([jets[index] for jets, index in zip(pt_jet, s_index_jet)])
                pt_isr_jet = np.array([jets[index] for jets, index in zip(pt_jet, isr_index_jet)])
                eta_s_jet = np.array([jets[index] for jets, index in zip(eta_jet, s_index_jet)])
                eta_isr_jet = np.array([jets[index] for jets, index in zip(eta_jet, isr_index_jet)])

                print '-> lep weights'
                lep_weight = np.array([np.array([np.float64(event)]*len(leps[~np.isnan(leps)])) for leps, event in zip(pt_lep, weight)]) 
                lep_weight = np.array([np.pad(w, (0, max_n_leps - len(w)), 'constant', constant_values=np.nan) for w in lep_weight]) 

                s_lep_weight = np.array([leps[index] for leps, index in zip(lep_weight, s_index_lep)])
                isr_lep_weight = np.array([leps[index] for leps, index in zip(lep_weight, isr_index_lep)])

                pt_s_lep = np.array([leps[index] for leps, index in zip(pt_lep, s_index_lep)])
                pt_isr_lep = np.array([leps[index] for leps, index in zip(pt_lep, isr_index_lep)])
                eta_s_lep = np.array([leps[index] for leps, index in zip(eta_lep, s_index_lep)])
                eta_isr_lep = np.array([leps[index] for leps, index in zip(eta_lep, isr_index_lep)])

                print '\napplying masks'
                print '-> jet pt'
                loose_pt_jet = pt_jet[loose_mask]
                medium_pt_jet = pt_jet[medium_mask]
                tight_pt_jet = pt_jet[tight_mask]

                print '-> S jet pt + padding'
                s_jet_len = [len(jets) for jets in pt_s_jet]
                max_n_s_jets = np.amax(s_jet_len)
                pt_s_jet = np.array([np.pad(jets, (0, max_n_s_jets - len(jets)), 'constant', constant_values=np.nan) for jets in pt_s_jet]) 
                eta_s_jet = np.array([np.pad(jets, (0, max_n_s_jets - len(jets)), 'constant', constant_values=np.nan) for jets in eta_s_jet]) 
                s_jet_weight = np.array([np.pad(jets, (0, max_n_s_jets - len(jets)), 'constant', constant_values=np.nan) for jets in s_jet_weight]) 
                loose_s_mask = np.array([np.pad(jets, (0, max_n_s_jets - len(jets)), 'constant', constant_values=False) for jets in loose_s_mask]) 
                medium_s_mask = np.array([np.pad(jets, (0, max_n_s_jets - len(jets)), 'constant', constant_values=False) for jets in medium_s_mask]) 
                tight_s_mask = np.array([np.pad(jets, (0, max_n_s_jets - len(jets)), 'constant', constant_values=False) for jets in tight_s_mask]) 

                loose_pt_s_jet = pt_s_jet[loose_s_mask]
                medium_pt_s_jet = pt_s_jet[medium_s_mask]
                tight_pt_s_jet = pt_s_jet[tight_s_mask]

                print '-> ISR jet pt + padding'
                isr_jet_len = [len(jets) for jets in pt_isr_jet]
                max_n_isr_jets = np.amax(isr_jet_len)
                pt_isr_jet = np.array([np.pad(jets, (0, max_n_isr_jets - len(jets)), 'constant', constant_values=np.nan) for jets in pt_isr_jet]) 
                eta_isr_jet = np.array([np.pad(jets, (0, max_n_isr_jets - len(jets)), 'constant', constant_values=np.nan) for jets in eta_isr_jet]) 
                isr_jet_weight = np.array([np.pad(jets, (0, max_n_isr_jets - len(jets)), 'constant', constant_values=np.nan) for jets in isr_jet_weight]) 
                loose_isr_mask = np.array([np.pad(jets, (0, max_n_isr_jets - len(jets)), 'constant', constant_values=False) for jets in loose_isr_mask]) 
                medium_isr_mask = np.array([np.pad(jets, (0, max_n_isr_jets - len(jets)), 'constant', constant_values=False) for jets in medium_isr_mask]) 
                tight_isr_mask = np.array([np.pad(jets, (0, max_n_isr_jets - len(jets)), 'constant', constant_values=False) for jets in tight_isr_mask]) 

                loose_pt_isr_jet = pt_isr_jet[loose_isr_mask]
                medium_pt_isr_jet = pt_isr_jet[medium_isr_mask]
                tight_pt_isr_jet = pt_isr_jet[tight_isr_mask]

                print '-> S lep pt + padding'
                s_lep_len = [len(leps) for leps in pt_s_lep]
                max_n_s_leps = np.amax(s_lep_len)
                pt_s_lep = np.array([np.pad(leps, (0, max_n_s_leps - len(leps)), 'constant', constant_values=np.nan) for leps in pt_s_lep]) 
                eta_s_lep = np.array([np.pad(leps, (0, max_n_s_leps - len(leps)), 'constant', constant_values=np.nan) for leps in eta_s_lep]) 
                s_lep_weight = np.array([np.pad(leps, (0, max_n_s_leps - len(leps)), 'constant', constant_values=np.nan) for leps in s_lep_weight]) 

                print '-> ISR lep pt + padding'
                isr_lep_len = [len(leps) for leps in pt_isr_lep]
                max_n_isr_leps = np.amax(isr_lep_len)
                pt_isr_lep = np.array([np.pad(leps, (0, max_n_isr_leps - len(leps)), 'constant', constant_values=np.nan) for leps in pt_isr_lep]) 
                eta_isr_lep = np.array([np.pad(leps, (0, max_n_isr_leps - len(leps)), 'constant', constant_values=np.nan) for leps in eta_isr_lep]) 
                isr_lep_weight = np.array([np.pad(leps, (0, max_n_isr_leps - len(leps)), 'constant', constant_values=np.nan) for leps in isr_lep_weight]) 

                print '-> N S jets'
                n_s_jet = [len(jets[~np.isnan(jets)]) for jets in pt_s_jet]
                n_loose_s_jet = [len(jets[mask]) for jets, mask in zip(pt_s_jet, loose_s_mask)]
                n_medium_s_jet = [len(jets[mask]) for jets, mask in zip(pt_s_jet, medium_s_mask)]
                n_tight_s_jet = [len(jets[mask]) for jets, mask in zip(pt_s_jet, tight_s_mask)]

                print '-> N ISR jets'
                n_isr_jet = [len(jets[~np.isnan(jets)]) for jets in pt_isr_jet]
                n_loose_isr_jet = [len(jets[mask]) for jets, mask in zip(pt_isr_jet, loose_isr_mask)]
                n_medium_isr_jet = [len(jets[mask]) for jets, mask in zip(pt_isr_jet, medium_isr_mask)]
                n_tight_isr_jet = [len(jets[mask]) for jets, mask in zip(pt_isr_jet, tight_isr_mask)]

                print '-> N S leps'
                n_s_lep = [len(leps[~np.isnan(leps)]) for leps in pt_s_lep]

                print '-> N ISR leps'
                n_isr_lep = [len(leps[~np.isnan(leps)]) for leps in pt_isr_lep]

                print '-> Event variables' 
                ptcm_div_ptisr = np.divide(ptcm, ptisr)

                print '-> lead jet pt'
                pt_lead_jet = [np.amax(jets[~np.isnan(jets)]) for jets in pt_jet if jets[~np.isnan(jets)].size != 0]
                lead_weight = [w for w, jets in zip(weight, pt_jet) if jets[~np.isnan(jets)].size != 0]
                loose_pt_lead_jet = [np.amax(jets[mask]) for jets, mask in zip(pt_jet, loose_mask) if jets[mask].size != 0]
                medium_pt_lead_jet = [np.amax(jets[mask]) for jets, mask in zip(pt_jet, medium_mask) if jets[mask].size != 0]
                tight_pt_lead_jet = [np.amax(jets[mask]) for jets, mask in zip(pt_jet, tight_mask) if jets[mask].size != 0]

                print '-> jet weights'
                loose_weight = weight[is_loose]
                medium_weight = weight[is_medium]
                tight_weight = weight[is_tight]

                loose_jet_weight = jet_weight[loose_mask]
                medium_jet_weight = jet_weight[medium_mask]
                tight_jet_weight = jet_weight[tight_mask]

                loose_s_jet_weight = s_jet_weight[loose_s_mask]
                medium_s_jet_weight = s_jet_weight[medium_s_mask]
                tight_s_jet_weight = s_jet_weight[tight_s_mask]

                loose_isr_jet_weight = isr_jet_weight[loose_isr_mask]
                medium_isr_jet_weight = isr_jet_weight[medium_isr_mask]
                tight_isr_jet_weight = isr_jet_weight[tight_isr_mask]

                print 'done applying mask'
                print '\nfilling histograms'

                #loose_weight = [w for w in loose_weight if w is not None]
                #medium_weight = [w for w in medium_weight if w is not None]
                #tight_weight = [w for w in tight_weight if w is not None]
               
                # loose_weight = np.concatenate(loose_weight)
                # medium_weight = np.concatenate(medium_weight)
                # tight_weight = np.concatenate(tight_weight)
                jet_weight = jet_weight[~np.isnan(pt_jet)]
                eta_jet = eta_jet[~np.isnan(pt_jet)]
                pt_jet = pt_jet[~np.isnan(pt_jet)]
                
                s_jet_weight = s_jet_weight[~np.isnan(pt_s_jet)]
                eta_s_jet = eta_s_jet[~np.isnan(pt_s_jet)]
                pt_s_jet = pt_s_jet[~np.isnan(pt_s_jet)]
                
                isr_jet_weight = isr_jet_weight[~np.isnan(pt_isr_jet)]
                eta_isr_jet = eta_isr_jet[~np.isnan(pt_isr_jet)]
                pt_isr_jet = pt_isr_jet[~np.isnan(pt_isr_jet)]
               
                lep_weight = lep_weight[~np.isnan(pt_lep)]
                eta_lep = eta_lep[~np.isnan(pt_lep)]
                pt_lep = pt_lep[~np.isnan(pt_lep)]
                
                s_lep_weight = s_lep_weight[~np.isnan(pt_s_lep)]
                eta_s_lep = eta_s_lep[~np.isnan(pt_s_lep)]
                pt_s_lep = pt_s_lep[~np.isnan(pt_s_lep)]
                
                isr_lep_weight = isr_lep_weight[~np.isnan(pt_isr_lep)]
                eta_isr_lep = eta_isr_lep[~np.isnan(pt_isr_lep)]
                pt_isr_lep = pt_isr_lep[~np.isnan(pt_isr_lep)]
 
                div_weight = weight[~np.isnan(ptcm_div_ptisr)]
                div_dphi = dphi[~np.isnan(ptcm_div_ptisr)]
                ptcm_div_ptisr = ptcm_div_ptisr[~np.isnan(ptcm_div_ptisr)]

                rnp.fill_hist(hist[sample][tree_name]['weight'], weight)
                rnp.fill_hist(hist[sample][tree_name]['MET'], met, weight)

                rnp.fill_hist(hist[sample][tree_name]['RISR'], risr, weight)
                rnp.fill_hist(hist[sample][tree_name]['PTISR'], ptisr, weight)
                rnp.fill_hist(hist[sample][tree_name]['PTCM'], ptcm, weight)

                rnp.fill_hist(hist[sample][tree_name]['S_ISR_N_jet'], np.swapaxes([n_s_jet,n_isr_jet],0,1), weight)
                rnp.fill_hist(hist[sample][tree_name]['S_ISR_N_lep'], np.swapaxes([n_s_lep,n_isr_lep],0,1), weight)

                rnp.fill_hist(hist[sample][tree_name]['S_ISR_N_loose_jet'], np.swapaxes([n_s_loose_jet,n_isr_loose_jet],0,1), weight)
                rnp.fill_hist(hist[sample][tree_name]['S_ISR_N_medium_jet'], np.swapaxes([n_s_medium_jet,n_isr_medium_jet],0,1), weight)
                rnp.fill_hist(hist[sample][tree_name]['S_ISR_N_tight_jet'], np.swapaxes([n_s_tight_jet,n_isr_tight_jet],0,1), weight)
                rnp.fill_hist(hist[sample][tree_name]['S_ISR_N_lep'], np.swapaxes([n_s_lep,n_isr_lep],0,1), weight)

                rnp.fill_hist(hist[sample][tree_name]['dphi_PTCM'], np.swapaxes([dphi,ptcm],0,1), weight)
                rnp.fill_hist(hist[sample][tree_name]['dphi_PTCM_div_PTISR'], np.swapaxes([div_dphi,ptcm_div_ptisr],0,1), div_weight)

                rnp.fill_hist(hist[sample][tree_name]['RISR_PTCM'], np.swapaxes([risr,ptcm],0,1), weight)

                rnp.fill_hist(hist[sample][tree_name]['RISR_PTISR'], np.swapaxes([risr,ptisr],0,1), weight)

                rnp.fill_hist(hist[sample][tree_name]['PTISR_PTCM'], np.swapaxes([ptisr,ptcm],0,1), weight)

                print 'finished filling'
    return hist


             

if __name__ == "__main__":

    signals = { 
    'SMS-T2bW' : ['/home/t3-ku/erichjs/work/Ewkinos/reducer/CMSSW_10_1_4_patch1/src/KUEWKinoAnalysis_dev_v2/output_samples/SMS-T2bW/root/SMS-T2bW_TuneCUETP8M1_13TeV-madgraphMLM-pythia8SMS-T2bW']
              }
    backgrounds = {
    'TTJets' : ['/home/t3-ku/erichjs/work/Ewkinos/reducer/CMSSW_10_1_4_patch1/src/KUEWKinoAnalysis_dev_v2/output_samples/ttbar_2016/root/TTJets_TuneCUETP8M2T4_13TeV-amcatnloFXFX-pythia8ttbar_2016'],
    'WJets' : [
              '/home/t3-ku/erichjs/work/Ewkinos/reducer/CMSSW_10_1_4_patch1/src/KUEWKinoAnalysis_dev_v2/output_samples/WJetsToLNu/root/WJetsToLNu_TuneCUETP8M1_13TeV-amcatnloFXFX-pythia8WJetsToLNu',
              #'/home/t3-ku/erichjs/work/Ewkinos/reducer/CMSSW_10_1_4_patch1/src/KUEWKinoAnalysis_dev_v2/output_samples/WJetsToLNu_70to100/root/WJetsToLNu_HT-70To100_TuneCUETP8M1_13TeV-madgraphMLM-pythia8WJetsToLNu_70to100',
              #'/home/t3-ku/erichjs/work/Ewkinos/reducer/CMSSW_10_1_4_patch1/src/KUEWKinoAnalysis_dev_v2/output_samples/WJetsToLNu_100to200/root/WJetsToLNu_HT-100To200_TuneCUETP8M1_13TeV-madgraphMLM-pythia8WJetsToLNu_100to200',
              #'/home/t3-ku/erichjs/work/Ewkinos/reducer/CMSSW_10_1_4_patch1/src/KUEWKinoAnalysis_dev_v2/output_samples/WJetsToLNu_200to400/root/WJetsToLNu_HT-200To400_TuneCUETP8M1_13TeV-madgraphMLM-pythia8WJetsToLNu_200to400',
              #'/home/t3-ku/erichjs/work/Ewkinos/reducer/CMSSW_10_1_4_patch1/src/KUEWKinoAnalysis_dev_v2/output_samples/WJetsToLNu_400to600/root/WJetsToLNu_HT-400To600_TuneCUETP8M1_13TeV-madgraphMLM-pythia8WJetsToLNu_400to600',
              #'/home/t3-ku/erichjs/work/Ewkinos/reducer/CMSSW_10_1_4_patch1/src/KUEWKinoAnalysis_dev_v2/output_samples/WJetsToLNu_600to800/root/WJetsToLNu_HT-600To800_TuneCUETP8M1_13TeV-madgraphMLM-pythia8WJetsToLNu_600to800',
              #'/home/t3-ku/erichjs/work/Ewkinos/reducer/CMSSW_10_1_4_patch1/src/KUEWKinoAnalysis_dev_v2/output_samples/WJetsToLNu_800to1200/root/WJetsToLNu_HT-800To1200_TuneCUETP8M1_13TeV-madgraphMLM-pythia8WJetsToLNu_800to1200',
              #'/home/t3-ku/erichjs/work/Ewkinos/reducer/CMSSW_10_1_4_patch1/src/KUEWKinoAnalysis_dev_v2/output_samples/WJetsToLNu_1200to2500/root/WJetsToLNu_HT-1200To2500_TuneCUETP8M1_13TeV-madgraphMLM-pythia8WJetsToLNu_1200to2500',
              #'/home/t3-ku/erichjs/work/Ewkinos/reducer/CMSSW_10_1_4_patch1/src/KUEWKinoAnalysis_dev_v2/output_samples/WJetsToLNu_2500toInf/root/WJetsToLNu_HT-2500ToInf_TuneCUETP8M1_13TeV-madgraphMLM-pythia8WJetsToLNu_2500toInf'
              ],
                  }
    variables = ['MET', 'PT_jet', 'index_jet_ISR', 'index_jet_S', 'Btag_ID_jet', 'Flavor_jet', 'PT_lep', 'Charge_lep', 'ID_lep', 'PDGID_lep', 'index_lep_ISR', 'index_lep_S', 'RISR', 'PTISR', 'PTCM', 'dphiCMI', 'weight']

    start_b = time.time()    
    background_list = process_the_samples(backgrounds, None, None)
    hist_background = get_histograms(background_list, variables, None)

    write_hists_to_file(hist_background, './output_background_cat1_hists.root') 
    stop_b = time.time()

    signal_list = process_the_samples(signals, None, None)
    hist_signal = get_histograms(signal_list, variables, None)

    write_hists_to_file(hist_signal, './output_signal_cat1_hists.root')  
    stop_s = time.time()

    print "background: ", stop_b - start_b
    print "signal:     ", stop_s - stop_b
    print "total:      ", stop_s - start_b
 
    print 'finished writing'
