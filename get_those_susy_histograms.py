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
from guppy import hpy

rt.gROOT.SetBatch()
rt.TH1.AddDirectory(rt.kFALSE)


def get_histograms(sample_array):
    hist = OrderedDict()
    for sample in sample_array:
        hist[sample] = OrderedDict()
        for tree_name in sample_array[sample]:
            print '\nGetting Histograms for:', sample, tree_name 
            hist[sample][tree_name] = OrderedDict()
  
            # Reserve histograms
            hist[sample][tree_name]['PT_jet'] = rt.TH1D('jetpt_'+sample+'_'+tree_name, 'jet p_{T} [GeV]', 1000, 0, 1000)
            hist[sample][tree_name]['loose_bjet_pt'] = rt.TH1D('loose_bjetpt_'+sample+'_'+tree_name, 'loose_bjet p_{T} [GeV]', 1000, 0, 1000)
            hist[sample][tree_name]['medium_bjet_pt'] = rt.TH1D('medium_bjetpt_'+sample+'_'+tree_name, 'medium_bjet p_{T} [GeV]', 1000, 0, 1000)
            hist[sample][tree_name]['tight_bjet_pt'] = rt.TH1D('tight_bjetpt_'+sample+'_'+tree_name, 'tight_bjet p_{T} [GeV]', 1000, 0, 1000)
            hist[sample][tree_name]['RISR'] = rt.TH1D('RISR_'+sample+'_'+tree_name, 'RISR_comb', 1000, 0, 2)
            hist[sample][tree_name]['loose_RISR'] = rt.TH1D('loose_RISR_'+sample+'_'+tree_name, 'loose bjet RISR_comb', 1000, 0, 2)
            hist[sample][tree_name]['medium_RISR'] = rt.TH1D('medium_RISR_'+sample+'_'+tree_name, 'medium bjet RISR_comb', 1000, 0, 2)
            hist[sample][tree_name]['tight_RISR'] = rt.TH1D('tight_RISR_'+sample+'_'+tree_name, 'tight bjet RISR_comb', 1000, 0, 2)
            hist[sample][tree_name]['PTISR'] = rt.TH1D('PTISR_'+sample+'_'+tree_name, 'PTISR_comb', 1000, 0, 1000)
            hist[sample][tree_name]['loose_PTISR'] = rt.TH1D('loose_PTISR_'+sample+'_'+tree_name, 'loose bjet PTISR_comb', 1000, 0, 1000)
            hist[sample][tree_name]['medium_PTISR'] = rt.TH1D('medium_PTISR_'+sample+'_'+tree_name, 'medium bjet PTISR_comb', 1000, 0, 1000)
            hist[sample][tree_name]['tight_PTISR'] = rt.TH1D('tight_PTISR_'+sample+'_'+tree_name, 'tight bjet PTISR_comb', 1000, 0, 1000)
            hist[sample][tree_name]['PTCM'] = rt.TH1D('PTCM_'+sample+'_'+tree_name, 'PTCM_comb', 1000, 0, 1000)
            hist[sample][tree_name]['loose_PTCM'] = rt.TH1D('loose_PTCM_'+sample+'_'+tree_name, 'loose bjet PTCM_comb', 1000, 0, 1000)
            hist[sample][tree_name]['medium_PTCM'] = rt.TH1D('medium_PTCM_'+sample+'_'+tree_name, 'medium bjet PTCM_comb', 1000, 0, 1000)
            hist[sample][tree_name]['tight_PTCM'] = rt.TH1D('tight_PTCM_'+sample+'_'+tree_name, 'tight bjet PTCM_comb', 1000, 0, 1000)

            hist[sample][tree_name]['RISR_jetpt'] = rt.TH2D('RISR_jetpt_'+sample+'_'+tree_name, 'loose bjet RISR_jetpt_comb', 1000, 0, 2, 1000, 0, 1000)
            hist[sample][tree_name]['loose_RISR_jetpt'] = rt.TH2D('loose_RISR_jetpt_'+sample+'_'+tree_name, 'loose bjet RISR_jetpt_comb',1000, 0, 2, 1000, 0, 1000)
            hist[sample][tree_name]['medium_RISR_jetpt'] = rt.TH2D('medium_RISR_jetpt_'+sample+'_'+tree_name, 'medium bjet RISR_jetpt_comb',1000, 0, 2, 1000, 0, 1000)
            hist[sample][tree_name]['tight_RISR_jetpt'] = rt.TH2D('tight_RISR_jetpt_'+sample+'_'+tree_name, 'tight bjet RISR_jetpt_comb',1000, 0, 2, 1000, 0, 1000)
    
            hist[sample][tree_name]['PTISR_jetpt'] = rt.TH2D('PTISR_jetpt_'+sample+'_'+tree_name, 'loose bjet PTISR_jetpt_comb', 1000, 0, 1000, 1000, 0, 1000)
            hist[sample][tree_name]['loose_PTISR_jetpt'] = rt.TH2D('loose_PTISR_jetpt_'+sample+'_'+tree_name, 'loose bjet PTISR_jetpt_comb',1000, 0, 1000, 1000, 0, 1000)
            hist[sample][tree_name]['medium_PTISR_jetpt'] = rt.TH2D('medium_PTISR_jetpt_'+sample+'_'+tree_name, 'medium bjet PTISR_jetpt_comb',1000, 0, 1000, 1000, 0, 1000)
            hist[sample][tree_name]['tight_PTISR_jetpt'] = rt.TH2D('tight_PTISR_jetpt_'+sample+'_'+tree_name, 'tight bjet PTISR_jetpt_comb',1000, 0, 1000, 1000, 0, 1000)
    
            hist[sample][tree_name]['PTCM_jetpt'] = rt.TH2D('PTCM_jetpt_'+sample+'_'+tree_name, 'loose bjet PTCM_jetpt_comb', 1000, 0, 1000, 1000, 0, 1000)
            hist[sample][tree_name]['loose_PTCM_jetpt'] = rt.TH2D('loose_PTCM_jetpt_'+sample+'_'+tree_name, 'loose bjet PTCM_jetpt_comb',1000, 0, 1000, 1000, 0, 1000)
            hist[sample][tree_name]['medium_PTCM_jetpt'] = rt.TH2D('medium_PTCM_jetpt_'+sample+'_'+tree_name, 'medium bjet PTCM_jetpt_comb',1000, 0, 1000, 1000, 0, 1000)
            hist[sample][tree_name]['tight_PTCM_jetpt'] = rt.TH2D('tight_PTCM_jetpt_'+sample+'_'+tree_name, 'tight bjet PTCM_jetpt_comb',1000, 0, 1000, 1000, 0, 1000)
    
            jet_pt = sample_array[sample][tree_name]['PT_jet'] 
            bjet_tag = sample_array[sample][tree_name]['Btag_jet']
            risr = sample_array[sample][tree_name]['RISR']
            ptisr = sample_array[sample][tree_name]['PTISR']
            ptcm = sample_array[sample][tree_name]['PTCM']
            weight = sample_array[sample][tree_name]['weight']

            risr_comb = []
            ptisr_comb = []
            ptcm_comb = []

            loose_mask = []
            medium_mask = []
            tight_mask = []

            test_loose = [1,2,3]
            test_medium = [2,3]
            test_tight = [3]
            #test_loose = [2,3,4]
            #test_medium = [3,4]
            #test_tight = [4]

            risr_comb_jets = []
            ptisr_comb_jets = []
            ptcm_comb_jets = []
            weight_jets = []

            bjet_loose_pt = []
            bjet_medium_pt = []
            bjet_tight_pt = []

            bjet_loose_eta = []
            bjet_medium_eta = []
            bjet_tight_eta = []
                
            bjet_loose_m = []
            bjet_medium_m = []
            bjet_tight_m = []

            bjet_loose_risr = []
            bjet_medium_risr = []
            bjet_tight_risr = []

            bjet_loose_ptisr = []
            bjet_medium_ptisr = []
            bjet_tight_ptisr = []

            bjet_loose_ptcm = []
            bjet_medium_ptcm = []
            bjet_tight_ptcm = []

            bjet_loose_risr_pt = []
            bjet_medium_risr_pt = []
            bjet_tight_risr_pt = []

            bjet_loose_ptisr_pt = []
            bjet_medium_ptisr_pt = []
            bjet_tight_ptisr_pt = []

            bjet_loose_ptcm_pt = []
            bjet_medium_ptcm_pt = []
            bjet_tight_ptcm_pt = []

            bjet_loose_weight = []
            bjet_medium_weight = []
            bjet_tight_weight = []

            loose_weight = []
            medium_weight = []
            tight_weight = []
            print '\n creating and applying masks'
            for ievent, event in enumerate(bjet_tag):
                # create arrays with combinatoric assignation for event observables
                risr_comb.append(np.array(risr[ievent][2]))
                ptisr_comb.append(np.array(ptisr[ievent][2]))
                ptcm_comb.append(np.array(ptcm[ievent][2]))

                # create b-tag masks
                loose_mask.append(np.isin(event, test_loose))
                medium_mask.append(np.isin(event, test_medium))
                tight_mask.append(np.isin(event, test_tight))

                jet_length = len(event)
                risr_comb_jets.append(np.array([np.float64(risr_comb[ievent])] * jet_length))
                ptisr_comb_jets.append(np.array([np.float64(ptisr_comb[ievent])] * jet_length))
                ptcm_comb_jets.append(np.array([np.float64(ptcm_comb[ievent])] * jet_length))
                weight_jets.append(np.array([np.float64(weight[ievent])] * jet_length))

                bjet_loose_pt.append(jet_pt[ievent][loose_mask[ievent]])
                bjet_medium_pt.append(jet_pt[ievent][medium_mask[ievent]])
                bjet_tight_pt.append(jet_pt[ievent][tight_mask[ievent]])

                bjet_loose_risr_pt.append(risr_comb_jets[ievent][loose_mask[ievent]])
                bjet_medium_risr_pt.append(risr_comb_jets[ievent][medium_mask[ievent]])
                bjet_tight_risr_pt.append(risr_comb_jets[ievent][tight_mask[ievent]])

                bjet_loose_ptisr_pt.append(ptisr_comb_jets[ievent][loose_mask[ievent]])
                bjet_medium_ptisr_pt.append(ptisr_comb_jets[ievent][medium_mask[ievent]])
                bjet_tight_ptisr_pt.append(ptisr_comb_jets[ievent][tight_mask[ievent]])

                bjet_loose_ptcm_pt.append(ptcm_comb_jets[ievent][loose_mask[ievent]])
                bjet_medium_ptcm_pt.append(ptcm_comb_jets[ievent][medium_mask[ievent]])
                bjet_tight_ptcm_pt.append(ptcm_comb_jets[ievent][tight_mask[ievent]])

                is_loose = np.any(loose_mask[ievent])
                is_medium = np.any(medium_mask[ievent])
                is_tight = np.any(tight_mask[ievent])
                
                bjet_loose_risr.append(risr_comb[ievent][is_loose])
                bjet_medium_risr.append(risr_comb[ievent][is_medium])
                bjet_tight_risr.append(risr_comb[ievent][is_tight])
                   
                bjet_loose_ptisr.append(ptisr_comb[ievent][is_loose])
                bjet_medium_ptisr.append(ptisr_comb[ievent][is_medium])
                bjet_tight_ptisr.append(ptisr_comb[ievent][is_tight])

                bjet_loose_ptcm.append(ptcm_comb[ievent][is_loose])
                bjet_medium_ptcm.append(ptcm_comb[ievent][is_medium])
                bjet_tight_ptcm.append(ptcm_comb[ievent][is_tight])

                loose_weight.append(weight[ievent][is_loose])
                medium_weight.append(weight[ievent][is_medium])
                tight_weight.append(weight[ievent][is_tight])

                bjet_loose_weight.append(weight_jets[ievent][loose_mask[ievent]])
                bjet_medium_weight.append(weight_jets[ievent][medium_mask[ievent]])
                bjet_tight_weight.append(weight_jets[ievent][tight_mask[ievent]])

            print 'done applying mask'
            print '\nfilling histograms'

            bjet_loose_weight = np.concatenate(bjet_loose_weight)
            bjet_medium_weight = np.concatenate(bjet_medium_weight)
            bjet_tight_weight = np.concatenate(bjet_tight_weight)

            loose_weight = [w for w in loose_weight if w is not None]
            medium_weight = [w for w in medium_weight if w is not None]
            tight_weight = [w for w in tight_weight if w is not None]

            loose_weight = np.concatenate(loose_weight)
            medium_weight = np.concatenate(medium_weight)
            tight_weight = np.concatenate(tight_weight)
            weight_jets = np.concatenate(weight_jets)

            rnp.fill_hist(hist[sample][tree_name]['PT_jet'], np.concatenate(jet_pt), weight_jets)
            rnp.fill_hist(hist[sample][tree_name]['loose_bjet_pt'], np.concatenate(bjet_loose_pt), bjet_loose_weight)
            rnp.fill_hist(hist[sample][tree_name]['medium_bjet_pt'],np.concatenate(bjet_medium_pt), bjet_medium_weight)
            rnp.fill_hist(hist[sample][tree_name]['tight_bjet_pt'] ,np.concatenate(bjet_tight_pt), bjet_tight_weight)

            rnp.fill_hist(hist[sample][tree_name]['RISR'], risr_comb, weight)
            rnp.fill_hist(hist[sample][tree_name]['PTISR'], ptisr_comb, weight)
            rnp.fill_hist(hist[sample][tree_name]['PTCM'], ptcm_comb, weight)

            rnp.fill_hist(hist[sample][tree_name]['RISR_jetpt'], np.swapaxes([np.concatenate(risr_comb_jets),np.concatenate(jet_pt)],0,1), weight_jets)
            rnp.fill_hist(hist[sample][tree_name]['PTISR_jetpt'], np.swapaxes([np.concatenate(ptisr_comb_jets),np.concatenate(jet_pt)],0,1), weight_jets)
            rnp.fill_hist(hist[sample][tree_name]['PTCM_jetpt'], np.swapaxes([np.concatenate(ptcm_comb_jets),np.concatenate(jet_pt)],0,1), weight_jets)

            bjet_loose_risr = [r for r in bjet_loose_risr if r is not None]
            bjet_loose_ptisr = [r for r in bjet_loose_ptisr if r is not None]
            bjet_loose_ptcm = [r for r in bjet_loose_ptcm if r is not None]
            bjet_loose_pt = np.concatenate(bjet_loose_pt)
 
            rnp.fill_hist(hist[sample][tree_name]['loose_RISR'], np.concatenate(bjet_loose_risr), loose_weight)
            rnp.fill_hist(hist[sample][tree_name]['loose_PTISR'], np.concatenate(bjet_loose_ptisr), loose_weight)
            rnp.fill_hist(hist[sample][tree_name]['loose_PTCM'], np.concatenate(bjet_loose_ptcm), loose_weight)
            rnp.fill_hist(hist[sample][tree_name]['loose_RISR_jetpt'], np.swapaxes([np.concatenate(bjet_loose_risr_pt), bjet_loose_pt],0,1), bjet_loose_weight)
            rnp.fill_hist(hist[sample][tree_name]['loose_PTISR_jetpt'], np.swapaxes([np.concatenate(bjet_loose_ptisr_pt), bjet_loose_pt],0,1), bjet_loose_weight)
            rnp.fill_hist(hist[sample][tree_name]['loose_PTCM_jetpt'], np.swapaxes([np.concatenate(bjet_loose_ptcm_pt), bjet_loose_pt],0,1), bjet_loose_weight)


            bjet_medium_risr = [r for r in bjet_medium_risr if r is not None]
            bjet_medium_ptisr = [r for r in bjet_medium_ptisr if r is not None]
            bjet_medium_ptcm = [r for r in bjet_medium_ptcm if r is not None]
            bjet_medium_pt = np.concatenate(bjet_medium_pt)
            rnp.fill_hist(hist[sample][tree_name]['medium_RISR'], np.concatenate(bjet_medium_risr), medium_weight)
            rnp.fill_hist(hist[sample][tree_name]['medium_PTISR'], np.concatenate(bjet_medium_ptisr), medium_weight)
            rnp.fill_hist(hist[sample][tree_name]['medium_PTCM'], np.concatenate(bjet_medium_ptcm), medium_weight)
            rnp.fill_hist(hist[sample][tree_name]['medium_RISR_jetpt'], np.swapaxes([np.concatenate(bjet_medium_risr_pt), bjet_medium_pt],0,1), bjet_medium_weight)
            rnp.fill_hist(hist[sample][tree_name]['medium_PTISR_jetpt'], np.swapaxes([np.concatenate(bjet_medium_ptisr_pt), bjet_medium_pt],0,1), bjet_medium_weight)
            rnp.fill_hist(hist[sample][tree_name]['medium_PTCM_jetpt'], np.swapaxes([np.concatenate(bjet_medium_ptcm_pt), bjet_medium_pt],0,1), bjet_medium_weight)


            bjet_tight_risr = [r for r in bjet_tight_risr if r is not None]
            bjet_tight_ptisr = [r for r in bjet_tight_ptisr if r is not None]
            bjet_tight_ptcm = [r for r in bjet_tight_ptcm if r is not None]
            bjet_tight_pt = np.concatenate(bjet_tight_pt)
            rnp.fill_hist(hist[sample][tree_name]['tight_RISR'], np.concatenate(bjet_tight_risr), tight_weight)
            rnp.fill_hist(hist[sample][tree_name]['tight_PTISR'], np.concatenate(bjet_tight_ptisr), tight_weight)
            rnp.fill_hist(hist[sample][tree_name]['tight_PTCM'], np.concatenate(bjet_tight_ptcm), tight_weight)
            rnp.fill_hist(hist[sample][tree_name]['tight_RISR_jetpt'], np.swapaxes([np.concatenate(bjet_tight_risr_pt), bjet_tight_pt],0,1), bjet_tight_weight)
            rnp.fill_hist(hist[sample][tree_name]['tight_PTISR_jetpt'], np.swapaxes([np.concatenate(bjet_tight_ptisr_pt), bjet_tight_pt],0,1), bjet_tight_weight)
            rnp.fill_hist(hist[sample][tree_name]['tight_PTCM_jetpt'], np.swapaxes([np.concatenate(bjet_tight_ptcm_pt), bjet_tight_pt],0,1), bjet_tight_weight)

            print 'finished filling'
    return hist


             

if __name__ == "__main__":

    # signals = { 
    # 'SMS-T2bW' : '/home/t3-ku/erichjs/work/Ewkinos/reducer/CMSSW_10_1_4_patch1/src/KUEWKinoAnalysis_dev/output_samples/SMS-T2bW_v2/root/SMS-T2bW_TuneCUETP8M1_13TeV-madgraphMLM-pythia8SMS-T2bW'
    #           }
    # backgrounds = {
    # 'TTJets' : '/home/t3-ku/erichjs/work/Ewkinos/reducer/CMSSW_10_1_4_patch1/src/KUEWKinoAnalysis_dev/output_samples/ttbar_2016_v2/root/TTJets_TuneCUETP8M2T4_13TeV-amcatnloFXFX-pythia8ttbar_2016'
    #               }
    signals = { 
    'SMS-T2bW' : '/home/t3-ku/erichjs/work/Ewkinos/reducer/CMSSW_10_1_4_patch1/src/KUEWKinoAnalysis_dev/output_samples/SMS-T2bW/root/SMS-T2bW_TuneCUETP8M1_13TeV-madgraphMLM-pythia8'
              }
    backgrounds = {
    'TTJets' : '/home/t3-ku/erichjs/work/Ewkinos/reducer/CMSSW_10_1_4_patch1/src/KUEWKinoAnalysis_dev/output_samples/ttbar_2016/root/TTJets_TuneCUETP8M2T4_13TeV-amcatnloFXFX-pythia8'
                  }
    variables = ['PT_jet', 'Btag_jet', 'RISR', 'PTISR', 'PTCM', 'weight']
    

    signal_array = process_the_samples(signals, variables, None, 50, None)
    hist_signal = get_histograms(signal_array)
    signal_array = None
    out_file = rt.TFile.Open("./output_signal_histos.root", "recreate")
    print 'writing histograms to: ', out_file.GetName()
    out_file.cd()
    for sample in hist_signal:
        sample_dir = out_file.mkdir(sample)
        for tree in hist_signal[sample]:
            sample_dir.cd()
            tmp_dir = sample_dir.mkdir(tree)
            tmp_dir.cd()
            for hist in hist_signal[sample][tree].values():
                hist.Write()
    hist_signal = None
    print 'finished writing signal histograms'

    # out_file = rt.TFile.Open("./output_background_histos.root", "recreate")
    # background_array = process_the_samples(backgrounds, variables, None, None, None)
    # hist_background = get_histograms(background_array)
    # background_array = None
    # for sample in hist_background:
    #     out_file.cd()
    #     sample_dir = out_file.mkdir(sample)
    #     sample_dir.cd()
    #     for tree in hist_background[sample]:
    #         sample_dir.cd()
    #         tmp_dir = sample_dir.mkdir(tree)
    #         tmp_dir.cd()
    #         for hist in hist_background[sample][tree].values():
    #             hist.Write()
  
    out_file.Close()
    print 'finished writing'
