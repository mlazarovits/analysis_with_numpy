import ROOT as rt
from time import strftime, localtime
from collections import OrderedDict
from plotting_susy_cff import plot_configurables as pc
from plotting_susy_cff import sample_configurables as sc
import imp, os

date = strftime('%d%b%y', localtime())


rt.gROOT.SetBatch()
rt.gROOT.SetStyle('Plain')
rt.gStyle.SetOptTitle(0)
rt.gStyle.SetOptStat(0000)
rt.gStyle.SetOptFit(0111)
rt.gStyle.SetPalette(rt.kBlueRedYellow)
rt.TH1.AddDirectory(rt.kFALSE)

helper   = imp.load_source('fix'     , './help.py')
tdrstyle = imp.load_source('tdrstyle', './tdrstyle.py')
CMS_lumi = imp.load_source('CMS_lumi', './CMS_lumi.py') 

tdrstyle.setTDRStyle()

def make_me_a_canvas():
   can = rt.TCanvas('canvas', 'canvas', 800, 600)
   can.SetLeftMargin(0.15);
   can.SetRightMargin(0.18);
   can.SetBottomMargin(0.15);
   can.SetGridx();
   can.SetGridy();
   can.SetLogz();
   return can

def make_2D_plots(hists_):
    rt.gROOT.SetBatch()
    rt.gROOT.SetStyle('Plain')
    rt.gStyle.SetOptTitle(0)
    rt.gStyle.SetOptStat(0000)
    rt.gStyle.SetOptFit(0111)
    rt.gStyle.SetPalette(rt.kBlueRedYellow)
    if not (os.path.isdir('./plots_'+date)): os.mkdir('./plots_'+date)
    for sample in hists_:
        if not (os.path.isdir('./plots_'+date+'/'+sample)): os.mkdir('./plots_'+date+'/'+sample)
        for tree in hists_[sample]:
            if not (os.path.isdir('./plots_'+date+'/'+sample+'/'+tree)): os.mkdir(os.path.join('./plots_'+date, sample, tree))
            out_dir = os.path.join('./plots_'+date, sample, tree)
            for hist_name, hist in hists_[sample][tree].items():
                if not hist.InheritsFrom(rt.TH2.Class()): continue
                can = make_me_a_canvas()
                can.cd() 
                hist.RebinX(4)
                hist.RebinY(4)
                hist.Draw("COLZ");
                hist.GetXaxis().CenterTitle();
                hist.GetXaxis().SetTitleFont(42);
                hist.GetXaxis().SetTitleSize(0.06);
                hist.GetXaxis().SetTitleOffset(1.06);
                hist.GetXaxis().SetLabelFont(42);
                hist.GetXaxis().SetLabelSize(0.05);
                hist.GetXaxis().SetTitle(pc[hist_name]['xlabel']);
                hist.GetYaxis().CenterTitle();
                hist.GetYaxis().SetTitleFont(42);
                hist.GetYaxis().SetTitleSize(0.06);
                hist.GetYaxis().SetTitleOffset(1.12);
                hist.GetYaxis().SetLabelFont(42);
                hist.GetYaxis().SetLabelSize(0.05);
                hist.GetYaxis().SetTitle(pc[hist_name]['ylabel']);
                hist.GetZaxis().CenterTitle();
                hist.GetZaxis().SetTitleFont(42);
                hist.GetZaxis().SetTitleSize(0.06);
                hist.GetZaxis().SetTitleOffset(1.1);
                hist.GetZaxis().SetLabelFont(42);
                hist.GetZaxis().SetLabelSize(0.05);
                hist.GetZaxis().SetTitle("a. u.");
                if pc[hist_name]['xmax'] is not None: 
                    xmin = pc[hist_name]['xmin']
                    xmax = pc[hist_name]['xmax']
                    hist.GetXaxis().SetRangeUser(xmin, xmax) 
                if pc[hist_name]['ymax'] is not None: 
                    ymin = pc[hist_name]['ymin']
                    ymax = pc[hist_name]['ymax']
                    hist.GetYaxis().SetRangeUser(ymin, ymax) 
                if ('loose' in hist_name) or ('medium' in hist_name) or ('tight' in hist_name):
                    if 'SMS' in sample:
                        hist_tmp_name = '_'.join(hist.GetName().split('_')[1:-3])
                    else:
                        hist_tmp_name = '_'.join(hist.GetName().split('_')[1:-2])
                    hist_tmp = hists_[sample][tree][hist_tmp_name]
                    hist.GetZaxis().SetRangeUser(0.9*hist_tmp.GetMinimum(0.0),1.1*hist_tmp.GetMaximum());
                    hist_tmp = None
                else:
                    hist.GetZaxis().SetRangeUser(0.9*hist.GetMinimum(0.0),1.1*hist.GetMaximum());
                CMS_lumi.writeExtraText = 1
                CMS_lumi.extraText = 'Simulation'
                CMS_lumi.CMS_lumi(can, 0, 0)
                can.Update()
                can.SaveAs(out_dir+'/h_'+hist.GetName()+'.root')
                can.SaveAs(out_dir+'/h_'+hist.GetName()+'.pdf')

def make_overlay_plot(hists_, samples_):
    print 'for posterity'

def make_1D_plots(hists_):
    print 'some more posterity here'
    if not (os.path.isdir('./plots_'+date)): os.mkdir('./plots_'+date)
    for sample in hists_:
        if not (os.path.isdir('./plots_'+date+'/'+sample)): os.mkdir('./plots_'+date+'/'+sample)
        for tree in hists_[sample]:
            if not (os.path.isdir('./plots_'+date+'/'+sample+'/'+tree)): os.mkdir(os.path.join('./plots_'+date, sample, tree))
            out_dir = os.path.join('./plots_'+date, sample, tree)
            for hist_name, hist in hists_[sample][tree].items():
                if not hist.InheritsFrom(rt.TH1.Class()): continue                    
                can = make_me_a_canvas()
                can.cd() 
                hist.SetLineColor(sc[sample]['color'])
                hist.SetLineStyle(sc[sample]['style'])
                if sc[sample]['fill']: hist.SetFillColor(sc[sample]['fill'])
                if sc[sample]['fill_style']: hist.SetFillStyle(sc[sample]['style'])
                hist.GetXaxis().CenterTitle();
                hist.GetXaxis().SetTitleFont(42);
                hist.GetXaxis().SetTitleSize(0.06);
                hist.GetXaxis().SetTitleOffset(1.06);
                hist.GetXaxis().SetLabelFont(42);
                hist.GetXaxis().SetLabelSize(0.05);
                hist.GetXaxis().SetTitle(pc[hist_name]['xlabel']);
                hist.GetYaxis().CenterTitle();
                hist.GetYaxis().SetTitleFont(42);
                hist.GetYaxis().SetTitleSize(0.06);
                hist.GetYaxis().SetTitleOffset(1.12);
                hist.GetYaxis().SetLabelFont(42);
                hist.GetYaxis().SetLabelSize(0.05);
                hist.GetYaxis().SetTitle(pc[hist_name]['ylabel']);
                CMS_lumi.writeExtraText = 1
                CMS_lumi.extraText = 'Simulation'
                CMS_lumi.CMS_lumi(can, 0, 0)
                l = rt.TLatex()
                l.SetTextFont(42);
                l.SetNDC();
                l.SetTextSize(0.035);
                l.SetTextFont(42);
                l.DrawLatex(0.41,0.943,sc[sample]['legend']);
                can.Update()
                can.SaveAs(out_dir+'/h_'+hist.GetName()+'.root')
                can.SaveAs(out_dir+'/h_'+hist.GetName()+'.pdf')

def read_in_hists(in_file_):
    print 'look at all that posterity' 
    in_file = rt.TFile(in_file_, 'r')
    hists = OrderedDict()
    for key in in_file.GetListOfKeys():
        print key.GetName()
        key_name = key.GetName()
        sample_dir = in_file.Get(key.GetName())
        hists[key_name] = OrderedDict()
        for tree_key in sample_dir.GetListOfKeys():
            print tree_key.GetName()
            tree_name = tree_key.GetName()
            tree_dir = sample_dir.Get(tree_key.GetName())
            hists[key_name][tree_name] = OrderedDict()
            for hist_key in tree_dir.GetListOfKeys():
                print hist_key.GetName()
                hist_name = hist_key.GetName()
                hist = tree_dir.Get(hist_key.GetName())
                if '_' in tree_name:
                    hists[key_name][tree_name]['_'.join(hist_name.split('_')[:-3])] = hist
                else:
                    hists[key_name][tree_name]['_'.join(hist_name.split('_')[:-2])] = hist

    return hists 



if __name__ == "__main__":
    signal_file = './output_signal_histos.root'
    background_file = './output_background_histos.root' 

    back_hists = read_in_hists(background_file)
    make_2D_plots(back_hists)
    
    sig_hists = read_in_hists(signal_file)
    make_2D_plots(sig_hists)
