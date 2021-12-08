import numpy as np
from scipy.stats.distributions import chi2
import glob
import os 
import pandas as pd
import copy
import pickle
from libs.filter import Filter
from libs.est import Est
from libs.glmb import GLMB
from libs.glmb import TT
from libs.model import Model

from libs.functions_plot import *
import argparse
import json

# Opening JSON file
import json


if __name__ == "__main__":  

    parser = argparse.ArgumentParser(description='JOINT GLMB ')
    parser.add_argument('--config_demo', type=str, help='Path to config_mot json', default='config_demo.json')
    args = parser.parse_args()

    config_mot=args.config_demo
    with open(config_mot) as json_file:
        config_mot = json.load(json_file)


    path_data=config_mot['path_data']  


    ## load measurements
    print('load measurements and ground truth')
    #path_data='dataset_milestone2\PETS09-S2L1'


    ##########
    path_data_meas=os.path.join(path_data,'meas')

    string_search=path_data_meas+'\*.txt'
    #files=sorted(glob.glob(string_search))
    files=sorted(glob.glob('meas/*.txt'))


    GT_files_X=sorted(glob.glob('ground_truth/X/*.txt'))
    GT_files_track_list=sorted(glob.glob('ground_truth/track_list/*.txt'))

    gt_X=[]
    gt_track_list=[]
    truth_total_tracks=0
    for ct in range(len(GT_files_X)):
        print(GT_files_X[ct])
        gt_X_ct=np.loadtxt(GT_files_X[ct])
        gt_track_list_ct=np.loadtxt(GT_files_track_list[ct]).astype('int')-1
        gt_X.append(gt_X_ct)
        gt_track_list.append(gt_track_list_ct)
        truth_total_tracks=int(max(truth_total_tracks, np.max(gt_track_list_ct)))
        print(ct,truth_total_tracks )
    truth_total_tracks=truth_total_tracks+1
    meas=[]
    for ct in range(len(files)):
        print(files[ct])
        M=np.loadtxt(files[ct])
        meas.append(M)

    K=len(meas)
    model = Model()
    filter= Filter(model=model)

    est = pickle.load( open( "est.pickle", "rb" ) )
    plot_results(model,gt_X,gt_track_list,truth_total_tracks,meas,est)

    print('done')
        




