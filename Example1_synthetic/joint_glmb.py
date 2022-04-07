import numpy as np
from numpy import random
from scipy.stats.distributions import chi2
import glob
import os 
import pandas as pd
import copy
import pickle
from libs.filter import Filter
from libs.est import Est
from libs.glmb import GLMB
from libs.model import Model
from libs.glmb import TT
from libs.model import Model

from libs.functions import *
import argparse
import json
if __name__ == "__main__":  

        parser = argparse.ArgumentParser(description='JOINT GLMB ')
        parser.add_argument('--config_demo', type=str, help='Path to config_mot json', default='config_demo.json')
        args = parser.parse_args()
    
        config_mot=args.config_demo

        ## load measurements

        # Opening JSON file
        
        with open(config_mot) as json_file:
            config_mot = json.load(json_file)


        path_data=config_mot['path_data']

        print('load measurements')
        #path_data='dataset_milestone2\PETS09-S2L1'
        path_data_meas=os.path.join(path_data,'meas')
        string_search=path_data_meas+'\*.txt'
        #files=sorted(glob.glob(string_search))
        files=sorted(glob.glob('meas/*.txt'))

        meas=[]
        #K=100
        for ct in range(len(files)):
            print(files[ct])
            M=np.loadtxt(files[ct])
            meas.append(M)
        meas,meas_min,meas_max,meas_range=normalize_meas(meas)
        K=len(meas)
        model = Model()
        filter= Filter(model=model)
        est=Est(K,filter)
        est.meas_min=meas_min
        est.meas_max=meas_max
        est.meas_range=meas_range

        glmb_update=GLMB()

        write_out=True

        start_time=0



        for k in range(K):
            #joint prediction and update
            print(k)
            # GLMB copy is for thesis 
            glmb_update= jointpredictupdate(glmb_update,model,filter,meas,k)
            H_posterior= len(glmb_update.w)
            
            # #pruning and truncation
            glmb_update= prune(glmb_update,filter);                   
            H_prune= len(glmb_update.w)
            glmb_update= cap(glmb_update,filter);                     
            H_cap= len(glmb_update.w)
            
            # #state estimation 
            est= extract_estimates_recursive(glmb_update,model,meas,est)
            

        #     if write_out==True:
        #         foldername='out'+str(k)
        #         if os.path.exists(foldername)==False:
        #             os.mkdir(foldername)
        #         for ct in range(len(est.X)):
        #             if len(est.X[ct])>0:
        #                 np.savetxt(foldername+'/'+str(ct)+'.txt',est.X[ct])


        # for ct in range(len(est.X)):
        #     np.savetxt('out/'+str(ct)+'.txt',est.X[ct])




        ### save output estimation
    
        for ct in range(len(est.X)):
            Xct=est.X[ct]
            if len(Xct)>0:
                Z=Xct[[0 ,2],:]
                Zd= denormalize(Z,meas_min,meas_max,meas_range)
                Xct[[0 ,2],:] =Zd
                est.X[ct]=Xct
        with open('est.pickle', 'wb') as f:
            pickle.dump(est, f)
        print('done')
            




