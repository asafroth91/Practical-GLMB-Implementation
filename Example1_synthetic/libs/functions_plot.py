import numpy as np
import cv2
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
import matplotlib.pyplot as plt
#debug_rand=np.loadtxt('debug_rand.txt')


def extract_tracks(X,track_list,total_tracks):
    K= len(X); 
    x_dim= X[K-1].shape[0] 
    k=K-2
    while x_dim==0:
        x_dim= X[k].shape[0]
        k= k-1
    X_track= np.nan*np.ones((x_dim,K,total_tracks))#NaN(x_dim,K,total_tracks)
    k_birth= np.zeros((total_tracks,1))
    k_death= np.zeros((total_tracks,1))


    max_idx= -1

    for k in range(K):
        if len(X[k])>0:
            X_track[:,k,track_list[k].astype('int')]= X[k]
        
            if np.max(track_list[k])> max_idx: #new target born?
                idx= np.where(track_list[k]> max_idx)[0]
                k_birth[track_list[k][idx]]= k
            
            if  len(track_list[k]):
                max_idx= np.max(track_list[k])
            k_death[track_list[k]]= k

    return X_track,k_birth,k_death
def unique_rows(a):
    df = pd.DataFrame(a)
    df2=df.drop_duplicates(inplace=False )
    I=np.array(df2.index)
    C=np.array(df2)
    return C,I

def countestlabels(meas,est ):
    labelstack=np.zeros( (2,0))

    for k in range(len(meas)):
            Lk=est.L[k]
            if len(Lk)>0:
                labelstack = np.append(labelstack,Lk,axis=1)
    
    C,I=unique_rows(labelstack.transpose())
    count=C.shape[0]
    return count



def makecolorarray(nlabels):
    lower= 0.1
    upper= 0.9
    rrr= np.random.rand(1,10)*(upper-lower)+lower
    ggg= np.random.rand(1,10)*(upper-lower)+lower
    bbb= np.random.rand(1,10)*(upper-lower)+lower
    ca_rgb= np.concatenate((rrr, ggg, bbb)).transpose()
    ca_lab=[None]*nlabels
    ca_cnt= -1;   
    return ca_rgb,ca_lab,ca_cnt

def assigncolor(label,ca_rgb,ca_lab,ca_cnt):
    s= np.array2string(label)#sprintf('%i*',label)
    try:
        idx=ca_lab.index(s)
    except:
        ca_cnt=ca_cnt+1
        ca_lab[ca_cnt]=s
        
        idx=ca_cnt

    # tmp= strcmp(str,colorarray.lab)
    # if any(tmp)
    #     idx= find(tmp);
    # else
    #     colorarray.cnt= colorarray.cnt + 1;
    #     colorarray.lab{colorarray.cnt}= str;
    #     idx= colorarray.cnt;
    return idx,ca_lab,ca_cnt


def plot_results(model,truth_X,truth_track_list,truth_total_tracks,meas,est):
    
    plot_max_time=len(meas)
    plot_max_x=800
    plot_max_y=800
    

    X_track,k_birth,k_death= extract_tracks(truth_X,truth_track_list,truth_total_tracks)

    labelcount= countestlabels(meas,est)
    ca_rgb,ca_lab,ca_cnt=makecolorarray(labelcount)

    est_total_tracks= labelcount
    truth_K=len(truth_X)
    est.track_list= []
    est.total_tracks=0
    for k in range(truth_K):
        est_track_list_k= []
        if len(est.X[k])>0:
            for eidx in range (est.X[k].shape[1]):
                label=est.L[k][:,eidx]
                idx,ca_lab,ca_cnt=assigncolor(label,ca_rgb,ca_lab,ca_cnt)
                est.total_tracks=max(est.total_tracks,idx)
                est_track_list_k.append(idx)# = [est_track_list{k} assigncolor(est.L[k][:,eidx])]
            
        est.track_list.append(np.array(est_track_list_k).astype('int'))
    
    est.total_tracks=est.total_tracks+1
    [Y_track,l_birth,l_death]= extract_tracks(est.X,est.track_list,est.total_tracks)
    #plot ground truths
    #limit= [ model.range_c[0,0], model.range_c[0,1], model.range_c[1,0] ,model.range_c[1,1] ]
    #figure; truths= gcf; hold on;
    for i in range(truth_total_tracks):
        k_birth_death=np.arange(k_birth[i],k_death[i]).astype('int')
        Pt= X_track[:,k_birth_death,i] 
        Pt=Pt[[0, 2],:]
        plt.plot(Pt[0,:],Pt[1,:])
        plt.plot( Pt[0,0], Pt[1,0], 'ko')
        plt.suptitle('Ground Truths')
        plt.xlabel('X')
        plt.ylabel('Y')

    plt.xlim([-800, plot_max_x])   
    plt.ylim([-1000, plot_max_y]) 
    plt.show()
       
    



    ##plot tracks and measurments in x/y
    
    ##%plot x measurement
    
    plt.figure(figsize=(18, 6))

    plt.subplot(131)
    for k in range(len(meas)):
        if len(meas[k])>0:
              x_line=k*np.ones( (meas[k].shape[1],1))
              y_line=meas[k][0,:]
              plt.plot(x_line, y_line, 'kx')
              plt.ylabel('X')
              plt.title('MEASURES')
              plt.xlabel('time')
              plt.xlim([0, plot_max_time])   
              plt.ylim([-plot_max_x, plot_max_x])     
          #  hlined= line(k*ones(size(meas.Z{k},2),1),meas.Z{k}(1,:),'LineStyle','none','Marker','x','Markersize',5,'Color',0.7*ones(1,3));
    
    

    ##%plot x track
    plt.subplot(132)
    for i in range(truth_total_tracks):
        k_birth_death=np.arange(k_birth[i],k_death[i]).astype('int')
        Px= X_track[:,k_birth_death,i]
        Px=Px[[0, 2],:]
        x_line=np.arange(k_birth[i],k_death[i])
        y_line=Px[0,:]
        plt.plot(x_line, y_line)
        plt.title('GROUND TRUTH')
        plt.ylabel('X')
        plt.xlabel('time')
        #hline1= line(,,'LineStyle','-','Marker','none','LineWidth',1,'Color',0*ones(1,3));
    
        plt.xlim([0, plot_max_time])   
        plt.ylim([-plot_max_x, plot_max_x])     
    #%plot x estimate
    plt.subplot(133)
    for t in range(Y_track.shape[2]):
        x_line=np.arange(0,truth_K)
        y_line=Y_track[0,:,t]
        plt.plot(x_line, y_line)
        #hline2= line(1:truth.K,Y_track(1,:,t),'LineStyle','none','Marker','.','Markersize',8,'Color',colorarray.rgb(t,:));
        plt.ylabel('X')
        plt.title('ESTIMATE')
        plt.xlabel('time')
        plt.xlim([0, plot_max_time])   
        plt.ylim([-plot_max_x, plot_max_x])     

    plt.suptitle('X ')
    plt.show() 
   ## %plot y measurement
  
    plt.figure(figsize=(18, 6))

    plt.subplot(131)
    for k in range(len(meas)):
        if len(meas[k])>0:
            x_line=k*np.ones( (meas[k].shape[1],1))
            y_line=meas[k][1,:]
            plt.plot(x_line, y_line, 'kx')
            plt.ylabel(' Y')
            plt.title('MEASURES')
            plt.xlabel('time')
            plt.xlim([0, plot_max_time])   
            plt.ylim([-plot_max_y, plot_max_y])     
            #yhlined= line(k*ones(size(meas.Z{k},2),1),meas.Z{k}(2,:),'LineStyle','none','Marker','x','Markersize',5,'Color',0.7*ones(1,3));
        
    

    #%plot y track
    plt.subplot(132)

    for i in range(truth_total_tracks):
            k_birth_death=np.arange(k_birth[i],k_death[i]).astype('int')
            Py= X_track[:,k_birth_death,i]
            Py=Py[[0 ,2],:]
            x_line=np.arange(k_birth[i],k_death[i])
            y_line=Py[1,:]
            #yhline1= line(k_birth(i):1:k_death(i),Py(2,:),'LineStyle','-','Marker','none','LineWidth',1,'Color',0*ones(1,3));
            plt.plot(x_line, y_line)
            plt.ylabel('Y')
            plt.title('GROUND TRUTH')
            plt.xlabel('time')
            plt.xlim([0, plot_max_time])   
            plt.ylim([-plot_max_y, plot_max_y])   

    #%plot y estimate
    plt.subplot(133)

    for t in range(Y_track.shape[2]):
        x_line=np.arange(0,truth_K)
        y_line=Y_track[2,:,t]
        plt.title('ESTIMATE')
        plt.plot(x_line, y_line)
        plt.ylabel('estimate Y')
        plt.xlabel('time')
        plt.xlim([0, plot_max_time])   
        plt.ylim([-plot_max_y, plot_max_y])     
    plt.suptitle('Y ')
    plt.show() 
        
    plt.figure(figsize=(18, 6))
    #plot cardinality
    truth_N=[]
    for i in range(len(truth_track_list)):
        truth_N.append(len(truth_track_list[i]))
    
    est_N=[]
    for i in range(len(est.track_list)):
        est_N.append(len(est.track_list[i]))

    x_step=np.arange(len(meas))
    plt.step(x_step,truth_N,'o',label='ground truth')
    plt.step(x_step,est_N,'-r',label='estimated')
    plt.title('CARDINALITY')
    plt.legend()
    plt.show()
    