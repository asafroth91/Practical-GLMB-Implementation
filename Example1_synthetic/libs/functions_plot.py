from datetime import time
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
    colors = ['#7e1e9c','#15b01a','#0343df','#ff81c0','#e50000','#f97306','#13eac9','#650021','#6e750e','#910951','#ad900d','#ff0000','#33ff99','#ff6666','#ff66b2','#009999']

    for i in range(truth_total_tracks):
        k_birth_death=np.arange(k_birth[i],k_death[i]).astype('int')
        Pt= X_track[:,k_birth_death,i] 
        Pt=Pt[[0, 2],:]
        plt.plot(Pt[0,:],Pt[1,:])
        plt.plot( Pt[0,0], Pt[1,0], 'k.')
        plt.suptitle('Ground Truth Tracks')
        plt.xlabel('X')
        plt.ylabel('Y')

    plt.xlim([-1000, 1000])   
    plt.ylim([-1000, 1000]) 
    plt.show()
       
    



    ##plot tracks and measurments in x/y
    
    ##%plot x measurement
    
    plt.figure()
    for k in range(len(meas)):
        if len(meas[k])>0:
              x_line=k*np.ones( (meas[k].shape[1],1))
              y_line=meas[k][0,:]
              plt.plot(x_line, y_line, 'kx')
              plt.ylabel('X')
              plt.title('X Measurements')
              plt.xlabel('time')
              plt.xlim([0, plot_max_time])   
              plt.ylim([-1000, 1000])     
          #  hlined= line(k*ones(size(meas.Z{k},2),1),meas.Z{k}(1,:),'LineStyle','none','Marker','x','Markersize',5,'Color',0.7*ones(1,3));
    plt.show()
    

    ##%plot x track
    plt.figure()
    for i in range(truth_total_tracks):
        k_birth_death=np.arange(k_birth[i],k_death[i]).astype('int')
        Px= X_track[:,k_birth_death,i]
        Px=Px[[0, 2],:]
        x_line=np.arange(k_birth[i],k_death[i])
        y_line=Px[0,:]
        plt.plot(x_line, y_line)
        plt.title('Ground Truth Tracks X')
        plt.ylabel('X')
        plt.xlabel('time')
        #hline1= line(,,'LineStyle','-','Marker','none','LineWidth',1,'Color',0*ones(1,3));
    
        plt.xlim([0, plot_max_time])   
        plt.ylim([-1000, 1000])       
    plt.show()   
    #%plot x estimate
    plt.figure()
    for t in range(Y_track.shape[2]):
        x_line=np.arange(0,truth_K)
        y_line=Y_track[0,:,t]
        plt.plot(x_line, y_line)
        #hline2= line(1:truth.K,Y_track(1,:,t),'LineStyle','none','Marker','.','Markersize',8,'Color',colorarray.rgb(t,:));
        plt.ylabel('X')
        plt.title('estimated X position')
        plt.xlabel('time')
        plt.xlim([0, plot_max_time])   
        plt.ylim([-1000, 1000])          
    plt.show() 
   ## %plot y measurement

   ## plot a subplot of ground truth x y versus time 
    plt.figure()
    for k in range(len(meas)):
        if len(meas[k])>0:
            x_line=k*np.ones( (meas[k].shape[1],1))
            y_line=meas[k][0,:]
            plt.subplot(121)
            plt.plot(x_line, y_line, 'kx')
            plt.ylabel('X')
            plt.title('X Measurements')
            plt.xlabel('Time')
            plt.xlim([0, plot_max_time])   
            plt.ylim([-1000, 1000])  
            x_line=k*np.ones( (meas[k].shape[1],1))
            y_line=meas[k][1,:]
            plt.subplot(122)
            plt.plot(x_line, y_line, 'kx')
            plt.ylabel(' Y')
            plt.title('Y Measurements')
            plt.xlabel('Time')
            plt.xlim([0, plot_max_time])   
            plt.ylim([-1000, 1000]) 
    plt.tight_layout()
    plt.show()

    ## plot a subplot of measurements x,y versues time 
    plt.figure()
    for i in range(truth_total_tracks):
        c = colors[i]
        k_birth_death=np.arange(k_birth[i],k_death[i]).astype('int')
        Px= X_track[:,k_birth_death,i]
        Px=Px[[0, 2],:]
        x_line=np.arange(k_birth[i],k_death[i])
        y_line=Px[0,:]
        plt.subplot(121)
        plt.plot(x_line, y_line)
        plt.title('Ground Truth Tracks X')
        plt.ylabel('X')
        plt.xlabel('Time')
        plt.xlim([0, plot_max_time])   
        plt.ylim([-1000, 1000])
        Py= X_track[:,k_birth_death,i]
        Py=Py[[0 ,2],:]
        x_line=np.arange(k_birth[i],k_death[i])
        y_line=Py[1,:]
        plt.subplot(122)
        plt.plot(x_line, y_line)
        plt.ylabel('Y')
        plt.title('Ground Truth Tracks Y')
        plt.xlabel('Time')
        plt.xlim([0, plot_max_time])   
        plt.ylim([-1000, 1000])
    plt.tight_layout()
    plt.show()

  
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
    plt.subplot(121)
    time_plot = 15
    handles=[]
    for t in range(Y_track.shape[2]):
        
        x_line=np.arange(0,time_plot)
        y_liney=Y_track[2,:time_plot,t]
        y_linex=Y_track[0,:time_plot,t]

        plt.subplot(121)
        plt.title('X Estimate & Ground Truth Vs. Time')
        if (t != 3 and t !=4):
            plt.plot(x_line, y_linex,'.')
        else:
            plt.plot(x_line, y_linex,'.r')
            plt.plot(x_line, y_linex,'Xk')
        plt.ylabel('X',labelpad=-15)
        plt.xlabel('Time')
        plt.xlim([0, time_plot+1])   
        plt.ylim([-1000, 1000])
        plt.xticks(np.arange(0,time_plot,2))

        plt.subplot(122) 
        plt.title('Y Estimate & Ground Truth Vs. Time')

        if (t != 3 and t !=4 and t <4):
            label_str = r'$Track\;\:\ell_'+str(t)+'$'
            track,=plt.plot(x_line, y_liney,'.',label=label_str)
            handles.append(track)

        elif(t==4 or t==3):
            plt.plot(x_line, y_liney,'.r')
            missed,=plt.plot(x_line, y_liney,'Xk',label='False Target')
            if (t==3):
                handles.append(missed)

        plt.xlabel('Time')
        plt.xlim([0, time_plot+1])   
        plt.ylim([-1000, 1000])
        plt.xticks(np.arange(0,time_plot,2))
        plt.yticks([])
        plt.ylabel('Y')
        plt.legend(handles=handles)
    for i in range(truth_total_tracks):
            k_birth_death=np.arange(k_birth[i],k_death[i]).astype('int')
            Py= X_track[:,k_birth_death,i]
            Py=Py[[0 ,2],:]
            x_line=np.arange(k_birth[i],k_death[i])
            y_liney=Py[1,:]
            y_linex=Py[0,:]
            plt.subplot(121)
            plt.plot(x_line[:time_plot], y_linex[:time_plot],':k')  
            plt.subplot(122)
            ground_truth,=plt.plot(x_line[:time_plot], y_liney[:time_plot],':k',label='Ground Truth')
            if (i==1):
                handles.append(ground_truth)
            plt.legend(handles=handles)

    plt.tight_layout()
    plt.show() 
        
    # plt.figure(figsize=(10, 10))
    for i in range(truth_total_tracks):
        k_birth_death=np.arange(k_birth[i],k_death[i]).astype('int')
        Pt= X_track[:,k_birth_death,i] 
        Pt=Pt[[0, 2],:]
        line,=plt.plot(Pt[0,:],Pt[1,:],'k')
        plt.plot( Pt[0,0], Pt[1,0], 'ko')
        plt.suptitle('Estimate Vs. Ground Truth ')
        plt.xlabel('X')
        plt.ylabel('Y')
    for t in range(Y_track.shape[2]):
        x_line=np.arange(0,truth_K)
        y_line=Y_track[2,:,t]
        x_line=Y_track[0,:,t]
        plt.plot(x_line, y_line,c=colors[t])
    plt.xlim([-1000, 1000])   
    plt.ylim([-1000, 1000]) 
    plt.show()

    plt.figure(figsize=(12,4))
    #plot cardinality
    truth_N=[]
    for i in range(len(truth_track_list)):
        truth_N.append(len(truth_track_list[i]))
    
    est_N=[]
    for i in range(len(est.track_list)):
        est_N.append(len(est.track_list[i]))

    x_step=np.arange(len(meas))
    plt.step(x_step,truth_N,'-k',label='Ground Truth')
    plt.step(x_step,est_N,'.r',label='Estimated')
    plt.title('Cardinality')
    plt.xlabel('Time')
    plt.ylabel('Cardinality')
    plt.xticks(np.arange(0,105,5))
    plt.legend()
    plt.tight_layout()
    plt.show()

    # ### COMPARE MHT AND GLMB ####
    plt.figure(figsize=(18, 6))
    with open('MHT_est_tracks.pickle', 'rb') as f:
        Y_MHT_tracks=pickle.load(f)
    for i in range(truth_total_tracks):
        k_birth_death=np.arange(k_birth[i],k_death[i]).astype('int')
        Pt= X_track[:,k_birth_death,i] 
        Pt=Pt[[0, 2],:]
        line,=plt.plot(Pt[0,:],Pt[1,:],'k')
        plt.plot( Pt[0,0], Pt[1,0], 'ko')
        plt.suptitle(r'$\delta-GLMB\:vs.\:MHT\:Estimates\:vs.\:Ground\:Truth$')
        plt.xlabel('X')
        plt.ylabel('Y')
    line,=plt.plot(Pt[0,:],Pt[1,:],'k',label=r'$Ground\:truth$')
    for t in range(Y_track.shape[2]):
        y_line=Y_track[2,:,t]
        x_line=Y_track[0,:,t]
        plt.plot(x_line, y_line,'or',mfc='none')
    glmb,=plt.plot(x_line, y_line,'or',mfc='none',label=r'$\delta-GLMB$')
    for t in range(Y_MHT_tracks.shape[2]):
        y_lineMHT=Y_MHT_tracks[2,:,t]
        x_lineMHT=Y_MHT_tracks[0,:,t]
        plt.plot(x_lineMHT, y_lineMHT,'.g')
    mht,=plt.plot(x_lineMHT, y_lineMHT,'.g',label=r'$MHT$')
    handles=[line,glmb,mht]
    plt.xlim([-1000, 1000])   
    plt.ylim([-1000, 1000]) 
    plt.legend(handles=handles)
    plt.show()

    # PLOT cardinalites of MHT and GLMB 
    data_plot_mht_fig3 = pickle.load( open( "data_plot_mht_fig3.pickle", "rb" ) )
    data_plot_glmb_fig3 = pickle.load( open( "data_plot_glmb_fig3.pickle", "rb" ) )
    plt.figure(figsize=(12,4))
    truth_N=[]
    for i in range(len(truth_track_list)):
        truth_N.append(len(truth_track_list[i]))
    
    est_N=[]
    for i in range(len(est.track_list)):
        est_N.append(len(est.track_list[i]))

    x_step=np.arange(len(meas))
    plt.step(x_step,truth_N,'-k',label='Ground Truth')
    plt.step(x_step,est_N,'.r',label='Estimated')
    plt.step(data_plot_glmb_fig3['x_step'],data_plot_mht_fig3['est_N'],'og',label='MHT',mfc='none')
    plt.title('Cardinality')
    plt.xlabel('Time')
    plt.ylabel('Cardinality')
    plt.xticks(np.arange(0,105,5))
    plt.legend()
    plt.tight_layout()
    plt.show()


def error_ellipse(P,mu):
    eigenvals, eigenvecs = np.linalg.eig(P)
    largest_eigvec_ind_c = np.where(eigenvals==max(eigenvals))[0]
    largest_eigenvec = eigenvecs[largest_eigvec_ind_c].T
    
    # get the largest eigenvalue
    largest_eigenval = max(eigenvals)
    
    # Get the smallest eigenvector and eigenvalue
    if (largest_eigvec_ind_c[0]==0 and largest_eigvec_ind_c.shape[0]==1):
        smallest_eigenval = max(eigenvals[:,2])
        smallest_eigenvec = eigenvecs[:,2]
    else:
        smallest_eigenval = max(eigenvals) # change here 
        smallest_eigenvec = eigenvecs[1,:]
        
    # Calculate the angle between the x-axis and the largest eigenvector
    angle = np.arctan2(largest_eigenvec[1,0], largest_eigenvec[0,0])
    
    # This angle is between -pi and pi.
    # Let's shift it such that the angle is between 0 and 2pi
    if (angle < 0):
        angle = angle + 2*np.pi;
    avg = mu
    
    # Get the 95% confidence interval error ellipse
    # chisquare_val = 2.4477
    chisquare_val = 5.959
    theta_grid = np.linspace(0,2*np.pi,100)
    phi = angle
    X0=avg[0]
    Y0=avg[1]
    a=chisquare_val*np.sqrt(largest_eigenval)
    b=chisquare_val*np.sqrt(smallest_eigenval)
    
    # the ellipse in x and y coordinates 
    ellipse_x_r  = a*np.cos( theta_grid )
    ellipse_y_r  = b*np.sin( theta_grid )
    
    #Define a rotation matrix
    R = np.array([ [np.cos(phi), np.sin(phi)], [-np.sin(phi), np.cos(phi)] ])
    
    # let's rotate the ellipse to some angle phi
    r_ellipse = np.dot(np.array([ellipse_x_r,ellipse_y_r]).T,R) # check this 
    out = np.array([r_ellipse[:,0]+X0,r_ellipse[:,1]+Y0]).T
    return out
    