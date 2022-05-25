import numpy as np
from scipy.stats.distributions import chi2
import glob
import os 
import pandas as pd
import copy
import pickle
from Glmb.filter import Filter
from Glmb.est import Est
from Glmb.glmb import GLMB, labeledTarget
from Glmb.model import Model
#debug_rand=np.loadtxt('debug_rand.txt')


def gate_meas_gms_idx(z,gamma,model,m,P) : #ok
    valid_idx = []
    zlength = z.shape[1]
    if zlength==0:
        z_gate= [] 
        return
    plength = m.shape[1]

    for j in range(plength):
        if plength<2:
            Pj=P
        else:
            Pj=P[:,:,j]
        HP=np.matmul( model.H,Pj)
        HPH=np.matmul( HP,model.H.transpose())
        Sj= model.R + HPH
        Vs= np.linalg.cholesky(Sj)
        det_Sj=np.power( np.prod(np.diag(Vs)),2) 
        inv_sqrt_Sj= np.linalg.inv(Vs)
        iSj= np.matmul(inv_sqrt_Sj,inv_sqrt_Sj.transpose())
        tmp=np.repeat(m[:,j].reshape(-1,1),zlength,1)
        #tmp=np.matlib.repmat(m[:,j].reshape(-1,1),1,zlength)
        nu= z- np.matmul(model.H,tmp)
        tmp=np.matmul(inv_sqrt_Sj.transpose(), nu)
        tmp=np.power(tmp,2)
        
        dist= np.sum(tmp,0)
        idx=np.where(dist<gamma)[0]
        tmp=np.array(valid_idx+idx.tolist())
        valid_idx= unique_faster(tmp)
    
        valid_idx=valid_idx.reshape(-1,1)
    return valid_idx

def sort_glmb(glmb_in):
    id_sort=np.argsort(glmb_in.w.squeeze())
    glmb_out=GLMB()
    glmb_out.set_tt(glmb_in.get_tt())
    glmb_out.set_cdn(glmb_in.get_cdn())
    glmb_out.set_w(glmb_in.w[id_sort])
    glmb_out.set_n(glmb_in.n[id_sort])
    I_out=[]
    for ct in range(len(id_sort)):
        i=id_sort[ct]
        Ict=glmb_in.I[i]
        I_out.append(Ict.copy())
    glmb_out.set_I(I_out)
    return glmb_out

        

def unique_faster(keys):#ok
    keys=np.sort(keys)
    tmp=np.array(keys.tolist()+[np.nan])
    difference=np.diff(tmp)
    keys=keys[difference!=0]
    return keys

def sum_sub2ind(X,rows,cols):
    s=0.0
    for ct in range(len(rows)):
        ith_cost = X[rows[ct],cols[ct]]
        s=s+ith_cost
    return s

def unique_rows(a):
    df = pd.DataFrame(a)
    df2=df.drop_duplicates(inplace=False )
    I=np.array(df2.index)
    C=np.array(df2)
    return C,I




def mbestwrap_updt_gibbsamp(P0,T): #ok
    # assignments=0
    # costs=0
    # m = T from algorithm 2 
    P = P0.shape[0] 
    n2 = P0.shape[1] # 

    if T<=1:
        assignments= np.zeros((1,P))
        costs= np.array([0.0])
    else:
        assignments= np.zeros((int(T),P))
        costs= np.zeros((1,int(T))).squeeze()
    
    currsoln= np.arange(P,2*P) #use all missed detections as initial solution
    assignments[0,:]= currsoln
    rows=np.arange(0,P)
    cols=currsoln
    costs[0]=sum_sub2ind(P0,rows,cols)
    #debug_rand=[]
    #debug_rand=np.loadtxt('debug_rand.txt')
    ct_rand=0
    # GIBBS ROUTINE
    for sol in range(1,int(T)):
        for n in range(P): 
            ## currsoln -> gamma_n
            tempsamp= np.exp(-P0[n,:]); #grab row of costs for current association variable
            i1=np.arange(0,n)
            i2=np.arange(n+1,len(currsoln))
            i=np.concatenate((i1,i2)) # this index is used to set all corresponding currsoln to 0, except for "n"
            tempsamp[currsoln[i]]= 0 # keeps n from currsoln (from tempsamp) and sets the other solutions in cursoln to 0 -> locks them out
            idxold= np.where(tempsamp>0)[0]
            tempsamp= tempsamp[idxold]
            bins=np.concatenate(([0],np.cumsum(tempsamp)/np.sum(tempsamp),[1+1e-6]))
            rand_num=np.random.rand()
            #debug_rand.append(rand_num)
            #rand_num=debug_rand[ct_rand]
            ct_rand=ct_rand+1

            
            hist=np.histogram(rand_num,bins)[0]
            ind=np.where(hist>0)[0]
            # we randomly choose a value from gamma, excluding gamma_n (i.e excluding currsoln)
            
            #currsoln[n]= 
            currsoln[n]= idxold[ind]
        
        assignments[sol,:]= currsoln
        rows=np.arange(0,P)
        cols=currsoln
        costs[sol]=sum_sub2ind(P0,rows,cols) # convert subscirpt indicies (row,col) to linear index and sum, see under equation 24
        ## this just sums the associated costs from the cost matrix
        
    
    C,I= unique_rows(assignments)
    assignments= C
    costs= costs[I]

    
    return assignments,costs 

def logsumexp(w):
    #performs log-sum-exp trick to avoid numerical underflow
    #input:  w weight vector assumed already log transformed
    #output: log(sum(exp(w)))

    val = np.max(w)
    logsum = np.log(np.sum(np.exp(w-val))) + val
    return logsum

def  kalman_predict_multiple(model,m,P):
    

    plength= m.shape[1]

    m_predict = np.zeros_like(m)
    P_predict = np.zeros_like(P)

    for idxp in range(plength):
        if plength<2:
            m_idxp=m
            P_idxp=P
        else:
            m_idxp=m[:,idxp]
            P_idxp=P[:,:,idxp]

        m_temp,P_temp = kalman_predict_single(model.F,model.Q,m_idxp,P_idxp)
        if plength<2:
            m_predict = m_temp
            P_predict = P_temp
        else:
            m_predict[:,idxp] = m_temp
            P_predict[:,:,idxp] = P_temp

    return m_predict,P_predict

def kalman_predict_single(F,Q,m,P):

    m_predict = np.matmul(F,m)
    FP=np.matmul(F,P)
    FPF=np.matmul(FP, F.transpose())
    P_predict = Q+FPF 

    return m_predict,P_predict

def kalman_update_multiple(z,model,m,P): #ok

        plength= m.shape[1]
        zlength= z.shape[1]

        qz_update= np.zeros((plength,zlength))
        m_update = np.zeros((model.x_dim,plength,zlength))
        P_update = np.zeros((model.x_dim,model.x_dim,plength))
        for idxp in range(plength):
            if plength>1:
                Pid=P[:,:,idxp]
                mid=m[:,idxp]
            else:
                Pid=P
                mid=m

            qz_temp,m_temp,P_temp= kalman_update_single(z,model.H,model.R,mid,Pid)

            if plength<2:
                qz_update  = qz_temp
                m_update = m_temp
                P_update = P_temp
            else:
                qz_update[idxp,:]   = qz_temp
                m_update[:,idxp,:] = m_temp
                P_update[:,:,idxp] = P_temp

        return qz_update,m_update,P_update

def kalman_update_single(z,H,R,m,P): #ok

    mu = np.matmul(H,m)
    HP=np.matmul(H,P)
    HPH=np.matmul(HP,H.transpose())
    S  = R+HPH
    Vs= np.linalg.cholesky(S) 
    det_S= np.prod(np.diag(Vs))
    det_S=np.power(det_S,2)
     
    inv_sqrt_S= np.linalg.inv(Vs)
    iS= np.matmul(inv_sqrt_S,inv_sqrt_S.transpose())
    PH=np.matmul(P,H.transpose() )
    K  = np.matmul(PH,iS)
    
    tmp1=   0.5*z.shape[0]*np.log(2*np.pi) #0.5*size(z,1)*log(2*pi)
    tmp2=0.5*np.log(det_S) 
    tmp3=np.repeat(mu,z.shape[1],1)  #repmat(mu,[1 size(z,2)])
    tmp4=z-tmp3
    tmp5=np.matmul(iS,tmp4)
    tmp6=0.5*np.dot(tmp4.squeeze(),tmp5.squeeze())
    tmp7=-tmp1-tmp2-tmp6
    qz_temp = np.exp(tmp7)
    #qz_temp = exp(-0.5*size(z,1)*log(2*pi) - 0.5*log(det_S) - 0.5*dot(z-repmat(mu,[1 size(z,2)]),iS*(z-repmat(mu,[1 size(z,2)]))))';
    tmp8=np.repeat(m,z.shape[1],1)  #repmat(m,[1 size(z,2)])
    tmp9=np.matmul(K,tmp4)
    m_temp =  tmp8 + tmp9 # repmat(m,[1 size(z,2)]) + K*(z-repmat(mu,[1 size(z,2)]))

    tmp10=np.eye(P.shape[0],P.shape[1])-np.matmul(K,H)
    P_temp = np.matmul(tmp10,P) #(eye(size(P))-K*H)*P

    return qz_temp,m_temp,P_temp

def insert_list(x,val,idx):
    if len(x)>idx:
        x[idx]=val
    else:
        if len(x)==idx:
            x.append(val)
        else:
            ndiff=idx-len(x)
            for ct in range(ndiff+1):
                x.append([])
            x[idx]=val
    return x


def clean_predict(glmb_raw):
    glmb_temp=GLMB()
    glmb_raw.hash=[]
    #hash label sets, find unique ones, merge all duplicates
    for hidx in range (len(glmb_raw.w)):
        s=str(np.sort(glmb_raw.I[hidx]))
        #s=(np.sort(glmb_raw.I[hidx]))
        glmb_raw.hash.append(s)

    df = pd.DataFrame(glmb_raw.hash)
    df2=df.drop_duplicates(inplace=False )
    ia=np.array(df2.index)
    cu=[]
    for ct in range(len(ia)):
        cu.append(glmb_raw.hash[ia[ct]])
    ic=[]
    for ct in range(len(glmb_raw.hash)):
        i=cu.index(glmb_raw.hash[ct])
        ic.append(i)

    ic=np.array(ic)



    glmb_temp.trackTable= glmb_raw.trackTable.copy()
    glmb_temp.w= np.zeros((len(ic),1))
    glmb_temp.I= []#cell(length(cu),1);
    for ct in range(len(ic)):
        glmb_temp.I.append([])
    glmb_temp.n= np.zeros((len(ic),1))
    for hidx in range( len(ic)):
            glmb_temp.w[ic[hidx]]= glmb_temp.w[ic[hidx]]+glmb_raw.w[hidx]
            glmb_temp.I[ic[hidx]]= glmb_raw.I[hidx].astype('int')
            glmb_temp.n[ic[hidx]]= glmb_raw.n[hidx]
    
    glmb_temp.cdn= glmb_raw.cdn
    
    
    #glmb_temp=0
    return glmb_temp

def clean_update(glmb_temp):
    glmb_clean=GLMB()
    #flag used tracks
    usedindicator=np.zeros((len(glmb_temp.trackTable),1))# np.zeros((len(glmb_temp.trackTable),1))
    for hidx in range(len(glmb_temp.w)):
        usedindicator[glmb_temp.I[hidx]]= usedindicator[glmb_temp.I[hidx]]+1
    
    id_trackcount=np.where(usedindicator>0)[0]

    #trackcount= np.sum(usedindicator>0)
    trackcount=len(id_trackcount)
    #remove unused tracks and reindex existing hypotheses/components
    newindices= np.zeros((len(glmb_temp.trackTable),1))

    #newindices[usedindicator>0]= np.arange(0,trackcount)
    for ct in range(trackcount):
            newindices[id_trackcount[ct]]=ct
            glmb_clean.trackTable.append(glmb_temp.trackTable[id_trackcount[ct]])
    #glmb_clean.trackTable= glmb_temp.trackTable[usedindicator>=0]
    glmb_clean.w= glmb_temp.w
    for hidx in range(len(glmb_temp.w)):
        if hidx==0:

            glmb_clean.I[0]= newindices[glmb_temp.I[hidx]]
        
        else:

            glmb_clean.I.append(newindices[glmb_temp.I[hidx]])
    
    glmb_clean.n= glmb_temp.n
    glmb_clean.cdn= glmb_temp.cdn
    
    return glmb_clean  

def prune(glmb_in,filter):
    #prune components with weights lower than specified threshold
    hypoPersist= np.where(glmb_in.w > filter.hyp_threshold)[0]
    glmb_out=GLMB()
    glmb_out.set_tt(glmb_in.trackTable)
    glmb_out.set_w(glmb_in.w[hypoPersist])
    Iout=[]
    for ct in range(len(hypoPersist)):
        Iout.append(glmb_in.I[hypoPersist[ct]])
    glmb_out.set_I(Iout)
    glmb_out.set_n(glmb_in.n[hypoPersist])

    glmb_out.w= glmb_out.w/np.sum(glmb_out.w)
    # for card in range(int(np.max(glmb_out.n))):
    #     glmb_out.cdn[card+1]= np.sum(glmb_out.w[glmb_out.n==card])


    for card in range(int(np.max(glmb_out.n)+1)):
        cdn=np.sum(glmb_out.w[glmb_out.n==card])
        if len(glmb_out.cdn)>card:
            glmb_out.cdn[card]= cdn
        else:
            glmb_out.cdn =  np.concatenate((glmb_out.cdn,[cdn])) #glmb_nextupdate.cdn.append(up_cdn)
   
    
    return glmb_out

def cap(glmb_in,filter):
    #cap total number of components to specified maximum
    if len(glmb_in.w) > filter.H_max:
        glmb_out=GLMB()
        idxsort= np.argsort(glmb_in.w)[::-1]
        hypoPersist=idxsort[1:filter.H_max]
        glmb_out.trackTable= glmb_in.trackTable
        glmb_out.w= glmb_in.w[hypoPersist]
        glmb_out.I= glmb_in.I[hypoPersist]
        glmb_out.n= glmb_in.n[hypoPersist]
        
        glmb_out.w= glmb_out.w/np.concatenatesum(glmb_out.w)
        for card in range(np.max(glmb_out.n)):
            glmb_out.cdn[card+1]= np.sum(glmb_out.w[glmb_out.n==card])
        
    else:
        glmb_out= glmb_in

    return glmb_out

def intersect(list1,list2):
    intersection_set = set.intersection(set(list1), set(list2))
    
    intersection_list = list(intersection_set)
    i1=[]
    i2=[]
    for ct in range(len(intersection_list)):
        el=intersection_list[ct]
        index1 = list1.index(el)
        index2 = list2.index(el)
        i1.append(index1)
        i2.append(index2)

    return intersection_list,i1,i2

def setxor(list1,list2):

    
    xor_set=set(list1) ^ set(list2)
    xor_list = list(xor_set)
    i1=[]
    i2=[]
    for ct in range(len(xor_list)):
        el=xor_list[ct]
        if el in list1:
            index1 = list1.index(el)
            i1.append(index1)
        else:
            index2 = list2.index(el)
            i2.append(index2)

    return xor_list,i1,i2

def extract_estimates_recursive(glmb,model,meas,est):

#extract estimates via recursive estimator, where  
#trajectories are extracted via association history, and
#track continuity is guaranteed with a non-trivial estimator

#extract MAP cardinality and corresponding highest weighted component
    mode = np.argmax(glmb.cdn); 
    M = mode
    T= []#cell(M,1);
    J= np.zeros((2,M)).astype(int)

    idxcmp= np.argmax(glmb.w*(glmb.n==M))
    #idxcmp -> idx of highest weighted
    for m in range(M):
        idxptr= int(glmb.I[idxcmp][m])
        T.append (glmb.trackTable[idxptr].ah)
        J[:,m]= glmb.trackTable[idxptr].l.squeeze()
    

    H= []#cell(M,1);
    for m in range(M):
        H.append ( str(J[0,m])+'.'+str(J[1,m]) ) 
    

    #compute dead & updated & new tracks
    int_list,IO,IS= intersect(est.H,H)
    xor_list,ID,IN= setxor(est.H,H)

    IO=np.sort(IO)
    IS=np.sort(IS)
    ID=np.sort(ID)
    IN=np.sort(IN)


    est.M= M
    Tid= [est.T[index] for index in ID]
    Tis= [T[index] for index in IS]
    Tin= [T[index] for index in IN]

    Hid= [est.H[index] for index in ID]
    His= [H[index] for index in IS]
    Hin= [H[index] for index in IN]

    Jid=np.zeros((2,0)).astype(int)
    Jis=np.zeros((2,0)).astype(int)
    Jin=np.zeros((2,0)).astype(int)

    if len(ID)>0:
        Jid=est.J[:,ID]
    if len(IS)>0:
        Jis=J[:,IS]
    if len(IN)>0:
        Jin=J[:,IN]

    est.T= Tid+Tis+Tin
    est.J= np.concatenate((Jid,Jis,Jin),1)
    est.H= Hid+His+Hin 

    #write out estimates in standard format
    meas_K=len(meas)
    est.N= np.zeros((meas_K,1))
    est.X= []#cell(meas.K,1)
    est.L= []#cell(meas.K,1)
    for ct in range(meas_K):
        est.X.append([])
        est.L.append([])

    for t in range( len(est.T)):
        ks= est.J[0,t]
        bidx= est.J[1,t]
        tah= est.T[t]
        
        w= model.w_birth[bidx]
        m= model.m_birth[bidx]
        P= model.P_birth[bidx]
        for u in range (len(tah)):
            m,P = kalman_predict_multiple(model,m,P)
            k= ks+u
            emm= tah[u]
            if emm >= 0:
                qz,m,P = kalman_update_multiple(meas[k][:,emm].reshape(-1,1),model,m,P)
                w= qz*w+np.finfo(float).eps
                w= w/np.sum(w)
            

            idxtrk= np.argmax(w)
            est.N[k]= est.N[k]+1
            if len(est.X[k])==0:
                est.X[k]=m[:,idxtrk].reshape(-1,1)
            else:
                est.X[k]= np.concatenate((est.X[k],m[:,idxtrk].reshape(-1,1)),1)
            
            if len(est.L[k])==0:
                est.L[k]= est.J[:,t].reshape(-1,1)
            else:    
                est.L[k]= np.concatenate((est.L[k],est.J[:,t].reshape(-1,1)),1)
        
    
    return est

def jointpredictupdate(glmb_update,model,filter,meas,k):
    #---generate next update
    #create birth tracks
    tt_birth= []#cell(length(model.r_birth),1); 
    for tabidx in range( len(model.r_birth)):
    
        tt_birth_new=labeledTarget(m=model.m_birth[tabidx],P=model.P_birth[tabidx],w= model.w_birth[tabidx],l=np.array([k,tabidx]).reshape(-1,1) , ah= [])
        tt_birth.append(tt_birth_new)     

    #create surviving tracks - via time prediction (single target CK)
    tt_survive=[]# cell(length(glmb_update.trackTable),1);  
                                                                                   #initialize cell array
    for tabsidx in range (len(glmb_update.trackTable)):
        [mtemp_predict,Ptemp_predict]= kalman_predict_multiple(model,glmb_update.trackTable[tabsidx].m,glmb_update.trackTable[tabsidx].P)     #kalman prediction for GM
        tt_survive_new=labeledTarget(mtemp_predict, Ptemp_predict,glmb_update.trackTable[tabsidx].w,glmb_update.trackTable[tabsidx].l,glmb_update.trackTable[tabsidx].ah )
        tt_survive.append(tt_survive_new)
    #create predicted tracks - concatenation of birth and survival
    glmb_predict=GLMB()
   # glmb_predict.trackTable= tt_birth+tt_survive                                                                                #copy track table back to GLMB struct
    glmb_predict.set_tt(tt_birth+tt_survive)
    #gating by tracks
    if filter.gate_flag:
        for tabidx  in range( len(glmb_predict.trackTable)):
            glmb_predict.trackTable[tabidx].gatemeas= gate_meas_gms_idx(meas[k],filter.gamma,model,glmb_predict.trackTable[tabidx].m,glmb_predict.trackTable[tabidx].P)
        
    else:
        for tabidx  in range(len(glmb_predict.trackTable)):
            start=1
            finish=meas[k].shape[1]
            glmb_predict.trackTable[tabidx].gatemeas= np.linspace( start,finish,finish-start+1)
        
    
    #precalculation loop for average survival/death probabilities
        #avps - average ps

    tmp=np.zeros((len(glmb_update.trackTable),1))
    avps= np.concatenate( (model.r_birth,tmp))
    for tabidx  in range(len(glmb_update.trackTable)):
        avps[model.T_birth+tabidx]= model.P_S
    avqs= 1-avps

    #precalculation loop for average detection/missed probabilities
    avpd= np.zeros((len(glmb_predict.trackTable),1))
    for tabidx in range(len(glmb_predict.trackTable)):
        avpd[tabidx]= model.P_D   
    avqd= 1-avpd; 

    #create updated tracks (single target Bayes update)
    m= meas[k].shape[1]                                                     # number of measurements for this time step (detections + clutter)
    n_tt_update=(1+m)*len(glmb_predict.trackTable)                                 #number of different ways to associate measurments to tracks
    tt_update=[]# cell((1+m)*length(glmb_predict.trackTable),1);       #initialize cell array
    for ct_tt_update in range(n_tt_update):
        tt_update.append(labeledTarget())
    #missed detection tracks (legacy tracks)
    for tabidx in range(len(glmb_predict.trackTable)):# 1:length(glmb_predict.trackTable)
        tt_update[tabidx]= copy.deepcopy(glmb_predict.trackTable[tabidx])       #same track table - copying over exisitng track tabel
        tt_update[tabidx].ah.append(-1)    #track association history (updated for missed detection)


    #measurement updated tracks (all pairs)
    # step 1 - producing signficant children - for each measurment inside a given tracks gate, update the track with the given measurment and create a new track with weight = 1
    # What is the signifcance of where they are stored 
    predLikelihood= np.zeros((len(glmb_predict.trackTable),m))
    for tabidx in range(len(glmb_predict.trackTable))  :
        for emm in glmb_predict.trackTable[tabidx].gatemeas: #gatemeas holds the valid measurements within the gate
                stoidx= len(glmb_predict.trackTable)*(emm[0]+1) + tabidx #index of predicted track i updated with measurement j is (number_predicted_tracks*j + i)
                qz_temp,m_temp,P_temp = kalman_update_multiple(meas[k][:,emm],model,glmb_predict.trackTable[tabidx].m,glmb_predict.trackTable[tabidx].P);   #kalman update for this track and this measurement
                # See equations 11,12,13
                w_temp= qz_temp*glmb_predict.trackTable[tabidx].w+np.finfo(float).eps;                          #unnormalized updated weights
                tt_update[stoidx].m= m_temp;                                                            #means of Gaussians for updated track
                tt_update[stoidx].P= P_temp;                                                            #covs of Gaussians for updated track
                tt_update[stoidx].w= w_temp/np.sum(w_temp)                                             #weights of Gaussians for updated track
                tt_update[stoidx].l =glmb_predict.trackTable[tabidx].get_l()# glmb_predict.trackTable[tabidx].l;        #track label
                tt_update[stoidx].ah= glmb_predict.trackTable[tabidx].get_ah() + emm.tolist() #glmb_predict.trackTable[tabidx].ah + emm.tolist()                    #track association history (updated with new measurement)
                predLikelihood[tabidx,emm]= np.sum(w_temp)                                     #predictive likelihood
    glmb_nextupdate=GLMB()     
    glmb_nextupdate.set_tt( tt_update) #glmb_nextupdate.trackTable= tt_update; 
                                                                                #copy track table back to GLMB struct
    #joint cost matrix - see equation 22
    joint1=np.diag(avqs.squeeze()) # died or not born 
    joint2=np.diag((avps*avqd).squeeze()) # survived and misdetected
    joint3=np.repeat(avps*avpd,m,1)*predLikelihood/(model.lambda_c*model.pdf_c) # survived and detected

    jointcostm=np.concatenate((joint1,joint2,joint3),1)

    #gated measurement index matrix - for each of the original tracks at time k, get the index from the meas[k] of the corresponding gated measurments per track
    # size = (num_tracks_orig, meas[k].shape[1])
    gatemeasidxs=-np.ones((len(glmb_predict.trackTable),m))# np.zeros((len(glmb_predict.trackTable),m))
    for tabidx in range(len(glmb_predict.trackTable)):
        gatemeasidxs[tabidx,np.arange(len(glmb_predict.trackTable[tabidx].gatemeas))]=glmb_predict.trackTable[tabidx].gatemeas.reshape(1,-1)
    
    # mask 
    gatemeasindc= gatemeasidxs>=0
            
    #component updates

    newHypIdx= 0
    for pidx in range(len(glmb_update.w)):
        #calculate best updated hypotheses/components
        cpreds= len(glmb_predict.trackTable)
        nbirths= model.T_birth
        nexists= len(glmb_update.I[pidx])
        ntracks= nbirths + nexists

        tindices=np.arange(0,nbirths)
        
        # for i in range(len(glmb_update.I)):
        #     tindices =np.concatenate((tindices,nbirths+glmb_update.I[pidx].astype('int')) )
        #
        if glmb_update.I[pidx].size>0: # remember, components in I[pidx] are indicies in track table
            # (birth indicies)+(I[pidx] indicies offset by bith indicies)
            # tindices here correspond to total indices of number of births + number of exisiting tracks for this hypothesis' componenets
            tindices =np.concatenate((tindices.reshape(-1,1),nbirths+glmb_update.I[pidx].reshape(-1,1)) ).astype('int').squeeze()       #indices of all births and existing tracks  for current component

        # size = (num_tracks_orig, meas[k].shape[1])
        lselmask= np.zeros((len(glmb_predict.trackTable),m))
        # mask based on zeros and ones - same as gatemeasidxs excpet with 0s and 1s
        lselmask[tindices,:]= gatemeasindc[tindices,:]                                        #logical selection mask to index gating matrices
        mindices= unique_faster(gatemeasidxs[lselmask>0]) #just get unique indicies of measurements from meas[k] -> no measurment repeated twice 
        
        indices_col=np.concatenate((np.array(tindices),np.array(tindices)+cpreds,np.array(mindices)+2*cpreds  ) ).astype('int')     #union indices of gated measurements for corresponding tracks
        costm= jointcostm[tindices,:]
        costm= costm[:,indices_col] 
        # COST MATRIX C                                                          #cost matrix - [no_birth/is_death | born/survived+missed | born/survived+detected]
        neglogcostm= -np.log(costm)
        # Recall tindices here correspond to total indices of number of births + number of exisiting tracks for this hypothesis' componenets                                                                                                          #negative log cost
        # Recall indices_col -  union indices of gated measurements for corresponding tracks
        # targets the rows, columns correspond to assocation of 1 of 3 categories

        
        ## Gibbs Sampling ##
        # number of requested / updated components - higher weighteted hypothesis get more components
        num_upd_comp = np.round(filter.H_upd*np.sqrt(glmb_update.w[pidx])/np.sum(np.sqrt(glmb_update.w)))
        gibbs_assignments,nlcost= mbestwrap_updt_gibbsamp(neglogcostm,num_upd_comp);#murty's algo/gibbs sampling to calculate m-best assignment hypotheses/components
        gibbs_assignments[gibbs_assignments<ntracks]= -np.inf                                                         #set not born/track deaths to -inf assignment
        gibbs_assignments[(gibbs_assignments>=ntracks) & (gibbs_assignments< 2*ntracks)]= -1                          #set survived+missed to 0 assignment
        gibbs_assignments[gibbs_assignments>=2*ntracks]= gibbs_assignments[gibbs_assignments>=2*ntracks]-2*ntracks;   #set survived+detected to assignment of measurement index from 1:|Z|    
        gibbs_assignments[gibbs_assignments>=0]= mindices[gibbs_assignments[gibbs_assignments>=0].astype('int')]      #restore original indices of gated measurements
        gibbs_assignments=gibbs_assignments+1
        #generate corrresponding jointly predicted/updated hypotheses/components
        for hidx  in range(len(nlcost)):
            update_hypcmp_tmp= gibbs_assignments[hidx,:].reshape(-1,1)
            tmp1= np.arange(nbirths).reshape(-1,1)
            tmp2=(glmb_update.I[pidx]+nbirths).reshape(-1,1)
            tmp=np.concatenate( (tmp1,tmp2),0 ) # indices of births and survived targets
            update_hypcmp_idx= cpreds*update_hypcmp_tmp+ tmp #
            up_w=-model.lambda_c+m*np.log(model.lambda_c*model.pdf_c)+np.log(glmb_update.w[pidx])-nlcost[hidx] # equation 53 in source 01
            up_I=update_hypcmp_idx[update_hypcmp_idx>=0] # equation 29
            up_n= np.sum(update_hypcmp_idx>=0)

            if len(glmb_nextupdate.w)>newHypIdx:
                glmb_nextupdate.w[newHypIdx]= up_w                                          #hypothesis/component weight
                glmb_nextupdate.I[newHypIdx]= up_I                                          #hypothesis/component tracks (via indices to track table)
                glmb_nextupdate.n[newHypIdx]= up_n
            else:
                 glmb_nextupdate.w=np.concatenate((glmb_nextupdate.w.reshape(-1,1),np.array([up_w]).reshape(-1,1)  )).squeeze()   
                 glmb_nextupdate.I.append(up_I)
                 glmb_nextupdate.n=np.concatenate(( glmb_nextupdate.n.reshape(-1,1),np.array([up_n]).reshape(-1,1) )).squeeze()       
                                                                                                               #hypothesis/component cardinality
            newHypIdx= newHypIdx+1

    glmb_nextupdate.w= np.exp(glmb_nextupdate.w-logsumexp(glmb_nextupdate.w))      #normalize weights
   
    #extract cardinality distribution
    for card in range(np.max(glmb_nextupdate.n)+1):
        up_cdn=np.sum(glmb_nextupdate.w[glmb_nextupdate.n==card])
        if len(glmb_nextupdate.cdn)>card:
            glmb_nextupdate.cdn[card]= up_cdn
        else:
            glmb_nextupdate.cdn =  np.concatenate((glmb_nextupdate.cdn,[up_cdn])) #glmb_nextupdate.cdn.append(up_cdn)
                                                                                                               #extract probability of n targets
    
    
    #remove duplicate entries and clean track table
    #glmb_nextupdate=sort_glmb(glmb_nextupdate)
    clp=clean_predict(glmb_nextupdate)
    glmb_nextupdate= clean_update(clp)
    glmb_nextupdate=sort_glmb(glmb_nextupdate)
    glmb_nextupdate.w[glmb_nextupdate.w<1e-90]=1e-90
    return glmb_nextupdate

def denormalize(z,minz,maxz,rangez):
    znew=z.copy()
    znew[0,:]= minz[0]+   (maxz[0]-minz[0])*(1+ z[0,:]/rangez)/2#          (2*(z[0,:]-minz[0])/(maxz[0]-minz[0])-1)*rangez
    znew[1,:]= minz[1]+   (maxz[1]-minz[1])*(1+ z[1,:]/rangez)/2#
    return znew

def normalize_meas(meas):
    z=np.zeros((2,0))
    for zk in meas:
        
        z=np.append(z,zk,1);
 

    minz=np.min(z,1)
    maxz=np.max(z,1)
    rangez=1000
    for ct in range(len(meas)):
        z=meas[ct]
       # zorig=z.copy()
        z[0,:]=(2*(z[0,:]-minz[0])/(maxz[0]-minz[0])-1)*rangez
        z[1,:]=(2*(z[1,:]-minz[1])/(maxz[1]-minz[1])-1)*rangez
        meas[ct]=z
        #zden=denormalize(z,minz,maxz,rangez)

    return meas,minz,maxz,rangez

