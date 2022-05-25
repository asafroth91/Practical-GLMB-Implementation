import numpy as np
import copy

class GLMB():
    def __init__(self):

        #initial prior
        self.trackTable= []      #track table for GLMB (cell array of structs for individual tracks)
        self.w= np.array([1])              #vector of GLMB component/hypothesis weights
        self.I= [np.array([])]            #cell of GLMB component/hypothesis labels (labels are indices/entries in track table)
        self.n= np.array([0])              #vector of GLMB component/hypothesis cardinalities
        self.cdn= np.array([1.0])   #
        self.hash=[]

    #### get
    def get_tt(self):
        return copy.deepcopy(self.trackTable)
    def get_I(self):
        return copy.deepcopy(self.I)
    def get_w(self):
        return copy.deepcopy(self.w)
    def get_n(self):
        return copy.deepcopy(self.n)
    def get_cdn(self):
        return copy.deepcopy(self.cdn)

    ####set
    def set_tt(self,tt):
        self.trackTable= copy.deepcopy(tt)
    def set_I(self,I):
        self.I= copy.deepcopy(I)
    def set_w(self,w):
        self.w= copy.deepcopy(w)
    def set_n(self,n):
        self.n= copy.deepcopy(n)
    def set_cdn(self,cdn):
        self.cdn= copy.deepcopy(cdn)

class labeledTarget():
    def __init__(self,m=[],P=[],w=[],l=[],ah=[]):
        # m means of Gaussians for  track
        #covs of Gaussians for  track
        #weights of Gaussians for  track
        #track label
        #track association history (empty at birth)
        #initialize cell array


        self.m=np.array(m)
        self.P=np.array(P)
        self.w=np.array(w)
        self.l=np.array(l)
        self.ah=ah # Var xi -> association history map
        self.gatemeas=-1

    #### get
    def get_m(self):
        return copy.deepcopy(self.m)
    def get_P(self):
        return copy.deepcopy(self.P)
    def get_w(self):
        return copy.deepcopy(self.w)
    def get_l(self):
        return copy.deepcopy(self.l)
    def get_ah(self):
        return copy.deepcopy(self.ah)
    def get_gatemeas(self):
        return copy.deepcopy(self.gatemeas)


    ####set

    def set_m(self):
        self.m= copy.deepcopy(self.m)
    def set_P(self):
        self.P= copy.deepcopy(self.P)
    def set_w(self):
        self.w= copy.deepcopy(self.w)
    def set_l(self):
        self.l= copy.deepcopy(self.l)
    def set_ah(self):
        self.ah= copy.deepcopy(self.ah)
    def set_gatemeas(self):
        self.gatemeas=copy.deepcopy(self.gatemeas)
