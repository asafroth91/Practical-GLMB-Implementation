import numpy as np
from libs.filter import Filter 

class Est():
    def __init__(self,K ,filter):
        self.X= []#cell(K,1);
        self.N= np.zeros((K,1))
        self.L= []#cell(K,1);
        self.T= []#{};
        self.M= 0
        self.J= []; 
        self.H= []#{};
        self.filter= filter                     #gating on or off 1/0

        