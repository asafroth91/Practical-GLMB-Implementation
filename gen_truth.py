import numpy as np
import sys

from model import model

class ground_truth(model):
    n_briths = 12
    x1_start = [0,0,0,-10]
    x2_start = [400, -10, -600, 5]
    x3_start= [-800, 20, -200, -5]
    x4_start = [400, -2.5, -600, 10]
    x5_start = [400, -2.5, -600, 10]
    x6_start = [0, 7.5, 0, -5]
    x7_start = [-800, 12, -200, 7]
    x8_start = [-200, 15, 800, -10]
    x9_start = [-800, 3, -200, 15]
    x10_start = [-200, -3, 800, -15]
    x11_start = [0, -20, 0, -15]
    x12_start = [-200, 15, 800, -5]
    start_locations = np.array([x1_start, x2_start,x3_start, x4_start,
                                x5_start, x6_start, x7_start, x8_start,
                                x9_start, x10_start, x11_start, x12_start])
    
    
    
    def __init__(self, model):
        self.K = 100
        n_births = 12
        x1_start = [0,0,0,-10]
        x2_start = [400, -10, -600, 5]
        x3_start= [-800, 20, -200, -5]
        x4_start = [400, -2.5, -600, 10]
        x5_start = [400, -2.5, -600, 10]
        x6_start = [0, 7.5, 0, -5]
        x7_start = [-800, 12, -200, 7]
        x8_start = [-200, 15, 800, -10]
        x9_start = [-800, 3, -200, 15]
        x10_start = [-200, -3, 800, -15]
        x11_start = [0, -20, 0, -15]
        x12_start = [-200, 15, 800, -5]
        self.start_locations = np.array([x1_start, x2_start,x3_start, x4_start,
                                    x5_start, x6_start, x7_start, x8_start,
                                    x9_start, x10_start, x11_start, x12_start])
        self.birth_time = np.array([1,1,1,20,20,20,40,40,60,60,80,80])
        self.death_time = np.ones((12,1))*(self.K+1)
        self.death_time[0] = 70
        self.death_time[2] = 70
        
        
        self.GLMBmodel = model()
        # Gen tracks
        for targetnum in range(n_births):
            target_state = self.start_locations[targetnum,:].reshape((4,1))
            self.truthK = []
            for k in range(self.birth_time[targetnum],min(((self.death_time[targetnum][0]).astype(np.int64())),self.K)):
                target_state = self.gen_newstate_fn(self.GLMBmodel, target_state, 'noiseless')
                self.truthK.append(target_state)
        
    def gen_newstate_fn(self,GLMBmodel,target_state, noise):
        if (noise == 'noise'):
            V = GLMBmodel.sigma_v*GLMBmodel.B*np.random.randn((GLMBmodel.B.shape[0],target_state.shape[1]))
        elif (noise == 'noiseless'):
            V = np.zeros((GLMBmodel.B.shape[0], target_state.shape[1]))
        X = GLMBmodel.F @ target_state + V
        return X
            
gt = ground_truth(model)
targt_state = gt.start_locations[0]
print(gt.birth_time[0])
a = (min(gt.death_time[0][0],gt.K))
print(a)