import numpy as np



class model():
    def __init__(self):
        self.x_dim = 4                              # dimension of state vector
        self.z_dim = 2                              # dimesion of observation vector
    
#    def constant_velocity_params(self):
        self.T = 1                                  # sampling period
        self.A0 = np.array([[1, self.T], [0 ,1]])   # Transition Matrix
        self.F = np.vstack((
            np.hstack((self.A0, np.zeros((2,2)))), 
            np.hstack((np.zeros((2,2)),self.A0 ))   
            ))
        self.B0 = np.array([[(self.T**2)/2],[self.T]])
        self.B = np.vstack((
            np.hstack((self.B0, np.zeros((2,1)))),
            np.hstack((np.zeros((2,1)), self.B0))
            ))
        self.sigma_v = 5
        self.Q = (self.sigma_v**2)*(self.B@self.B.T)  # Process noise covariance


#    def survival_and_death_params(self):
        self.Ps = 0.99                              # probability of survival
        self.Qs = 1 - self.Ps                       # vprobability of death
    
#    def birth_params(self):
        self.T_birth = 4                            # Number of LMB birth terms
        self.L_birth = np.ones((4,1))               # number of GM components in each LMB brith term
        self.r_birth = np.ones((4,1))*0.03
        self.w_birth = np.ones((4,1))               # weights of Gaussians
        mean1 = np.array([0.1, 0, 0.1, 0]).reshape(4,1)
        mean2 = np.array([400, 0, -600, 0]).reshape(4,1)
        mean3 = np.array([-800, 0, -200, 0]).reshape(4,1)
        mean4 = np.array([-200, 0, 800, 0]).reshape(4,1)
        self.m_birth = np.hstack((  
            mean1,mean2,mean3, mean4))              # mean of GM models, 1 vector per 1 GMM
        std_per_gmm = np.diag([10,10,10,10])        
        self.B_birth = np.broadcast_to(             #std per gmm, each 4x4 array
            std_per_gmm, (4,4,4))
        cov_per_gmm = std_per_gmm @ std_per_gmm.T   # cov of GMM each 4x4
        self.P_birth = np.broadcast_to(
            cov_per_gmm, (4,4,4))
        
#    def observation model params
        
        self.H = np.array([[1,0,0,0],[0,0,1,0]])      # observation matrix
        self.D = np.diag([10,10])
        self.R = self.D @ self.D.T              #observation noise covariance

#   def detection_params
        self.Pd = 0.98                                # probability of detection in measurements
        self.Qd = 1-self.Pd                           #probability of missed detections in measurements

 # def clutter_params
        self.lambda_c = 30                           #poisson averagte rate of unifrom clutter (per scan)
        self.range_c = np.array([[-1000, 1000],[-1000, 1000]]) # clutter bounds
        self.pdf_c = 1/np.prod((self.range_c[:,1]-self.range_c[:,0])) # uniform clutter density

    def getX_dim(self):
        return self.x_dim

