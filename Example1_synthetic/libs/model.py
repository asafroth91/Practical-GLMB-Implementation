import numpy as np


class Model():
    def __init__(self,x_dim= 4,z_dim= 2,T=1,sigma_v = 5,P_S= 0.99 ):
        self.x_dim=x_dim #dimension of state vector
        self.z_dim=z_dim #dimension of observation vector

        #dynamical model parameters (CV model)
        self.T=1 #sampling period
        self.sigma_v=sigma_v
        self.P_S=P_S
        
        self.A0=np.array([[1,1],[0,1]]) #transition matrix  
        F=np.zeros((4,4))
        F[0:2,0:2]=self.A0
        F[2:4,2:4]=self.A0
        self.F=F
        self.B0=np.array(  [ (self.T*self.T )/2, self.T ] )
        self.B0=self.B0.reshape(2,1)
        
        B=np.zeros((4,2) )
        B[0:2,0]=self.B0.squeeze()
        B[2:4,1]=self.B0.squeeze()
        self.B=B

        self.Q=np.matmul((self.sigma_v)*(self.sigma_v)*B,np.transpose(B)) #process noise covariance
        #survival/death parameters
        self.P_S= .99
        self.Q_S= 1-self.P_S

       #######################################################
        #birth parameters (LMB birth model, single component only)
        self.T_birth= 4                                          ##no. of LMB birth terms
        self.L_birth=np.zeros((self.T_birth,1))                  ##no of Gaussians in each LMB birth term
        self.r_birth=np.zeros((self.T_birth,1))                  ##prob of birth for each LMB birth term
        self.w_birth=[]                                          ##weights of GM for each LMB birth term
        self.m_birth=[]                                          # means of GM for each LMB birth term
        self.B_birth=[]                                          #std of GM for each LMB birth term
        self.P_birth=[]                                          #cov of GM for each LMB birth term
      
        self.L_birth[0]=1                                                            #no of Gaussians in birth term 1
        self.r_birth[0]=0.03                                                         #prob of birth
        self.w_birth.append(1)                                                       #weight of Gaussians - must be column_vector
        self.m_birth.append( np.array([ 0.1, 0, 0.1, 0 ]).reshape(4,1))                                      #mean of Gaussians
        self.B_birth.append( np.diag([10,10,10,10]))                                 #std of Gaussians
        self.P_birth.append( np.matmul(self.B_birth[0],self.B_birth[0].transpose() ))    #cov of Gaussians

        self.L_birth[1]=1                                                            #no of Gaussians in birth term 2
        self.r_birth[1]=0.03                                                         #prob of birth
        self.w_birth.append(1)                                                       #weight of Gaussians - must be column_vector
        self.m_birth.append( np.array([ 400, 0, -600, 0 ]).reshape(4,1))                                      #mean of Gaussians
        self.B_birth.append( np.diag([10,10,10,10]))                                 #std of Gaussians
        self.P_birth.append( np.matmul(self.B_birth[0],self.B_birth[0].transpose() ))    #cov of Gaussians


        self.L_birth[2]=1                                                            #no of Gaussians in birth term 3
        self.r_birth[2]=0.03                                                         #prob of birth
        self.w_birth.append(1)                                                       #weight of Gaussians - must be column_vector
        self.m_birth.append( np.array([ -800, 0, -200, 0 ]).reshape(4,1))                                      #mean of Gaussians
        self.B_birth.append( np.diag([10,10,10,10]))                                 #std of Gaussians
        self.P_birth.append( np.matmul(self.B_birth[0],self.B_birth[0].transpose() ))    #cov of Gaussians

        self.L_birth[3]=1                                                            #no of Gaussians in birth term 4
        self.r_birth[3]=0.03                                                         #prob of birth
        self.w_birth.append(1)                                                       #weight of Gaussians - must be column_vector
        self.m_birth.append( np.array([ -200, 0, 800, 0 ]).reshape(4,1))                                      #mean of Gaussians
        self.B_birth.append( np.diag([10,10,10,10]))                                  #std of Gaussians
        self.P_birth.append( np.matmul(self.B_birth[0],self.B_birth[0].transpose() ))    #cov of Gaussians

      
       ##################################
        # observation model parameters (noisy x/y only)
        self.H= np.array([[ 1 ,0, 0, 0] ,[ 0, 0, 1, 0 ]]) #observation matrix
        self.D=np.diag([10,10])
        self.R=np.matmul(self.D, np.transpose(self.D)) #observation noise covariance
        
        #detection parameters
        self.P_D=0.98
        self.P_D= .98 #probability of detection in measurements
        self.Q_D= 1-self.P_D; #probability of missed detection in measurements
        # clutter parameters
        self.lambda_c= 30                             #poisson average rate of uniform clutter (per scan)
        self.range_c= np.array([[ -1000.0, 1000.0], [-1000.0, 1000.0 ]])    #uniform clutter region
        self.pdf_c=  1/np.prod(self.range_c[:,1]- self.range_c[:,0]); #uniform clutter density
    # def predict(self, frame, time):
    #     #img=np.copy(frame)
    #     img_orig=np.copy(img)
    #     height=img.shape[0]
    #     width=img.shape[1]