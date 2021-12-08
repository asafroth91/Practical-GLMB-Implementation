from scipy.stats.distributions import chi2
from libs.model import Model

class Filter():
    def __init__(self,model,H_upd= 1000,H_max= 1000,hyp_threshold= 1e-15,L_max= 100,elim_threshold= 1e-5,merge_threshold= 4,P_G= 0.9999999,gate_flag= 1):
        self.H_upd= H_upd                #requested number of updated components/hypotheses
        self.H_max= H_max                #cap on number of posterior components/hypotheses
        self.hyp_threshold= hyp_threshold        #pruning threshold for components/hypotheses

        self.L_max= L_max                  #limit on number of Gaussians in each track - not implemented yet
        self.elim_threshold= elim_threshold        #pruning threshold for Gaussians in each track - not implemented yet
        self.merge_threshold= merge_threshold         #merging threshold for Gaussians in each track - not implemented yet

        self.P_G=P_G                          #gate size in percentage
        self.gamma=chi2.ppf(self.P_G, df=model.z_dim)#chi2inv(self.P_G,model.z_dim);   #inv chi^2 dn gamma value
        self.gate_flag= gate_flag                            #gating on or off 1/0
