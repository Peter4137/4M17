import numpy as np

def new_theta(theta):
    n1 = (m*v_cg**2)/(R_w - h_cg*np.sin(theta-kappa))*np.cos(kappa)
    n2 = F_d*np.sin(eta)*np.cos(beta)
    d1 = m*g*np.cos(local_slope())
    d2 = (m*v_cg**2)/(R_w - h_cg*np.sin(theta-kappa))*np.sin(kappa)
    d3 = F_d*np.sin(eta)*np.sin(beta)
    return np.atan((n1+n2)/(d1-d2-d3))

all_theta = [0]
for i in range(20):
    all_theta.append(new_theta(all_theta[-1]))
