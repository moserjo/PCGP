import numpy as np
from scipy.optimize import minimize
import os

def load_method_data(method, npoints, sigma, run, dataset_version, end = 6):
    input_path = os.path.join(
                    os.path.dirname(__file__),
                    "output_data",
                    f"{dataset_version}/Ex2_{method}_n{npoints}_sigma{sigma:.2f}_end{end}_{run}.npz")
    data = np.load(input_path)
    return {
        "MAP": data["learned_parameters"][:2],
        "hessian": data["hessian"][:2, :2],
        "cov": data["laplace_covariance"][:2, :2],
        "train_x": data["train_x"],
        "train_y": data["train_y"],
    }


def analytic_DKL(mu_p, hessian_p, mu_q, hessian_q):
    det_Hp = hessian_p[0,0]*hessian_p[1,1] - hessian_p[0,1]*hessian_p[1,0]
    cov_p = np.array([[hessian_p[1,1], -hessian_p[0,1]],
                  [-hessian_p[1,0], hessian_p[0,0]]]) / det_Hp
    det_Hq = hessian_q[0,0]*hessian_q[1,1] - hessian_q[0,1]*hessian_q[1,0]
    
    trace_term = np.trace(np.dot(hessian_q, cov_p))
    diff = mu_q - mu_p
    quad_term = diff @ hessian_q @ diff
    
    DKL = 0.5 * (np.log(det_Hp / det_Hq) - 2 + trace_term + quad_term)
    return DKL

def ex2_analytic_solution(x, l1, l2, g=9.81, X=10.0):
    a1 = np.sqrt(g / l1)
    a2 = np.sqrt(g / l2)
    f1 = 3 * np.sin(a1 * x) + X / l1 / (a1**2 - 4) * np.sin(2 * x)
    f2 = -5 * np.cos(a2 * x) + X / l2 / (a2**2 - 4) * np.sin(2 * x)
    u1 = X * np.sin(2 * x)
    return np.stack([f1, f2, u1], axis=-1) 

def neg_log_likelihood(theta, x, y, sigma):
    l1, l2 = theta
    residuals = y[:,:2] - ex2_analytic_solution(x, l1, l2)[:,:2]
    return np.sum(residuals**2)/(2*sigma**2)

def get_mu_bayes(train_x, train_y, sigma):
    res = minimize(neg_log_likelihood, x0=[1.0, 2.0], args=(train_x, train_y, sigma), bounds=[(0.5, 4.), (0.5, 4.)] )
    mu = res.x
    return mu

def get_hessian_bayes(train_x, train_y, l1, l2, sigma):
    g=9.81
    X=10.0
    x = np.asarray(train_x)
    y = np.asarray(train_y)
    a1 = np.sqrt(g / l1)
    a2 = np.sqrt(g / l2)
    f1 = 3*np.sin(a1*x) + X/(l1*(a1**2 - 4))*np.sin(2*x)
    f2 = -5*np.cos(a2*x) + X/(l2*(a2**2 - 4))*np.sin(2*x)

    r1 = y[:,0] - f1
    r2 = y[:,1] - f2

    # First derivatives
    df1_dl1 = - (3*np.sqrt(g)/(2*l1**(3/2))) * x * np.cos(a1*x) + (4*X)/((g - 4*l1)**2) * np.sin(2*x)
    df2_dl2 = - (5*np.sqrt(g)/(2*l2**(3/2))) * x * np.sin(a2*x) + (4*X)/((g - 4*l2)**2) * np.sin(2*x)

    # Second derivatives
    d2f1_dl12 = (9*np.sqrt(g)/(4*l1**(5/2))) * x * np.cos(a1*x) + (3*g/(4*l1**3)) * x**2 * np.sin(a1*x) + (32*X)/((g - 4*l1)**3) * np.sin(2*x)
    d2f2_dl22 = (15*np.sqrt(g)/(4*l2**(5/2))) * x * np.sin(a2*x) - (5*g/(4*l2**3)) * x**2 * np.cos(a2*x) + (32*X)/((g - 4*l2)**3) * np.sin(2*x)

    # Hessian components
    H11 = np.sum(df1_dl1**2 + r1*d2f1_dl12)
    H22 = np.sum(df2_dl2**2 + r2*d2f2_dl22)
    H12 = H21 = 0.0  # Cross derivatives are zero since f1 and f2 depend on l1 and l2 separately

    H = np.array([[H11, H12],
                  [H21, H22]]) / sigma**2
    return H
    


sigma_list = [0.0, 0.01, 0.1, 0.5]
n_list = [2, 3, 4, 5, 7, 10, 25, 50]
all_DKL_end6 = np.zeros((len(n_list), len(sigma_list), 2, 10))
all_DKL_end2 = np.zeros((5, len(sigma_list), 2, 10))  # for n=2,3,4,5,7

for n_idx, n in enumerate(n_list):
    for sigma_idx, sigma in enumerate(sigma_list):
        if sigma == 0: #to prevent kernel instability
                likelihood_sigma = 1e-3
        else: 
             likelihood_sigma = sigma
        for run in range(10):
            data_NSB = load_method_data("NSB", n, sigma, run, dataset_version="fixedAl", end=6)
            data_PIGP = load_method_data("PIGP", n, sigma, run, dataset_version="fixedAl", end = 6)
            
            train_x = data_NSB["train_x"]
            train_y = data_NSB["train_y"] 
            
            mu_bayes = get_mu_bayes(train_x, train_y, likelihood_sigma)
            hessian_bayes = get_hessian_bayes(train_x, train_y, mu_bayes[0], mu_bayes[1], likelihood_sigma)
            all_DKL_end6[n_idx, sigma_idx, 0, run] = analytic_DKL(mu_bayes, hessian_bayes, data_PIGP["MAP"], data_PIGP["hessian"])
            all_DKL_end6[n_idx, sigma_idx, 1, run] = analytic_DKL(mu_bayes, hessian_bayes, data_NSB["MAP"], data_NSB["hessian"])
            #print(all_DKL_end6[n_idx, sigma_idx, 1, run], all_DKL_end6[n_idx, sigma_idx, 0, run])
            if n in [2,3,4,5, 7]:
                data_NSB = load_method_data("NSB", n, sigma, run, dataset_version="fixedAl", end=2)
                data_PIGP = load_method_data("PIGP", n, sigma, run, dataset_version="fixedAl", end=2)
                
                train_x = data_NSB["train_x"]
                train_y = data_NSB["train_y"] 
                
                mu_bayes = get_mu_bayes(train_x, train_y, likelihood_sigma)
                hessian_bayes = get_hessian_bayes(train_x, train_y, mu_bayes[0], mu_bayes[1], likelihood_sigma)
                all_DKL_end2[n_idx, sigma_idx, 0, run] = analytic_DKL(mu_bayes, hessian_bayes, data_PIGP["MAP"], data_PIGP["hessian"])
                all_DKL_end2[n_idx, sigma_idx, 1, run] = analytic_DKL(mu_bayes, hessian_bayes, data_NSB["MAP"], data_NSB["hessian"])

output_path = os.path.join(
                os.path.dirname(__file__),
                "output_data",
                "Ex2_DKL.npz")            
np.savez_compressed(
            output_path,
            sigma_list = np.array(sigma_list),
            n_list = np.array(n_list),
            all_DKL_end2 = all_DKL_end2,
            all_DKL_end6 = all_DKL_end6
        )                
                
print("done")
