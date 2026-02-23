import os
import torch
import numpy as np
import math
torch.set_default_dtype(torch.float64)
np.random.seed(123)



g_true = 9.81
def ex2_analytic_solution(x, l1 = 1, l2 = 2., g = 9.81, noise = 0):
    X = 10
    a1 = math.sqrt(g/l1)
    a2 = math.sqrt(g/l2)
    f1 = 3*np.sin(a1*x) + X/l1/(a1**2 - 4)*np.sin(2*x) + noise*np.random.randn(x.shape[0])
    f2 = -5*np.cos(a2*x)  + X/l2/(a2**2 - 4)*np.sin(2*x)+ noise*np.random.randn(x.shape[0])
    u1 = X*np.sin(2*x)+ noise*np.random.randn(x.shape[0])
    y = {}
    y["NSB"] = np.stack([f1, f2, u1], axis = -1)
    y["PIGP"] = np.stack([f1, f2, u1, u1], axis = -1)
    return y

end = 2
sigma_list = [0., 0.01, 0.1, 0.5]
n_list = [2, 3, 5, 7]
for sigma in sigma_list:
    for npoints in n_list:
        train_x =  np.linspace(1, end, npoints)
        train_y = ex2_analytic_solution(train_x, noise = sigma)
        test_x =  np.linspace(1, end, 201)
        test_y = ex2_analytic_solution(test_x,  noise = 0)
        data_path = os.path.join(
                    os.path.dirname(__file__),
                    "input_data",
                    "Ex2_n%i_sigma%.2f_end%i.npz"%(npoints, sigma, end))
        np.savez_compressed(
                    data_path,
                    train_x = train_x,
                    test_x = test_x,
                    train_y_NSB = train_y["NSB"],
                    train_y_PIGP = train_y["PIGP"],
                    test_y_NSB = test_y["NSB"],
                    test_y_PIGP = test_y["PIGP"],
                )
        print("saved", npoints, end, sigma)
end = 6
sigma_list = [0., 0.01, 0.1, 0.5]
n_list = [2, 3, 5, 7, 10, 25, 50]
for sigma in sigma_list:
    for npoints in n_list:
        train_x =  np.linspace(1, end, npoints)
        train_y = ex2_analytic_solution(train_x, noise = sigma)
        test_x =  np.linspace(1, end, 201)
        test_y = ex2_analytic_solution(test_x,  noise = 0)
        data_path = os.path.join(
                    os.path.dirname(__file__),
                    "input_data",
                    "Ex2_n%i_sigma%.2f_end%i.npz"%(npoints, sigma, end))
        np.savez_compressed(
                    data_path,
                    train_x = train_x,
                    test_x = test_x,
                    train_y_NSB = train_y["NSB"],
                    train_y_PIGP = train_y["PIGP"],
                    test_y_NSB = test_y["NSB"],
                    test_y_PIGP = test_y["PIGP"],
                )
        print("saved", npoints, end, sigma)