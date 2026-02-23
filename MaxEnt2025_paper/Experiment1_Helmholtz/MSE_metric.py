import os
import torch
import numpy as np

n_r, n_p, n_bc, ntest = 10, 15, 30, 21
output_path = os.path.join(
    os.path.dirname(__file__),
    "output_data",
    "Ex1_nr%i_np%i_nbc%i_ntest%i.npz" % (n_r, n_p, n_bc, ntest)
)

data = np.load(output_path)
error = data["error"]
test_y = data["test_y"]


residuum_1 = np.sqrt(torch.mean(torch.tensor(error[0,:])**2))
residuum_2 = np.sqrt(torch.mean(torch.tensor(error[1,:])**2))
inhomogeneity_error = np.sqrt(torch.mean(torch.tensor(error[2,:])**2))
boundary_error_1 = np.sqrt(np.mean(test_y[0, -1, :]**2))
boundary_error_2 = np.sqrt(np.mean(test_y[1, -1, :]**2))
print("residuum_1: ", residuum_1.item())
print("residuum_2: ", residuum_2.item())
print("inhomogeneity_error: ", inhomogeneity_error.item())
print("boundary_error_1: ", boundary_error_1.item())
print("boundary_error_2: ", boundary_error_2.item())