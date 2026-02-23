import torch
import numpy as np
import sys
import os
if len(sys.argv) != 4:
    print("Usage: python script.py <npoints> <sigma> <end>")
    sys.exit(1)

npoints = int(sys.argv[1])
sigma = float(sys.argv[2])
end = int(sys.argv[3])

l1_true = 1.
l2_true = 2.
def ex2_analytic_solution(x, l1_grid, l2_grid, g=9.81):
    X = 10
    a1 = torch.sqrt(g / l1_grid)
    a2 = torch.sqrt(g / l2_grid)
    f1 = 3*torch.sin(a1[:, None, None] * x[None, :, None]) + \
         X / l1_grid[:, None, None] / (a1[:, None, None]**2 - 4) * torch.sin(2 * x)[None, :, None]
    f2 = -5 * torch.cos(a2[:, None, None] * x[None, :, None]) + \
         X / l2_grid[:, None, None] / (a2[:, None, None]**2 - 4) * torch.sin(2 * x)[None, :, None]
    n_param_comb = l1_grid.shape[0]
    u1 = X * torch.sin(2 * x)[None, :, None].expand(n_param_comb, -1, -1)
    return torch.cat([f1, f2, u1], dim=-1)  

def draw_samples_from_posterior(P, mesh_1, mesh_2):
    P = P.numpy()
    P_flat = P.ravel()
    N_samples = 1000
    cdf = np.cumsum(P_flat)
    cdf = cdf / cdf[-1]
    u = np.random.rand(N_samples)
    sample_indices = np.searchsorted(cdf, u)
    i, j = np.unravel_index(sample_indices, P.shape)
    theta1_samples = mesh_1[i]
    theta2_samples = mesh_2[j]
    return theta1_samples, theta2_samples

#load data
input_path = os.path.join(
                os.path.dirname(__file__),
                "input_data",
                f"Ex2_n{npoints}_sigma{sigma:.2f}_end{end}.npz")   

data = np.load(input_path)
train_x = torch.tensor(data["train_x"])
test_x = torch.tensor(data["test_x"])
train_y = torch.tensor(data["train_y_NSB"])
test_y = torch.tensor(data["test_y_NSB"])


#define grid
N = 3000
grid_l1 = torch.linspace(0.5, 4.0, N)
grid_l2 = torch.linspace(0.5, 4.0, N)
mesh_1, mesh_2 = torch.meshgrid(grid_l1, grid_l2, indexing='ij')

#calculate posterior
if sigma == 0:
    sigma = 1e-3 #to prevent division by zero in likelihood
model_outputs = ex2_analytic_solution(train_x, mesh_1.flatten(), mesh_2.flatten(), g=9.81)  #param_combos[:,0], param_combos[:,1]
train_y_expanded = train_y.unsqueeze(0).expand(model_outputs.shape)
squared_error = ((train_y_expanded - model_outputs) ** 2).sum(dim=(1, 2))
log_likelihood = -squared_error / (2 * sigma ** 2)
log_likelihood_grid = log_likelihood.reshape(mesh_1.shape)
dx = (grid_l1[1]-grid_l1[0])*(grid_l2[1]-grid_l2[0])
posterior = torch.exp(log_likelihood_grid- torch.max(log_likelihood_grid))
posterior /= torch.sum(posterior*dx) 

# Find MAP, posterior and expectation estimate
best_idx = torch.argmax(log_likelihood_grid)
i, j = divmod(best_idx.item(), mesh_1.shape[1])
l1_MAP = grid_l1[i]
l2_MAP = grid_l2[j]
expect_l1 = torch.sum(mesh_1* posterior*dx)
expect_l2 = torch.sum(mesh_2 * posterior*dx)


#calculate mean and uncertainties
theta1_samples, theta2_samples = draw_samples_from_posterior(posterior, grid_l1, grid_l2)
model_outputs_samples = ex2_analytic_solution(test_x, theta1_samples, theta2_samples, g=9.81) 
mean = torch.mean(model_outputs_samples, dim = 0)
lower = np.percentile(model_outputs_samples.numpy(), 2.5, axis=0)
upper = np.percentile(model_outputs_samples.numpy(), 97.5, axis=0)



output_path = os.path.join(
                os.path.dirname(__file__),
                "output_data",
                "Bayes",
                "Ex2_Bayes_n%i_sigma%.2f_end%i.npz"%(npoints, sigma, end))    
        
np.savez_compressed(
            output_path,
            mean= mean.numpy(),
            lower= lower,
            upper= upper,
            test_x = test_x.numpy(),
            test_y = test_y.numpy(),
            train_x = train_x.numpy(),
            train_y = train_y.numpy(),
            expected_parameters = np.array([expect_l1.item(), expect_l2.item()]),
            MAP_parameters = np.array([l1_MAP, l2_MAP]),
            true_parameters = np.array([l1_true, l2_true]),
            grid_l1 = grid_l1.numpy(),
            grid_l2 = grid_l2.numpy(),
            posterior = posterior.numpy(),
        )
print("done")
