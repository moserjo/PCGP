import os
import torch
import numpy as np
torch.set_default_dtype(torch.float64)
import math

np.random.seed(123)
def f(x):
    return 50*torch.exp(-((x[:,0] - 0.5)**2 + (x[:,1] - 0.5)**2)/0.09) 

n_r = 10
n_p = 15
n_bc = 30
ntest = 21

r = torch.tensor(np.linspace(0.01, 0.5, n_r, endpoint = False))
p = torch.linspace(0, 2*math.pi, n_p)
R, P = torch.meshgrid(r, p, indexing = "ij")

r_bc = 0.5
p_bc = torch.linspace(0, math.pi*2, n_bc)

train_x_inside = torch.stack([(R*torch.cos(P)+0.5).flatten(), (R*torch.sin(P)+0.5).flatten()], dim = -1)
train_x_boundary = torch.stack([(r_bc*torch.cos(p_bc)+0.5).flatten(), (r_bc*torch.sin(p_bc)+0.5).flatten()], dim = -1)    
train_x = torch.cat([train_x_inside, train_x_boundary], dim = 0)

output_first_task = torch.full(train_x[:,0].shape, float('nan'))
output_first_task[-n_bc:] = torch.zeros((n_bc,))
train_y = torch.stack([output_first_task, output_first_task, f(train_x)], dim = -1)


r_test = torch.linspace(0.01, 0.5, ntest)
p_test = torch.tensor(np.linspace(0.0, 2*math.pi+0.0, ntest))
R_test, P_test = torch.meshgrid(r_test, p_test, indexing = "ij")

xx_test, yy_test = R_test*torch.cos(P_test)+0.5, R_test*torch.sin(P_test)+0.5
test_x = torch.stack([xx_test.flatten(), yy_test.flatten()], dim=-1).requires_grad_(True)
f_true = f(test_x).detach().reshape(r_test.shape[0], p_test.shape[0])

data_path = os.path.join(
    os.path.dirname(__file__),
    "input_data",
    "Ex1_nr%i_np%i_nbc%i_ntest%i.npz" % (n_r, n_p, n_bc, ntest)
)
with torch.no_grad():
    np.savez_compressed(
                    data_path,
                    train_x = train_x.numpy(),
                    test_x = test_x.numpy(),
                    train_y = train_y.numpy(),
                    f_true = f_true.numpy(),
                    )
print("saved", n_r, n_p, n_bc, ntest)
