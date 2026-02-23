import os
import numpy as np

np.random.seed(123)

N_timesteps = 2101*4
skip_train = 56
skip_test = 7
time = np.linspace(0, 20, N_timesteps)

def ex3_analytic_solution(time, R, Ta, boundary_condition = 0): 
    dt = time[1]-time[0] 
    R_inverted = 1/R
    R_cumsum = np.cumsum(R_inverted*dt)
    Ti_homogenous = np.exp(-R_cumsum)
    Ct = np.cumsum(np.exp(R_cumsum)*R_inverted*Ta*dt)
    Ti_particular = Ct*Ti_homogenous
    Ti = Ti_homogenous/Ti_homogenous[0]*(boundary_condition - Ti_particular[0]) + Ti_particular
    return Ti

R_true = np.where(time<8, 1, time/16+0.5)
Ta = np.sin(2*time)
Ti = ex3_analytic_solution(time, R_true, Ta, boundary_condition=0)

sigma = 0.05
train_x = time[::skip_train]
test_x = time[::skip_test]
train_y = np.stack([(Ti + sigma*np.random.randn(N_timesteps))[::skip_train],
                   (Ta + sigma*np.random.randn(N_timesteps))[::skip_train]],
                    axis = -1)
test_y = np.stack([(Ti)[::skip_test],
                   (Ta)[::skip_test]],
                    axis = -1)

data_path = os.path.join(
    os.path.dirname(__file__),
    "input_data",
    "Ex3_sigma%.2f.npz"%sigma
)
np.savez_compressed(
    data_path,
    train_x = train_x,
    test_x = test_x,
    train_y = train_y,
    test_y = test_y,
    R_true = R_true[::skip_test],
    )

print("saved")