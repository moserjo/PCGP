import numpy as np
import os

sigma_list = [0.0, 0.01, 0.1, 0.5]
n_list = [2,3,4,5, 7, 10, 25, 50]
labels = ["l1 PIGP", "l1 NSB", "l2 PIGP", "l2 NSB"]

all_AE_end6 = np.zeros((len(n_list), len(sigma_list), 4, 10))
all_AE_end2 = np.zeros((5, len(sigma_list), 4, 10)) #5 for n=2,3,4,5,7

input_dir = os.path.join(
                os.path.dirname(__file__),
                "output_data",
                "fixedAl")   
for n_idx, n in enumerate(n_list):
    for sigma_idx, sigma in enumerate(sigma_list):
        for run in range(10):
            data_NSB = np.load(os.path.join(input_dir, f"Ex2_NSB_n{n}_sigma{sigma:.2f}_end6_{run}.npz"))
            data_PIGP = np.load(os.path.join(input_dir, f"Ex2_PIGP_n{n}_sigma{sigma:.2f}_end6_{run}.npz"))
            parameters_NSB = data_NSB["parameters_during_training"]
            parameters_PIGP = data_PIGP["parameters_during_training"]
 
            all_AE_end6[n_idx, sigma_idx, 0, run] = abs(parameters_PIGP[0][-1, 0] - 1)
            all_AE_end6[n_idx, sigma_idx, 1, run] = abs(parameters_NSB[0][-1, 0] - 1)
            all_AE_end6[n_idx, sigma_idx, 2, run] = abs(parameters_PIGP[1][-1, 0] - 2)
            all_AE_end6[n_idx, sigma_idx, 3, run] = abs(parameters_NSB[1][-1, 0] - 2)
            
            if n in [2,3,4,5,7]:   
                data_NSB_end2 = np.load(os.path.join(input_dir, f"Ex2_NSB_n{n}_sigma{sigma:.2f}_end2_{run}.npz"))
                data_PIGP_end2 = np.load(os.path.join(input_dir, f"Ex2_PIGP_n{n}_sigma{sigma:.2f}_end2_{run}.npz"))
                parameters_NSB_end2 = data_NSB_end2["parameters_during_training"]
                parameters_PIGP_end2 = data_PIGP_end2["parameters_during_training"]

                all_AE_end2[n_idx, sigma_idx, 0, run] = abs(parameters_PIGP_end2[0][-1, 0] - 1)
                all_AE_end2[n_idx, sigma_idx, 1, run] = abs(parameters_NSB_end2[0][-1, 0] - 1)
                all_AE_end2[n_idx, sigma_idx, 2, run] = abs(parameters_PIGP_end2[1][-1, 0] - 2)
                all_AE_end2[n_idx, sigma_idx, 3, run] = abs(parameters_NSB_end2[1][-1, 0] - 2)

output_path = os.path.join(
                os.path.dirname(__file__),
                "output_data",
                "Ex2_AE.npz")            
np.savez_compressed(
            output_path,
            sigma_list = np.array(sigma_list),
            n_list = np.array(n_list),
            all_AE_end2 = all_AE_end2,
            all_AE_end6 = all_AE_end6
        )                
                
print("done")