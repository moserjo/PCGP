
import sympy
from PCGP import write_kernel_and_model, generate_forward_function
from PCGP import write_numpyro_forward_body, write_numpyro_kernel_and_model, write_numpyro_parameter_sampling


def B(X, x):
    R = sympy.symbols("R")
    return sympy.matrices.Matrix([[1], [R*X[0]+1]])

priors = {"l": "dist.Uniform(0.1, 15)",
          "A": "dist.Uniform(0.1, 15)",
          "R": "dist.Uniform(0.1, 15)",}


forward_body = generate_forward_function(B, priors, number_of_input_dimensions=1)
write_kernel_and_model(
     class_name = "Experiment3_LumpedHeat",
     forward_body_code = forward_body,
     input_dims = 1,
     num_tasks = 2,
)

parameter_sampling = write_numpyro_parameter_sampling(priors)
forward_body_numpyro = write_numpyro_forward_body(B, priors, 1)

write_numpyro_kernel_and_model("Experiment3_LumpedHeat", 
                               forward_body_numpyro, 
                               parameter_sampling, 
                               number_of_input_dimensions=1, 
                               num_tasks=2,   
)