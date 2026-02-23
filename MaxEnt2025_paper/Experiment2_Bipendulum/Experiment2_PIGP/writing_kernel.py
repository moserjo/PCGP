import sympy
from PCGP import write_kernel_and_model, generate_forward_function
from PCGP import write_numpyro_forward_body, write_numpyro_kernel_and_model, write_numpyro_parameter_sampling


#define B
def B(X, t):
    g, l1, l2 = sympy.symbols("g, l1, l2")
    return sympy.matrices.Matrix([[1, 0],
                                  [0,1],
                                  [l1*X[0]**2 + g, 0],
                                  [0, l2*X[0]**2 + g]
                                  ])


priors = {"l": "dist.Uniform(0.1, 15)",
          "A": "dist.Uniform(0.1, 15)",
          "g": "dist.Uniform(0.1, 15)",
          "l1": "dist.Uniform(0.5, 4.)",
          "l2": "dist.Uniform(0.5, 4.)"}
num_tasks = 4
number_of_input_dimensions = 1

forward_body = generate_forward_function(B, priors, number_of_input_dimensions=number_of_input_dimensions)
write_kernel_and_model(
     class_name = "Experiment2_PIGP",
     forward_body_code = forward_body,
     input_dims = number_of_input_dimensions,
     num_tasks = num_tasks,
)

parameter_sampling = write_numpyro_parameter_sampling(priors)
forward_body_numpyro = write_numpyro_forward_body(B, priors, 1)

write_numpyro_kernel_and_model("Experiment2_PIGP", 
                               forward_body_numpyro, 
                               parameter_sampling, 
                               number_of_input_dimensions=number_of_input_dimensions, 
                               num_tasks=num_tasks,   
                              
)