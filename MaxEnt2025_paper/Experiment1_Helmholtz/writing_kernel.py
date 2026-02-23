
import sympy
from PCGP import write_kernel_and_model, generate_forward_function


def B(X, x): 
    a = 30*sympy.cos(x[0]*2)
    b = 50*(x[1]+1)**3
    c = 10
    d = 1
    return sympy.matrices.Matrix([
            [d + X[0]**2 + X[1]**2],
            [-c],
            [-b*c + a*d + (a+d)*X[0]**2 + (a+d)*X[1]**2+ X[0]**4 + X[1]**4 +2*X[0]**2*X[1]**2],
        ])


A, l = sympy.symbols("A, l")
parameters = {"A": A, "l": l, }
number_of_input_dimensions = 2

forward_body = generate_forward_function(B, parameters, number_of_input_dimensions=number_of_input_dimensions)

write_kernel_and_model(
     class_name = "Experiment1_Helmholtz",
     forward_body_code = forward_body,
     input_dims = number_of_input_dimensions,
     num_tasks = 3,
)