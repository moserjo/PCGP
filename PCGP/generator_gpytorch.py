from jinja2 import Template
import os
import sympy
import inspect

from sympy.printing.pycode import pycode
from .diffeq_kernel import kernel_matrix


def generate_forward_function(B, parameters, number_of_input_dimensions):
    """
    Generates needed forward function code for the GPyTorch kernel based on the parametrization matrix B as a string.
    
    :param B: sympy.matrices.Matrix, user specified parametrization matrix
    :param parameters: dictionary including all parameters in B and the base-kernel hyperparameters A (amplitude) and l (lengthscale)
    :param number_of_input_dimensions: int, number of input dimensions
    :return: str, code for the body of the forward function
    """
    kernel_object = kernel_matrix(B, parameters, number_of_input_dimensions=number_of_input_dimensions) 
    symbolic_kernel = kernel_object.get_symbolic_kernel()
    num_tasks = symbolic_kernel.shape[0]

    # Define symbol substitutions
    substitutions = {}
    for i in range(1, number_of_input_dimensions + 1):
            substitutions[sympy.Symbol(f"x{i}")] = sympy.Symbol(f"xx[{i-1}]")
            substitutions[sympy.Symbol(f"x{i}_")] = sympy.Symbol(f"yy[{i-1}]")

    lines = [
        "if x1.dim() == 1:",
        "    mesh = torch.meshgrid(x1.flatten(), x2.flatten(), indexing='xy')",
        "    xx, yy = mesh[0].T.unsqueeze(0), mesh[1].T.unsqueeze(0)",
        "elif x1.dim() == 2 and x1.shape[1] == self.number_of_input_dimensions:",
        "    xx = torch.zeros((self.number_of_input_dimensions, x1.shape[0], x2.shape[0]), device=x1.device)",
        "    yy = torch.zeros((self.number_of_input_dimensions, x1.shape[0], x2.shape[0]), device=x1.device)",
        "    for i in range(self.number_of_input_dimensions):",
        "        mesh = torch.meshgrid(torch.squeeze(x1[:,i]), torch.squeeze(x2[:,i]), indexing='xy')",
        "        xx[i] = mesh[0].T",
        "        yy[i] = mesh[1].T",
    ]
    parameter_lines = []
    for p in parameters:
        parameter_lines.append(f"{p} = self.get_param('{p}')")
    lines.extend(parameter_lines)    

    # Convert symbolic matrix into code for PyTorch
    code_lines = []
    for i in range(num_tasks):
        for j in range(num_tasks):
            expr = sympy.simplify(symbolic_kernel[i, j]).subs(substitutions)
            torch_expr = pycode(expr).replace("math.", "torch.")
            if torch_expr == "0":
                 torch_expr = "torch.zeros_like(xx[0], device=x1.device)"
            code_lines.append(f"k{i}{j} = {torch_expr}")

    lines.extend(code_lines)

    # Assemble the cov matrix
    cat_rows = []
    for i in range(num_tasks):
        row_expr = ", ".join([f"k{i}{j}" for j in range(num_tasks)])
        cat_rows.append(f"torch.cat([{row_expr}], dim=-1)")
    full_mat = "torch.cat([\n    " + ",\n    ".join(cat_rows) + "], dim=-2)" 

    lines.append("cov_m = torch.squeeze(" + full_mat + ")")
    lines.append(f"cov_f = rearrange(cov_m, \"(t1 w1) (t2 w2)-> (w1 t1) (w2 t2)\", t1={num_tasks}, t2={num_tasks})")
    lines.append("return torch.diag(cov_f) if diag else cov_f")

    return "\n".join([" " * 8 + l for l in lines]) 


KERNEL_TEMPLATE = Template("""
import torch
import gpytorch
from einops import rearrange
from PCGP import ConstraintsModifications

class PCGP_Kernel(gpytorch.kernels.Kernel):
    def __init__(self, input_parameters, number_of_input_dimensions={{ input_dims }}, num_tasks={{ num_tasks }}, **kwargs):
        super().__init__()
        self.num_tasks = num_tasks
        self.input_parameters = input_parameters
        self.param_constraints = {}
        self.number_of_input_dimensions = number_of_input_dimensions

        for param_name in input_parameters:
            raw_name = f"raw_{param_name}"
            value, requires_grad, constraint = input_parameters[param_name]
            param = torch.nn.Parameter(torch.tensor([value]), requires_grad=requires_grad)
            self.register_parameter(raw_name, param)
            if constraint and requires_grad:
                CM = ConstraintsModifications(constraint)
                init_val = value if CM.is_fulfilled(value) else CM.init_val_from_constraint()
                self.register_constraint(raw_name, constraint)
                self.param_constraints[param_name] = constraint
                self.set_param(param_name, init_val)

    def _set_param(self, param_name, value):
        raw_name = f"raw_{param_name}"
        param = getattr(self, raw_name)
        value = torch.as_tensor(value).to(param)
        if param_name in self.param_constraints:
            value = self.param_constraints[param_name].inverse_transform(value)
        self.initialize(**{raw_name: value})

    def get_param(self, param_name):
        raw_name = f"raw_{param_name}"
        raw_param = getattr(self, raw_name)
        if param_name in self.param_constraints:
            return self.param_constraints[param_name].transform(raw_param)
        return raw_param

    def get_raw_param(self, param_name):
        return getattr(self, f"raw_{param_name}")

    def set_param(self, param_name, value):
        if param_name in self.param_constraints:
            self._set_param(param_name, value)
        else:
            raw_name = f"raw_{param_name}"
            value_tensor = torch.nn.Parameter(torch.tensor([value]))
            self.register_parameter(raw_name, value_tensor)

    def num_outputs_per_input(self, x1, x2):
        return self.num_tasks

    def forward(self, x1, x2, diag=False, **params):
{{ forward_body }}


class PCGP_Model(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood,  model_parameters, number_of_input_dimensions = {{ input_dims }}, num_tasks = {{ num_tasks }}, priors = {{ priors }}):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.ZeroMean(), num_tasks=num_tasks 
        )
        self.num_tasks = num_tasks
        self.number_of_input_dimensions = number_of_input_dimensions
        self.covar_module = PCGP_Kernel(model_parameters, number_of_input_dimensions=self.number_of_input_dimensions, num_tasks = self.num_tasks)
        
        if priors:
            for name, prior in priors.items():
                self.register_prior(
                    name+"_prior",
                    prior,
                    lambda m: m.covar_module.get_param(name),
                    lambda m, val: m.covar_module._set_param(name, val),
                )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x, interleaved = True) 
""")

def write_kernel_and_model(class_name, forward_body_code, input_dims=1, num_tasks=1,  priors=None, output_dir=None):
    """
    Writes a file class_name.py containing a GPyTorch kernel and model class for the specified forward function body. 
    
    :param class_name: file and class name for the generated kernel and model
    :param forward_body_code: str, code for the body of the forward function
    :param input_dims: int, number of input dimensions
    :param num_tasks: int, number of output tasks
    :param priors: dictionary of priors to include in the model, all parameters must be included
    :param output_dir: directory to save the generated file
    """
    rendered = KERNEL_TEMPLATE.render(
        class_name=class_name,
        forward_body=forward_body_code,
        input_dims=input_dims,
        num_tasks=num_tasks,
        priors=priors
    )
    if output_dir is None:
        caller_frame = inspect.stack()[1]
        caller_file = caller_frame.filename
        output_dir = os.path.dirname(os.path.abspath(caller_file))
    file_path = os.path.join(output_dir, f"{class_name}.py")
    with open(file_path, "w") as f:
        f.write(rendered)
    print(f"Kernel and model written to: {file_path}")
