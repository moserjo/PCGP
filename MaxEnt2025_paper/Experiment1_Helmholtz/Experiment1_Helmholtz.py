
import torch
import gpytorch
from einops import rearrange
from PCGP.constraint_handling import ConstraintsModifications

class PCGP_Kernel(gpytorch.kernels.Kernel):
    def __init__(self, input_parameters, number_of_input_dimensions=2, num_tasks=3, **kwargs):
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
        if x1.dim() == 1:
            mesh = torch.meshgrid(x1.flatten(), x2.flatten(), indexing='xy')
            xx, yy = mesh[0].T.unsqueeze(0), mesh[1].T.unsqueeze(0)
        elif x1.dim() == 2 and x1.shape[1] == self.number_of_input_dimensions:
            xx = torch.zeros((self.number_of_input_dimensions, x1.shape[0], x2.shape[0]), device=x1.device)
            yy = torch.zeros((self.number_of_input_dimensions, x1.shape[0], x2.shape[0]), device=x1.device)
            for i in range(self.number_of_input_dimensions):
                mesh = torch.meshgrid(torch.squeeze(x1[:,i]), torch.squeeze(x2[:,i]), indexing='xy')
                xx[i] = mesh[0].T
                yy[i] = mesh[1].T
        A = self.get_param('A')
        l = self.get_param('l')
        k00 = A*(l**4 + 2*l**2*(-2*l + (xx[0] - yy[0])**2 + (xx[1] - yy[1])**2) + 4*l**2 - 4*l*(xx[0] - yy[0])**2 - 4*l*(xx[1] - yy[1])**2 + (l - (xx[0] - yy[0])**2)**2 + 2*(l - (xx[0] - yy[0])**2)*(l - (xx[1] - yy[1])**2) + (l - (xx[1] - yy[1])**2)**2)*torch.exp(-1/2*((xx[0] - yy[0])**2 + (xx[1] - yy[1])**2)/l)/l**4
        k01 = 10*A*(-l**2 + 2*l - (xx[0] - yy[0])**2 - (xx[1] - yy[1])**2)*torch.exp(-1/2*((xx[0] - yy[0])**2 + (xx[1] - yy[1])**2)/l)/l**2
        k02 = A*(10*l**6*(-50*yy[1]**3 - 150*yy[1]**2 - 150*yy[1] + 3*torch.cos(2*yy[0]) - 50) + l**4*(998*l + 500*yy[1]**3*(l - (xx[0] - yy[0])**2) + 500*yy[1]**3*(l - (xx[1] - yy[1])**2) + 1500*yy[1]**2*(l - (xx[0] - yy[0])**2) + 1500*yy[1]**2*(l - (xx[1] - yy[1])**2) + 1500*yy[1]*(l - (xx[0] - yy[0])**2) + 1500*yy[1]*(l - (xx[1] - yy[1])**2) + 60*(-l + (xx[0] - yy[0])**2)*torch.cos(2*yy[0]) + 60*(-l + (xx[1] - yy[1])**2)*torch.cos(2*yy[0]) - 499*(xx[0] - yy[0])**2 - 499*(xx[1] - yy[1])**2) - 28*l**3 + 4*l**2*(l - (xx[0] - yy[0])**2)*(l - (xx[1] - yy[1])**2)*(15*torch.cos(2*yy[0]) + 1) + 12*l**2*(xx[0] - yy[0])**2 + 12*l**2*(xx[1] - yy[1])**2 + 4*l**2*(2*(xx[0] - yy[0])**2 + (xx[1] - yy[1])**2) + l**2*(10*l**2 - 10*l*(xx[0] - yy[0])**2 - 10*l*(xx[1] - yy[1])**2 + (l - (xx[0] - yy[0])**2)**2 + (l - (xx[1] - yy[1])**2)**2 + (xx[0] - yy[0])**4 + (xx[1] - yy[1])**4 + 30*(2*l**2 - 4*l*(xx[0] - yy[0])**2 + (l - (xx[0] - yy[0])**2)**2)*torch.cos(2*yy[0]) + 30*(2*l**2 - 4*l*(xx[1] - yy[1])**2 + (l - (xx[1] - yy[1])**2)**2)*torch.cos(2*yy[0])) - 8*l*(-3*l + (xx[0] - yy[0])**2)*(xx[0] - yy[0])**2 - 8*l*(-3*l + (xx[1] - yy[1])**2)*(xx[1] - yy[1])**2 - 2*l*((l - (xx[0] - yy[0])**2)**2 + 4*(xx[0] - yy[0])**2*(xx[1] - yy[1])**2) + 2*(l - (xx[0] - yy[0])**2)**2*(xx[1] - yy[1])**2 - 2*(l - (xx[0] - yy[0])**2)*(2*l**2 - 4*l*(xx[1] - yy[1])**2 + (l - (xx[1] - yy[1])**2)**2) - (l - (xx[0] - yy[0])**2)*(3*l**2 - 6*l*(xx[0] - yy[0])**2 + (xx[0] - yy[0])**4) - (l - (xx[0] - yy[0])**2)*(3*l**2 - 6*l*(xx[1] - yy[1])**2 + (xx[1] - yy[1])**4) - (l - (xx[1] - yy[1])**2)*(3*l**2 - 6*l*(xx[0] - yy[0])**2 + (xx[0] - yy[0])**4) - (l - (xx[1] - yy[1])**2)*(3*l**2 - 6*l*(xx[1] - yy[1])**2 + (xx[1] - yy[1])**4))*torch.exp(-1/2*((xx[0] - yy[0])**2 + (xx[1] - yy[1])**2)/l)/l**6
        k10 = 10*A*(-l**2 + 2*l - (xx[0] - yy[0])**2 - (xx[1] - yy[1])**2)*torch.exp(-1/2*((xx[0] - yy[0])**2 + (xx[1] - yy[1])**2)/l)/l**2
        k11 = 100*A*torch.exp(-1/2*((xx[0] - yy[0])**2 + (xx[1] - yy[1])**2)/l)
        k12 = 10*A*(10*l**4*(50*yy[1]**3 + 150*yy[1]**2 + 150*yy[1] - 3*torch.cos(2*yy[0]) + 50) + l**2*(2*l + 30*(l - (xx[0] - yy[0])**2)*torch.cos(2*yy[0]) + 30*(l - (xx[1] - yy[1])**2)*torch.cos(2*yy[0]) - (xx[0] - yy[0])**2 - (xx[1] - yy[1])**2) - 6*l**2 + 6*l*(xx[0] - yy[0])**2 + 6*l*(xx[1] - yy[1])**2 - 2*(l - (xx[0] - yy[0])**2)*(l - (xx[1] - yy[1])**2) - (xx[0] - yy[0])**4 - (xx[1] - yy[1])**4)*torch.exp(-1/2*((xx[0] - yy[0])**2 + (xx[1] - yy[1])**2)/l)/l**4
        k20 = A*(10*l**6*(-50*xx[1]**3 - 150*xx[1]**2 - 150*xx[1] + 3*torch.cos(2*xx[0]) - 50) + l**4*(998*l + 500*xx[1]**3*(l - (xx[0] - yy[0])**2) + 500*xx[1]**3*(l - (xx[1] - yy[1])**2) + 1500*xx[1]**2*(l - (xx[0] - yy[0])**2) + 1500*xx[1]**2*(l - (xx[1] - yy[1])**2) + 1500*xx[1]*(l - (xx[0] - yy[0])**2) + 1500*xx[1]*(l - (xx[1] - yy[1])**2) + 60*(-l + (xx[0] - yy[0])**2)*torch.cos(2*xx[0]) + 60*(-l + (xx[1] - yy[1])**2)*torch.cos(2*xx[0]) - 499*(xx[0] - yy[0])**2 - 499*(xx[1] - yy[1])**2) - 28*l**3 + 4*l**2*(l - (xx[0] - yy[0])**2)*(l - (xx[1] - yy[1])**2)*(15*torch.cos(2*xx[0]) + 1) + 12*l**2*(xx[0] - yy[0])**2 + 12*l**2*(xx[1] - yy[1])**2 + 4*l**2*(2*(xx[0] - yy[0])**2 + (xx[1] - yy[1])**2) + l**2*(10*l**2 - 10*l*(xx[0] - yy[0])**2 - 10*l*(xx[1] - yy[1])**2 + (l - (xx[0] - yy[0])**2)**2 + (l - (xx[1] - yy[1])**2)**2 + (xx[0] - yy[0])**4 + (xx[1] - yy[1])**4 + 30*(2*l**2 - 4*l*(xx[0] - yy[0])**2 + (l - (xx[0] - yy[0])**2)**2)*torch.cos(2*xx[0]) + 30*(2*l**2 - 4*l*(xx[1] - yy[1])**2 + (l - (xx[1] - yy[1])**2)**2)*torch.cos(2*xx[0])) - 8*l*(-3*l + (xx[0] - yy[0])**2)*(xx[0] - yy[0])**2 - 8*l*(-3*l + (xx[1] - yy[1])**2)*(xx[1] - yy[1])**2 - 2*l*((l - (xx[0] - yy[0])**2)**2 + 4*(xx[0] - yy[0])**2*(xx[1] - yy[1])**2) + 2*(l - (xx[0] - yy[0])**2)**2*(xx[1] - yy[1])**2 - 2*(l - (xx[0] - yy[0])**2)*(2*l**2 - 4*l*(xx[1] - yy[1])**2 + (l - (xx[1] - yy[1])**2)**2) - (l - (xx[0] - yy[0])**2)*(3*l**2 - 6*l*(xx[0] - yy[0])**2 + (xx[0] - yy[0])**4) - (l - (xx[0] - yy[0])**2)*(3*l**2 - 6*l*(xx[1] - yy[1])**2 + (xx[1] - yy[1])**4) - (l - (xx[1] - yy[1])**2)*(3*l**2 - 6*l*(xx[0] - yy[0])**2 + (xx[0] - yy[0])**4) - (l - (xx[1] - yy[1])**2)*(3*l**2 - 6*l*(xx[1] - yy[1])**2 + (xx[1] - yy[1])**4))*torch.exp(-1/2*((xx[0] - yy[0])**2 + (xx[1] - yy[1])**2)/l)/l**6
        k21 = 10*A*(10*l**4*(50*xx[1]**3 + 150*xx[1]**2 + 150*xx[1] - 3*torch.cos(2*xx[0]) + 50) + l**2*(2*l + 30*(l - (xx[0] - yy[0])**2)*torch.cos(2*xx[0]) + 30*(l - (xx[1] - yy[1])**2)*torch.cos(2*xx[0]) - (xx[0] - yy[0])**2 - (xx[1] - yy[1])**2) - 6*l**2 + 6*l*(xx[0] - yy[0])**2 + 6*l*(xx[1] - yy[1])**2 - 2*(l - (xx[0] - yy[0])**2)*(l - (xx[1] - yy[1])**2) - (xx[0] - yy[0])**4 - (xx[1] - yy[1])**4)*torch.exp(-1/2*((xx[0] - yy[0])**2 + (xx[1] - yy[1])**2)/l)/l**4
        k22 = A*(100*l**8*(2500*xx[1]**3*yy[1]**3 + 7500*xx[1]**3*yy[1]**2 + 7500*xx[1]**3*yy[1] - 150*xx[1]**3*torch.cos(2*yy[0]) + 2500*xx[1]**3 + 7500*xx[1]**2*yy[1]**3 + 22500*xx[1]**2*yy[1]**2 + 22500*xx[1]**2*yy[1] - 450*xx[1]**2*torch.cos(2*yy[0]) + 7500*xx[1]**2 + 7500*xx[1]*yy[1]**3 + 22500*xx[1]*yy[1]**2 + 22500*xx[1]*yy[1] - 450*xx[1]*torch.cos(2*yy[0]) + 7500*xx[1] - 150*yy[1]**3*torch.cos(2*xx[0]) + 2500*yy[1]**3 - 450*yy[1]**2*torch.cos(2*xx[0]) + 7500*yy[1]**2 - 450*yy[1]*torch.cos(2*xx[0]) + 7500*yy[1] + 9*torch.cos(2*xx[0])*torch.cos(2*yy[0]) - 150*torch.cos(2*xx[0]) - 150*torch.cos(2*yy[0]) + 2500) + 10*l**6*(200*l + 1500*xx[1]**3*(l - (xx[0] - yy[0])**2)*torch.cos(2*yy[0]) + 50*xx[1]**3*(l - (xx[0] - yy[0])**2) + 1500*xx[1]**3*(l - (xx[1] - yy[1])**2)*torch.cos(2*yy[0]) + 50*xx[1]**3*(l - (xx[1] - yy[1])**2) + 4500*xx[1]**2*(l - (xx[0] - yy[0])**2)*torch.cos(2*yy[0]) + 150*xx[1]**2*(l - (xx[0] - yy[0])**2) + 4500*xx[1]**2*(l - (xx[1] - yy[1])**2)*torch.cos(2*yy[0]) + 150*xx[1]**2*(l - (xx[1] - yy[1])**2) + 4500*xx[1]*(l - (xx[0] - yy[0])**2)*torch.cos(2*yy[0]) + 150*xx[1]*(l - (xx[0] - yy[0])**2) + 4500*xx[1]*(l - (xx[1] - yy[1])**2)*torch.cos(2*yy[0]) + 150*xx[1]*(l - (xx[1] - yy[1])**2) + 1500*yy[1]**3*(l - (xx[0] - yy[0])**2)*torch.cos(2*xx[0]) + 50*yy[1]**3*(l - (xx[0] - yy[0])**2) + 1500*yy[1]**3*(l - (xx[1] - yy[1])**2)*torch.cos(2*xx[0]) + 50*yy[1]**3*(l - (xx[1] - yy[1])**2) + 4500*yy[1]**2*(l - (xx[0] - yy[0])**2)*torch.cos(2*xx[0]) + 150*yy[1]**2*(l - (xx[0] - yy[0])**2) + 4500*yy[1]**2*(l - (xx[1] - yy[1])**2)*torch.cos(2*xx[0]) + 150*yy[1]**2*(l - (xx[1] - yy[1])**2) + 4500*yy[1]*(l - (xx[0] - yy[0])**2)*torch.cos(2*xx[0]) + 150*yy[1]*(l - (xx[0] - yy[0])**2) + 4500*yy[1]*(l - (xx[1] - yy[1])**2)*torch.cos(2*xx[0]) + 150*yy[1]*(l - (xx[1] - yy[1])**2) + 180*(-l + (xx[0] - yy[0])**2)*torch.cos(2*xx[0])*torch.cos(2*yy[0]) + 180*(-l + (xx[1] - yy[1])**2)*torch.cos(2*xx[0])*torch.cos(2*yy[0]) + 1497*(l - (xx[0] - yy[0])**2)*torch.cos(2*xx[0]) + 1497*(l - (xx[0] - yy[0])**2)*torch.cos(2*yy[0]) + 1497*(l - (xx[1] - yy[1])**2)*torch.cos(2*xx[0]) + 1497*(l - (xx[1] - yy[1])**2)*torch.cos(2*yy[0]) - 100*(xx[0] - yy[0])**2 - 100*(xx[1] - yy[1])**2) + 2*l**4*(l - (xx[0] - yy[0])**2)*(l - (xx[1] - yy[1])**2)*(-500*xx[1]**3 - 1500*xx[1]**2 - 1500*xx[1] - 500*yy[1]**3 - 1500*yy[1]**2 - 1500*yy[1] + 900*torch.cos(2*xx[0])*torch.cos(2*yy[0]) + 60*torch.cos(2*xx[0]) + 60*torch.cos(2*yy[0]) - 999) + l**4*(-5996*l**2 + 5996*l*(xx[0] - yy[0])**2 + 5996*l*(xx[1] - yy[1])**2 - 500*xx[1]**3*(3*l**2 - 6*l*(xx[0] - yy[0])**2 + (xx[0] - yy[0])**4) - 500*xx[1]**3*(3*l**2 - 6*l*(xx[1] - yy[1])**2 + (xx[1] - yy[1])**4) - 1500*xx[1]**2*(3*l**2 - 6*l*(xx[0] - yy[0])**2 + (xx[0] - yy[0])**4) - 1500*xx[1]**2*(3*l**2 - 6*l*(xx[1] - yy[1])**2 + (xx[1] - yy[1])**4) - 1500*xx[1]*(3*l**2 - 6*l*(xx[0] - yy[0])**2 + (xx[0] - yy[0])**4) - 1500*xx[1]*(3*l**2 - 6*l*(xx[1] - yy[1])**2 + (xx[1] - yy[1])**4) - 500*yy[1]**3*(3*l**2 - 6*l*(xx[0] - yy[0])**2 + (xx[0] - yy[0])**4) - 500*yy[1]**3*(3*l**2 - 6*l*(xx[1] - yy[1])**2 + (xx[1] - yy[1])**4) - 1500*yy[1]**2*(3*l**2 - 6*l*(xx[0] - yy[0])**2 + (xx[0] - yy[0])**4) - 1500*yy[1]**2*(3*l**2 - 6*l*(xx[1] - yy[1])**2 + (xx[1] - yy[1])**4) - 1500*yy[1]*(3*l**2 - 6*l*(xx[0] - yy[0])**2 + (xx[0] - yy[0])**4) - 1500*yy[1]*(3*l**2 - 6*l*(xx[1] - yy[1])**2 + (xx[1] - yy[1])**4) + (l - (xx[0] - yy[0])**2)**2 + (l - (xx[1] - yy[1])**2)**2 - 1000*(xx[0] - yy[0])**4 - 1000*(xx[1] - yy[1])**4 + 900*(2*l**2 - 4*l*(xx[0] - yy[0])**2 + (l - (xx[0] - yy[0])**2)**2)*torch.cos(2*xx[0])*torch.cos(2*yy[0]) + 30*(2*l**2 - 4*l*(xx[0] - yy[0])**2 + (l - (xx[0] - yy[0])**2)**2)*torch.cos(2*xx[0]) + 30*(2*l**2 - 4*l*(xx[0] - yy[0])**2 + (l - (xx[0] - yy[0])**2)**2)*torch.cos(2*yy[0]) + 900*(2*l**2 - 4*l*(xx[1] - yy[1])**2 + (l - (xx[1] - yy[1])**2)**2)*torch.cos(2*xx[0])*torch.cos(2*yy[0]) + 30*(2*l**2 - 4*l*(xx[1] - yy[1])**2 + (l - (xx[1] - yy[1])**2)**2)*torch.cos(2*xx[0]) + 30*(2*l**2 - 4*l*(xx[1] - yy[1])**2 + (l - (xx[1] - yy[1])**2)**2)*torch.cos(2*yy[0]) + 30*(3*l**2 - 6*l*(xx[0] - yy[0])**2 + (xx[0] - yy[0])**4)*torch.cos(2*xx[0]) + 30*(3*l**2 - 6*l*(xx[0] - yy[0])**2 + (xx[0] - yy[0])**4)*torch.cos(2*yy[0]) + 30*(3*l**2 - 6*l*(xx[1] - yy[1])**2 + (xx[1] - yy[1])**4)*torch.cos(2*xx[0]) + 30*(3*l**2 - 6*l*(xx[1] - yy[1])**2 + (xx[1] - yy[1])**4)*torch.cos(2*yy[0])) + 120*l**4 - 144*l**3*(xx[0] - yy[0])**2 - 96*l**3*(xx[1] - yy[1])**2 - 48*l**3*((xx[0] - yy[0])**2 + (xx[1] - yy[1])**2) + 72*l**2*(l - (xx[0] - yy[0])**2)**2 + 72*l**2*(l - (xx[1] - yy[1])**2)**2 - 16*l**2*(3*(l - (xx[0] - yy[0])**2)*(xx[1] - yy[1])**2 + 2*(3*l - (xx[0] - yy[0])**2)*(xx[0] - yy[0])**2) + 4*l**2*(3*(l - (xx[0] - yy[0])**2)**2 + 24*(xx[0] - yy[0])**2*(xx[1] - yy[1])**2 + 2*(xx[1] - yy[1])**4) + 4*l**2*(30*torch.sin(xx[0])**2 + 30*torch.sin(yy[0])**2 - 31)*(2*l**3 - 2*l**2*(2*(xx[0] - yy[0])**2 + (xx[1] - yy[1])**2) + l*((l - (xx[0] - yy[0])**2)**2 + 4*(xx[0] - yy[0])**2*(xx[1] - yy[1])**2) - (l - (xx[0] - yy[0])**2)**2*(xx[1] - yy[1])**2) - 30*l**2*(12*l**3 - 12*l**2*(xx[0] - yy[0])**2 + 8*l*(-3*l + (xx[0] - yy[0])**2)*(xx[0] - yy[0])**2 + (l - (xx[0] - yy[0])**2)*(3*l**2 - 6*l*(xx[0] - yy[0])**2 + (xx[0] - yy[0])**4))*torch.cos(2*xx[0]) - 30*l**2*(12*l**3 - 12*l**2*(xx[0] - yy[0])**2 + 8*l*(-3*l + (xx[0] - yy[0])**2)*(xx[0] - yy[0])**2 + (l - (xx[0] - yy[0])**2)*(3*l**2 - 6*l*(xx[0] - yy[0])**2 + (xx[0] - yy[0])**4))*torch.cos(2*yy[0]) - 2*l**2*(12*l**3 - 12*l**2*(xx[0] - yy[0])**2 + 8*l*(-3*l + (xx[0] - yy[0])**2)*(xx[0] - yy[0])**2 + (l - (xx[0] - yy[0])**2)*(3*l**2 - 6*l*(xx[0] - yy[0])**2 + (xx[0] - yy[0])**4)) - 30*l**2*(12*l**3 - 12*l**2*(xx[1] - yy[1])**2 + 8*l*(-3*l + (xx[1] - yy[1])**2)*(xx[1] - yy[1])**2 + (l - (xx[1] - yy[1])**2)*(3*l**2 - 6*l*(xx[1] - yy[1])**2 + (xx[1] - yy[1])**4))*torch.cos(2*xx[0]) - 30*l**2*(12*l**3 - 12*l**2*(xx[1] - yy[1])**2 + 8*l*(-3*l + (xx[1] - yy[1])**2)*(xx[1] - yy[1])**2 + (l - (xx[1] - yy[1])**2)*(3*l**2 - 6*l*(xx[1] - yy[1])**2 + (xx[1] - yy[1])**4))*torch.cos(2*yy[0]) - 2*l**2*(12*l**3 - 12*l**2*(xx[1] - yy[1])**2 + 8*l*(-3*l + (xx[1] - yy[1])**2)*(xx[1] - yy[1])**2 + (l - (xx[1] - yy[1])**2)*(3*l**2 - 6*l*(xx[1] - yy[1])**2 + (xx[1] - yy[1])**4)) - 2*l**2*(30*(l - (xx[0] - yy[0])**2)*(2*l**2 - 4*l*(xx[1] - yy[1])**2 + (l - (xx[1] - yy[1])**2)**2)*torch.cos(2*xx[0]) + 30*(l - (xx[0] - yy[0])**2)*(2*l**2 - 4*l*(xx[1] - yy[1])**2 + (l - (xx[1] - yy[1])**2)**2)*torch.cos(2*yy[0]) + 2*(l - (xx[0] - yy[0])**2)*(2*l**2 - 4*l*(xx[1] - yy[1])**2 + (l - (xx[1] - yy[1])**2)**2) + 15*(l - (xx[0] - yy[0])**2)*(3*l**2 - 6*l*(xx[1] - yy[1])**2 + (xx[1] - yy[1])**4)*torch.cos(2*xx[0]) + 15*(l - (xx[0] - yy[0])**2)*(3*l**2 - 6*l*(xx[1] - yy[1])**2 + (xx[1] - yy[1])**4)*torch.cos(2*yy[0]) + (l - (xx[0] - yy[0])**2)*(3*l**2 - 6*l*(xx[1] - yy[1])**2 + (xx[1] - yy[1])**4) + 15*(l - (xx[1] - yy[1])**2)*(3*l**2 - 6*l*(xx[0] - yy[0])**2 + (xx[0] - yy[0])**4)*torch.cos(2*xx[0]) + 15*(l - (xx[1] - yy[1])**2)*(3*l**2 - 6*l*(xx[0] - yy[0])**2 + (xx[0] - yy[0])**4)*torch.cos(2*yy[0]) + (l - (xx[1] - yy[1])**2)*(3*l**2 - 6*l*(xx[0] - yy[0])**2 + (xx[0] - yy[0])**4)) - 32*l*(-3*l + (xx[0] - yy[0])**2)*(xx[0] - yy[0])**2*(xx[1] - yy[1])**2 + 4*l*(l - (xx[0] - yy[0])**2)*(3*l**2 - 6*l*(xx[0] - yy[0])**2 + (xx[0] - yy[0])**4) - 16*l*(3*l - (xx[0] - yy[0])**2)**2*(xx[0] - yy[0])**2 - 16*l*(3*l - (xx[1] - yy[1])**2)**2*(xx[1] - yy[1])**2 - 8*l*(xx[1] - yy[1])**2*(3*(l - (xx[0] - yy[0])**2)**2 + 2*(xx[0] - yy[0])**2*(xx[1] - yy[1])**2) + 4*(l - (xx[0] - yy[0])**2)**2*(xx[1] - yy[1])**4 - 4*(l - (xx[0] - yy[0])**2)*(xx[1] - yy[1])**2*(3*l**2 - 6*l*(xx[0] - yy[0])**2 + (xx[0] - yy[0])**4) + 4*(l - (xx[0] - yy[0])**2)*(12*l**3 - 12*l**2*(xx[1] - yy[1])**2 + 8*l*(-3*l + (xx[1] - yy[1])**2)*(xx[1] - yy[1])**2 + (l - (xx[1] - yy[1])**2)*(3*l**2 - 6*l*(xx[1] - yy[1])**2 + (xx[1] - yy[1])**4)) + (3*l**2 - 6*l*(xx[0] - yy[0])**2 + (xx[0] - yy[0])**4)**2 + 2*(3*l**2 - 6*l*(xx[0] - yy[0])**2 + (xx[0] - yy[0])**4)*(3*l**2 - 6*l*(xx[1] - yy[1])**2 + (xx[1] - yy[1])**4) + (3*l**2 - 6*l*(xx[1] - yy[1])**2 + (xx[1] - yy[1])**4)**2)*torch.exp(-1/2*((xx[0] - yy[0])**2 + (xx[1] - yy[1])**2)/l)/l**8
        cov_m = torch.squeeze(torch.cat([
    torch.cat([k00, k01, k02], dim=-1),
    torch.cat([k10, k11, k12], dim=-1),
    torch.cat([k20, k21, k22], dim=-1)], dim=-2))
        cov_f = rearrange(cov_m, "(t1 w1) (t2 w2)-> (w1 t1) (w2 t2)", t1=3, t2=3)
        return torch.diag(cov_f) if diag else cov_f


class PCGP_Model(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood,  model_parameters, number_of_input_dimensions = 2, num_tasks = 3, priors = None):
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