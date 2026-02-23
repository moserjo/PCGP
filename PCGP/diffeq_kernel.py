import sympy as sp

class kernel_matrix:
    """
    Base class that calculates the symbolic kernel matrix based on a user specified parametrization matrix B and the RBF kernel in number_of_input_dimensions dimensions.
    """
     
    def __init__(self, B, parameters, number_of_input_dimensions = 1):
        """
        Parameters
        
        :param B: sympy.matrices.Matrix, user specified parametrization matrix
        :param parameters: dictionary including all parameters (as keys) in B and the base-kernel hyperparameters A (amplitude) and l (lengthscale)
        :param number_of_input_dimensions: int, number of input dimensions
        """
        self.number_of_input_dimensions = number_of_input_dimensions
        self.x = sp.symbols(f'x1:{number_of_input_dimensions + 1}')     # creates (x1, x2,..)
        self.x_ = sp.symbols(f'x1:{number_of_input_dimensions + 1}_')  # creates (x1_, x2_,...)
        A, l = sp.symbols("A, l")
        self.base_kernel = A*sp.exp(-sum((xi - xi_)**2/(2*l) for xi, xi_ in zip(self.x, self.x_)))
        self.X = sp.symbols(f'X1:{number_of_input_dimensions + 1}')
        self.Y = sp.symbols(f'Y1:{number_of_input_dimensions + 1}')
        self.B = B
        self.parameters = parameters 
        self.constraints = {}
        self.num_tasks = self.B(self.X, self.x).shape[0]

    def get_symbolic_kernel(self): 
        """
        Calculates the symbolic kernel matrix based on the parametrization matrix B and the base kernel.
        """
        operator_kernel = self.B(self.X, self.x)*self.B(self.Y, self.x_).T 
        diff_conversion = {X: x for X, x in zip(self.X+self.Y, self.x+self.x_)}
        used_differential_terms = {}
        for i in range(self.num_tasks):
            for j in range(self.num_tasks):
                symbolic_expr = sp.expand(operator_kernel[i,j])
                terms = sp.Add.make_args(symbolic_expr)
                for term in terms:
                    factors = sp.Mul.make_args(term)
                    diff_factors = []
                    needed_differentiations = []
                    for f in factors:
                        if isinstance(f, sp.Symbol) and (f in self.X or f in self.Y):
                            diff_factors.append(f)
                            needed_differentiations.append((diff_conversion[f], 1))
                        elif isinstance(f, sp.Pow):
                            base, exp = f.args
                            if base in self.X or base in self.Y:
                                diff_factors.append(f)
                                needed_differentiations.append((diff_conversion[base], exp))
                    if diff_factors:
                        op = sp.Mul(*diff_factors)
                        used_differential_terms[op] = needed_differentiations
                   
        sorted_items = sorted(used_differential_terms.items(), key=lambda item: sorted([p for _, p in item[1]], reverse=True), reverse=True)
        differential_substitutions = {}
        for key, value in sorted_items:
            differential_substitutions[key] = sp.diff(self.base_kernel, *value)
        
        #define differentials correctly:
        for i in range(self.num_tasks):
            for j in range(self.num_tasks):
                terms_of_BBT_entry = sp.matrices.Matrix(sp.Add.make_args(sp.expand(operator_kernel[i,j])))
                for l in range(len(terms_of_BBT_entry)):
                    term = terms_of_BBT_entry[l]
                    if all(str(Xi) not in str(term) for Xi in self.X) and all(str(Yi) not in str(term) for Yi in self.Y):#need to be able to handle symbols 
                            terms_of_BBT_entry[l] = terms_of_BBT_entry[l].subs(term, term*self.base_kernel)
                    else:
                        for key, value in sorted_items:
                            if str(key) in str(terms_of_BBT_entry[l]):
                                terms_of_BBT_entry[l] = terms_of_BBT_entry[l].subs(key, differential_substitutions[key])
                               

               
                operator_kernel[i,j] = sum(terms_of_BBT_entry)
        return operator_kernel

