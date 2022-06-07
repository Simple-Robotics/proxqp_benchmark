import numpy as np
import scipy.sparse as spa
import cvxpy
import sklearn.datasets as skd
from scipy.stats import ortho_group


class RandomMixedQPExample(object):
    '''
    Random Mixed QP example
    '''
    def __init__(self, n,seed=1):
        '''
        Generate problem in QP format and CVXPY format
        '''
        np.random.seed(seed)

        m =  int(n/4) + int(n/4)
        #m  = n 
        n_eq = int(n/4)
        n_in = int(n/4)

        # Generate problem data
        self.n = int(n)
        self.m = m
        
        P = spa.random(n, n, density=0.075,
                       data_rvs=np.random.randn,
                       format='csc').toarray()
        P = (P+P.T)/2.  
     
        s = max(np.absolute(np.linalg.eigvals(P)))
        self.P = spa.coo_matrix(P) + (abs(s)+1e-02) * spa.eye(n) # to be sure being strictly convex
        print("sparsity of P : {}".format((self.P.nnz)/(n**2)))
        self.q = np.random.randn(n)
        self.A = spa.random(m, n, density=0.15,data_rvs=np.random.randn,format='csc')
        v = np.random.randn(n)   # Fictitious solution
        delta = np.random.rand(m)  # To get inequality
        self.u = self.A@v 

        self.l = (- np.inf * np.ones(m)) 
        
        self.u[n_in:] += delta[n_in:]
        self.l[:n_eq] = self.u[:n_eq]
        
        self.qp_problem = self._generate_qp_problem()
        self.cvxpy_problem = self._generate_cvxpy_problem()

    @staticmethod
    def name():
        return 'Random Mixed QP'

    def _generate_qp_problem(self):
        '''
        Generate QP problem
        '''
        problem = {}
        problem['P'] = self.P
        problem['q'] = self.q
        problem['A'] = self.A
        problem['l'] = self.l
        problem['u'] = self.u
        problem['m'] = self.A.shape[0]
        problem['n'] = self.A.shape[1]

        return problem

    def _generate_cvxpy_problem(self):
        '''
        Generate QP problem
        '''
        x_var = cvxpy.Variable(self.n)
        objective = .5 * cvxpy.quad_form(x_var, self.P) + self.q * x_var
        constraints = [self.A * x_var <= self.u, self.A * x_var >= self.l]
        problem = cvxpy.Problem(cvxpy.Minimize(objective), constraints)

        return problem

    def revert_cvxpy_solution(self):
        '''
        Get QP primal and duar variables from cvxpy solution
        '''

        variables = self.cvxpy_problem.variables()
        constraints = self.cvxpy_problem.constraints

        # primal solution
        x = variables[0].value

        # dual solution
        y = constraints[0].dual_value - constraints[1].dual_value

        return x, y
