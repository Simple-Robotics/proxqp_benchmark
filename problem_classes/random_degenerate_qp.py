import numpy as np
import scipy.sparse as spa
import cvxpy


class RandomDegenerateQPExample(object):
    '''
    Random Degenerate QP example
    '''
    def __init__(self, n, seed=1):
        '''
        Generate problem in QP format and CVXPY format
        '''
        # Set random seed
        np.random.seed(seed)

        n_in = int(n/4)
        m = 2 * n_in
        

        # Generate problem data
        self.n = int(n)
        self.m = m
        P = spa.random(n, n, density=0.15,
                       data_rvs=np.random.randn,
                       format='csc')
        self.P = P.dot(P.T).tocsc() + 1e-2 * spa.eye(n)
        self.q = np.random.randn(n)
        C = spa.random(n_in, n, density=0.15,
                            data_rvs=np.random.randn,
                            format='csc')
        # make sure the matrix rank deficient
        self.A = spa.csc_matrix(np.vstack([C.toarray(),C.toarray()]))
        
        v = np.random.randn(n)   # Fictitious solution
        delta = np.random.rand(m)  # To get inequality

        sol = self.A@v

        self.u = sol + delta
        self.l =  - np.inf * np.ones(m) 

        self.qp_problem = self._generate_qp_problem()
        self.cvxpy_problem = self._generate_cvxpy_problem()

    @staticmethod
    def name():
        return 'Random Degenerate QP'

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
