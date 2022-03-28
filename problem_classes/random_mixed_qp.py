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
        # Set random seed
        np.random.seed(seed)

        m = 2 * int(n/4)

        n_eq = int(n/4)
        n_in = int(n/4)

        # Generate problem data
        self.n = int(n)
        self.m = m
        
        '''
        P = spa.random(n, n, density=0.0125,
                       data_rvs=np.random.randn,
                       format='csc')
        self.P = P.dot(P.T).tocsc() + 1e-02 * spa.eye(n)
        '''

        ''' other idea but gives full dense P...
        s = np.random.exponential(size = n)
        print("s : {}".format(s))
        for i in range(n):
            u = np.random.rand()
            print("u : {}".format(u))
            input()
            if (u>0.15):
                s[i] = 0.
        o = ortho_group.rvs(n)
        print("s : {}".format(s))
        P = o.dot(np.diag(s).dot(o.T)) + 1e-02 * spa.eye(n)
        #P = (P+P.T)/2. 
        print("P : {}".format(P))
        self.P = spa.coo_matrix(P)
        '''
        P = spa.random(n, n, density=0.075,
                       data_rvs=np.random.randn,
                       format='csc').toarray()
        P = (P+P.T)/2.  
        s = min(np.linalg.eigvals(P))        
        #s = max(np.absolute(np.linalg.eigvals(P)))
        self.P = spa.coo_matrix(P) + (abs(s)+1e-02) * spa.eye(n) # to be sure being strictly convex

        #self.P = spa.coo_matrix(skd.make_sparse_spd_matrix(dim=n, alpha=1. - 0.15))
        #P = (P+P.T)/2. 
        #U,S,Ut = np.linalg.svd(P.toarray(), full_matrices=True)
        #S = np.maximum(S,0.) + 1e-02 # ensure matrix is positive definite
        #print("s : {}".format(S))
        #P = U @ np.diag(S) @ Ut
        #P = (P+P.T)/2. 
        #self.P = spa.coo_matrix(P)

        print("sparsity of P : {}".format((self.P.nnz)/(n**2)))
        #input()
        self.q = np.random.randn(n)
        self.A = spa.random(m, n, density=0.15,
                            data_rvs=np.random.randn,
                            format='csc')
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
