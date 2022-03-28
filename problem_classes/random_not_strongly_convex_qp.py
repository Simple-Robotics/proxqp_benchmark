import numpy as np
import scipy.sparse as spa
import cvxpy


class RandomNotStronglyConvexQPExample(object):
    '''
    Random Mixed QP example
    '''
    def __init__(self, n,seed=1):
        '''
        Generate problem in QP format and CVXPY format
        '''
        # Set random seed
        np.random.seed(seed)

        m = int(n/2)

        # Generate problem data
        self.n = int(n)
        self.m = m
        '''
        if (n<=100):
            P = spa.random(n, n, density=0.025,
                        data_rvs=np.random.randn,
                        format='csc')
        elif (n>100 and n<=400):
            P = spa.random(n, n, density=0.015,
                        data_rvs=np.random.randn,
                        format='csc')
        else:
            P = spa.random(n, n, density=0.0125,
                        data_rvs=np.random.randn,
                        format='csc')
        self.P = P.dot(P.T).tocsc()

        '''
        P = spa.random(n, n, density=0.075,
                       data_rvs=np.random.randn,
                       format='csc').toarray()
        P = (P+P.T)/2.         
        s = min(np.linalg.eigvals(P)) 
        if (s<0.):
            self.P = spa.coo_matrix(P) + abs(s) * spa.eye(n) # get eigenvalue = 0 to be not strictly convex
        elif (s==0.):
            self.P = spa.coo_matrix(P) # already not strictly convex
        else:
            self.P = spa.coo_matrix(P) - abs(s) * spa.eye(n)
        '''
        P = spa.random(n, n, density=0.025,
                       data_rvs=np.random.randn,
                       format='csc').toarray()
        P = (P+P.T)/2. 

        
        U,S,Ut = np.linalg.svd(P, full_matrices=True,hermitian=True)
        S = np.maximum(S,0.) # to ensure the matrix is psd and not pd
        
        P = U.dot(np.diag(S).dot(U.T))
        print("S : {}".format(S))
        P = (P+P.T)/2. 
        '''
        #self.P = spa.coo_matrix(P)

        self.A = spa.random(m, n, density=0.15,
                            data_rvs=np.random.randn,
                            format='csc')
        #print("min(np.linalg.eigvals(P)) : {}".format(min(np.linalg.eigvals((self.P).toarray()))))
        print("sparsity of P : {}".format((self.P.nnz)/(n**2)))
        #input()
        v = np.random.randn(n)   # Fictitious solution
        delta = np.random.rand(m)  # To get inequality

        sol_dual = np.random.randn(m)
        sol = self.A@v

        self.q = - (self.P @ v + self.A.T @ sol_dual) # make sure a solution exists

        # if the constraint is a box the problem won't be dually infeasible (otherwise it can be possible to find a feasible direction s.t Px = 0 and qTx <0)
        self.u = sol + delta
        self.l = sol - delta

        self.qp_problem = self._generate_qp_problem()
        self.cvxpy_problem = self._generate_cvxpy_problem()

    @staticmethod
    def name():
        return 'Random Not Strongly Convex QP'

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
