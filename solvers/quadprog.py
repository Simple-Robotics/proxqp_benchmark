from quadprog import solve_qp
import numpy as np
from . import statuses as s
from .results import Results
from utils.general import is_qp_solution_optimal
from typing import Optional
from warnings import warn
import time

class QUADPROGSolver(object):

    STATUS_MAP = {2: s.OPTIMAL,
                  3: s.PRIMAL_INFEASIBLE,
                  5: s.DUAL_INFEASIBLE,
                  4: s.PRIMAL_OR_DUAL_INFEASIBLE,
                  6: s.SOLVER_ERROR,
                  7: s.MAX_ITER_REACHED,
                  8: s.SOLVER_ERROR,
                  9: s.TIME_LIMIT,
                  10: s.SOLVER_ERROR,
                  11: s.SOLVER_ERROR,
                  12: s.SOLVER_ERROR,
                  13: s.OPTIMAL_INACCURATE}

    def __init__(self, settings={}):
        '''
        Initialize solver object by setting require settings
        '''
        self._settings = settings

    @property
    def settings(self):
        """Solver settings"""
        return self._settings


    def solve(self, example,n_average,eps):
        """
        Adapted from the wrapper written by Stéphane Caron : https://github.com/stephane-caron/qpsolvers/blob/master/qpsolvers/solvers/quadprog_.py 
        """

        problem = example.qp_problem
        H = problem['P'].toarray()
        g = problem['q']
        ub = problem['u']
        lb = problem['l']
    
        eq_ids = lb == ub
        in_ids = lb != ub
        
        A = problem['A'][eq_ids,:].toarray()
        b = lb[eq_ids]
        C = problem['A'][in_ids,:].toarray()
        l = lb[in_ids]
        u = ub[in_ids]

        qp_G = H
        qp_a = -g
        qp_C: Optional[np.ndarray] = None
        qp_b: Optional[np.ndarray] = None
        if A is not None and b is not None:
            if C is not None and u is not None:
                qp_C = -np.vstack([A,C, -C]).T
                qp_b = -np.hstack([b, u, -l])
            else:
                qp_C = A.T
                qp_b = b
            meq = A.shape[0]
        else:  # no equality constraint
            if C is not None and u is not None:
                qp_C = -np.vstack([C, -C]).T
                qp_b = -np.hstack([u, -l])
            meq = 0
        try:         
            #qp_G = scipy.linalg.inv(scipy.linalg.cholesky(H))

            factorize = False
            tic = time.time()
            for i in range(n_average):
                x, objval, xu, niter, y, iact = solve_qp(qp_G, qp_a, qp_C, qp_b, meq, factorize )
            toc = time.time()
            run_time = (toc - tic)/n_average
            n_in = sum(in_ids)
            y_sol = np.zeros(meq+n_in)

            if (meq>0 and n_in==0):
                y_sol = -y
            if (meq>0 and n_in>0):
                y_sol[eq_ids] = y[:meq]
                y_sol[in_ids] = (-y[meq+n_in:meq+2*n_in] + y[meq:meq+n_in])

            if (meq==0 and n_in>0):
                y_sol[in_ids] = -y[meq+n_in:meq+2*n_in] + y[meq:meq+n_in]

            if not is_qp_solution_optimal(problem, x, y_sol, eps):
                status = s.SOLVER_ERROR
            # Verify solver time
                if 'time_limit' in self._settings:
                    if run_time > self._settings['time_limit']:
                        status = s.TIME_LIMIT
            else:
                status = s.OPTIMAL

                if 'time_limit' in self._settings:
                    if run_time > self._settings['time_limit']:
                        status = s.TIME_LIMIT

            return Results(status, objval, x, y_sol,
                            run_time, niter)


        except ValueError as e:
            error = str(e)
            if "matrix G is not positive definite" in error:
                # quadprog writes G the cost matrix that we write P in this package
                print("matrix P is not positive definite")
                #raise ValueError("matrix P is not positive definite") from e
                return Results(s.PRIMAL_OR_DUAL_INFEASIBLE, None, None, None,
                            None, None)
            if "no solution" in error:
                return Results(s.SOLVER_ERROR, None, None, None,
                            None, None)
            warn("quadprog raised a ValueError: {}".format(e))

            return Results(s.PRIMAL_OR_DUAL_INFEASIBLE, None, None, None,
                            None, None)

