#import proxsuite 
import proxsuite_pywrap as proxsuite
import numpy as np
from . import statuses as s
from .results import Results
from utils.general import is_qp_solution_optimal
import math 
import scipy
import numpy

def normInf(x):
  if x.shape[0] == 0:
    return 0.
  else:
    return np.linalg.norm(x,np.inf)


class PROXQPSolver(object):

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

    def solve(self, example, n_average, eps):
        '''
        Solve problem

        Args:
            example: example object

        Returns:
            Results structure
        '''
        problem = example.qp_problem

        # preprocessing
        ub = np.copy(problem['u'])
        lb = np.copy(problem['l'])

        eq_ids = lb == ub
        in_ids = lb != ub

        PlusInfId = ub == math.inf 
        NegInfId = lb == -math.inf 

        ub[PlusInfId] = 1.e20
        lb[NegInfId] = -1.e20

        H = problem['P']
        g = problem['q']
        A = problem['A'][eq_ids,:]
        b = lb[eq_ids]
        C = problem['A'][in_ids,:]
        l = lb[in_ids]
        u = ub[in_ids]

        n = H.shape[0]
        n_eq = A.shape[0]
        n_in = C.shape[0]

        if (self._settings['dense']):
            Qp = proxsuite.qp.dense.QP(n,n_eq,n_in)
            if (self._settings['bcl_update']):
                Qp.settings.bcl_update= True
            else:
                Qp.settings.bcl_update= False
        else:
            #Qp = proxsuite.qp.sparse.QP(n,n_eq,n_in)
            H_ = H!=0.
            A_ = A!=0.
            C_ = C!=0.
            Qp = proxsuite.qp.sparse.QP(H_,A_,C_)
            
        Qp.settings.eps_abs = self._settings['eps_abs']
        Qp.settings.eps_rel = self._settings['eps_rel']
        Qp.settings.verbose = self._settings['verbose'] 
        Qp.settings.initial_guess = self._settings["initial_guess"]

        Qp.init(
                H,
                np.asfortranarray(g),
                A,
                np.asfortranarray(b),
                C,
                np.asfortranarray(u),
                np.asfortranarray(l)
        )
        run_time = 0
        n_solving = n_average
        for i in range(n_solving):
            Qp.solve()
            run_time += Qp.results.info.solve_time + Qp.results.info.setup_time
            #Qp.update(update_preconditioner = True)
        Qp.solve()
        run_time += Qp.results.info.solve_time + Qp.results.info.setup_time
        n_solving += 1
    
        # duration time
        run_time /= (1.e6 * n_solving) # the run_time is measured in microseconds 

        # Obj val
        objval = Qp.results.info.objValue

        # Total Number of iterations
        niter = Qp.results.info.iter

        # Get solution
        x = Qp.results.x 
        y = np.zeros(n_eq+n_in)

        y[eq_ids] = Qp.results.y
        y[in_ids] = Qp.results.z

        if not is_qp_solution_optimal(problem, x, y, eps):
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

        return Results(status, objval, x, y,
                        run_time, niter)



