import inria_ldlt_py 
import numpy as np
from . import statuses as s
from .results import Results
from utils.general import is_qp_solution_optimal
import math 


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

    def solve(self, example):
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

        model = inria_ldlt_py.QPData(n,n_eq,n_in)
        results = inria_ldlt_py.QPResults(n,n_eq,n_in)
        prox_settings = inria_ldlt_py.QPSettings()
        work = inria_ldlt_py.QPWorkspace(n,n_eq,n_in)
        #prox_settings.max_iter = 1000
        #prox_settings.max_iter_in = 1500

        inria_ldlt_py.QPsetup(
                np.asfortranarray(H.toarray()),
                np.asfortranarray(g),
                np.asfortranarray(A.toarray()),
                np.asfortranarray(b),
                np.asfortranarray(C.toarray()),
                np.asfortranarray(u),
                np.asfortranarray(l),
                prox_settings,
                model,
                work,
                results,

                self._settings['eps_abs'],
                self._settings['eps_rel'],
                self._settings['verbose']  
        )

        #print("prox_settings._eps_abs : {}".format(prox_settings._eps_abs))
        #print("prox_settings._eps_rel : {}".format(prox_settings._eps_rel))
        run_time = 0
        n_solving = 10
        for i in range(n_solving):
            inria_ldlt_py.QPsolve(
                model,
                results,
                work,
                prox_settings)
            run_time += results.timing
            inria_ldlt_py.QPreset(
                model,
                prox_settings,
                results,
                work
            )
        inria_ldlt_py.QPsolve(
            model,
            results,
            work,
            prox_settings)
        run_time += results.timing
        n_solving += 1
    
        # duration time
        #run_time = results.timing * 1.e-6 # the run_time is measured in microseconds 
        run_time /= (1.e6 * n_solving) # the run_time is measured in microseconds 

        # Obj val
        objval = results.objValue

        # Total Number of iterations
        niter = results.n_tot

        # Get solution
        x = results.x 
        y = np.zeros(n_eq+n_in)
        y[eq_ids] = results.y
        y[in_ids] = results.z

        if not is_qp_solution_optimal(problem, x, y,
                                        high_accuracy=self._settings.get('high_accuracy')):
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



