import inria_ldlt_py 
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

    def solve(self, example,n_average):
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

        #print("ub :{}".format(ub))
        #print("lb: {}".format(lb))
        #input()

        eq_ids = lb == ub
        in_ids = lb != ub

        #print("eq_ids : {}".format(eq_ids))
        #print("in_ids : {}".format(in_ids))
        #input()

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
        #print("n : {} ; n_eq : {} ; n_in : {}".format(n,n_eq,n_in))
        #input()
        model = inria_ldlt_py.QPData(n,n_eq,n_in)
        results = inria_ldlt_py.QPResults(n,n_eq,n_in)
        prox_settings = inria_ldlt_py.QPSettings()
        work = inria_ldlt_py.QPWorkspace(n,n_eq,n_in)
        prox_settings.max_iter = 1000
        prox_settings.eps_IG = min(self._settings['eps_abs'],1.e-9)
        prox_settings.max_iter_in = 1500
        prox_settings.warm_start = self._settings['warm_start']
        prox_settings.verbose = self._settings['verbose']
        prox_settings.eps_abs = self._settings['eps_abs']
        prox_settings.eps_rel = self._settings['eps_rel']


        inria_ldlt_py.QPsetup(
                #np.asfortranarray(H.toarray()),
                H,
                np.asfortranarray(g),
                #np.asfortranarray(A.toarray()),
                A,
                np.asfortranarray(b),
                #np.asfortranarray(C.toarray()),
                C,
                np.asfortranarray(u),
                np.asfortranarray(l),
                prox_settings,
                model,
                work,
                results
        )

        run_time = 0
        n_solving = n_average
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

        #print("primal equality : {}".format(np.linalg.norm(A@x-b,np.inf)))
        #print("primal inequality : {}".format(np.linalg.norm(np.maximum(C@x-u,0.),np.inf)))
        #print(f"dual residual {np.linalg.norm(H@x + g + A.T@results.y + C.T @ results.z,np.inf)}")
        #print("x : {} ;  y : {} ; z : {}".format(x, results.y, results.z))
        #input()

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



