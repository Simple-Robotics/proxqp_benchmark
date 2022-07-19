import qpoases
import numpy as np
from . import statuses as s
from .results import Results
from utils.general import is_qp_solution_optimal, stdout_redirected
import time

class qpOASESSolver(object):

    # Get return value class
    _PyReturnValue = qpoases.PyReturnValue()
    STATUS_MAP = {_PyReturnValue.SUCCESSFUL_RETURN: s.OPTIMAL,
                  _PyReturnValue.INIT_FAILED_INFEASIBILITY:
                  s.PRIMAL_INFEASIBLE,
                  _PyReturnValue.INIT_FAILED_UNBOUNDEDNESS: s.DUAL_INFEASIBLE,
                  _PyReturnValue.MAX_NWSR_REACHED: s.MAX_ITER_REACHED,
                  _PyReturnValue.INIT_FAILED: s.SOLVER_ERROR
                  }

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
        '''
        Solve problem

        Args:
            example: example object

        Returns:
            Results structure
        '''
        p = example.qp_problem
        n, m = p['n'], p['m']

        # Get number of nonzeros to check if problem is too big
        N = 0
        if p['P'] is not None:
            N += p['P'].nnz
        if p['A'] is not None:
            N += p['A'].nnz


        # Too large for qpOASES crashing the server (it reaches anyway the time-limit way before!)
        if example.name == 'Huber' and N > 250000:
            return Results(s.MAX_ITER_REACHED, None, None, None,
                           1000.0, 1000)

        if p['P'] is not None:
            P = np.ascontiguousarray(p['P'].todense())

        if 'A_nobounds' in p:
            A = np.ascontiguousarray(p['A_nobounds'].todense())
            m = A.shape[0]
        elif p['A'] is not None:
            A = np.ascontiguousarray(p['A'].todense())

        # Define contiguous array vectors
        q = np.ascontiguousarray(p['q'])

        if 'l_nobounds' in p:
            l = np.ascontiguousarray(p['l_nobounds'])
        else:
            l = np.ascontiguousarray(p['l'])

        if 'u_nobounds' in p:
            u = np.ascontiguousarray(p['u_nobounds'])
        else:
            u = np.ascontiguousarray(p['u'])

        if 'lx' in p:
            lx = np.ascontiguousarray(p['lx'])
        else:
            lx = np.ascontiguousarray(-np.inf * np.ones(n))

        if 'ux' in p:
            ux = np.ascontiguousarray(p['ux'])
        else:
            ux = np.ascontiguousarray(np.inf * np.ones(n))

        # Redirect output if verbose is False
        if 'verbose' in self._settings:
            if self._settings['verbose'] is False:
                with stdout_redirected():
                    qpoases_m = qpoases.PyQProblem(n, m)
            else:
                qpoases_m = qpoases.PyQProblem(n, m)
        else:
            # Suppress output also if we do not specify verbose
            with stdout_redirected():
                qpoases_m = qpoases.PyQProblem(n, m)

        options = qpoases.PyOptions()

        for param, value in self._settings.items():
            if param == 'verbose':
                if value is False:
                    options.printLevel = qpoases.PyPrintLevel.NONE
            elif param == 'time_limit':
                qpoases_cpu_time = np.array([value])
            elif param == 'nWSR':
                qpoases_nWSR = np.array([value])
            elif param != "high_accuracy":
                exec("options.%s = %s" % (param, value))

        qpoases_m.setOptions(options) 

        if 'time_limit' not in self._settings:
            # Set default to max 10 seconds in runtime
            qpoases_cpu_time = np.array([10.])

        if 'nWSR' not in self._settings:
            # Set default to max 1000000 working set recalculations
            qpoases_nWSR = np.array([1000000])

        # Solve problem
        tic = time.time()
        status = qpoases_m.init(P, q, A, lx, ux, l, u,
                                qpoases_nWSR, qpoases_cpu_time)
        toc = time.time()
        total_time = toc-tic
        # Check status
        status = self.STATUS_MAP.get(status, s.SOLVER_ERROR)

        # run_time
        run_time = total_time#qpoases_cpu_time[0]

        # number of iterations
        niter = qpoases_nWSR[0]

        if status in s.SOLUTION_PRESENT:
            x = np.zeros(n)
            y_temp = np.zeros(n + m)
            obj_val = qpoases_m.getObjVal()
            qpoases_m.getPrimalSolution(x)
            qpoases_m.getDualSolution(y_temp)

            # Change sign
            y = -y_temp[n:]

            if 'A_nobounds' in p:
                # If explicit bounds provided, reconstruct dual variable
                # Bounds on all the variables
                y_bounds = -y_temp[:n]
                # Limit to only the actual bounds
                y_bounds = y_bounds[p['bounds_idx']]

                y = np.concatenate((y, y_bounds))

            if not is_qp_solution_optimal(p, x, y, eps):
                status = s.SOLVER_ERROR

            return Results(status, obj_val,
                           x, y,
                           run_time, niter)
        else:
            return Results(status, None, None, None,
                           run_time, niter)
