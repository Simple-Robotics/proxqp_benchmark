import qpalm_biding as qp
import numpy as np
from . import statuses as s
from .results import Results
from utils.general import is_qp_solution_optimal
import math 


class QPALMSolver(object):

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
        #documentation https://benny44.github.io/QPALM_vLADEL/
        problem = example.qp_problem

        solver = qp.Qpalm()
        solver._settings.contents.eps_abs = self._settings['eps_abs']
        solver._settings.contents.eps_rel = self._settings['eps_rel']
        solver._settings.contents.eps_prim_inf = self._settings['eps_prim_inf']
        solver._settings.contents.max_iter =  self._settings['max_iter']
        solver._settings.contents.verbose = False
        solver._settings.contents.max_iter =  int(self._settings['time_limit'])
        
        ub = np.copy(problem['u'])
        lb = np.copy(problem['l'])

        PlusInfId = ub == math.inf 
        NegInfId = lb == -math.inf 

        ub[PlusInfId] = 1.e20
        lb[NegInfId] = -1.e20

        P_qpalm = problem['P']
        q_qpalm = problem['q']
        A_qpalm = problem['A']
        lb_qpalm = lb
        ub_qpalm = ub

        solver.set_data(Q=P_qpalm, A=A_qpalm, q=q_qpalm, bmin=lb_qpalm, bmax=ub_qpalm)
        solver._solve()

        it_tot = solver._work.contents.info.contents.iter
        #it_out = solver._work.contents.info.contents.iter_out
        timing = solver._work.contents.info.contents.solve_time
        objval = solver._work.contents.info.contents.objective

        n = P_qpalm.shape[0]
        m = A_qpalm.shape[0]


        x_ = solver._work.contents.solution.contents.x
        x = np.zeros(n)
        y = np.zeros( int(m) )
        y_ = solver._work.contents.solution.contents.y
        for i in range(n):
            x[i] = x_[i]
        for i in range( int(m)):
            y[i] = y_[i]

        if not is_qp_solution_optimal(problem, x, y,
                                        high_accuracy=self._settings.get('high_accuracy')):
            status = s.SOLVER_ERROR
        # Verify solver time
            if 'time_limit' in self._settings:
                if timing > self._settings['time_limit']:
                    status = s.TIME_LIMIT
        else:
            # see https://benny44.github.io/QPALM_vLADEL/constants_8h.html 
           
            status = s.OPTIMAL

            if 'time_limit' in self._settings:
                if timing > self._settings['time_limit']:
                    status = s.TIME_LIMIT

            if (solver._work.contents.info.contents.status_val == -2):
                status = s.MAX_ITER_REACHED
            elif (solver._work.contents.info.contents.status_val == -3):
                status = 3
            elif (solver._work.contents.info.contents.status_val == -4):
                status = 5
            elif (solver._work.contents.info.contents.status_val == -10):
                status = 10
            elif (solver._work.contents.info.contents.status_val == 0):
                status = 10

        return Results(status, objval, x, y,
                        timing, it_tot)
