from quadprog import solve_qp
import numpy as np
from . import statuses as s
from .results import Results
from utils.general import is_qp_solution_optimal
import math 
from typing import Optional
from warnings import warn
from numpy import hstack,vstack, ndarray
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


    def solve(self, example):
        """
        Solve a Quadratic Program defined as:
        .. math::
            \\begin{split}\\begin{array}{ll}
            \\mbox{minimize} &
                \\frac{1}{2} x^T P x + q^T x \\\\
            \\mbox{subject to}
                & G x \\leq h                \\\\
                & A x = h
            \\end{array}\\end{split}
        using `quadprog <https://pypi.python.org/pypi/quadprog/>`_.
        Parameters
        ----------
        P :
            Symmetric quadratic-cost matrix.
        q :
            Quadratic-cost vector.
        G :
            Linear inequality constraint matrix.
        h :
            Linear inequality constraint vector.
        A :
            Linear equality constraint matrix.
        b :
            Linear equality constraint vector.
        initvals :
            Warm-start guess vector (not used).
        verbose :
            Set to `True` to print out extra information.
        Returns
        -------
        :
            Solution to the QP, if found, otherwise ``None``.
        Note
        ----
        The quadprog solver only considers the lower entries of :math:`P`,
        therefore it will use a different cost than the one intended if a
        non-symmetric matrix is provided.
        Notes
        -----
        All other keyword arguments are forwarded to the quadprog solver. For
        instance, you can call ``quadprog_solve_qp(P, q, G, h, factorized=True)``.
        See the solver documentation for details.

        quadprog solve_qp function returns (see : https://github.com/quadprog/quadprog/blob/master/quadprog/quadprog.pyx)

        x : array, shape=(n,)
            vector containing the solution of the quadratic programming problem.
        f : float
            the value of the quadratic function at the solution.
        xu : array, shape=(n,)
            vector containing the unconstrained minimizer of the quadratic function
        iterations : tuple
            2-tuple. the first component contains the number of iterations the
            algorithm needed, the second indicates how often constraints became
            inactive after becoming active first.
        lagrangian : array, shape=(m,)
            vector with the Lagragian at the solution.
        iact : array
            vector with the indices of the active constraints at the solution.


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
        qp_C: Optional[ndarray] = None
        qp_b: Optional[ndarray] = None
        if A is not None and b is not None:
            if C is not None and u is not None:
                qp_C = -vstack([-A,C, -C]).T
                qp_b = -hstack([-b, u, -l])
            else:
                qp_C = -A.T
                qp_b = -b
            meq = A.shape[0]
        else:  # no equality constraint
            if C is not None and u is not None:
                qp_C = -vstack([C, -C]).T
                qp_b = -hstack([u, -l])
            meq = 0
        try:
            print("H shape : {} ; g : {} ; A : {} ; u : {} ; eq : {}".format(problem['P'].shape,problem['q'].shape,problem['A'].shape,problem['u'].shape,A.shape[0]))
            print("qp_G shape : {} ; qp_a : {} ; qp_C : {} ; qp_b : {} ; meq : {}".format(qp_G.shape,qp_a.shape,qp_C.shape,qp_b.shape,meq))
            tic = time.time()
            x, objval, xu, niter, y, iact = solve_qp(qp_G, qp_a, qp_C, qp_b, meq)
            toc = time.time()
            run_time = toc - tic
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


        except ValueError as e:
            error = str(e)
            if "matrix G is not positive definite" in error:
                # quadprog writes G the cost matrix that we write P in this package
                raise ValueError("matrix P is not positive definite") from e
            if "no solution" in error:
                return Results(s.SOLVER_ERROR, None, None, None,
                            None, None)
            warn("quadprog raised a ValueError: {}".format(e))

            return Results(s.PRIMAL_OR_DUAL_INFEASIBLE, None, None, None,
                            None, None)

