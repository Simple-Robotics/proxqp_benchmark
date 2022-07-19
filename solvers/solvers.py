from solvers.gurobi import GUROBISolver
from solvers.mosek import MOSEKSolver
from solvers.osqp import OSQPSolver
from solvers.qpoases import qpOASESSolver
from solvers.proxqp import PROXQPSolver
from solvers.quadprog import QUADPROGSolver
import proxsuite_pywrap as proxsuite

GUROBI = 'GUROBI'
OSQP = 'OSQP'
MOSEK = 'MOSEK'
qpOASES = 'qpOASES'
PROXQP = 'PROXQP'
PROXQP_Martinez = 'PROXQP_Martinez'
PROXQP_sparse = 'PROXQP_sparse'
quadprog = 'quadprog'

SOLVER_MAP = {OSQP: OSQPSolver,
              GUROBI: GUROBISolver,
              MOSEK: MOSEKSolver,
              qpOASES: qpOASESSolver,
              PROXQP : PROXQPSolver,
              PROXQP_Martinez : PROXQPSolver,
              PROXQP_sparse:PROXQPSolver,
              quadprog : QUADPROGSolver
              }

time_limit = 1000. # Seconds
eps_high = 1e-09

# Solver settings
settings = {
    PROXQP: {'eps_abs': eps_high,
              'eps_rel': 0.,
              'verbose':False,
              'dense' : True,
              'initial_guess': proxsuite.qp.NO_INITIAL_GUESS,
              'bcl_update' : True
    },
    PROXQP_sparse: {'eps_abs': eps_high,
              'eps_rel': 0.,
              'verbose':False,
              'dense':False,
              'initial_guess': proxsuite.qp.NO_INITIAL_GUESS,
              'bcl_update' : True
    },
    OSQP: {'eps_abs': eps_high,
           'eps_rel': 0.,
           'polish': True,
           'max_iter': int(1e09),
           'eps_prim_inf': 1e-15,  # Disable infeas check
           'eps_dual_inf': 1e-15,
    },
    GUROBI: {'TimeLimit': time_limit,
                  'FeasibilityTol': eps_high,
                  'OptimalityTol': eps_high,
                  },
    MOSEK: {'MSK_DPAR_OPTIMIZER_MAX_TIME': time_limit,
                 'MSK_DPAR_INTPNT_CO_TOL_PFEAS': eps_high,   # Primal feasibility tolerance
                 'MSK_DPAR_INTPNT_CO_TOL_DFEAS': eps_high,   # Dual feasibility tolerance
                },
    qpOASES: {},
    quadprog: {}
}

for key in settings:
    settings[key]['verbose'] = False
    settings[key]['time_limit'] = time_limit
