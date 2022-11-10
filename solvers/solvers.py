GUROBI = 'GUROBI'
OSQP = 'OSQP'
MOSEK = 'MOSEK'
qpOASES = 'qpOASES'
PROXQP = 'PROXQP'
PROXQP_sparse = 'PROXQP_sparse'
quadprog = 'quadprog'

SOLVER_MAP = dict()

try:
  from solvers.gurobi import GUROBISolver
  SOLVER_MAP[GUROBI] = GUROBISolver
except ModuleNotFoundError:
  pass
try:
  from solvers.mosek import MOSEKSolver
  SOLVER_MAP[MOSEK] = MOSEKSolver
except ModuleNotFoundError:
  pass
try:
  from solvers.osqp import OSQPSolver
  SOLVER_MAP[OSQP] = OSQPSolver
except ModuleNotFoundError:
  pass
try:
  from solvers.qpoases import qpOASESSolver
  SOLVER_MAP[qpOASES] = qpOASESSolver
except ModuleNotFoundError:
  pass
try:
  from solvers.proxqp import PROXQPSolver
  SOLVER_MAP[PROXQP] = PROXQPSolver
  SOLVER_MAP[PROXQP_sparse] = PROXQPSolver
except ModuleNotFoundError:
  pass
try:
  from solvers.quadprog import QUADPROGSolver
  SOLVER_MAP[quadprog] = QUADPROGSolver
except ModuleNotFoundError:
  pass

time_limit = 1000. # Seconds
eps_high = 1e-09

# Solver settings
settings = {
    PROXQP: {'eps_abs': eps_high,
              'eps_rel': 0.,
              'verbose':False
    },
    OSQP: {'eps_abs': eps_high,
           'eps_rel': 0.,
           'polish': False, # true polish slows down OSQP
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
