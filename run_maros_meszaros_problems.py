'''
Run Maros-Meszaros problems for the PROXQP paper

This code tests the solvers:
    - PROXQP
    - OSQP
    - GUROBI
    - MOSEK
    - qpOASES
'''
from maros_meszaros_problems.maros_meszaros_problem import MarosMeszarosRunner
import solvers.solvers as s
from utils.benchmark import compute_stats_info,plot_performance_profiles,compute_performance_profiles
import os
import argparse


parser = argparse.ArgumentParser(description='Maros Meszaros Runner')
parser.add_argument('--high_accuracy', help='Test with high accuracy', default=True,
                    action='store_true')
parser.add_argument('--verbose', help='Verbose solvers', default=False,
                    action='store_true')
parser.add_argument('--parallel', help='Parallel solution', default=False,
                    action='store_true')
args = parser.parse_args()
high_accuracy = args.high_accuracy
verbose = args.verbose
parallel = args.parallel

print('high_accuracy', high_accuracy)
print('verbose', verbose)
print('parallel', parallel)

# Add high accuracy solvers when accuracy
if high_accuracy:
    solvers = [s.PROXQP,s.OSQP,s.qpOASES,s.GUROBI,s.quadprog,s.MOSEK]
    OUTPUT_FOLDER = 'maros_meszaros_problems_high_accuracy'
    for key in s.settings:
        s.settings[key]['high_accuracy'] = True
else:
    solvers = [s.OSQP, s.OSQP_polish, s.GUROBI, s.MOSEK]
    OUTPUT_FOLDER = 'maros_meszaros_problems'

# Shut up solvers
if verbose:
    for key in s.settings:
        s.settings[key]['verbose'] = True

# Run all examples

maros_meszaros_runner = MarosMeszarosRunner(solvers,
                                            s.settings,
                                            OUTPUT_FOLDER)
eps = 1.E-9
maros_meszaros_runner.solve(parallel=parallel, cores=12,n_average=1,eps=eps)

# Compute results statistics
compute_stats_info(solvers, OUTPUT_FOLDER,
                   high_accuracy=high_accuracy)
compute_performance_profiles(solvers, OUTPUT_FOLDER,'')
plot_performance_profiles(OUTPUT_FOLDER, solvers , '')
