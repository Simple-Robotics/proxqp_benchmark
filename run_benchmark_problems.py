'''
Run all benchmarks for the PROXQP paper

This code tests the solvers:
    - PROXQP
    - OSQP
    - GUROBI
    - MOSEK
    - qpOASES
    - quadprog

'''
from benchmark_problems.example import Example
import solvers.solvers as s
from utils.general import gen_int_log_space
from utils.benchmark import compute_stats_info,plot_performance_profiles,compute_time_series_plot
import os
import argparse
import inria_ldlt_py 

parser = argparse.ArgumentParser(description='Benchmark Problems Runner')
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
    #solvers = [s.OSQP_high, s.OSQP_polish_high, s.GUROBI_high, s.MOSEK_high, s.ECOS_high, s.qpOASES,s.quadprog] # ECOS returns nans... ; quadprog gives always an error (dimension mismatch..)
    solvers =  [s.MOSEK_high,s.qpOASES,s.GUROBI_high,s.OSQP_high,s.PROXQP] #[s.OSQP_high,s.PDALM] #[ s.OSQP_high,s.PROXQP,s.GUROBI_high,s.MOSEK_high,s.qpOASES,s.quadprog] # qpalm crashes as well (segfault..)  , ,
    OUTPUT_FOLDER ='benchmark_problems_high_accuracy'
    for key in s.settings:
        s.settings[key]['high_accuracy'] = True
else:
    solvers = [s.OSQP, s.OSQP_polish, s.GUROBI, s.MOSEK, s.ECOS, s.qpOASES]
    OUTPUT_FOLDER = 'benchmark_problems'

if verbose:
    for key in s.settings:
        s.settings[key]['verbose'] = True

# Number of instances per different dimension
n_instances = 5 #
n_dim = 10 # 20 à la base
n_average = 10
sparsity = 0.15

# Run benchmark problems
problems = [
            #'Random QP',
            #'Random Degenerate QP'
            'Random Not Strongly Convex QP'
            #'Random Mixed QP'
            #'Eq QP'
            #'Eq QP_m=0.5n_density0.15'
            #'Random QP_m0.5_density0.15'
            #'Random_Mixed_QP'
            #'Random_Degenerate_QP'
            #'Random Not Strongly Convex QP_density0.15_m=0.5n'
            ]

problem_dimensions = {'Random QP': gen_int_log_space(10, 1000, n_dim),
                      'Random Mixed QP': gen_int_log_space(10, 1000, n_dim),
                      'Random Degenerate QP': gen_int_log_space(10, 1000, n_dim),
                      'Random Not Strongly Convex QP': gen_int_log_space(10, 1000, n_dim),
                      'Eq QP': gen_int_log_space(10, 1000, n_dim)
                      }

# Some problems become too big to be executed in parallel and we solve them
# serially
problem_parallel = {'Random QP': parallel,
                    'Random Mixed QP': parallel,
                    'Random Degenerate QP': parallel,
                    'Random Not Strongly Convex QP': parallel,
                    'Eq QP': parallel
                    }

# Run all examples

for problem in problems:
    example = Example(problem,
                      problem_dimensions[problem],
                      #[11],
                      solvers,
                      s.settings,
                      OUTPUT_FOLDER,
                      n_instances,
                      n_average
                      #1,
                      #0
                      )
    example.solve(parallel=problem_parallel[problem])

# Compute results statistics

compute_stats_info(solvers, OUTPUT_FOLDER,
                   problems=problems,
                   high_accuracy=high_accuracy)

# plots 

#suffix = '_sparsity_' + sparsity + '_high_accuracy'

suffix = ''
for problem in problems:
    compute_time_series_plot(solvers, problem, suffix)
