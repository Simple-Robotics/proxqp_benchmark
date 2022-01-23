'''
Run all benchmarks for the OSQP paper

This code tests the solvers:
    - OSQP
    - GUROBI
    - MOSEK
    - ECOS (crash always..)
    - qpOASES
    - quadprog


    - QPALM (error to fix)

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
    solvers =  [s.GUROBI_high,s.MOSEK_high,s.quadprog]#[s.OSQP_high,s.PROXQP,s.GUROBI_high,s.MOSEK_high,s.qpOASES] #[ s.OSQP_high,s.PROXQP,s.GUROBI_high,s.MOSEK_high,s.qpOASES,s.quadprog] # qpalm crashes as well (segfault..)  , ,
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
n_instances = 5 # 10 à la base #1 
n_dim = 10 # 20 à la base
n_average = 10


# Run benchmark problems
problems = [
            'Random QP'
            #'Random Degenerate QP'
            #'Random Not Strongly Convex QP'
            #'Random Mixed QP'
            #'Eq QP'
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
                      #[10],
                      solvers,
                      s.settings,
                      OUTPUT_FOLDER,
                      n_instances,
                      #1,
                      #0
                      n_average
                      )
    example.solve(parallel=problem_parallel[problem])

# Compute results statistics
'''
compute_stats_info(solvers, OUTPUT_FOLDER,
                   problems=problems,
                   high_accuracy=high_accuracy)

plot_performance_profiles(problems, solvers)


# plots 

problems_type = 'Random_Mixed_QP'
suffix = ''
#_m=0.5n_density1
compute_time_series_plot(solvers, problems_type, suffix)
'''