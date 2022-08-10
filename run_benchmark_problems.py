'''
Run all benchmarks for the PROXQP paper

This code tests the solvers:
    - PROXQP (with dense backend)
    - OSQP
    - GUROBI
    - MOSEK
    - qpOASES
    - quadprog

'''
from benchmark_problems.example import Example
import solvers.solvers as s
from utils.general import gen_int_log_space
from utils.benchmark import compute_stats_info,compute_time_series_plot
import argparse
import proxsuite_pywrap as proxsuite

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
    solvers = [s.PROXQP]#s.MOSEK,s.qpOASES,s.GUROBI,s.quadprog,s.OSQP,s.PROXQP
    OUTPUT_FOLDER ='benchmark_problems_high_accuracy'
    for key in s.settings:
        s.settings[key]['high_accuracy'] = True
else:
    solvers = []
    OUTPUT_FOLDER = 'benchmark_problems'

if verbose:
    for key in s.settings:
        s.settings[key]['verbose'] = True

# Number of instances per different dimension
n_instances = 5
n_dim = 10
n_average = 10
sparsity = 0.15 # control problem sparsity
accuracies = [1.e-9] # control accuracy asked

# Run benchmark problems
problems = [
            'Random Mixed QP'
            #'Random Not Strongly Convex QP',
            #'Random Degenerate QP'
            ]

problem_dimensions = {
                      'Random Mixed QP': gen_int_log_space(10, 1000, n_dim),
                      'Random Degenerate QP': gen_int_log_space(10, 1000, n_dim),
                      'Random Not Strongly Convex QP': gen_int_log_space(10, 1000, n_dim)
                      }

# Some problems become too big to be executed in parallel and we solve them
# serially
problem_parallel = {
                    'Random Mixed QP': parallel,
                    'Random Degenerate QP': parallel,
                    'Random Not Strongly Convex QP': parallel
                    }

# Run all examples
for problem in problems:
        example = Example(problem,
                        problem_dimensions[problem],
                        accuracies,
                        solvers,
                        s.settings,
                        OUTPUT_FOLDER,
                        n_instances,
                        n_average,
                        sparsity
                        )
        example.solve(parallel=problem_parallel[problem])

# Compute results statistics

compute_stats_info(solvers, OUTPUT_FOLDER,
                   problems=problems,
                   high_accuracy=high_accuracy)

# plots 

suffix = ''
for problem in problems:
    compute_time_series_plot(solvers, problem, suffix)