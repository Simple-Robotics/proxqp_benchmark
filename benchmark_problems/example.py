import os
from multiprocessing import Pool, cpu_count
from itertools import repeat
import pandas as pd

from solvers.solvers import SOLVER_MAP
from problem_classes.random_mixed_qp import RandomMixedQPExample
from problem_classes.random_degenerate_qp import RandomDegenerateQPExample
from problem_classes.random_not_strongly_convex_qp import RandomNotStronglyConvexQPExample
from problem_classes.eq_qp import EqQPExample
from utils.general import make_sure_path_exists



examples = [
            RandomMixedQPExample,
            RandomDegenerateQPExample,
            RandomNotStronglyConvexQPExample,
            EqQPExample
            ]


EXAMPLES_MAP = {example.name(): example for example in examples}


class Example(object):
    '''
    Examples runner
    '''
    def __init__(self, name,
                 dims,
                 accuracies,
                 solvers,
                 settings,
                 output_folder,
                 n_instances=10,
                 n_average=10,
                 sparsity=0.15):
        self.name = name
        self.dims = dims
        self.sparsity = sparsity
        self.n_instances = n_instances
        self.n_average = n_average
        self.accuracies = accuracies
        self.solvers = solvers
        self.settings = settings
        self.output_folder = output_folder

    def solve(self, parallel=True,n_average=10):
        '''
        Solve problems of type example

        The results are stored as

            ./results/{self.output_folder}/{solver}/{class}/n{dimension}.csv

        using a pandas table with fields
            - 'class': example class
            - 'solver': solver name
            - 'status': solver status
            - 'run_time': execution time
            - 'iter': number of iterations
            - 'obj_val': objective value
            - 'n': leading dimension
            - 'N': nnz dimension (nnz(P) + nnz(A))
        '''

        print("Solving %s" % self.name)
        print("-----------------")

        if parallel:
            pool = Pool(processes=min(self.n_instances, cpu_count()))

        # Iterate over all solvers
        for solver in self.solvers:
            
            settings = self.settings[solver]
            # Initialize solver results
            results_solver = []

            # Solution directory
            path = os.path.join('.', 'results', self.output_folder,
                                solver,
                                self.name
                                )

            # Create directory for the results
            make_sure_path_exists(path)

            # Get solver file name
            solver_file_name = os.path.join(path, 'full.csv')

            for eps in self.accuracies:
                if solver in ['PROXQP',"PROXQP_sparse","OSQP"]:
                    settings['eps_abs'] = eps
                    settings['eps_rel'] = 0
                elif solver == "MOSEK":
                    settings["MSK_DPAR_INTPNT_CO_TOL_PFEAS"] = eps # cannot be put to 0..
                    settings["MSK_DPAR_INTPNT_CO_TOL_DFEAS"] = eps # cannot be put to 0..
                elif solver == "GUROBI":
                    settings["FeasibilityTol"] = eps # cannot be put to 0..
                    settings["OptimalityTol"] = eps # cannot be put to 0..

                print("solver : {} ; solvers:{} ; accuracy : {}".format(solver,self.solvers,eps))
                for n in self.dims:

                    # Check if solution already exists
                    n_file_name = os.path.join(path, 'n%i.csv' % n)
                    #n_file_name = os.path.join(path, 'eps%s.csv' % eps)

                    if not os.path.isfile(n_file_name):

                        if parallel and solver not in ['qpOASES']:
                            # NB. ECOS and qpOASES crahs if the problem sizes are too large
                            instances_list = list(range(self.n_instances))
                            n_results = pool.starmap(self.solve_single_example,
                                                    zip(repeat(n),repeat(self.sparsity),
                                                        instances_list,
                                                        repeat(solver),
                                                        repeat(settings),repeat(eps)))
                        else:
                            n_results = []
                            for instance in range(self.n_instances):
                                    # solve n_solving times the same problem for having a good average of the solving time
                                    run_time = 0
                                    n_solving = n_average
                                    #n_solving = 0
                                    for i in range(n_solving-1):
                                        res = self.solve_single_example(n,self.sparsity,
                                                            instance,
                                                            solver,
                                                            settings,
                                                            eps
                                                            )
                                        run_time+=res.run_time
                                        
                                    res = self.solve_single_example(n,self.sparsity,
                                                            instance,
                                                            solver,
                                                            settings,
                                                            eps
                                                            )
                                    run_time += res.run_time
                                    run_time/= n_solving
                                    res.run_time = run_time
                                    n_results.append(
                                        res
                                    )   

                        # Combine n_results
                        df = pd.concat(n_results)

                        # Store n_results
                        df.to_csv(n_file_name, index=False)

                    else:
                        # Load from file
                        df = pd.read_csv(n_file_name)

                    # Combine list of dataframes
                    results_solver.append(df)

            # Create total dataframe for the solver from list
            df_solver = pd.concat(results_solver)

            # Store dataframe
            df_solver.to_csv(solver_file_name, index=False)

        if parallel:
            pool.close()  # Not accepting any more jobs on this pool
            pool.join()   # Wait for all processes to finish

    def solve_single_example(self,
                             dimension,sparsity, instance_number,
                             solver, settings,eps):
        '''
        Solve 'example' with 'solver'

        Args:
            dimension: problem leading dimension
            instance_number: number of the instance
            solver: solver name
            settings: settings dictionary for the solver

        '''

        # Create example instance
        example_instance = EXAMPLES_MAP[self.name](dimension,sparsity,
                                                   instance_number)

        print(" - Solving %s with n = %i, instance = %i with solver %s with accuracy %s" %
              (self.name, dimension, instance_number, solver, eps))

        # Solve problem
        s = SOLVER_MAP[solver](settings)
        results = s.solve(example_instance,self.n_average,eps)

        # Create solution as pandas table
        P = example_instance.qp_problem['P']
        A = example_instance.qp_problem['A']
        N = P.nnz + A.nnz
        solution_dict = {'class': [self.name],
                         'solver': [solver],
                         'status': [results.status],
                         'run_time': [results.run_time],
                         'iter': [results.niter],
                         'obj_val': [results.obj_val],
                         'n': [dimension],
                         'N': [N],
                         'eps': [eps]}

        # Return solution
        return pd.DataFrame(solution_dict)
