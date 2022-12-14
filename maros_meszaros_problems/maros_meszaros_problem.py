import os
from multiprocessing import Pool, cpu_count
from itertools import repeat
import pandas as pd

from solvers.solvers import SOLVER_MAP
from problem_classes.maros_meszaros import MarosMeszaros
from utils.general import make_sure_path_exists
from utils.maros_meszaros import OPT_COST_MAP

import numpy as np

PROBLEMS_FOLDER = "maros_meszaros_data"


class MarosMeszarosRunner(object):
    '''
    Examples runner
    '''
    def __init__(self,
                 solvers,
                 settings,
                 output_folder):
        self.solvers = solvers
        self.settings = settings
        self.output_folder = output_folder

        # Get maros problems list
        problems_dir = os.path.join(".", "problem_classes", PROBLEMS_FOLDER)
        # List of problems in .mat format
        lst_probs = [f for f in os.listdir(problems_dir) if \
            f.endswith('.mat')]
        self.problems = [f[:-4] for f in lst_probs]   # List of problem names

    def solve(self, parallel=True, cores=32,n_average=0,eps=1.E-9):
        '''
        Solve problems of type example

        The results are stored as

            ./results/{self.output_folder}/{solver}/results.csv

        using a pandas table with fields
            - 'name': Maros problem name
            - 'solver': solver name
            - 'status': solver status
            - 'run_time': execution time
            - 'iter': number of iterations
            - 'obj_val': objective value from solver
            - 'obj_opt': optimal objective value
            - 'n': leading dimension
            - 'N': nnz dimension (nnz(P) + nnz(A))
        '''

        print("Solving Maros Meszaros problems")
        print("-------------------------------")

        if parallel:
            pool = Pool(processes=min(cores, cpu_count()))

        # Iterate over all solvers
        for solver in self.solvers:
            settings = self.settings[solver]
            if solver in ['PROXQP',"PROXQP_sparse","OSQP"]:
                settings['eps_abs'] = eps
                settings['eps_rel'] = 0
            elif solver == "MOSEK":
                settings["MSK_DPAR_INTPNT_CO_TOL_PFEAS"] = eps 
                settings["MSK_DPAR_INTPNT_CO_TOL_DFEAS"] = eps 
            elif solver == "GUROBI":
                settings["FeasibilityTol"] = eps 
                settings["OptimalityTol"] = eps 

            #  # Initialize solver results
            #  results_solver = []

            # Solution directory
            path = os.path.join('.', 'results', self.output_folder,
                                solver)

            # Create directory for the results
            make_sure_path_exists(path)

            # Get solver file name
            results_file_name = os.path.join(path, 'results.csv')

            # Check if file name already exists
            if not os.path.isfile(results_file_name):
                # Solve Maros Meszaros problems
                if parallel:
                    results = pool.starmap(self.solve_single_example,
                                           zip(self.problems,
                                               repeat(solver),
                                               repeat(settings)))
                else:
                    results = []
                    for problem in self.problems:
                        # preprocessing
                        full_name = os.path.join(".", "problem_classes",
                                 PROBLEMS_FOLDER, problem)
                        instance = MarosMeszaros(full_name)
                        if (instance.qp_problem['P'].shape[0]<=1000 and instance.qp_problem['A'].shape[0] <= 1000 ): # to get a subset wchich dimension is bounded by 1000
                            results.append(self.solve_single_example(problem,
                                                                 solver,
                                                                 settings,n_average,eps))
                # Create dataframe
                df = pd.concat(results)

                # Store results
                df.to_csv(results_file_name, index=False)

            #  else:
            #      # Load from file
            #      df = pd.read_csv(results_file_name)
            #
            #      # Combine list of dataframes
            #      results_solver.append(df)

        if parallel:
            pool.close()  # Not accepting any more jobs on this pool
            pool.join()   # Wait for all processes to finish

    def solve_single_example(self,
                             problem,
                             solver, settings,n_average,eps):
        '''
        Solve Maros Meszaro 'problem' with 'solver'

        Args:
            dimension: problem leading dimension
            instance_number: number of the instance
            solver: solver name
            settings: settings dictionary for the solver

        '''
        # Create example instance
        full_name = os.path.join(".", "problem_classes",
                                 PROBLEMS_FOLDER, problem)
        instance = MarosMeszaros(full_name)

        # Create solution as pandas table
        P = instance.qp_problem['P']
        A = instance.qp_problem['A']
        N = P.nnz + A.nnz     

        # Solve problem

        s = SOLVER_MAP[solver](settings)
        print(" - Solving %s with solver %s P.nnz %s A.nnz %s eps %s" % (problem, solver, P.nnz, A.nnz,eps))  

        results = s.solve(instance,n_average,eps)

        # Add constant part to objective value
        # NB. This is needed to match the objective in the original
        # Maros Meszaros paper
        obj = results.obj_val
        if results.obj_val is not None:
            obj += instance.qp_problem["r"]

        # Optimal cost distance from Maros Meszaros results
        # (For DEBUG)
        # ( obj - opt_obj )/(|opt_obj|)
        #  if obj is not None:
        #      obj_dist = abs(obj - OPT_COST_MAP[problem])
        #      if abs(OPT_COST_MAP[problem]) > 1e-20:
        #          # Normalize cost distance
        #          obj_dist /= abs(OPT_COST_MAP[problem])
        #  else:
        #      obj_dist = np.inf

        solution_dict = {'name': [problem],
                         'solver': [solver],
                         'status': [results.status],
                         'run_time': [results.run_time],
                         'iter': [results.niter],
                         'obj_val': [obj],
                         'obj_opt': [OPT_COST_MAP[problem]],
                         #  'obj_dist': [obj_dist],
                         'n': [instance.qp_problem["n"]],
                         'm': [instance.qp_problem["m"]],
                         'N': [N],
                         'eps' : [eps]}

        print(" - Solved %s with solver %s" % (problem, solver))

        # Return solution
        return pd.DataFrame(solution_dict)
