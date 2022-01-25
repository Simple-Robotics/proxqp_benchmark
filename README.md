# Benchmark examples for the OSQP solver

These are the scripts to compare the following Quadratic Program (QP) solvers

-   PROXQP
-   OSQP
-   GUROBI
-   MOSEK
-   qpOASES
-   quadprog

The detailed description of these tests is available in [this paper](https://arxiv.org/pdf/1711.08013.pdf).
To run these scripts you need `pandas`, `cvxpy` (and solvers `quadprog`, `gurobi`, `mosek`, `qpOASES`, `OSQP`) installed.

All the scripts come with options (default to `False`)

- `--parallel` for parallel execution across instances
- `--verbose` for verbose solvers output (they  can be slower than necessary while printing)
- `--high_accuracy` for high accuracy `eps=1e-09` solver settings + optimality checks


## Benchmark problems
The problems are all randomly generated as described in the PROXQP paper.
Problem instances include

-   Pure Inequality constrained QP
-   Pure Equality constrained QP
-   Inequality and Equality Constrained QP
-   Pure Inequality Degenerate QP
-   Pure Inequality QP without strictly convex Hessian matrix

We generate the problems using the scripts in the `problem_classes` folder.

To execute these tests run
```python
python run_benchmark_problems.py
```

### Results
The resulting [shifted geometric means](http://plato.asu.edu/ftp/shgeom.html) are

| PROXQP | OSQP              | GUROBI          | MOSEK              | qpOASES            |
| -----  | ----------------- | --------------- | ------------------ | ------------------ |
|        |                   |                 |                    |                    |


## Maros Meszaros problems
These are the hard problems from the [Maros Meszaros testset](http://www.cuter.rl.ac.uk/Problems/marmes.shtml) converted using [CUTEst](https://ccpforge.cse.rl.ac.uk/gf/project/cutest/wiki) and the scripts in the [maros_meszaros_data/](./problem_classes/maros_meszaros_data) folder.
In these benchmarks we compare OSQP with GUROBI and MOSEK.

To execute these tests run
```python
python run_maros_meszaros_problems.py
```

### Results
The resulting [shifted geometric means](http://plato.asu.edu/ftp/shgeom.html) are

| PROXQP             | OSQP   | GUROBI            |
| ------------------ | ------ | ----------------- |
|                    |        |                   |

## Citing

If you are using these benchmarks for your work, please cite the [OSQP paper](https://osqp.org/citing/).
