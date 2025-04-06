import math
import numpy as np
import time
from scipy.stats import iqr
import sys
import os
import argparse

sys.path.append(os.path.abspath("./src"))

from emibbo.lbvd.alg.algo import lbvdcma
from emibbo.lbfmnes.alg.algo import lbfmnes


def main(args):
    def sphere_int(x):
        xbar = np.array(x)
        xbar[dim_co:] = np.round(xbar[dim_co:])
        return np.sum(xbar**2)
    def sphere_one_max(x):
        xbar = np.array(x)
        xbar[dim_co:] = np.where(xbar[dim_co:] > 0, 1.0, 0.0)
        return np.sum(xbar[:dim_co]**2) + dim_int - np.sum(xbar[dim_co:])
    def n_int_tablet(x):
        xbar = np.array(x)
        xbar[dim_co:] = np.round(xbar[dim_co:])
        xbar[:dim_co] *= 100
        return np.sum(xbar**2)
    def ellipsoid_int(x):
        xbar = np.array(x)
        xbar[dim_co:] = np.round(xbar[dim_co:])
        coefficients = np.array([math.pow(1e3, i / (dim - 1.)) for i in range(dim)]).reshape(-1,1)
        return np.sum((coefficients * xbar)**2)
    def reversed_ellipsoid_int(x):
        xbar = np.array(x)
        xbar[dim_co:] = np.round(xbar[dim_co:])
        coefficients = np.array([math.pow(1e3, i / (dim - 1.)) for i in range(dim)]).reshape(-1,1)
        return np.sum((coefficients[dim_co:] * xbar[:dim_co])**2) + np.sum((coefficients[:dim_co] * xbar[dim_co:])**2)
    def different_powers(x):
        xbar = np.array(x)
        xbar = np.abs(xbar)
        xbar[dim_co:] = np.round(xbar[dim_co:])
        power = np.array([2 + 10 * i / (dim - 1.) for i in range(dim)]).reshape(-1, 1)
        return np.sum(np.power(xbar, power))
    def sphere_leading_one(x):
        xbar = np.array(x)
        xbar[dim_co:] = np.where(xbar[dim_co:] > 0, 1.0, 0.0)
        Prod = np.array([np.prod(xbar[dim_co:i+1]) for i in range(dim_co, dim)])
        return np.sum((xbar[:dim_co])**2) + dim_int - np.sum(Prod)
    def ellipsoid_one_max(x):
        xbar = np.array(x)
        xbar[dim_co:] = np.where(xbar[dim_co:] > 0, 1.0, 0.0)
        coefficients = np.array([math.pow(1e3, i / (dim_co - 1.)) for i in range(dim_co)]).reshape(-1,1)
        return np.sum((coefficients * xbar[:dim_co])**2) + dim_int - np.sum(xbar[dim_co:])
    
    func_name = {
        'sphere_int': sphere_int,
        'sphere_one_max': sphere_one_max,
        'ellipsoid_int': ellipsoid_int,
        'reversed_ellipsoid_int': reversed_ellipsoid_int,
        'different_powers': different_powers,
        'n_int_tablet': n_int_tablet,
        'sphere_leading_one': sphere_leading_one,
        'ellipsoid_one_max': ellipsoid_one_max
    }
    if args.list_funcs:
        print("Available functions:")
        for name in func_name.keys():
            print(f" - {name}")
        selected_func = input("Please enter the function name to execute: ")
        if selected_func not in func_name:
            print(f"Error: '{selected_func}' is not a valid function name.")
            sys.exit(1)
        args.func = selected_func
    func = func_name.get(args.func)
    dim = args.dim
    if args.max_evals is None:
        args.max_evals = args.dim * 10 * 10 * 10 * 10
    if args.dim_co is None:
        args.dim_co = dim // 2
    if args.dim_int is None:
        args.dim_int = dim - args.dim_co
    dim_int = args.dim_int
    dim_co = args.dim_co
    assert args.dim_co + args.dim_int == dim
    if args.func in ['sphere_one_max', 'sphere_leading_one', 'ellipsoid_one_max']:
        domain_int = [[0,1] for _ in range(dim_int)]
        mean_NES = np.ones([dim, 1])
        mean_NES[:dim_co] *= 2.
        mean_NES[dim_co:] *= 0.5

        mean_VD = np.ones(dim)
        mean_VD[:dim_co] *= 2.
        mean_VD[dim_co:] *= 0.5
    else:
        domain_int = [list(range(-10, 11)) for _ in range(dim_int)]
        mean_VD = np.ones(dim) * 2.
        mean_NES = np.ones([dim, 1]) * 2.
    target = args.target
    if args.population_size_VD is None:
        args.population_size_VD = 4 + int(3 * np.log(dim))
    if args.population_size_NES is None:
        log_d = np.floor(3 * np.log(dim))
        args.population_size_NES = 4 + int(log_d) if int(log_d) % 2 == 0 else 5 + int(log_d)
    
    lamb_VD = args.population_size_VD
    budget = args.max_evals
    sigma_VD = args.sigma_VD
    margin_VD = 1.0 / (lamb_VD * dim)
    begin = time.time()
    vd = lbvdcma(dim_int=dim_int, dim_co=dim_co, func=func, xmean0=mean_VD, sigma0=sigma_VD, domain_int=domain_int, margin=margin_VD, lamb=lamb_VD, ssa=args.step_size_control, maxeval=budget, ftarget=target)
    is_success, x_best, f_best, evals = vd.run()
    end = time.time()

    print(f'Algorithm: LBVDCMA')
    print(f'Step-size control by: {args.step_size_control}')
    print(f'Success: {is_success}')
    print(f'Eval: {evals}        fbest: {f_best}')
    print(f'x_best: {x_best}')
    print(f'Time: {end - begin} seconds')
    print("===========================================================================")

    lamb_NES = args.population_size_NES
    sigma_NES = args.sigma_NES
    margin_NES = 1.0 / (dim * lamb_NES)
    begin = time.time()
    nes = lbfmnes(dim_int=dim_int, dim_co=dim_co, f=func, m=mean_NES, sigma=sigma_NES, lamb=lamb_NES, domain_int=domain_int, margin=margin_NES, ftarget=target, budget=budget)
    is_success, x_best, f_best, evals = nes.optimize(target)
    end = time.time()

    print(f'Algorithm: LBFMNES')
    print(f'Success: {is_success}')
    print(f'Eval: {evals}        fbest: {f_best}')
    print(f'x_best: {x_best}')
    print(f'Time: {end - begin} seconds')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--list_funcs", action="store_true", help="Show all available function names.")
    parser.add_argument("--func", type = str, default="ellipsoid_int", help="Objective function to use.")
    parser.add_argument("--dim", type=int, default=80, help="Problem dimension (dim = dim_co + dim_int).")
    parser.add_argument("--dim_co", type=int, default=None, help="Continuous part of the problem dimension.")
    parser.add_argument("--dim_int", type=int, default=None, help="Integer part of the problem dimension.")
    parser.add_argument("--max_evals", type=int, default=None, help="Maximum number of evaluations.")
    parser.add_argument("--target", type=float, default=1e-10, help="Target value.")
    parser.add_argument("--population_size_VD", type=int, default=None, help="Population size for LB-VD-CMA.")
    parser.add_argument("--population_size_NES", type=int, default=None, help="Population size for LB-FM-NES.")
    parser.add_argument("--sigma_VD", type=float, default=0.5, help="Initial sigma for LB-VD-CMA.")
    parser.add_argument("--sigma_NES", type=float, default=0.5, help="Initial sigma for LB-FM-NES.")
    parser.add_argument("--step_size_control", type=str, default="MCSA", help="Step size control for LB-VD-CMA: MCSA or TPA")

    args = parser.parse_args()
    main(args)