from firedrake import *
from time import perf_counter as time
from sys import argv
N = int(argv[1])
Re = 50
save_output = True


M = UnitSquareMesh(N, N)

Vu = VectorFunctionSpace(M, 'CG', 1)
Vp = FunctionSpace(M, 'CG', 1)
V = Vu * Vp

sol = Function(V)
soln = Function(V)
u, p = split(sol)
un, pn = split(soln)
v, q = TestFunctions(V)

F = Constant(1/Re) * \
    inner(grad(u), grad(v)) * dx - div(v) * p * dx + div(u) * q * dx
L = dot(Constant((0, 0)), v) * dx
F_schur = Constant(Re) * p * q * dx
FJ = F
FP = FJ + F_schur

# BCs
zero = Constant((0, 0))
v_in = Constant((1, 0))
bc_noslip = DirichletBC(V.sub(0), zero, (1, 2, 3))  # x in {0,L} and y = 0
bc_in = DirichletBC(V.sub(0), v_in, 4)
bcs = [bc_noslip, bc_in]


params = {
    "ksp_type": "gmres",
    "mat_type": "nest",
    "ksp_norm_type": "unpreconditioned",
    "ksp_converged_reason": None,
    "ksp_max_it": 1000,
    "ksp_atol": 1e-14,
    "ksp_rtol": 1e-6,
    "ksp_gmres_restart": "1000",
    "pc_type": "fieldsplit",
    "pc_fieldsplit_type": "schur",
    "pc_fieldsplit_schur_type": "diag",
    "pc_fieldsplit_schur_precondition": "a11",
    "fieldsplit_0": {"ksp_type": "preonly",
                         "pc_type": "hypre"},
    "fieldsplit_1": {"ksp_type": "preonly",
                     "pc_type": "jacobi"}
}

AJ = derivative(FJ, sol, TrialFunction(V))
AP = derivative(FP, sol, TrialFunction(V))
A = lhs(AJ)
b = L
problem = LinearVariationalProblem(A, b, sol, bcs=bcs, aP=AP)
solver = LinearVariationalSolver(problem, solver_parameters=params)

t = time()
solver.solve()
t_solve = time() - t

if save_output:
    outfile = File("output/stokes.pvd")
    uout, pout = sol.subfunctions
    outfile.write(uout, pout)

if COMM_WORLD.rank == 0:
    print("Solved in {:.2e}s".format(t_solve), flush=True)
    print("DoFs:", V.dim())

