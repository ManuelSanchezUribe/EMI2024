from firedrake import *
from sys import argv
N = int(argv[1])
KSP = argv[2]
PC = argv[3]
mesh = UnitSquareMesh(N, N)

V = FunctionSpace(mesh, "CG", 1)
print("DoFs:", V.dim())
u = TrialFunction(V)
v = TestFunction(V)

a = inner(grad(u), grad(v)) * dx
f = Function(V)
x,y = SpatialCoordinate(mesh)
f.interpolate((1+8*pi*pi)*cos(x*pi*2)*cos(y*pi*2))
L = f * v * dx

bcs = DirichletBC(V, Constant(0), "on_boundary")
params={"ksp_type": KSP, 
        "ksp_converged_reason": None, 
        "ksp_max_it": 10000, 
        "pc_type": PC}

sol = Function(V)
solve(a==L, sol, bcs=bcs, solver_parameters=params)
