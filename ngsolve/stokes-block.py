from ngsolve import *
from netgen.occ import *
from mpi4py import MPI
from sys import argv

comm = MPI.COMM_WORLD
maxh = float(argv[1])
# Geometry
shape = Rectangle(2,0.41).Circle(0.2,0.2,0.05).Reverse().Face()
shape.edges.name="wall"
shape.edges.Min(X).name="inlet"
shape.edges.Max(X).name="outlet"

geo = OCCGeometry(shape, dim=2)
if comm.rank == 0:
    mesh = Mesh(geo.GenerateMesh(maxh=maxh).Distribute(comm))
else:
    mesh = Mesh(netgen.meshing.Mesh.Receive(comm))


mesh.Curve(3)
V = VectorH1(mesh, order=1, dirichlet="wall|inlet|cyl")
Q = H1(mesh, order=1)

#if comm.rank==0: print("u DoFs:", V.ndofglobal)
#if comm.rank==0: print("p DoFs:", Q.ndofglobal)
if comm.rank==0: print("Total DoFs:", V.ndofglobal + Q.ndofglobal)

u,v = V.TnT()
p,q = Q.TnT()


a = BilinearForm(InnerProduct(Grad(u),Grad(v))*dx, symmetric=True)
b = BilinearForm(div(u)*q*dx).Assemble()
h = specialcf.mesh_size
c = BilinearForm(-0.1*h*h*grad(p)*grad(q)*dx, symmetric=True).Assemble()

mp = BilinearForm(p*q*dx, symmetric=True)
f = LinearForm(V).Assemble()
g = LinearForm(Q).Assemble();
gfu = GridFunction(V, name="u")
gfp = GridFunction(Q, name="p")
uin = CoefficientFunction( (1.5*4*y*(0.41-y)/(0.41*0.41), 0) )
gfu.Set(uin, definedon=mesh.Boundaries("inlet"))


Qjacobi = Preconditioner(mp, "local")
Amg = Preconditioner(a, "hypre")
a.Assemble()
mp.Assemble()
f.Assemble()
g.Assemble()
K = BlockMatrix( [ [a.mat, b.mat.T], [b.mat, c.mat] ] )
C = BlockMatrix( [ [Amg.mat, None], [None, Qjacobi.mat] ] )
#C = BlockMatrix( [ [a.mat.Inverse(freedofs=V.FreeDofs()), None], [None, mp.mat.Inverse()] ] )

rhs = BlockVector ( [f.vec, g.vec] )
sol = BlockVector( [gfu.vec, gfp.vec] )

from time import perf_counter as time
t0 = time()
solvers.MinRes (mat=K, pre=C, rhs=rhs, sol=sol, printrates=comm.rank==0, initialize=False, maxsteps=500);
tf = time() - t0
sol_time = 0.0
sol_time = comm.allreduce(tf, op=MPI.MAX)

#vtk = VTKOutput(ma=mesh,
                #coefs=[gfu, gfp],
                #names = ["u", "p"],
                #filename="stokes")
# Exporting the results:
#vtk.Do()
if comm.rank==0: print("Solved in {:1.4f} seconds".format(sol_time))
