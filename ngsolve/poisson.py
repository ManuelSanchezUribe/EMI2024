from ngsolve import *
from netgen.occ import *
from sys import argv
from mpi4py import MPI
comm = MPI.COMM_WORLD

## Geometry and Mesh
shape = Rectangle(2,0.41).Circle(0.2,0.2,0.05).Reverse().Face()
shape.edges.name="wall"
shape.edges.Min(X).name="inlet"
shape.edges.Max(X).name="outlet"

geo = OCCGeometry(shape, dim=2)
if comm.rank == 0:
    mesh = Mesh(geo.GenerateMesh(maxh=float(argv[1])).Distribute(comm))
else:
    mesh = Mesh(netgen.meshing.Mesh.Receive(comm))
mesh.Curve(3)
#Draw (mesh);

# Functional space, trial and test functions
V = H1(mesh, order=2, dirichlet="wall|inlet|cyl")
u, v = V.TnT()

# Problem definition
form = InnerProduct(Grad(u), Grad(v))*dx
a = BilinearForm(form)
f = LinearForm(v*dx)
gf = GridFunction(V)

# Boundary condition
uin = CF(1.5*4*y*(0.41-y)/(0.41*0.41))
gf.Set(uin, definedon=mesh.Boundaries("inlet"))

# Solution with simple iterative method
pre = Preconditioner(a, "bddc")
a.Assemble()
f.Assemble()
res = f.vec -a.mat * gf.vec
inv = solvers.CG(a.mat, f.vec, pre=pre.mat, sol=gf.vec, tol=1e-10, printrates='\r', maxsteps=500, printing=comm.rank==0)
# Uncomment to use direct solver instead
#inv = a.mat.Inverse(freedofs=V.FreeDofs(), inverse="umfpack")
#gf.vec.data += inv * res

fesL2 = L2(mesh, order=0)
gfL2 = GridFunction(fesL2)
gfL2.vec.local_vec[:] = comm.rank

#vtk = VTKOutput(ma=mesh,
                #coefs=[gf, gfL2],
                #names = ["u", "domain"],
                #filename="poisson")
# Exporting the results:
#vtk.Do()
