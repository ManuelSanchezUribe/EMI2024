from ngsolve import *
from netgen.occ import *

# Geometry
shape = Rectangle(2,0.41).Circle(0.2,0.2,0.05).Reverse().Face()
shape.edges.name="wall"
shape.edges.Min(X).name="inlet"
shape.edges.Max(X).name="outlet"

geo = OCCGeometry(shape, dim=2)
mesh = Mesh(geo.GenerateMesh(maxh=0.1))
mesh.Curve(3)

# Function Spaces (Taylor-Hood)
V = VectorH1(mesh, order=2, dirichlet="wall|inlet|cyl")
Q = L2(mesh, order=1)
X = V*Q
(u,p),(v,q) = X.TnT()

# Problem definition
stokes = InnerProduct(Grad(u), Grad(v))*dx + div(u)*q*dx + div(v)*p*dx
a = BilinearForm(stokes)
gf = GridFunction(X)
gfu, gfp = gf.components
uin = CF( (1.5*4*y*(0.41-y)/(0.41*0.41), 0) )
gfu.Set(uin, definedon=mesh.Boundaries("inlet"))

pre = Preconditioner(a, "local")
a.Assemble()
res = -a.mat * gf.vec
inv = CGSolver(a.mat, pre.mat, precision=1e-6, printrates=True)
gf.vec.data += inv * res

vtk = VTKOutput(ma=mesh,
                coefs=[gfu, gfp],
                names = ["u", "p"],
                filename="stokes")
# Exporting the results:
vtk.Do()

print("CG iterations")
