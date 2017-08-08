from firedrake import *

from os import listdir, environ
from os.path import isfile, join
import sys
import mpi4py

from seigen.helpers import get_dofs


def print_info(mesh, sd, p, cells, U_tot_dofs, S_tot_dofs):
    if op2.MPI.comm.rank == 0:
        print info % {
            'mesh': mesh,
            'sd': sd,
            'p': p,
            'cells': cells,
            'U': U_tot_dofs,
            'S': S_tot_dofs,
            'Unp': U_tot_dofs / nprocs,
            'Snp': S_tot_dofs / nprocs
        }
    op2.MPI.comm.barrier()


expected = "<mesh,sd,p,Nelements,U,S,U/np,S/np>"
info = "    <%(mesh)s, %(sd)d, %(p)d, %(cells)d, %(U)d, %(S)d, %(Unp)d, %(Snp)d>"

poly = [1, 2, 3]
s_depths = [1, 2, 3, 4]

nprocs = op2.MPI.comm.size

#################################


print "Printing info for RectangleMesh (%s):" % expected
Lx, Ly = 300.0, 150.0
all_h = [2.5, 2.0, 1.0, 0.8, 0.4, 0.2]

for p in poly:
    for sd in s_depths:
        for h in all_h:
            mesh = RectangleMesh(int(Lx/h), int(Ly/h), Lx, Ly)
            mesh.topology.init(s_depth=sd)
            S_tot_dofs, U_tot_dofs = get_dofs(mesh, p)
            print_info(str((Lx, Ly, h)), sd, p, mesh.num_cells(), U_tot_dofs, S_tot_dofs)


#################################

meshes_dir = environ.get('SEIGEN_MESHES')
if not meshes_dir:
    print "Set the environment variable SEIGEN_MESHES to the unstructured meshes directory"
    sys.exit(0)
print "Printing info for UnstructuredMesh in %s (%s):" % (meshes_dir, expected)
meshes = [f for f in listdir(meshes_dir) if isfile(join(meshes_dir, f))]

for p in poly:
    for sd in s_depths:
        for m in meshes:
            mesh = Mesh(join(meshes_dir, m))
            mesh.topology.init(s_depth=sd)
            S_tot_dofs, U_tot_dofs = get_dofs(mesh, p)
            print_info(m, sd, p, mesh.num_cells(), U_tot_dofs, S_tot_dofs)
