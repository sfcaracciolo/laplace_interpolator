from half_edge import HalfEdgeModel
from matplotlib import pyplot as plt
from surface_laplacian import SurfaceLaplacian, operators
from isotropic_remesher import IsotropicRemesher
from open3d.geometry import TriangleMesh
import scipy as sp 
import numpy as np 
import geometric_tools
from geometric_plotter import Plotter
from src.laplace_interpolator import laplace_interp
import pathlib 

rng = np.random.default_rng(seed=0)
filename = pathlib.Path(__file__).stem

# sphere mesh
path = pathlib.Path('data/mesh.npz')
if not path.exists(): 
    sphere = TriangleMesh().create_sphere(radius=1, resolution=20)
    model = HalfEdgeModel(sphere.vertices, sphere.triangles)
    remesher = IsotropicRemesher(model)
    remesher.isotropic_remeshing(
        .1, # L_target, 
        iter=5, 
        explicit=False, 
        foldover=10, # degrees
        sliver=False
    )

    remesher.model.topology_checker(clean=True)
    np.savez(path, vertices=model.vertices, triangles=model.triangles)
else:
    npz = np.load(path)
    model = HalfEdgeModel(npz['vertices'], npz['triangles'])
mesh = TriangleMesh(model.vertices, model.triangles)

N = np.asarray(model.vertices).shape[0]
B = int(N*.05) # amount of bad channels
α = np.unique(rng.integers(0, N, B))

# Laplacian matrix
sl = SurfaceLaplacian(model)
L = sl.stiffness(operators.cotan)
M = sl.mass(operators.mixed)
Δ = sp.sparse.linalg.inv(M) @ L

# smooth functions
spherical = geometric_tools.cartesian_to_spherical_coords(model.vertices)
ρ, θ, φ = spherical[:,0], spherical[:,1], spherical[:,2]
fs = (  θ, np.cos(φ), np.cos(θ), np.cos(φ)*np.sin(θ) )

ylabels = [
    '$θ$',
    '$\cos(φ)$',
    '$\cos(θ)$',
    '$\cos(φ)\sin(θ)$',
]


for method in ('a', 'b'):
    fig, axs = plt.subplots(len(fs), 2, sharex='col', sharey=False, width_ratios=[6, 1], figsize=(10,8))
    for i, f in enumerate(fs):
        Δf = Δ @ f
        f_interp = laplace_interp(f, α, L, method=method)
        error = f[α] - f_interp

        
        axs[i, 0].plot(f[α],'-k')
        axs[i, 0].plot(error,'--k')
        axs[i, 0].set_ylabel(ylabels[i])
        axs[i, 1].boxplot(error, flierprops=dict(marker='.', markerfacecolor='k', markersize=5, markeredgecolor='none'))
        # assert np.allclose(f[α], f_interp, rtol=1e-1)


    axs[0, 0].set_title(f'method {method}')
    axs[-1, 0].set_xlabel('nodes')
    axs[-1, 1].set_xticks([])

    plt.savefig(f"figs/{filename}_{method}.png", dpi = 300, orientation = 'portrait', bbox_inches = 'tight')
    plt.show()

# Plotter.set_export()

# p = Plotter(figsize=(5,5))
# p.add_trisurf(np.asarray(mesh.vertices), np.asarray(mesh.triangles), vertex_values=f, vmin=f.min(), vmax=f.max(), colorbar=True)
# p.camera(view=(0, 0, 0), zoom=1.5)
# p.save(folder='figs/', name=f'{filename}')

# p = Plotter(figsize=(5,5))
# p.add_trisurf(np.asarray(mesh.vertices), np.asarray(mesh.triangles), vertex_values=error, vmin=error.min(), vmax=error.max(), colorbar=True)
# p.camera(view=(0, 0, 0), zoom=1.5)
# p.save(folder='figs/', name=f'{filename}')

# p = Plotter(figsize=(5,5))
# p.add_trisurf(np.asarray(mesh.vertices), np.asarray(mesh.triangles), vertex_values=Δf, vmin=Δf.min(), vmax=Δf.max(), colorbar=True)
# p.camera(view=(0, 0, 0), zoom=1.5)
# p.save(folder='figs/', name=filename)

# Plotter.show()