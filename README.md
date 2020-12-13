# laplace_interpolator
Laplace interpolator based on "Interpolation on a triangulated 3D surface" paper written by Thom F. Oostendorp et al. The source code compute an interpolation of bad channels based on Laplace operator. The only dependece is Numpy and the implementation use broadcasting rules.

This code is used to ECGI (Electrocardiographic Imaging) research. Possibly some improvements/optimizations can be made using symmetric and sparse matrix structure in cases with geometries has several points but typical cases with 100 to 2000 nodes not requiere this optimizations.

# Usage

Clone the repository in some PATH and import it in your proyect as:

```python
import sys
sys.path.insert(0,PATH+'laplace_interpolator/src/') 
from laplace_interpolator import laplace_interpolator

interpolated = laplace_interpolator((nodes, faces), measured, bad_channels, copy=True)
```
Please, if you use this fragment, contact me at scaracciolo@conicet.gov.ar