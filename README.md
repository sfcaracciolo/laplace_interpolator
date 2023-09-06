# Laplace Interpolator
Laplace interpolator based on ["Interpolation on a triangulated 3D surface"](https://doi.org/10.1016/0021-9991(89)90103-4) paper written by Thom F. Oostendorp et al .

### Usage

```python
from laplace_interpolator import laplace_interp

f_interp = laplace_interp(
    f, # values on each vertex
    α, # array of vertex indices where interpolate
    L, # discrete laplace matrix. See https://github.com/sfcaracciolo/surface_laplacian
    method='a' # can be a or b. See https://doi.org/10.1016/0021-9991(89)90103-4
)
```
### Examples
Both figures show the $f$ at $\alpha$ nodes in solid line and the absolute error in dashed line. Each row is a example with different $f$ function. The boxplots represent the error distribution.
<img src="/figs/interp_a.png" alt="drawing" width=""/>
<img src="/figs/interp_b.png" alt="drawing" width=""/>