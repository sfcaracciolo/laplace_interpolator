from typing import Literal
import numpy as np
import scipy as sp

def laplace_interp(values: np.ndarray, α: np.ndarray, L: sp.sparse.sparray, method: Literal['a', 'b'] = 'a'):
    """
    α: indices to interpolate
    """
    N = values.shape[0]
    ix = np.arange(N)
    β = np.delete(ix, α)
    Fβ = values[β]

    if method == 'a':
        L11 = L[np.ix_(α, α)]
        L12 = L[np.ix_(α, β)]
        B = - L12 @ Fβ
        Fα = sp.linalg.solve(L11.todense(), B, assume_a='gen')

    if method == 'b':
        B = - L[:, β] @ Fβ
        Fα, _, _, _ = sp.linalg.lstsq(L[:, α].todense(), B)
    
    return Fα