import numpy as np

def neighbour_distance_matrix(nodes: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """
    Retorna la matriz de distancia para los nodos vecinos. 

    Parámetros
    ----------

    Recibe los nodos y caras que conforman el mesh de una geometría.

    nodes : np.ndarray, shape: (N, 3), dtype=np.float32

    faces : np.ndarray, shape: (M, 3), dtype=np.int32 
    
    Salida
    ------

    Retorna una matriz simétrica con valores nulos para los nodos no vecinos y la distancia entre nodos vecinos. Además, la diagonal es nula.
    
    H : np.ndarray, shape: (N, N), dtype=np.float32 

    """
    distance = lambda x, y: np.linalg.norm(nodes[x]-nodes[y])

    n = nodes.shape[0]
    H = np.zeros((n, n), dtype=np.float32)
    for p1, p2, p3 in faces:

        if H[p1, p2] == 0:
            H[p1, p2] = distance(p1, p2)
            H[p2, p1] = H[p1, p2]
            
        if H[p1, p3] == 0:
            H[p1, p3] = distance(p1, p3)
            H[p3, p1] = H[p1, p3]
            
        if H[p2, p3] == 0:
            H[p2, p3] = distance(p2, p3)
            H[p3, p2] = H[p2, p3]
        
    return H

def laplace_operator(nodes: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """
    Retorna el operador de Laplace para una dada geometría. La construcción del operador se basa en la descripción de (1).
 
    Parámetros
    ----------
    
    Recibe los nodos y caras que conforman el mesh de una geometría.

    nodes : np.ndarray, shape: (N, 3), dtype=np.float32 

    faces : np.ndarray, shape: (M, 3), dtype=np.int32 
    
    Salida
    ------
    
    Retorna una matriz no simétrica con valores nulos para los nodos no vecinos y valores no nulos entre nodos vecinos. Además, la diagonal es no nula.

    L : np.ndarray, shape: (N, N), dtype=np.float32 
    
    1. Oostendorp TF, van Oosterom A, Huiskamp G. Interpolation on a triangulated 3D surface. Journal of Computational Physics. 1989 Feb 1;80(2):331–43. 
    """

    H = neighbour_distance_matrix(nodes, faces)
    
    # a = np.count_nonzero(H, axis=1) # cantidad de vecinos directos de cada nodo.
    b = 1/np.sum(H, axis=1) # promedio de distancias de vecinos directos de cada nodo, sin dividir por el total de vecinos e invertido
    C = np.divide(1, H, out=np.zeros_like(H), where=H!=0) # inversión punto a punto de H, usando 0 en 1/0 = inf.
    c = np.sum(C, axis=1) # promedio de distancias invertidas de vecinos directos de cada nodo, sin dividir por el total de vecinos.

    L = C*b[:, np.newaxis] # cómputo de lij/4
    np.fill_diagonal(L, -c*b) # cómputo de lii/4
    L *= 4
    
    return L

def laplace_interpolation(nodes: np.ndarray, faces: np.ndarray, measured: np.ndarray, bad_channels: np.ndarray, in_place: bool = False) -> np.ndarray:
    """
    Retorna las mediciones interpoladas sobre la geometría utilizando el método B descripto en (1).

    Parámetros
    ----------

    nodes : np.ndarray, shape: (N, 3), dtype=np.float32 

    faces : np.ndarray, shape: (M, 3), dtype=np.int32 
    
    measured : np.ndarray, shape: (N, T), dtype=np.float32.
        Son las mediciones de cada nodo a lo largo del tiempo.

    bad_channels : np.ndarray, shape: (P, ), dtype=np.int32.
        Son los índices de los nodos a interpolar.

    in_place: bool, opcional
        Si es False, retorna una nueva matriz, si es True, modifica "measured" en las filas de los canales mal medidos.

    Salida
    ------

    Retorna una matriz no simétrica con valores nulos para los nodos no vecinos y valores no nulos entre nodos vecinos. Además, la diagonal es no nula.

    interpolated : np.ndarray, shape: (N, T), dtype=np.float32.
        Son las mediciones de la variable "measured" con las señales de los canales malos interpoladas.

    1. Oostendorp TF, van Oosterom A, Huiskamp G. Interpolation on a triangulated 3D surface. Journal of Computational Physics. 1989 Feb 1;80(2):331–43. 
    """

    L = laplace_operator(nodes, faces)

    channels = np.arange(L.shape[0], dtype=np.int32)
    good_channels = np.delete(channels, bad_channels) # [:, np.newaxis]

    L11 = L[np.ix_(bad_channels, bad_channels)]
    L12 = L[np.ix_(bad_channels, good_channels)]
    L21 = L[np.ix_(good_channels, bad_channels)]
    L22 = L[np.ix_(good_channels, good_channels)]

    f2 = np.delete(measured, bad_channels, axis=0)
    Y = -np.vstack((L12, L22))@f2
    A = np.vstack((L11, L21))
    f1 = np.linalg.inv(A.T@A)@A.T@Y
    
    if not in_place:
        interpolated = measured.copy()
        interpolated[bad_channels] = f1
    else:
        measured[bad_channels] = f1
        interpolated = measured

    return interpolated