# Unassigned Distance Geometry

Uma nova formulação a partir do artigo

*Duxbury, Lavor, Liberti, Salles-Neto, Unassigned distance geometry and molecular conformation problems, Journal of Global Optimization, v.83, pp: 73-82, 2022.*

A ideia é substituir a função objetivo do modelo (3) no artigo acima por:
$$
    \min \sum_{i=1}^{n-1} \sum_{j=i+1}^{n} \left( \sum_{k=1}^{m} a_{ij}^k \big\vert \Vert x_i - x_j \Vert_2 - d_k \big\vert \right),
$$
onde $x_i = (x_{i,1}, x_{i,2}, x_{i,3})^\mathsf{T}$, $i = 1, 2, \ldots, n$.

O novo modelo:
$$
    \text{(NP):} \quad \min \sum_{i=1}^{n-1} \sum_{j=i+1}^{n} \left( \sum_{k=1}^{m} z_{ijk} \right) - S,
$$
sujeito a:
$$
\begin{aligned}
    &\text{(C1):} & z_{ijk}^2 &= a_{ij}^k \sum_{l=1}^3 ( x_{i,l} - x_{j,l} )^2 \\
    &\text{(C2):} & z_{ijk} &\leq a_{ij}^k D \\
    &\text{(C3):} & z_{ijk} &\leq d_k + (1-a_{ij}^k) D \\
    &\text{(C4):} & z_{ijk} &\geq d_k - (1-a_{ij}^k) D \\
\end{aligned}
$$
para $i = 1, 2, \ldots, n−1,\; j = i+1, i+2, \ldots, n,\; k = 1, 2, \ldots, m$

Mantendo as restrições do modelo (3):
$$
\begin{aligned}
    &\text{(C5):} & \sum_{i=1}^{n-1} \sum_{j=1+1}^{n} a_{ij}^k &= 1 && k = 1, 2, \ldots, m, \\
    &\text{(C6):} & \sum_{k=1}^{m} a_{ij}^k &\leq 1 && i = 1, 2, \ldots, n−1,\; j = i+1, i+2, \ldots, n,
\end{aligned}
$$