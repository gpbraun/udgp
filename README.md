# Novas formulações para o uDGP

Uma nova formulação a partir do artigo

*Duxbury, Lavor, Liberti, Salles-Neto, Unassigned distance geometry and molecular conformation problems, Journal of Global Optimization, v.83, pp: 73-82, 2022.*

A ideia é substituir a função objetivo do modelo (3) no artigo acima por:
$$
    \min \sum_{i=1}^{n-1} \sum_{j=i+1}^{n} \left( \sum_{k=1}^{m} a_{ij}^k \big\vert \Vert x_i - x_j \Vert_2 - d_k \big\vert \right),
$$
onde $x_i = (x_{i,1}, x_{i,2}, x_{i,3})^\mathsf{T}$, $i = 1, 2, \ldots, n$.

## Instalação

Ativar o ambiente do conda.

```shell
conda env create --file=environment.yml
```

