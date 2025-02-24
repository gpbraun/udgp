"""
Gabriel Braun, 2024

Este módulo implementa o modelo M3 para instâncias do problema uDGP utilizando Pyomo,
com a implementação do algoritmo DCA (Difference-of-Convex Algorithm) e a restrição
SOC escrita na forma:
    sqrt(sum(v[i,j,l]^2 for l in L)) <= r[i,j],
garantindo que r[i,j] >= 0.
"""

import pyomo.environ as pyo
from pyomo.environ import sqrt

from .base_model import BaseModel


class M3(BaseModel):
    """Modelo M3 para o uDGP com algoritmo DCA e restrição SOC explícita para r."""

    def __init__(self, *args, **kwargs):
        super(M3, self).__init__(*args, **kwargs)

        # Forçar o bound inferior de r a 0 para garantir a equivalência com a restrição SOC
        for i, j in self.IJ:
            self.r[i, j].setlb(0)

        # Variáveis auxiliares adicionais
        self.p = pyo.Var(
            self.IJ,
            self.K,
            within=pyo.Reals,
            bounds=(-self.max_gap, self.max_gap),
        )
        self.w = pyo.Var(
            self.IJ,
            self.K,
            within=pyo.NonNegativeReals,
            bounds=(0, self.max_gap),
        )
        self.z = pyo.Var(
            self.IJ,
            self.K,
            within=pyo.NonNegativeReals,
            bounds=(0, self.d_max),
        )

        # Remover a restrição original de r definida na classe base
        # (a restrição original era: r[i,j]^2 == sum(v[i,j,l]^2 for l in L))
        self.del_component(self.constr_r)

        # Nova restrição SOC: sqrt(sum(v[i,j,l]^2 for l in L)) <= r[i,j]
        @self.Constraint(self.IJ)
        def soc_constr(model, i, j):
            return sqrt(sum(model.v[i, j, l] ** 2 for l in model.L)) <= model.r[i, j]

        # Outras restrições Big‑M que ligam (p, w, z) à atribuição a e à distância r
        @self.Constraint(self.IJ, self.K)
        def constr_x1(model, i, j, k):
            return model.w[i, j, k] >= model.p[i, j, k]

        @self.Constraint(self.IJ, self.K)
        def constr_x2(model, i, j, k):
            return model.w[i, j, k] >= -model.p[i, j, k]

        @self.Constraint(self.IJ, self.K)
        def constr_x3(model, i, j, k):
            return model.z[i, j, k] >= model.r[i, j] - model.d_max * (
                1 - model.a[i, j, k]
            )

        @self.Constraint(self.IJ, self.K)
        def constr_x4(model, i, j, k):
            return model.z[i, j, k] <= model.r[i, j] + model.d_max * (
                1 - model.a[i, j, k]
            )

        @self.Constraint(self.IJ, self.K)
        def constr_x5(model, i, j, k):
            return model.z[i, j, k] >= -model.d_max * model.a[i, j, k]

        @self.Constraint(self.IJ, self.K)
        def constr_x6(model, i, j, k):
            return model.z[i, j, k] <= model.d_max * model.a[i, j, k]

        @self.Constraint(self.IJ, self.K)
        def constr_x7(model, i, j, k):
            return model.z[i, j, k] >= model.dists[k] + model.p[
                i, j, k
            ] - model.d_max * (1 - model.a[i, j, k])

        @self.Constraint(self.IJ, self.K)
        def constr_x8(model, i, j, k):
            return model.z[i, j, k] <= model.dists[k] + model.p[
                i, j, k
            ] + model.d_max * (1 - model.a[i, j, k])

        # Função objetivo original (modelo DC):
        # F(x) = 1 + ∑_{(i,j,k) ∈ IJ×K} w[i,j,k]^2 + ∑_{(i,j) ∈ IJ} r[i,j]^2 - ∑_{(i,j) ∈ IJ} ∑_{l ∈ L} v[i,j,l]^2
        def objective_original(model):
            term1 = 1
            term2 = sum(model.w[i, j, k] ** 2 for (i, j) in model.IJ for k in model.K)
            term3 = sum(model.r[i, j] ** 2 for (i, j) in model.IJ)
            term4 = sum(model.v[i, j, l] ** 2 for (i, j) in model.IJ for l in model.L)
            return term1 + term2 + term3 - term4

        self.obj = pyo.Objective(rule=objective_original, sense=pyo.minimize)

    def solve(self, solver="gurobi", log=False, max_iter=20, tol=1e-5):
        """
        Implementa o algoritmo DCA para resolver o modelo.

        Em cada iteração, resolve-se um problema convexo (com a restrição SOC para r) e lineariza-se a parte
        não convexa da função objetivo, isto é, -∑_{(i,j) ∈ IJ}∑_{l ∈ L} v[i,j,l]^2.

        Parâmetros:
            solver (str): nome do solver (por exemplo, "gurobi").
            log (bool): se True, exibe os logs do solver.
            max_iter (int): número máximo de iterações.
            tol (float): tolerância para convergência da função objetivo.

        Retorna:
            True se encontrar uma solução viável; False caso contrário.
        """
        # Atenção: para que a restrição SOC seja reconhecida, use o solver "gurobi"
        # (e não "gurobi_direct") ou outro solver que suporte SOC de forma nativa.
        opt = pyo.SolverFactory(solver, solver_io="python")
        if "gurobi" in solver.lower():
            opt.options["TimeLimit"] = self.time_limit

        # Resolver inicialmente com o objetivo original
        results = opt.solve(self, tee=log)
        if results.solver.termination_condition == pyo.TerminationCondition.infeasible:
            print("Problema inviável na iteração inicial.")
            return False

        prev_obj = pyo.value(self.obj)
        print(f"Iteração 0: Objetivo = {prev_obj}")

        # Iterações do DCA
        for it in range(1, max_iter + 1):
            # Obter os valores atuais das variáveis v para cada (i,j,l)
            v_current = {
                (i, j, l): pyo.value(self.v[i, j, l])
                for (i, j) in self.IJ
                for l in self.L
            }

            # Construir novo objetivo com linearização da parte não convexa:
            # Linearizamos cada v[i,j,l]^2 por 2*v_current[i,j,l]*v[i,j,l] (desprezando constantes)
            expr = 1
            expr += sum(self.w[i, j, k] ** 2 for (i, j) in self.IJ for k in self.K)
            expr += sum(self.r[i, j] ** 2 for (i, j) in self.IJ)
            expr -= sum(
                2 * v_current[(i, j, l)] * self.v[i, j, l]
                for (i, j) in self.IJ
                for l in self.L
            )

            # Atualizar a função objetivo
            self.del_component(self.obj)
            self.obj = pyo.Objective(expr=expr, sense=pyo.minimize)
            self.reclassify_block()

            results = opt.solve(self, tee=log)
            if (
                results.solver.termination_condition
                == pyo.TerminationCondition.infeasible
            ):
                print(f"Problema inviável na iteração {it}.")
                return False

            curr_obj = pyo.value(self.obj)
            print(f"Iteração {it}: Objetivo = {curr_obj}")

            if abs(prev_obj - curr_obj) < tol:
                print("Critério de convergência atingido.")
                break

            prev_obj = curr_obj

        return True
