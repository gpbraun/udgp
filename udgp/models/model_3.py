import numpy as np
import pyomo.environ as pyo

from .base_model import BaseModel


class M3(BaseModel):
    """
    Modelo M3 'do zero', com:
      - Primeira resolução do problema não convexo (f_expr - g_expr),
        usando Gurobi NonConvex=2.
      - Iterações subsequentes de DCA (Difference of Convex functions Algorithm),
        onde o objetivo é f_expr + aproximacao_linear(-g_expr).
    """

    def __init__(self, *args, **kwargs):
        super(M3, self).__init__(*args, **kwargs)

        # 1) Variáveis adicionais (p, w, z), seguindo a ideia do model_3.py
        self.p = pyo.Var(
            self.IJ, self.K, within=pyo.Reals, bounds=(-self.max_gap, self.max_gap)
        )
        self.w = pyo.Var(
            self.IJ, self.K, within=pyo.NonNegativeReals, bounds=(0, self.max_gap)
        )
        self.z = pyo.Var(
            self.IJ, self.K, within=pyo.NonNegativeReals, bounds=(0, self.d_max)
        )

        # 2) Ajustar restrição r[i,j]^2 >= sum(v[i,j,l]^2)
        if hasattr(self, "constr_r"):
            self.del_component(self.constr_r)

        @self.Constraint(self.IJ)
        def constr_r(m, i, j):
            return m.r[i, j] ** 2 >= sum(m.v[i, j, l] ** 2 for l in m.L)

        # 3) Demais restrições x1..x8
        @self.Constraint(self.IJ, self.K)
        def constr_x1(m, i, j, k):
            return m.w[i, j, k] >= m.p[i, j, k]

        @self.Constraint(self.IJ, self.K)
        def constr_x2(m, i, j, k):
            return m.w[i, j, k] >= -m.p[i, j, k]

        @self.Constraint(self.IJ, self.K)
        def constr_x3(m, i, j, k):
            return m.z[i, j, k] >= m.r[i, j] - m.d_max * (1 - m.a[i, j, k])

        @self.Constraint(self.IJ, self.K)
        def constr_x4(m, i, j, k):
            return m.z[i, j, k] <= m.r[i, j] + m.d_max * (1 - m.a[i, j, k])

        @self.Constraint(self.IJ, self.K)
        def constr_x5(m, i, j, k):
            return m.z[i, j, k] >= -m.d_max * m.a[i, j, k]

        @self.Constraint(self.IJ, self.K)
        def constr_x6(m, i, j, k):
            return m.z[i, j, k] <= m.d_max * m.a[i, j, k]

        @self.Constraint(self.IJ, self.K)
        def constr_x7(m, i, j, k):
            return m.z[i, j, k] >= (
                m.dists[k] + m.p[i, j, k] - m.d_max * (1 - m.a[i, j, k])
            )

        @self.Constraint(self.IJ, self.K)
        def constr_x8(m, i, j, k):
            return m.z[i, j, k] <= (
                m.dists[k] + m.p[i, j, k] + m.d_max * (1 - m.a[i, j, k])
            )

        # 4) Definições f_expr, g_expr
        def _f_rule(m):
            # f(X) = 1 + Σ(w^2) + Σ(r^2)
            return (
                1
                + sum(m.w[i, j, k] ** 2 for i, j, k in m.IJ * m.K)
                + sum(m.r[i, j] ** 2 for i, j in m.IJ)
            )

        self.f_expr = pyo.Expression(rule=_f_rule)

        def _g_rule(m):
            # g(X) = Σ(v^2)
            return sum(m.v[i, j, l] ** 2 for i, j in m.IJ for l in m.L)

        self.g_expr = pyo.Expression(rule=_g_rule)

        # Removemos qualquer objeto "obj" se existir (para segurança)
        if hasattr(self, "obj"):
            self.del_component(self.obj)

    def solve(self, solver="gurobi", log=False, max_dca_iters=100, dca_tol=1e-6):
        """
        Passo 1: Resolve o problema não convexo  min [f_expr - g_expr].
                 (Gurobi com NonConvex=2, se solver=gurobi)
        Passo 2: Itera DCA, substituindo -g_expr por sua aproximação linear
                 no ponto v^k, para k=1..max_dca_iters.

        Retorna True se não houve inviabilidade; False caso contrário.
        """
        opt = pyo.SolverFactory(solver, solver_io="python")
        if solver.lower().startswith("gurobi"):
            opt.options["TimeLimit"] = self.time_limit

        # --------------------------------------------------
        # (A) Primeira fase: problema não convexo original
        #     obj = f_expr - g_expr
        # --------------------------------------------------
        # 1) Se existir self.obj, remover
        if hasattr(self, "obj"):
            self.del_component(self.obj)

        opt.options["NonConvex"] = 2
        opt.options["MIPFocus"] = 1
        opt.options["Heuristics"] = 0.5
        opt.options["SolutionLimit"] = 20

        # 2) Criar e anexar a objective
        self.obj = pyo.Objective(expr=self.f_expr - self.g_expr, sense=pyo.minimize)

        # 3) Resolver
        results = opt.solve(self, tee=log)
        if results.solver.termination_condition == pyo.TerminationCondition.infeasible:
            print("[M3] ERRO: primeira fase inviável.")
            return False

        # 4) Guardar old_v
        old_v = {}
        for i, j in self.IJ:
            for l in self.L:
                val = self.v[i, j, l].value
                if val is None:
                    val = 0.0
                old_v[(i, j, l)] = val

        # 5) Remover a objective para não causar 'multiple objectives'
        self.del_component(self.obj)

        # --------------------------------------------------
        # (B) Iterações de DCAt
        # --------------------------------------------------
        opt.options["NonConvex"] = 1
        opt.options["SolutionLimit"] = 1000
        opt.options["MIPFocus"] = 0
        opt.options["Heuristics"] = 0.05

        for it in range(max_dca_iters):
            # (1) Subgradiente de -g(X) = -2 * v^k
            subgrad = {}
            for key, v_val in old_v.items():
                subgrad[key] = -2.0 * v_val

            # (2) Definir objective = f_expr + Σ(subgrad·v).
            if hasattr(self, "obj"):
                self.del_component(self.obj)

            def dca_objective_rule(m):
                linear_part = 0
                for (i, j, l), grad_val in subgrad.items():
                    linear_part += grad_val * m.v[i, j, l]
                return m.f_expr + linear_part

            self.obj = pyo.Objective(rule=dca_objective_rule, sense=pyo.minimize)

            # (3) Resolve
            results = opt.solve(self, tee=log, warmstart=True)
            if (
                results.solver.termination_condition
                == pyo.TerminationCondition.infeasible
            ):
                print(f"[DCA] it={it}: inviável.")
                return False

            # (4) Calcula diferença
            new_v = {}
            max_diff = 0.0
            for i, j in self.IJ:
                for l in self.L:
                    val = self.v[i, j, l].value
                    if val is None:
                        val = 0.0
                    new_v[(i, j, l)] = val
                    diff = abs(val - old_v[(i, j, l)])
                    if diff > max_diff:
                        max_diff = diff

            if log:
                print(f"[DCA] it={it}, max_diff={max_diff:.4g}")

            # Atualiza old_v
            old_v = new_v

            # (5) Critério de parada
            if it >= 1 and max_diff < dca_tol:
                if log:
                    print(f"[DCA] Convergência na iteração {it}, diff={max_diff:.2e}")
                break

            # Remover obj para não manter vários de uma só vez
            self.del_component(self.obj)

        return True
