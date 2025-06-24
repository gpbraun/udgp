"""
udgp.solvers.gp_m3
──────────────────
Modelo M3 - warm-start com M2 (não-convexo) seguido de
Difference-of-Convex Algorithm usando a relaxação convexa: ||v_ij|| ≤ r_ij.
"""

import logging

import gurobipy as gp
import numpy as np

from udgp.utils import points_dists

from .gp_model_2 import gpM2

logger = logging.getLogger("gpM3")


class gpM3(gpM2):
    """
    uDGP model M3.
    """

    NAME = "M3"
    PARAMS = {
        "Mu": 0.05,
        "MaxIter": 100,
        "DCATol": 1.0e-4,
    }

    def dca_init(
        self,
        *,
        solver_params: dict | None = None,
    ) -> bool:
        """
        Encontra o ponto de partida.

        Retorna: verdadeiro se a solução inicial foi encontrada com sucesso.
        """
        return super().solve(stage="dca_init", solver_params=solver_params)

    def solve(
        self,
        *,
        stage: str | None = None,
        solver_params: dict | None = None,
    ) -> bool:
        """
        Retorna: verdadeiro se uma solução foi encontrada
        """
        dca_init_ok = self.dca_init()

        if not dca_init_ok:
            logger.warning("warm-start não achou solução.")
            return False

        logger.debug("it 0 (M2)  obj %.6g", self.ObjVal)

        # ponteiros para v
        v_vars = [self.v[i, j][l] for (i, j) in self.IJ for l in range(3)]
        v_prev = np.array([var.X for var in v_vars], dtype=float)
        prev_obj = self.ObjVal

        # ---------- relaxação convexa ||v|| ≤ r -----------------------
        self.remove(self._constr_r)
        self._constr_r = self.addConstrs(
            self.v[i, j] @ self.v[i, j] <= self.r[i, j] ** 2 for i, j in self.IJ
        )
        self.update()

        # parte convexa f
        mu = self.model_params["Mu"]
        f_expr = gp.quicksum(w**2 for w in self.w.values()) + mu * gp.quicksum(
            r**2 for r in self.r.values()
        )

        # ---------- parâmetros DCA -----------------------------------
        self.set_solver_params(stage="dca_iter", overrides=solver_params)

        max_iter = self.model_params["MaxIter"]
        tol = self.model_params["DCATol"]

        # ---------- loop DCA -----------------------------------------
        for it in range(1, max_iter + 1):
            lin = gp.quicksum(2.0 * vv * var for vv, var in zip(v_prev, v_vars))
            self.setObjective(f_expr - mu * lin, gp.GRB.MINIMIZE)
            super().optimize()
            if self.SolCount == 0:
                return False

            obj = self.ObjVal
            logger.debug("it %-2d  obj=%.6g", it, obj)

            if abs(prev_obj - obj) / prev_obj < tol:
                logger.info("Convergiu (Δ<%g) em %d it.", tol, it)
                break

            prev_obj = obj
            v_prev = np.array([var.X for var in v_vars], dtype=float)

            self.total_runtime += self.Runtime
            self.total_work += self.Work

        print("distâncias:")
        d = np.array([self.r[i, j].X for i, j in self.IJ])
        d.sort()
        print(d)

        print("distâncias calculadas:")
        x = np.array([self.x[i].X for i in self.Ix])
        x = np.vstack((x, np.array([0, 0, 0])))
        print(points_dists(x))

        return True
