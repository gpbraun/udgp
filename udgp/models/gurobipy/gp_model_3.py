"""
udgp.solvers.gp_m3
──────────────────
Modelo M3 - warm-start com M2 (não-convexo) seguido de
Difference-of-Convex Algorithm usando a relaxação convexa: ||v_ij|| ≤ r_ij.
"""

import logging

import gurobipy as gp
import numpy as np

from .gp_model_2 import gpM2

logger = logging.getLogger("gpM3")


class gpM3(gpM2):
    """
    uDGP model M3.
    """

    NAME = "M3"
    PARAMS = {
        **gpM2.PARAMS,
        "Mu": 0.001,
        "DCAMaxIter": 1000,
        "DCATol": 1.0e-6,
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
        v_vars = self.v.values()
        v_prev = np.array([v.X for v in self.v.values()])
        prev_obj = self.ObjVal

        # ---------- relaxação convexa ||v|| ≤ r -----------------------
        self.remove(self._constr_r)
        self._constr_r = self.addConstrs(
            self.v[i, j] @ self.v[i, j] <= self.r[i, j] for i, j in self.IJ
        )

        self.update()

        # parte convexa f
        mu = self.model_params["Mu"]

        # ---------- parâmetros DCA -----------------------------------
        self._set_solver_params(stage="dca_iter", overrides=solver_params)

        max_iter = self.model_params["DCAMaxIter"]
        tol = self.model_params["DCATol"]

        # ---------- loop DCA -----------------------------------------
        for it in range(1, max_iter + 1):

            self.objective = (
                gp.quicksum(w for w in self.w.values())
                + mu * gp.quicksum(r for r in self.r.values())
                - mu * gp.quicksum(2.0 * vv @ var for vv, var in zip(v_prev, v_vars))
            )

            super().optimize()
            if self.SolCount == 0:
                return False

            obj = self.ObjVal
            logger.debug("it %-2d  obj=%.6g", it, obj)

            if abs(prev_obj - obj) / (prev_obj + 1e-9) < tol:
                logger.info("Convergiu (Δ<%g) em %d it.", tol, it)
                break

            prev_obj = obj
            v_prev = np.array([v.X for v in self.v.values()])

            self.total_runtime += self.Runtime
            self.total_work += self.Work

        # LOGS AUXILIARES PARA DESENVOLVIMENTO!
        np.set_printoptions(
            precision=3,
            suppress=True,
            linewidth=120,
        )

        def _log_section():
            logger.info("")
            logger.info("=" * 80)
            logger.info("")

        _log_section()

        logger.info(f"OBJECTIVE: {self.ObjVal}")

        _log_section()

        var_r = np.sqrt(np.sort(np.array([r.X for r in self.r.values()])))
        calc_r = np.sort(np.linalg.norm(self.sol_v_array, axis=1))
        input_r = np.sqrt(np.sort(np.repeat(self.dists, self.freqs)))

        logger.info(f"Rv   {var_r}")
        logger.info(f"Rc   {calc_r}")
        logger.info(f"Ri   {input_r}")

        logger.info("")

        err2_vc = np.sort(abs(calc_r**2 - var_r**2))
        err2_ic = np.sort(abs(calc_r**2 - input_r**2))
        err2_iv = np.sort(abs(input_r**2 - var_r**2))

        logger.info(f"E2vc {err2_vc}")
        logger.info(f"E2ic {err2_ic}")
        logger.info(f"E2iv {err2_iv}")

        _log_section()

        w = np.sort(np.array([w.X for w in self.w.values()]))
        err_iv = np.sort(abs(input_r - var_r))

        logger.info(f"Eiv  {err_iv}")
        logger.info(f"y:   {w[w != 0]}")

        _log_section()

        return True
