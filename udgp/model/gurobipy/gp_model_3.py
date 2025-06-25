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
        "Mu": 1.2,
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
            # self.objective = 0

            # self.setObjectiveN(
            #     self.w.sum(),
            #     index=0,
            #     priority=1,
            #     abstol=1e-2,
            #     name="assign_err",
            # )
            # self.setObjectiveN(
            #     self.r.sum()
            #     - gp.quicksum(2.0 * vv @ var for vv, var in zip(v_prev, v_vars)),
            #     index=1,
            #     priority=2,
            #     weight=mu,
            #     abstol=1e-2,
            #     name="geom_energy",
            # )

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
        def _log_section():
            logger.info("")
            logger.info("=" * 80)
            logger.info("")

        def _log_array_row(header, array):
            logger.info(
                f"{header:<4}"
                + "".join(
                    [
                        f"{0:>7}" if np.isclose(x, 0, atol=1e-4) else f"{x:>7.3f}"
                        for x in array
                    ]
                )
            )

        _log_section()

        logger.info(f"OBJECTIVE: {self.ObjVal}")

        _log_section()

        ij_indices = self.IJ
        k_indices = [self.assignments.get((i, j)) for i, j in ij_indices]

        logger.info(f"i, j" + "".join([f"    {i},{j}" for i, j in ij_indices]))
        logger.info(f"k   " + "".join([f"{k:>7}" for k in k_indices]))

        logger.info("-" * (len(ij_indices) + 1) * 7)

        # r - variables
        r_v = np.sqrt(np.array([r.X for r in self.r.values()]))
        # r - calculated from points
        r_c = np.linalg.norm(self.sol_v_array, axis=1)
        # r - input
        r_i = np.sqrt([self.dists[k] for k in k_indices])

        _log_array_row("Rv", r_v)
        _log_array_row("Rc", r_c)
        _log_array_row("Ri", r_i)

        logger.info("-" * (len(ij_indices) + 1) * 7)

        err2_vc = abs(r_c**2 - r_v**2)
        err2_ic = abs(r_c**2 - r_i**2)
        err2_iv = abs(r_i**2 - r_v**2)
        w = np.array([w.X for w in self.w.values()])
        p = np.array([p.X for p in self.p.values()])

        _log_array_row("E2ci", err2_ic)
        _log_array_row("E2cv", err2_vc)
        _log_array_row("E2vi", err2_iv)
        _log_array_row("y", w)
        _log_array_row("alph", p)

        # _log_section()

        # err_vc = abs(calc_r - var_r)
        # err_ic = abs(calc_r - input_r)
        # err_iv = abs(input_r - var_r)

        # logger.info(f"Eci  {err_ic}")
        # logger.info(f"Ecv  {err_vc}")
        # logger.info(f"Evi  {err_iv}")

        _log_section()

        return True
