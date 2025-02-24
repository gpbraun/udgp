"""
Gabriel Braun, 2023
This module implements the M3 model for instances of the uDGP using gurobipy,
with a DCA (Difference-of-Convex Algorithm) approach for the continuous relaxation.
The model inherits from GPBaseModel.

IMPORTANT:
To avoid the trivial solution, fix only a minimal number of vertices (e.g. fix vertex 0 at (0,0,0)
and leave others free or fix them to nonzero positions). Alternatively, use the dispersion constraint.
"""

import gurobipy as gp
import numpy as np

from .gp_base_model import GPBaseModel

# SIMP penalty parameter: higher values penalize fractional assignment variables.
PENALTY_MU = 10.0  # adjust as needed

# Dispersion threshold: forces free vertices (Ix) to have a minimum spread.
DISPERSION_THRESHOLD = 1.0  # adjust as needed


class M3gp(GPBaseModel):
    """
    M3 model for uDGP (continuous relaxation with SIMP-like penalty and SOC constraints).

    In this formulation:
      - The original constraint r[i,j]^2 = v[i,j]^T v[i,j] is removed and replaced by a
        second-order cone constraint enforcing:
            r[i,j] >= || v[i,j] ||_2,
        with r[i,j] forced to be nonnegative.
      - The binary assignment variables a are relaxed (a ∈ [0,1]).
      - A new continuous variable yy[k] is added for each k in K.
      - A penalty term is added on (a[i,j,k])² (with weight PENALTY_MU) to encourage integrality.
      - A dispersion constraint is added to force free vertices to have a nontrivial geometry.

    The full DC objective is defined as:
         Obj = [∑ₖ yy[k] + ∑_(i,j ∈ IJ) r[i,j]²]
               - [∑_(i,j ∈ IJ) ∑_(l=0)² (v[i,j][l])² + PENALTY_MU * ∑_(i,j,k ∈ IJK) (a[i,j,k])²].

    Then, a convex surrogate (dropping the concave part) is solved for initialization,
    and DCA iterations linearize the concave part around the current solution.
    """

    def __init__(self, *args, **kwargs):
        # Ensure relaxed mode so that a ∈ [0,1]
        kwargs["relaxed"] = True
        super(M3gp, self).__init__(*args, **kwargs)

        # Force r's lower bound to 0 so that r is nonnegative.
        for i, j in self.IJ:
            self.r[i, j].LB = 0

        # ----------------------------------------------------------------
        # Remove the original r constraint and replace it with a SOC constraint.
        # We enforce:
        #       r[i,j] >= sqrt( (v[i,j][0])² + (v[i,j][1])² + (v[i,j][2])² )
        # by calling addGenConstrSOC.
        # ----------------------------------------------------------------
        self.remove(self.constr_r)
        self.constr_r = {}
        for i, j in self.IJ:
            self.constr_r[(i, j)] = self.addGenConstrSOC(
                self.r[i, j],
                [self.v[i, j][0], self.v[i, j][1], self.v[i, j][2]],
                name=f"soc_{i}_{j}",
            )

        # ----------------------------------------------------------------
        # Add a new continuous variable yy for each k in K.
        # ----------------------------------------------------------------
        self.yy = self.addVars(self.K, name="yy", lb=0, vtype=gp.GRB.CONTINUOUS)

        # ----------------------------------------------------------------
        # Add a dispersion constraint on the free vertices (Ix) to avoid the trivial solution.
        # This forces the sum of squared norms of the free vertices to be at least DISPERSION_THRESHOLD.
        # ----------------------------------------------------------------
        disp_expr = gp.quicksum(
            self.x[i][l] * self.x[i][l] for i in self.Ix for l in range(3)
        )
        self.addConstr(disp_expr >= DISPERSION_THRESHOLD, name="dispersion")

        # ----------------------------------------------------------------
        # Define the full DC objective.
        # Let:
        #   f(x) = ∑ₖ yy[k] + ∑_(i,j ∈ IJ) r[i,j]²    (convex part)
        #   g(x) = ∑_(i,j ∈ IJ)∑_(l=0)² (v[i,j][l])² + PENALTY_MU * ∑_(i,j,k ∈ IJK) (a[i,j,k])²  (convex)
        # Then, Obj = f(x) - g(x).
        # ----------------------------------------------------------------
        convex_part = gp.quicksum(self.yy[k] for k in self.K) + gp.quicksum(
            self.r[i, j] * self.r[i, j] for (i, j) in self.IJ
        )
        concave_part = gp.quicksum(
            self.v[i, j][l] * self.v[i, j][l] for (i, j) in self.IJ for l in range(3)
        ) + PENALTY_MU * gp.quicksum(
            self.a[i, j, k] * self.a[i, j, k] for (i, j, k) in self.IJK
        )
        full_obj = convex_part - concave_part
        self.setObjective(full_obj, gp.GRB.MINIMIZE)
        self.update()

    def solve(self, max_iter=20, tol=1e-5, log=False):
        """
        Solves the continuous relaxation using a DCA approach.

        Overall, the DC objective is:
             Obj = f(x) - g(x), where
             f(x) = ∑ₖ yy[k] + ∑_(i,j ∈ IJ) r[i,j]²    (convex)
             g(x) = ∑_(i,j ∈ IJ)∑_(l=0)² (v[i,j][l])² +
                    PENALTY_MU * ∑_(i,j,k ∈ IJK) (a[i,j,k])²    (convex).

        Step 1: Convex Initialization.
          Solve the convex surrogate by dropping g(x):
              Obj_init = ∑ₖ yy[k] + ∑_(i,j ∈ IJ) r[i,j]².
          This provides an initial solution.

        Step 2: DCA Iterations.
          At each iteration, linearize the concave part g(x) around the current solution.
          For each (i,j) and each component l, the derivative of (v[i,j][l])² is 2*v[i,j][l]_t.
          For each (i,j,k), the derivative of (a[i,j,k])² is 2*a[i,j,k]_t.
          Replace -g(x) by:
              - ∑_(i,j)∑_(l) 2*v_t[i,j][l]*v[i,j][l] - PENALTY_MU * ∑_(i,j,k) 2*a_t[i,j,k]*a[i,j,k].
          The updated objective becomes:
              Obj = f(x) + (linearized - g(x)),
          which is convex.
        """
        # --- Step 1: Convex Initialization ---
        init_obj = gp.quicksum(self.yy[k] for k in self.K) + gp.quicksum(
            self.r[i, j] * self.r[i, j] for (i, j) in self.IJ
        )
        self.setObjective(init_obj, gp.GRB.MINIMIZE)
        self.update()

        # Set solver parameters for convex subproblems.
        self.Params.LogToConsole = log
        self.Params.TimeLimit = self.time_limit
        self.Params.NonConvex = 0  # Force convex mode.
        self.Params.MIPGap = 1e-6
        self.Params.IntFeasTol = self.max_tol
        self.Params.FeasibilityTol = self.max_tol
        self.Params.OptimalityTol = self.max_tol

        self.optimize()
        if self.Status == gp.GRB.INFEASIBLE or self.SolCount == 0:
            print("Initial convex solution infeasible.")
            return False
        prev_obj = self.ObjVal
        print("Iteration 0 (convex init): Objective =", prev_obj)

        # --- Step 2: DCA Iterations ---
        for it in range(1, max_iter + 1):
            # Extract current values of v and a.
            v_vals = {
                (i, j): [self.v[i, j][l].X for l in range(3)] for (i, j) in self.IJ
            }
            a_vals = {(i, j, k): self.a[i, j, k].X for (i, j, k) in self.IJK}

            # f(x) remains:
            f_part = gp.quicksum(self.yy[k] for k in self.K) + gp.quicksum(
                self.r[i, j] * self.r[i, j] for (i, j) in self.IJ
            )
            # Linearize the concave part g(x):
            lin_v = gp.quicksum(
                -2 * v_vals[(i, j)][l] * self.v[i, j][l]
                for (i, j) in self.IJ
                for l in range(3)
            )
            lin_a = gp.quicksum(
                -2 * PENALTY_MU * a_vals[(i, j, k)] * self.a[i, j, k]
                for (i, j, k) in self.IJK
            )
            lin_concave = lin_v + lin_a

            new_obj = f_part + lin_concave
            self.setObjective(new_obj, gp.GRB.MINIMIZE)
            self.update()

            self.optimize()
            if self.Status == gp.GRB.INFEASIBLE or self.SolCount == 0:
                print(f"Subproblem infeasible at iteration {it}.")
                return False
            curr_obj = self.ObjVal
            print(f"Iteration {it}: Objective =", curr_obj)
            print("x solution:", np.array([self.x[i].X for i in self.Ix]))
            if abs(prev_obj - curr_obj) < tol:
                print("Convergence reached.")
                break
            prev_obj = curr_obj

        return True
