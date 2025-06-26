import gurobipy as gp
import numpy as np

from udgp import Instance

env = gp.Env()


# HISTÓRICO:
# NÃO ACHO MAIS UMA BOA IDEIA PRA ITERAÇÃO DA HEURÍSTICA...
def add_previous_solution_constrs(self, previous_a: list[tuple[int]]):
    """
    Adds constraints to prevent previous solutions.
    """
    self._constr_previous_a = self.addConstrs(
        gp.quicksum(self.a[ijk] for ijk in a_ijk_idx) <= len(a_ijk_idx) - 1
        for a_ijk_idx in previous_a
    )


def np_value(x):
    return np.format_float_positional(
        x, precision=3, unique=False, fractional=False, trim="k"
    )


def get_results(runtimes):
    if not runtimes:
        print("todos falharam", flush=True)
        return
    runtime_mean = np_value(np.mean(runtimes))
    runtime_std = np_value(np.std(runtimes))
    runtime_min = np_value(np.min(runtimes))
    runtime_max = np_value(np.max(runtimes))
    print(f"{runtime_mean} & {runtime_std} & {runtime_min} & {runtime_max}", flush=True)


def solve_heuristic(instance, model):
    i = 0
    previous_a = []
    instance.reset(reset_runtime=True)

    while instance.fixed_n < instance.n:
        if instance.runtime > 5000:
            return False, 0

        i += 1

        # instance.reset_with_core("mock")

        instance.reset()
        instance.solve_step(model, nx=4, max_gap=1e-2, previous_a=previous_a)
        previous_a.append(instance.a_indices.tolist())

        broken = False
        while instance.fixed_n < instance.n and not broken:
            solved = False
            solved = instance.solve_step(
                model,
                nx=1,
                # ny=4,
                max_gap=1e-1,
                max_threshold=100,
            )

            if not solved:
                broken = True
                break

    return True, i


SEEDS = np.array([12345678910, 12345, 123456])


def tests(N_range, seeds, model, freq, instance):
    for N in N_range:
        print(f"========== {N} átomos ==========")
        times = []
        nums_steps = []

        for seed in seeds:
            if instance == "LJ":
                instance = Instance.lj_cluster(N, freq=freq)
            else:
                instance = Instance.artificial_molecule(N, freq=freq, seed=seed)

            solved, num_steps = solve_heuristic(instance, model)

            if solved:
                print(f"{num_steps} steps - {np_value(instance.runtime)} s", flush=True)
                nums_steps.append(num_steps)
                times.append(instance.runtime)
            else:
                print("timeout", flush=True)

        get_results(nums_steps)
        get_results(times)


N_range = [6, 7, 8, 9, 10, 20, 50]

print(f"\n\n========== NT2: LJ ==========")
tests(N_range, SEEDS, "M2GP", freq=True, instance="LJ")

print(f"\n\n========== NT2: Lavor ==========")
tests(N_range, SEEDS, "M2GP", freq=True, instance="LAVOR")

print(f"\n\n========== NT1: LJ ==========")
tests(N_range, SEEDS, "M1RGP", freq=False, instance="LJ")

print(f"\n\n========== NT1: Lavor ==========")
tests(N_range, SEEDS, "M1RGP", freq=False, instance="LAVOR")
