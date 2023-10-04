import numpy as np

from udgp import Instance


def np_value(x):
    return np.format_float_positional(
        x, precision=3, unique=False, fractional=False, trim="k"
    )


def get_results(runtimes):
    runtime_mean = np_value(np.mean(runtimes))
    runtime_std = np_value(np.std(runtimes))
    runtime_min = np_value(np.min(runtimes))
    runtime_max = np_value(np.max(runtimes))
    print(f"{runtime_mean} & {runtime_std} & {runtime_min} & {runtime_max}")


def solve_heuristic(instance):
    i = 0
    instance.reset(reset_runtime=True)

    while instance.fixed_n < instance.n:
        i += 1
        instance.reset_with_core("mock")
        broken = False
        while instance.fixed_n < instance.n and not broken:
            solved = False
            solved = instance.solve_step(
                "M2GP",
                nx=1,
                # ny=4,
                max_gap=1e-1,
                max_threshold=1,
            )

            if not solved:
                broken = True
                break

    return i


SEEDS = np.array([12345678910, 123456, 1234567, 12345678, 12345])

N_range = [6, 7, 8, 9, 10, 20, 30, 40, 50]

for N in N_range:
    print(N)
    times = []
    nums_steps = []

    for seed in SEEDS:
        instance = Instance.artificial_molecule(N, freq=True, seed=12345678910)
        print(seed)

        num_steps = solve_heuristic(instance)

        nums_steps.append(num_steps)
        times.append(instance.runtime)

    get_results(num_steps)
    get_results(times)
