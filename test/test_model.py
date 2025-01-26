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


SEEDS = np.array([12345, 123456, 1234567, 12345678])

if __name__ == "__main__":
    runtimes = np.empty(4)

    N = 5
    for i, seed in enumerate(SEEDS):
        instance = Instance.artificial_molecule(N, freq=False, seed=seed)
        instance.reset(reset_runtime=True)
        instance.solve_step("M1", max_gap=5e-3)
        runtimes[i] = instance.runtime
        print(f"{i} - {instance.runtime}")

    get_results(runtimes)
