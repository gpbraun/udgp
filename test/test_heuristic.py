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


def get_results_core(runtimes):
    runtime_low = np_value(np.mean(runtimes) - np.std(runtimes))
    runtime_mean = np_value(np.mean(runtimes))
    runtime_high = np_value(np.mean(runtimes) + np.std(runtimes))
    print(f"{runtime_low}, {runtime_mean}, {runtime_high}")


SEEDS = np.array([12345678910, 123456, 1234567, 12345678, 12345])

N_range = range(6, 51, 1)

low_times = []
high_times = []
mean_times = []

for N in N_range:
    print(N)
    times = []

    for seed in SEEDS:
        previous_a = []

        instance = Instance.artificial_molecule(N, freq=True)
        i = 1
        instance.reset(reset_runtime=True)
        # while not instance.is_solved():
        # while instance.fixed_n < instance.n:
        i += 1

        instance.reset()

        instance.solve_step(
            "M2GP", nx=4, max_gap=1e-2, max_threshold=1e-1, previous_a=previous_a
        )
        previous_a.append(instance.a_indices.tolist())

        # print("CORE")

        # # instance.reset_with_core("mock")

        # broken = False
        # while instance.fixed_n < instance.n and not broken:
        #     print(f"  {instance.fixed_n} átomos")

        #     solved = False
        #     tries = 0

        #     # iter_gap = 5e-3 + 1e-3 * instance.fixed_n
        #     solved = instance.solve_step("M2", nx=1, max_gap=5e-2, max_threshold=1e-1)

        #     if not solved:
        #         broken = True
        #         break

        times.append(instance.runtime)

    low_time = np_value(np.mean(times) - np.std(times))
    high_time = np_value(np.mean(times) + np.std(times))
    mean_time = np_value(np.mean(times))

    print(f"low:  ({N}, {low_time})")
    print(f"high: ({N}, {high_time})")
    print(f"mean: ({N}, {mean_time})")

    low_times.append(low_time)
    high_times.append(high_time)
    mean_times.append(mean_time)

print(f"low =======")

for i, N in enumerate(N_range):
    print(f"({N}, {low_times[i]})")

print(f"high =======")

for i, N in enumerate(N_range):
    print(f"({N}, {high_times[i]})")

print(f"mean =======")

for i, N in enumerate(N_range):
    print(f"({N}, {mean_times[i]})")
