from udgp import generate_random_instance, solve_M4

N = 5
input_instance = generate_random_instance(N)
input_instance.view()


solved_instance = solve_M4(input_instance)

print(repr(solved_instance[0]))
