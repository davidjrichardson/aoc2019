# Open the file and turn it into a list of ints
from intcode import Machine

with open('input.txt', 'r') as input_file:
    input_program = input_file.readline()

# Question 1
q1_machine = Machine(input_program, param_ranges=[(1, [12]), (2, [2])])
q1_output, _ = q1_machine.run_one()
q1_out_state, _, _ = q1_output
print(f'Q1 answer: {q1_out_state[0]}')

# Question 2
# q2_machine = Machine(input_program, param_ranges=[(1,[89]),(2,[76])])
q2_machine = Machine(input_program, param_ranges=[
    (1, range(0, 100)),
    (2, range(0, 100))
])
q2_output, q2_params = q2_machine.run_till_predicate(lambda x: x == 19690720)
print(f'Q2 answers: {q2_params}')
