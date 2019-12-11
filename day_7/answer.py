from itertools import permutations

from intcode import Machine

# Open the file and turn it into a list of ints
with open('input.txt', 'r') as input_file:
    input_program = input_file.readline()

# Question 1
q1_answer = ((), 0)
for a, b, c, d, e in list(permutations(range(0, 5))):
    machineA = Machine(input_program)
    machineB = Machine(input_program, chained_pc=machineA)
    machineC = Machine(input_program, chained_pc=machineB)
    machineD = Machine(input_program, chained_pc=machineC)
    machineE = Machine(input_program, chained_pc=machineD)

    inputs = [[e], [d], [c], [b], [a, 0]]
    print(f'inputs: {a}, {b}, {c}, {d}, {e}')
    output, _ = machineE.run_chain(inputs)
    _, _, output_buf = output
    q1_answer = max(q1_answer, ((a, b, c, d, e), output_buf[0]), key=lambda x: x[1])

print(q1_answer)
