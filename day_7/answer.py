import asyncio
import sys

from intcode_async import Machine

sys.setrecursionlimit(10 ** 6)

# Open the file and turn it into a list of ints
with open('input.txt', 'r') as input_file:
    program_str = input_file.readline()

# Question 1
# q1_answer = ((), 0)
# for a, b, c, d, e in list(permutations(range(0, 5))):
#     machineA = Machine('A', program_str)
#     machineB = Machine('B', program_str)
#     machineC = Machine('C', program_str)
#     machineD = Machine('D', program_str)
#     machineE = Machine('E', program_str)
#
#     machineA.set_input_values([a, 0])
#     machineB.set_input_values([b])
#     machineC.set_input_values([c])
#     machineD.set_input_values([d])
#     machineE.set_input_values([e])
#
#     machineB.set_pipe(machineA.program)
#     machineC.set_pipe(machineB.program)
#     machineD.set_pipe(machineC.program)
#     machineE.set_pipe(machineD.program)
#
#     output_value = asyncio.run(machineE.run_program())[0]
#     q1_answer = max(q1_answer, ((a, b, c, d, e), output_value), key=lambda x: x[1])
#
# print('Q1:', q1_answer)

# Question 2
q2_answer = ((), 0)
# for a, b, c, d, e in list(permutations(range(5, 10))):
#     print(a,b,c,d,e, '\n\n\n')
a = 9
b = 8
c = 7
d = 6
e = 5

machineA2 = Machine('A', program_str)
machineB2 = Machine('B', program_str)
machineC2 = Machine('C', program_str)
machineD2 = Machine('D', program_str)
machineE2 = Machine('E', program_str)

machineA2.set_input_values([a, 0])
machineB2.set_input_values([b])
machineC2.set_input_values([c])
machineD2.set_input_values([d])
machineE2.set_input_values([e])

machineB2.set_pipe(machineA2.program)
machineC2.set_pipe(machineB2.program)
machineD2.set_pipe(machineC2.program)
machineE2.set_pipe(machineD2.program)
machineA2.set_pipe(machineE2.program)

machineE2.run_program()
output_value = machineE2.program[9]
print(output_value)
# q2_answer = max(q2_answer, ((a, b, c, d, e), output_value), key=lambda x: x[1])

# print(q2_answer)
