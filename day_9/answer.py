import sys

sys.path.insert(1, '../util')
from intcode_async import Program

# Open the file and turn it into a list of ints
with open('input.txt', 'r') as input_file:
    program_str = input_file.readline()

original = Program('OG', input_str=program_str)

# Question 1
q1_program = Program('Q1', input_program=original)
q1_program.set_input_values([1])
q1_program.execute()
print(f'Q1: {q1_program.output_buf[0]}')

# Question 2
q2_program = Program('Q1', input_program=original)
q2_program.set_input_values([2])
q2_program.execute()
print(f'Q1: {q2_program.output_buf[0]}')
