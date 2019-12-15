import sys
from itertools import permutations

sys.path.insert(1, '../util')
from intcode_async import Program

# Open the file and turn it into a list of ints
with open('input.txt', 'r') as input_file:
    program_str = input_file.readline()

original = Program('OG', input_str=program_str)
amp_a = Program('A', input_program=original)
amp_b = Program('B', input_program=original)
amp_c = Program('C', input_program=original)
amp_d = Program('D', input_program=original)
amp_e = Program('E', input_program=original)

# Question 1
q1_answer = ((), 0)
for a, b, c, d, e in list(permutations(range(0, 5))):
    amp_a.reset(original)
    amp_b.reset(original)
    amp_c.reset(original)
    amp_d.reset(original)
    amp_e.reset(original)

    amp_a.set_input_values([a, 0])
    amp_b.set_input_values([b])
    amp_c.set_input_values([c])
    amp_d.set_input_values([d])
    amp_e.set_input_values([e])

    amp_a.pipe_into(amp_b) \
        .pipe_into(amp_c) \
        .pipe_into(amp_d) \
        .pipe_into(amp_e)

    amp_a.execute()
    q1_answer = max(q1_answer, ((a, b, c, d, e), amp_e.output_buf[-1]), key=lambda x: x[1])

print(f'Q1: {q1_answer}')

# Question 2
q2_answer = ((), 0)
for a, b, c, d, e in list(permutations(range(5, 10))):
    amp_a.reset(original)
    amp_b.reset(original)
    amp_c.reset(original)
    amp_d.reset(original)
    amp_e.reset(original)

    amp_a.set_input_values([a, 0])
    amp_b.set_input_values([b])
    amp_c.set_input_values([c])
    amp_d.set_input_values([d])
    amp_e.set_input_values([e])

    amp_a.pipe_into(amp_b) \
        .pipe_into(amp_c) \
        .pipe_into(amp_d) \
        .pipe_into(amp_e) \
        .pipe_into(amp_a)

    amp_a.execute()
    q2_answer = max(q2_answer, ((a, b, c, d, e), amp_e.output_buf[-1]), key=lambda x: x[1])

print(f'Q2: {q2_answer}')
