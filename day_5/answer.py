from intcode import Machine

# Open the file and turn it into a list of ints
with open('input.txt', 'r') as input_file:
    input_program = input_file.readline()

# Question 1
q1_machine = Machine(input_program)
q1_output, _ = q1_machine.run_one(inputs=[1])
q1_state, _, q1_output_buf = q1_output
print(q1_output_buf)

# Question 2
q2_machine = Machine(input_program)
q2_output, _ = q2_machine.run_one(inputs=[5])
q2_state, _, q2_output_buf = q2_output
print(q2_output_buf)
