from itertools import product
from typing import List, Dict, Tuple, Callable


class Instruction:
    """
    Instruction class to allow for operator overrides
    """

    def __init__(self, args: Tuple[int, ...]):
        self.opcode, self.verb, self.noun, self.place = args

    def get_op(self) -> Callable[[int, int], int]:
        return {
            1: lambda x, y: x + y,
            2: lambda x, y: x * y,
        }[self.opcode]


# Open the file and turn it into a list of ints
with open('input.txt', 'r') as input_file:
    input_program = input_file.readline()

input_program = list(map(lambda x: int(x), input_program.split(',')))


def glob_program(program: List[int]) -> Tuple[Instruction, List[int]]:
    return Instruction(tuple(program[0:4])), program[4:]


def intcode(program: List[int], memory: Dict[int, int]):
    # Recursive base case
    if len(program) < 4:
        return memory

    if program[0] == 99:
        return memory

    # Pattern match over the input
    ins, rest = glob_program(program)
    memory[ins.place] = ins.get_op()(memory[ins.verb], memory[ins.noun])

    # Perform the operation
    return intcode(rest, memory)


# Question 1
q1_memory = dict(enumerate(input_program))
q1_state = intcode(input_program, q1_memory)
q1_result = q1_state[0]
print(f'Q1 answer: {q1_result}')

# Question 2
q2_argspace = list(product(range(0, 100), repeat=2))
for verb, noun in q2_argspace:
    q2_program = [1, verb, noun] + input_program[3:]
    q2_memory = dict(enumerate(q2_program))

    q2_state = intcode(q2_program, q2_memory)
    if q2_state[0] == 19690720:
        print('Q2 answer: {answer}'.format(answer=(100 * verb) + noun))
        break
