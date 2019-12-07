from itertools import product
from typing import List, Dict, Tuple, Callable

# Parameter pair of value, indexing mode
ParamPair = Tuple[int, int]


# TODO: Create namedtuple for parameter pairs
# TODO: Create program class to encapsulate instructions etc


class Instruction:
    """
    Instruction class to allow for operator overrides
    """

    def __init__(self, args: Tuple[int, ...]):
        # TODO: Deconstruct the opcode into addressing modes
        opcode, verb, noun, place = args
        opstr = str(opcode).zfill(5)

        self.opcode = int(opstr[-2:])
        self.verb = (verb, int(opstr[2]))
        self.noun = (noun, int(opstr[1]))
        self.place = (place, int(opstr[0]))

    def get_op(self) -> Callable[[int, int], int]:
        return {
            1: lambda x, y: x + y,
            2: lambda x, y: x * y,
        }[self.opcode]

    @staticmethod
    def get_value(param, memory):
        return {
            0: memory[param[0]],
            1: param[param[1]],
        }[param[1]]


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
    memory[ins.place[0]] = ins.get_op()(Instruction.get_value(ins.verb, memory),
                                        Instruction.get_value(ins.noun, memory))

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
