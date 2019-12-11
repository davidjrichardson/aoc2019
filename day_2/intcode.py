from abc import ABC, abstractmethod
from collections import namedtuple
from itertools import product
from typing import List, Dict, Tuple, TypeVar, Callable, Iterable

# TypeVar to get around Python's type hinting shenanigans
Ins = TypeVar('Ins', bound='Instruction')
OutState = Tuple[Dict[int, int], List[int], List[int]]
SubList = List[Tuple[int, int]]


class Parameter(namedtuple('Parameter', ['value', 'addr_mode'])):
    """
    Helper class to abstract/encapsulate parameters
    """

    def get_value(self, memory: Dict[int, int]) -> int:
        return {
            0: memory[self.value],
            1: self.value,
            2: memory[-1] + self.value,
        }[self.addr_mode]


class Instruction(ABC):
    """
    Abstract instruction class to allow for operator overrides
    """

    def __init__(self, opcode: int, params: List[Parameter] = []):
        self.opcode = opcode
        self.params = params

    def __repr__(self) -> str:
        return f'Instruction(opcode={self.opcode}, parameters={self.params})'

    @abstractmethod
    def execute(self, memory: Dict[int, int], inputs: List[int], outputs: List[int]) -> OutState:
        pass

    @staticmethod
    def from_int_list(ints: List[int]) -> Ins:
        if not ints:
            return HaltInstruction(99)

        opcode_full, args = str(ints[0]).zfill(5), ints[1:]
        opcode = int(opcode_full[-2:])

        if opcode == 99:
            return HaltInstruction(opcode)
        else:
            if opcode <= 2:
                verb, noun, place, rest = args[0], args[1], args[2], args[3:]
                params = [Parameter(value=verb, addr_mode=int(opcode_full[2])),
                          Parameter(value=noun, addr_mode=int(opcode_full[1])),
                          Parameter(value=place, addr_mode=int(opcode_full[0]))]

                return BinaryOpInstruction(opcode, params)
            elif 2 < opcode <= 4 or opcode == 9:
                place, rest = args[0], args[1:]
                params = [Parameter(value=place, addr_mode=opcode_full[0])]
                return UnaryOpInstruction(opcode, params)
            else:
                raise NotImplementedError(f'Opcode: {opcode}')


class HaltInstruction(Instruction):
    """
    Halt instruction to indicate to the interpreter that the program has finished
    """

    def execute(self, memory: Dict[int, int], inputs: List[int], outputs: List[int]) -> OutState:
        # Halt producs a no-op
        return memory, inputs, outputs


class UnaryOpInstruction(Instruction):
    """
    Unary operation for the interpreter - may consume an input token
    """

    def place(self, memory: Dict[int, int]) -> int:
        if self.params[0].addr_mode == 2:
            # memory[-1] is the relative base for execution
            return memory[-1] + self.params[0].value
        elif self.params[0].addr_mode == 1:
            return self.params[0].value
        elif self.params[0].value == 0:
            return self.params[0].value

    def execute(self, memory: Dict[int, int], inputs: List[int], outputs: List[int]) -> OutState:
        if self.opcode == 3:
            if inputs:
                # Save the first item of the inputs list to memory and cosnume it from the input list
                memory[self.place(memory)] = inputs[0]

                return memory, inputs[1:], outputs
            else:
                raise IndexError(f'Out of inputs to consume for instruction: {self.__repr__()}')
        elif self.opcode == 4:
            # TODO: Think about how to implement immediate addressing
            pass
            # outputs.append()
        else:
            raise NotImplementedError(f'Opcode not implemented: {self.opcode}')


class BinaryOpInstruction(Instruction):
    """
    Binary operation (add, mult) for the interpreter
    """

    def verb(self, memory: Dict[int, int]) -> int:
        if self.params[0].addr_mode == 0:
            return memory[self.params[0].value]
        elif self.params[0].addr_mode == 1:
            return self.params[0].value
        else:
            raise NotImplementedError(f'Addressing mode {self.params[0].addr_mode} not supported')

    def noun(self, memory: Dict[int, int]) -> int:
        if self.params[1].addr_mode == 0:
            return memory[self.params[1].value]
        elif self.params[1].addr_mode == 1:
            return self.params[1].value
        else:
            raise NotImplementedError(f'Addressing mode {self.params[0].addr_mode} not supported')

    @property
    def place(self) -> Parameter:
        return self.params[2]

    @property
    def operation(self) -> Callable[[int, int], int]:
        return {
            1: lambda x, y: x + y,
            2: lambda x, y: x * y,
        }[self.opcode]

    def execute(self, memory: Dict[int, int], inputs: List[int], outputs: List[int]) -> OutState:
        if self.place.addr_mode == 0:
            memory[self.place.value] = self.operation(self.verb(memory), self.noun(memory))

            return memory, inputs, outputs
        else:
            raise NotImplementedError(f'Addressing mode for place={self.place.addr_mode} is unsupported')


def run_instruction(ins_ptr: int, memory: Dict[int, int], inputs: List[int], outputs: List[int]) -> OutState:
    if not memory:
        return memory, inputs, outputs

    # Decode the memory into an instruction
    ints = list(map(lambda x: x[1], sorted(memory.items(), key=lambda x: x[0])))
    instruction = Instruction.from_int_list(ints[ins_ptr:])

    if isinstance(instruction, BinaryOpInstruction):
        new_ptr = ins_ptr + 4
    elif isinstance(instruction, UnaryOpInstruction):
        new_ptr = ins_ptr + 2
    else:
        new_ptr = ins_ptr

    # Update the program and run the next instruction or stop the program if we hit a halt
    if isinstance(instruction, HaltInstruction):
        return memory, inputs, outputs
    else:
        return run_instruction(new_ptr, *(instruction.execute(memory, inputs, outputs)))


class Program:
    def __init__(self, in_program: Dict[int, int], substitutes: SubList = []):
        # Substitute any changes to the program at the position provided
        self.memory = in_program
        self.memory.update(substitutes)

    def __repr__(self):
        return f'Program {self.memory}'

    def execute(self, inputs: List[int], outputs: List[int]) -> Tuple[Dict[int, int], List[int], List[int]]:
        #  Start with an offset of 1 since index -1 stores the relative base
        return run_instruction(1, self.memory, inputs, outputs)


class Machine:
    def __init__(self, input_str: str, param_ranges: List[Tuple[int, Iterable[int]]] = []):
        # param_range is tuple of (instruction index, range of values)
        self.base_program = dict(enumerate(map(lambda x: int(x), input_str.split(","))))
        self.base_program.update([(-1, 0)])  # Add the relative addressing base

        # Convert any substitutions to a parameter space
        params = []
        for index, param_range in param_ranges:
            params.append(list(zip([index for i in range(len(param_range))], param_range)))

        self.param_space = list(map(lambda x: list(x), product(*params)))

    def run_one(self, inputs: List[int] = [], outputs: List[int] = []) -> Tuple[OutState, SubList]:
        parameters = self.param_space[0]
        if len(self.param_space) > 1:
            self.param_space = self.param_space[1:]
        else:
            self.param_space = []

        program = Program(dict(self.base_program), parameters)
        return program.execute(inputs, outputs), parameters

    def run_all(self, inputs: List[int] = [], outputs: List[int] = []) -> List[Tuple[OutState, SubList]]:
        runs = []
        for parameters in self.param_space:
            try:
                program = Program(dict(self.base_program), parameters)
                runs.append((program.execute(inputs, outputs), parameters))
            except NotImplementedError:
                print(f'Parameters {parameters} creates invalid program')

        return runs

    def run_till_predicate(self, predicate: Callable[[int], bool], inputs: List[int] = [],
                           outputs: List[int] = []) -> Tuple[OutState, SubList]:
        for parameters in self.param_space:
            program = Program(dict(self.base_program), parameters)

            try:
                out_state, inputs, outputs = program.execute(inputs, outputs)

                if predicate(out_state[0]):
                    return (out_state, inputs, outputs), parameters
            except NotImplementedError:
                print(f'Parameters {parameters} creates invalid program')

        return None, None
