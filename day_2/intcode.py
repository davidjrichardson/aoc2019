from abc import ABC, abstractmethod
from collections import namedtuple, defaultdict
from itertools import product
from typing import List, Dict, Tuple, TypeVar, Callable, Iterable

# TypeVar to get around Python's type hinting shenanigans
Ins = TypeVar('Ins', bound='Instruction')
Computer = TypeVar('Computer', bound='Machine')
ProgramState = Tuple[int, Dict[int, int], List[int], List[int]]
ProgramOutput = Tuple[Dict[int, int], List[int], List[int]]
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

    @property
    def place(self) -> Parameter:
        return self.params[-1]

    def memory_index(self, memory: Dict[int, int]) -> int:
        if self.place.addr_mode == 2:
            return memory[-1] + self.place.value
        else:
            return self.place.value

    @abstractmethod
    def execute(self, in_state: ProgramState) -> ProgramState:
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
            if opcode == 1 or opcode == 2 or opcode == 7 or opcode == 8:
                verb, noun, place, rest = args[0], args[1], args[2], args[3:]
                params = [Parameter(value=verb, addr_mode=int(opcode_full[2])),
                          Parameter(value=noun, addr_mode=int(opcode_full[1])),
                          Parameter(value=place, addr_mode=int(opcode_full[0]))]

                return BinaryOpInstruction(opcode, params)
            elif opcode == 3 or opcode == 4:
                place, rest = args[0], args[1:]
                params = [Parameter(value=place, addr_mode=int(opcode_full[2]))]
                return IoInstruction(opcode, params)
            elif opcode == 5 or opcode == 6:
                verb, noun, rest = args[0], args[1], args[2:]
                params = [Parameter(value=verb, addr_mode=int(opcode_full[2])),
                          Parameter(value=noun, addr_mode=int(opcode_full[1]))]
                return JumpInstruction(opcode, params)
            else:
                raise NotImplementedError(f'Opcode: {opcode}')


class HaltInstruction(Instruction):
    """
    Halt instruction to indicate to the interpreter that the program has finished
    """

    def execute(self, in_state: ProgramState) -> ProgramState:
        # Halt producs a no-op
        return in_state


class IoInstruction(Instruction):
    """
    I/O operation for the interpreter - may consume an input token or output to the output buffer
    """

    def take_input(self, memory: Dict[int, int], inputs: List[int]) -> List[int]:
        if self.place.addr_mode == 1:
            raise NotImplementedError(f'Addressing mode {self.place.addr_mode} unsupported for opcode {self.opcode}')

        memory[self.memory_index(memory)] = inputs[0]
        return inputs[1:]

    def execute(self, in_state: ProgramState) -> ProgramState:
        ins_ptr, memory, inputs, outputs = in_state
        if self.opcode == 3:
            if inputs:
                # Save the first item of the inputs list to memory and cosnume it from the input list
                return ins_ptr, memory, self.take_input(memory, inputs), outputs
            else:
                raise IndexError(f'Out of inputs to consume for instruction: {self}')
        elif self.opcode == 4:
            if self.place.addr_mode == 1:
                outputs.append(self.place.value)
            else:
                outputs.append(memory[self.memory_index(memory)])

            return ins_ptr, memory, inputs, outputs
        else:
            raise NotImplementedError(f'Instruction not implemented: {self}')


class JumpInstruction(Instruction):
    """
    Jump the instruction pointer based for execution based on current state
    """

    def verb(self, memory: Dict[int, int]) -> int:
        if self.params[0].addr_mode == 1:
            return self.params[0].value
        elif self.params[0].addr_mode == 2:
            return memory[memory[-1] + self.params[0].value]
        else:
            return memory[self.params[0].value]

    def noun(self, memory: Dict[int, int]) -> int:
        if self.params[1].addr_mode == 1:
            return self.params[1].value
        elif self.params[1].addr_mode == 2:
            return memory[memory[-1] + self.params[1].value]
        else:
            return memory[self.params[1].value]

    def execute(self, in_state: ProgramState) -> ProgramState:
        ins_ptr, memory, inputs, outputs = in_state
        # Offset of +1 because -1 is the relative base for relative addressing
        if self.opcode == 5:
            new_ptr = self.noun(memory) + 1 if self.verb(memory) > 0 else ins_ptr
            return new_ptr, memory, inputs, outputs
        elif self.opcode == 6:
            new_ptr = self.noun(memory) + 1 if self.verb(memory) == 0 else ins_ptr
            return new_ptr, memory, inputs, outputs
        else:
            raise NotImplementedError(f'Instruction not implemented: {self}')


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
    def operation(self) -> Callable[[int, int], int]:
        return {
            1: lambda x, y: x + y,
            2: lambda x, y: x * y,
            7: lambda x, y: 1 if x < y else 0,
            8: lambda x, y: 1 if x == y else 0,
        }[self.opcode]

    def execute(self, in_state: ProgramState) -> ProgramState:
        ins_ptr, memory, inputs, outputs = in_state

        if self.place.addr_mode == 1:
            raise NotImplementedError(f'Addressing mode for place={self.place.addr_mode} is unsupported')

        memory[self.memory_index(memory)] = self.operation(self.verb(memory), self.noun(memory))
        return ins_ptr, memory, inputs, outputs


def run_instruction(in_state: ProgramState) -> ProgramState:
    ins_ptr, memory, inputs, outputs = in_state

    if not memory:
        return ins_ptr, memory, inputs, outputs

    # Decode the memory into an instruction
    ints = list(map(lambda x: x[1], sorted(memory.items(), key=lambda x: x[0])))
    instruction = Instruction.from_int_list(ints[ins_ptr:])

    if isinstance(instruction, BinaryOpInstruction):
        new_ptr = ins_ptr + 4
    elif isinstance(instruction, IoInstruction):
        new_ptr = ins_ptr + 2
    elif isinstance(instruction, JumpInstruction):
        new_ptr = ins_ptr + 3
    else:
        new_ptr = ins_ptr

    # Update the program and run the next instruction or stop the program if we hit a halt
    if isinstance(instruction, HaltInstruction):
        return ins_ptr, memory, inputs, outputs
    else:
        return run_instruction(instruction.execute((new_ptr, memory, inputs, outputs)))


class Program:
    def __init__(self, in_program: Dict[int, int], substitutes: SubList = []):
        # Substitute any changes to the program at the position provided
        self.memory = in_program
        self.memory.update(substitutes)

    def __repr__(self):
        return f'Program {self.memory}'

    def execute(self, inputs: List[int], outputs: List[int]) -> Tuple[Dict[int, int], List[int], List[int]]:
        #  Start with an offset of 1 since index -1 stores the relative base
        _, memory, inputs, outputs = run_instruction((1, self.memory, inputs, outputs))
        return memory, inputs, outputs


class Machine:
    def __init__(self, input_str: str, param_ranges: List[Tuple[int, Iterable[int]]] = [], chained_pc: Computer = None):
        # param_range is tuple of (instruction index, range of values)
        self.base_program = defaultdict(int)
        self.base_program.update(enumerate(map(lambda x: int(x), input_str.split(","))))
        self.base_program.update([(-1, 0)])  # Add the relative addressing base

        # Convert any substitutions to a parameter space
        params = []
        for index, param_range in param_ranges:
            params.append(list(zip([index for i in range(len(param_range))], param_range)))

        self.param_space = list(map(lambda x: list(x), product(*params)))
        self.chained_machine = chained_pc

    def run_chain(self, inputs: List[List[int]] = []) -> Tuple[ProgramOutput, SubList]:
        """
        inputs is in reverse execution order; the last item in the list is the first to be used
        """
        if self.chained_machine:
            chain_out, _ = self.chained_machine.run_chain(inputs[1:])
            _, chain_inputs, chain_outputs = chain_out
            our_input = inputs[0] + chain_outputs
            return self.run_one(our_input, outputs=[])
            # Input: [signal phase, output from previous machine]
        else:
            return self.run_one(inputs[0], [])

    def run_one(self, inputs: List[int] = [], outputs: List[int] = []) -> Tuple[ProgramOutput, SubList]:
        parameters = self.param_space[0]
        if len(self.param_space) > 1:
            self.param_space = self.param_space[1:]
        else:
            self.param_space = []

        program = Program(dict(self.base_program), parameters)
        return program.execute(inputs, outputs), parameters

    def run_all(self, inputs: List[int] = [], outputs: List[int] = []) -> List[Tuple[ProgramOutput, SubList]]:
        runs = []
        for parameters in self.param_space:
            try:
                program = Program(dict(self.base_program), parameters)
                runs.append((program.execute(inputs, outputs), parameters))
            except NotImplementedError:
                print(f'Parameters {parameters} creates invalid program')

        return runs

    def run_till_predicate(self, predicate: Callable[[int], bool], inputs: List[int] = [],
                           outputs: List[int] = []) -> Tuple[ProgramOutput, SubList]:
        for parameters in self.param_space:
            program = Program(dict(self.base_program), parameters)

            try:
                out_state, inputs, outputs = program.execute(inputs, outputs)

                if predicate(out_state[0]):
                    return (out_state, inputs, outputs), parameters
            except NotImplementedError:
                print(f'Parameters {parameters} creates invalid program')

        return None, None
