from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Tuple, List, Dict, NamedTuple, Union, TypeVar, Callable

Memory = Dict[int, int]
MemoryAddr = Tuple[int, int]

# TypeVars to help with type hinting
State = TypeVar('State', bound='Program')  # Program type variable
Ins = TypeVar('Ins', bound='Instruction')


class Parameter(NamedTuple):
    addr_mode: int
    value: int


class Instruction(ABC):
    """
    Abstract instruction class to allow for operator overrides
    """

    def __init__(self, opcode: int, params: List[Parameter]):
        self.opcode = opcode
        self.params = params

    def __repr__(self) -> str:
        return f'Instruction(opcode={self.opcode}, parameters={self.params})'

    @property
    def place(self) -> Union[Parameter, None]:
        if self.opcode != 99:
            return self.params[-1]
        else:
            return None

    @property
    def length(self) -> int:
        return len(self.params) + 1  # The number of parameters +1 for the opcode

    @abstractmethod
    def execute(self, state: State, input_val: Union[int, None] = None) -> Union[int, None]:
        pass

    @staticmethod
    def from_int_list(ints: List[int]) -> Ins:
        """
        Instruction factory function that returns an instruction object and the amount that the instruction pointer
        needs to be advanced by
        """
        if not ints:
            return HaltInstruction(99, params=[])

        opcode_full, args = str(ints[0]).zfill(5), ints[1:]
        opcode = int(opcode_full[-2:])

        if opcode == 99:
            return HaltInstruction(99, params=[])
        else:
            if opcode == 1 or opcode == 2 or opcode == 7 or opcode == 8:
                verb, noun, place, rest = args[0], args[1], args[2], args[3:]
                params = [Parameter(value=verb, addr_mode=int(opcode_full[2])),
                          Parameter(value=noun, addr_mode=int(opcode_full[1])),
                          Parameter(value=place, addr_mode=int(opcode_full[0]))]

                return BinaryOpInstruction(opcode, params)
            elif opcode == 3 or opcode == 4:
                print(f'Generating instruction for opcode {opcode}')
                place, rest = args[0], args[1:]
                params = [Parameter(value=place, addr_mode=int(opcode_full[2]))]
                return IoInstruction(opcode, params)
            elif opcode == 5 or opcode == 6:
                verb, noun, rest = args[0], args[1], args[2:]
                params = [Parameter(value=verb, addr_mode=int(opcode_full[2])),
                          Parameter(value=noun, addr_mode=int(opcode_full[1]))]
                return JumpInstruction(opcode, params)
            else:
                raise NotImplementedError(f'Opcode {opcode} not implemented for {ints[:5]}')


class HaltInstruction(Instruction):
    """
    Halt instruction to indicate to the interpreter that the program has finished
    """

    def execute(self, state: State, input_val: Union[int, None] = None) -> Union[int, None]:
        print('halt program')
        return None  # Halting causes a no-op in execution


class JumpInstruction(Instruction):
    """
    Jump the instruction pointer based for execution based on current state
    """

    @property
    def verb(self):
        return self.params[0]

    @property
    def noun(self):
        return self.params[1]

    @property
    def compare(self) -> Callable[[int, int], Union[int, None]]:
        return {  # +1 offset is used here because of the relative_base being at the -1 index
            5: lambda x, y: y + 1 if x != 0 else None,
            6: lambda x, y: y + 1 if x == 0 else None,
        }.get(self.opcode)

    def execute(self, state: State, input_val: Union[int, None] = None) -> Union[int, None]:
        """
        Returns the new instruction pointer value or None if there is no change
        """
        print(f'jumping to {self.compare(state.resolve_value(self.verb), state.resolve_value(self.noun))}')
        return self.compare(state.resolve_value(self.verb), state.resolve_value(self.noun))


class IoInstruction(Instruction):
    """
    I/O instruction for intcode. The program will have already resolved the _value of input_ without factoring in
    addressing mode. Values returned are yielded to the output buffer.
    """

    def execute(self, state: State, input_val: Union[int, None] = None) -> Union[int, None, State]:
        """
        If the opcode is for value input, then None is returned. If the opcode is for value output, input_val is None.
        """
        if self.opcode == 3:
            print(f'inserting {input_val} at location {state.resolve_index(self.place)}')
            # Update the internal state then return None to indicate an input operation
            state.update([(state.resolve_index(self.place), input_val)])
            return state
        elif self.opcode == 4:
            print(f'outputting value {state.resolve_value(self.place)} from location {state.resolve_index(self.place)}')
            return state.resolve_value(self.place)
        else:
            raise NotImplementedError(f'I/O is not implemented for instruction: {self}')


class BinaryOpInstruction(Instruction):
    """
    A binary operation instruction that takes two operands and applies a function to it. None is always returned since
    these instructions do not relate to the output buffer.
    """

    @property
    def verb(self):
        return self.params[0]

    @property
    def noun(self):
        return self.params[1]

    @property
    def operation(self) -> Callable[[int, int], int]:
        """
        Function that determines the binary operation to use based on the opcode
        """
        return {
            1: lambda x, y: x + y,
            2: lambda x, y: x * y,
            7: lambda x, y: 1 if x < y else 0,
            8: lambda x, y: 1 if x == y else 0,
        }.get(self.opcode)

    def execute(self, state: State, input_val: Union[int, None] = None) -> Union[int, None]:
        print(f'result {self.operation(state.resolve_value(self.verb), state.resolve_value(self.noun))} at location {state.resolve_index(self.place)}')
        state[state.resolve_index(self.place)] = self.operation(state.resolve_value(self.verb),
                                                                state.resolve_value(self.noun))
        return None


class Program(defaultdict):
    """
    A helper class to implement methods over the program memory state
    """

    def __init__(self, initial_state: Memory, label: str, inputs: List[int] = None, pipe: State = None):
        # Initialise this program's state
        super(Program, self).__init__(int)
        super(Program, self).update(initial_state)

        self.label = label
        self.initial = inputs
        self.inputs = inputs
        self.pipe = pipe
        self.ins_ptr = 1

    @property
    def int_list(self):
        return list(map(lambda x: x[1], sorted(self.items(), key=lambda x: x[0])))

    def is_finished(self):
        return self.int_list[self.ins_ptr] == 99

    def set_pipe(self, pipe: State):
        self.pipe = pipe

    def set_input_values(self, inputs: List[int]):
        self.initial = list(inputs)
        self.inputs = list(inputs)

    def reset(self, new_state: Memory = []):
        """
        A function to clear the internal state and then set it to the provided new state
        """
        super(Program, self).clear()
        super(Program, self).update(new_state)

    def execute(self):
        """
        Execute the memory state, starting at the instruction pointer and has some inputs. Will return some output
        value(s) as an iterable
        """
        print(f'begin {self.label} exec with ins_ptr: {self.ins_ptr}')
        while instruction := Instruction.from_int_list(self.int_list[self.ins_ptr:]):
            print(f'P:{self.label} ins_ptr: {self.ins_ptr}, instruction: {instruction}')
            # Exit early if it's a halt instruction
            if instruction.opcode == 99:
                print(f'Halting program {self.label}')
                break

            # Update the value of instruction pointer
            self.ins_ptr = self.ins_ptr + instruction.length

            if instruction.opcode == 3:  # If we're an input instruction then handle the special case of inputs
                if not self.inputs:
                    # Execute the piped-in program first
                    if not self.pipe:
                        raise AttributeError(
                            f'There is no input buffer available for ptr: {self.ins_ptr}; instruction {instruction}')

                    if self.pipe.is_finished():
                        print(f'Program {self.pipe.label} has finished, continuing onto next instruction')
                        continue
                    else:
                        print(f'Running sub-program {self.pipe.label}')
                        self.inputs += [self.pipe.execute()]
                        print(f'Resuming {self.label}:{self.ins_ptr - instruction.length} with output from {self.pipe.label}: {self.inputs}')
                        # print(Instruction.from_int_list(self.int_list[self.ins_ptr - instruction.length:]))
                        # print(self[28])
                        print(f'{instruction}')

                self.update(instruction.execute(self, input_val=self.inputs.pop(0)))
            elif instruction.opcode == 4:
                value = instruction.execute(self)
                return value
            elif instruction.opcode == 5 or instruction.opcode == 6:  # Jump instructions change the instruction pointer
                new_ptr = instruction.execute(self)
                self.ins_ptr = new_ptr if new_ptr else self.ins_ptr
            else:
                instruction.execute(self)

    def resolve_index(self, parameter: Parameter) -> int:
        """
        A function that resolves the index in memory to access something at.
        """
        if parameter.addr_mode == 1:
            return parameter.addr_mode

        if parameter.addr_mode == 2:
            # Add the relative base to the parameter
            return parameter.value + self.get(-1, 0)
        else:
            return parameter.value

    def resolve_value(self, parameter: Parameter) -> int:
        """
        A function that resolves the value of the parameter. This **does not** return the index to store the value at
        """
        if parameter.addr_mode == 1:
            return parameter.value

        return self.get(self.resolve_index(parameter), 0)


class Machine:
    def __init__(self, label: str, input_str: str = None, input_program: Memory = None):
        if input_str:
            # Convert the program string to a default dictionary
            self.initial_state = defaultdict(int)
            self.initial_state.update(enumerate(map(lambda x: int(x), input_str.split(","))))
        elif input_program:
            # Make a copy of the input state as a default dictionary
            self.initial_state = defaultdict(int)
            self.initial_state.update(input_program)
        else:
            raise NotImplementedError('Instantiating a machine requires an initial program string or memory state')

        self.initial_state.update([(-1, 0)])  # Always initialise the relative base
        self.program = Program(self.initial_state, label)

    def reset_memory(self):
        """
        A function to restore this machine's state to how it was initialised
        """
        self.program.reset(self.initial_state)

    def update_memory(self, changes: List[MemoryAddr]):
        """
        A function to make updates to specific memory addresses in the machine
        """
        self.program.update(changes)

    def set_input_values(self, inputs: List[int]):
        self.program.set_input_values(inputs)

    def set_pipe(self, pipe: Program):
        self.program.set_pipe(pipe)

    def run_program(self):
        output = []
        while not self.program.is_finished():
            output.append(self.program.execute())
        return output
