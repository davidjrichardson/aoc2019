import sys

import numpy as np

sys.path.insert(1, '../util')
from intcode_async import Program


def print_grid(grid):
    for row in grid:
        print('|' + ''.join(map(lambda x: '  ' if x < 1 else '##', row)) + '|')


class Grid:
    def __init__(self):
        self.dimension = (7, 43)
        self.coordinates = (0, -2)
        self.grid = np.zeros(self.dimension, dtype=np.int8)
        self.direction = 0

    def __str__(self):
        return '\n'.join(map(lambda x: '|' + ''.join(
            map(lambda y: '..' if y < 0 else ('  ' if y < 1 else '##'), x)) + '|', self.grid))

    def painted_count(self):
        """
        Function that checks how many squares were painted. Black is -1, white is 1 so it just sums the magnitude of
        every value in the grid
        """
        return np.sum(np.abs(self.grid))

    def advance(self, colour: int, direction: int):
        self.paint(colour)
        # If direction == 1 then turn right 90 degrees, else turn left 90 degrees
        self.direction += 90 if direction else -90

        if self.direction % 360 == 0:
            offset = (0, 1)
        elif self.direction % 360 == 90:
            offset = (1, 0)
        elif self.direction % 360 == 180:
            offset = (0, -1)
        else:  # The west case
            offset = (-1, 0)

        self.coordinates = (self.coordinates[0] + offset[0], self.coordinates[1] + offset[1])

    def get_colour(self):
        return self.grid[self.coordinates[1], self.coordinates[0]]

    def paint(self, colour: int):
        if colour == 0:
            self.grid[self.coordinates[1], self.coordinates[0]] = -1
        else:
            self.grid[self.coordinates[1], self.coordinates[0]] = 1


# Open the file and turn it into a list of ints
with open('input.txt', 'r') as input_file:
    program_str = input_file.readline()

original = Program('OG', input_str=program_str)

# Question 1
q1_program = Program('Q1', input_program=original)
# Map of location pair -> paint colour
colour_ship_map = dict()
direction = 0
# Clockwise by index: N, E, S, W
movements = [(0, 1), (1, 0), (0, -1), (-1, 0)]
position = (0, 0)

while not q1_program.is_finished():
    in_colour = colour_ship_map.get(position, 0)
    q1_program.send(in_colour)
    q1_program.execute()
    out_colour, d = q1_program.output_buf[-2:]
    colour_ship_map[position] = out_colour
    direction = (direction + (1 if d else - 1)) % 4
    position = (position[0] + movements[direction][0], position[1] + movements[direction][1])

print(f'Q1: {len(colour_ship_map)}')

# Question 2
q2_program = Program('Q2', input_program=original)
q2_grid = Grid()
q2_grid.paint(1)

while not q2_program.is_finished():
    in_colour = q2_grid.get_colour()
    q2_program.send(in_colour)
    q2_program.execute()
    out_colour, d = q2_program.output_buf[-2:]
    q2_grid.advance(out_colour, d)

print(str(q2_grid))
