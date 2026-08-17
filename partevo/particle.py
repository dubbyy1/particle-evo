import math
from typing import override

class World:
    def __init__(self, size:tuple[int, int]):
        self.size = size

        self.species:dict[str, dict[str, tuple[int, int, int]]] = {}
        self.bonds:dict[str, dict[str, float]] = {}
        self.particles:dict[str, Particle] = {}
        self.cells:dict[tuple[int, int], list[Particle]] = {}

        self.deadzone:float = 0
        self.force_scale:int = 0
        self.detection_radius:int = 0
        self.cell_size:int = 0

    def pos_to_cell(self, pos:tuple[int, int]):
        return (pos[0] // self.cell_size, pos[1] // self.cell_size)

class Particle:
    def __init__(self, id:str, species:str, world:World):
        self.id = id
        self.species = species
        self.world:World = world

        self.bonds:dict[str, float] = world.bonds[species]

        self.x:float = 0
        self.y:float = 0
        self.cell:tuple[int, int] = (0, 0)

        self.vx:float = 0
        self.vy:float = 0

        self.detection_radius:float = world.detection_radius
        self.deadzone:float = world.deadzone # beta = 0.3
        self.friction:float = 0.5

    @override
    def __repr__(self) -> str:
        return str((self.id, self.species, self.x, self.y))

    def apply_friction(self, delta):
        friction_factor = self.friction ** (60 * delta)
        self.vx *= friction_factor
        self.vy *= friction_factor

    def get_interaction_force(self, distance, bond):
        norm = distance / self.detection_radius

        if norm < self.deadzone:
            return norm / self.deadzone - 1
        else:
            return bond * (1 - abs(1 + self.deadzone - 2 * norm) / (1 - self.deadzone))

    def apply_bonds(self, delta):
        neighbours = []
        for cell_y in (-1, 0, 1):
            for cell_x in (-1, 0, 1):
                raw_target_cell:tuple[int, int] = (self.cell[0] + cell_x, self.cell[1] + cell_y)
                target_cell = raw_target_cell
                far_x_cell = self.world.pos_to_cell((self.world.size[0] - 1, 0))[0]
                far_y_cell = self.world.pos_to_cell((0, self.world.size[1] - 1))[1]

                wx = raw_target_cell[0] % (far_x_cell + 1)
                wy = raw_target_cell[1] % (far_y_cell + 1)
                target_cell = (wx, wy)

                for particle in self.world.cells.setdefault(target_cell, []):
                    if particle.id == self.id:
                        continue
                    neighbours.append(particle)


        applied_neighbours = 0
        dvx, dvy = (0, 0)
        for neighbour in neighbours:
            raw_x, raw_y = neighbour.get_position()
            raw_dx, raw_dy = raw_x - self.x, raw_y - self.y
            nx, ny = (raw_x, raw_y)

            if abs(raw_dx) > self.world.size[0] / 2:
                nx = raw_x + self.world.size[0] if raw_dx < 0 else raw_x - self.world.size[0]
            if abs(raw_dy) > self.world.size[1] / 2:
                ny = raw_y + self.world.size[1] if raw_dy < 0 else raw_y - self.world.size[1]
            dx, dy = nx - self.x, ny - self.y

            distance_sq = self.get_distance_squared(self.get_position(), (nx, ny))
            if distance_sq > (self.detection_radius**2):
                continue
            if distance_sq == 0:
                continue

            distance = math.sqrt(distance_sq)
            bond_force = self.bonds[neighbour.species]
            force = self.get_interaction_force(distance, bond_force)

            if distance:
                dvx += (dx / distance) * force * self.detection_radius * self.world.force_scale * delta
                dvy += (dy / distance) * force * self.detection_radius * self.world.force_scale * delta
            applied_neighbours += 1

        self.vx += dvx
        self.vy += dvy

    def tick(self, delta):
        self.apply_friction(delta)
        self.apply_bonds(delta)

        new_x = self.x + self.vx * delta
        new_y = self.y + self.vy * delta

        if new_x < 0:
            new_x = self.world.size[0]
        elif new_x > self.world.size[0]:
            new_x = 0
        if new_y < 0:
            new_y = self.world.size[1]
        elif new_y > self.world.size[1]:
            new_y = 0

        self.set_position((new_x, new_y))

    def get_distance(self, pos1:tuple[float, float], pos2:tuple[float, float]):
        return math.hypot(pos1[0] - pos2[0], pos1[1] - pos2[1])
    def get_distance_squared(self, pos1:tuple[float, float], pos2:tuple[float, float]):
        return (pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2

    def get_position(self):
        return (self.x, self.y)
    def set_position(self, position:tuple[float, float]):
        self.x, self.y = position

        if self in self.world.cells.setdefault(self.cell, []):
            self.world.cells[self.cell].remove(self)
        self.cell = self.world.pos_to_cell((self.x, self.y))
        self.world.cells.setdefault(self.cell, []).append(self)
