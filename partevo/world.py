from .particle import Particle

import random
import string

class World:
    def __init__(self, size:tuple[int, int]):
        self.size = size

        self.species:dict[str, dict[str, tuple[int, int, int]|str]] = {}
        self.bonds = self.generate_bonds()
        self.particles:dict[str, Particle] = {}
        self.cells:dict[tuple[int, int], list[Particle]] = {}

        self.deadzone = 0.3
        self.detection_radius:float = 150.0
        self.force_scale:int = 1
        self.cell_size = self.detection_radius

    def add_species(self, name:str, colour:tuple[int, int, int]|str):
        self.species[name] = {
            "colour": colour
        }
        self.bonds = self.generate_bonds()

    def tick(self, delta):
        for id, particle in self.particles.items():
            particle.tick(delta)

    def spawn_particle(self, species:str, position:tuple[int, int]):
        id = ''.join(random.choices(string.ascii_letters + string.digits, k=4))
        part = Particle(id=id, species=species, world=self)
        part.set_position(position)

        self.particles[id] = part

    def populate(self, count, ratio=None):
        if ratio:
            pass
        else: # UNIFORM
            for i in range(count):
                species = list(self.species.keys())[i % len(self.species)]

                position = (
                    random.randint(0, self.size[0]),
                    random.randint(0, self.size[1])
                )
                self.spawn_particle(species=species, position=position)

    def generate_bonds(self, default_value=0.0):
        species_names = list(self.species.keys())

        return {
            from_sp: {to_sp: default_value for to_sp in species_names}
            for from_sp in species_names
        }

    def set_matrix_preset(self, name):
        species_count = len(self.species)
        if name == "chains":
            pattern = [1.0, 1.0, -1.0, -1.0, -1.0, 1.0]
            for i in range(species_count**2):
                species_x = list(self.species.keys())[i // species_count]
                species_y = list(self.species.keys())[i % species_count]
                self.bonds[species_x][species_y] = pattern[(i % species_count) - (i // species_count)]

    def randomize_bonds(self):
        for from_sp in self.species:
            for to_sp in self.species:
                self.bonds[from_sp][to_sp] = round(random.uniform(-1.0, 1.0), 2)

    def set_bond(self, bond:tuple[str, str], value:float):
        self.bonds[bond[0]][bond[1]] = value

    def get_bond(self, bond:tuple[str, str]):
        return self.bonds[bond[0]][bond[1]]

    def update_detection(self, deadzone:float, detection_radius:int):
        self.deadzone = deadzone
        self.detection_radius = detection_radius
        self.cell_size = self.detection_radius * 2

        for part in self.particles.values():
            part.detection_radius = self.detection_radius
            part.deadzone = self.deadzone

    def pos_to_cell(self, pos:tuple[int, int]):
        return (pos[0] // self.cell_size, pos[1] // self.cell_size)
