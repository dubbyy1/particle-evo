import taichi as ti
from taichi.ui.gui import ScalarField
import easygui

from time import time
import numpy as np
import json

ti.init(arch=ti.gpu, random_seed=int(time()))

@ti.data_oriented
class World:
    def __init__(self, width:int, height:int, species_count=5):
        capacity = 100_000
        species_capacity = 10
        self.width:int = width
        self.height:int = height

        self.visualize:ti.MatrixField|ScalarField = ti.field(ti.i32, shape=())
        self.visualize[None] = 0
        self.population:ti.MatrixField|ScalarField = ti.field(ti.i32, shape=())
        self.population[None] = capacity
        self.species_count:ti.MatrixField|ScalarField = ti.field(ti.i32, shape=())
        self.species_count[None] = species_count

        self.traits:ti.MatrixField = ti.Vector.field(3, ti.f32, shape=capacity)
        self.receptors:ti.MatrixField = ti.Vector.field(3, ti.f32, shape=capacity)

        self.pos:ti.MatrixField = ti.Vector.field(2, dtype=ti.f32, shape=capacity)
        self.force:ti.MatrixField = ti.Vector.field(2, dtype=ti.f32, shape=capacity)
        self.vel:ti.MatrixField = ti.Vector.field(2, dtype=ti.f32, shape=capacity)

        self.rmax = ti.field(dtype=ti.f32, shape=())
        self.rmax[None] = 0.2
        self.beta = ti.field(dtype=ti.f32, shape=())
        self.beta[None] = 0.35
        self.friction = ti.field(dtype=ti.f32, shape=())
        self.friction[None] = 0.8
        self.species = ti.Vector.field(3, ti.f32, shape=(species_capacity, 2))

        self.screen_pos:ti.MatrixField = ti.Vector.field(2, dtype=ti.f32, shape=capacity)
        self.colors:ti.MatrixField = ti.Vector.field(3, ti.f32, shape=capacity)

        self.p:ti.MatrixField|ScalarField = ti.field(ti.f32, shape=())

    @ti.kernel
    def populate(self, count:int, species_count:int):
        self.species_count[None] = species_count
        self.population[None] = count

        for i in range(10): # max species
            self.species[i, 0] = ti.Vector([
                ti.random(), ti.random(), ti.random()
            ])
            self.species[i, 1] = ti.Vector([
                (ti.random() * 2) - 1, (ti.random() * 2) - 1, (ti.random() * 2) - 1
            ])

        for i in range(count):
            self.pos[i] = ti.Vector([ti.random()*2 - 1, ti.random()*2 - 1])
            self.force[i] = ti.Vector([0.0, 0.0])
            self.vel[i] = ti.Vector([0.0, 0.0])

    @ti.kernel
    def random_genome(self):
        for i in range(10): # max species
            self.species[i, 0] = ti.Vector([
                ti.random(), ti.random(), ti.random()
            ])
            self.species[i, 1] = ti.Vector([
                (ti.random() * 2) - 1, (ti.random() * 2) - 1, (ti.random() * 2) - 1
            ])
    @ti.kernel
    def random_traits(self):
        for i in range(10): # max species
            self.species[i, 0] = ti.Vector([
                ti.random(), ti.random(), ti.random()
            ])
    @ti.kernel
    def random_receptors(self):
        for i in range(10): # max species
            self.species[i, 1] = ti.Vector([
                (ti.random() * 2) - 1, (ti.random() * 2) - 1, (ti.random() * 2) - 1
            ])

    @ti.kernel
    def random_pos(self):
        for i in range(self.population[None]):
            self.pos[i] = ti.Vector([ti.random()*2 - 1, ti.random()*2 - 1])

    @ti.kernel
    def update_population(self, old_pop:int, new_pop:int):
        if old_pop < new_pop:
            for i in range(new_pop - old_pop):
                self.pos[i + old_pop] = ti.Vector([ti.random()*2 - 1, ti.random()*2 - 1])
                self.force[i + old_pop] = ti.Vector([0.0, 0.0])
                self.vel[i + old_pop] = ti.Vector([0.0, 0.0])
            self.population[None] = new_pop
        if new_pop < old_pop:
            for i in range(old_pop - new_pop):
                self.pos[new_pop + i] = ti.Vector([-2.0, -2.0])
                self.screen_pos[new_pop + i] = ti.Vector([-2.0, -2.0])
                self.force[new_pop + i] = ti.Vector([0.0, 0.0])
                self.vel[new_pop + i] = ti.Vector([0.0, 0.0])
            self.population[None] = new_pop

    @ti.func
    def apply_friction(self, dt:float, pop:int):
        for i in range(pop):
            self.vel[i] *= self.friction[None]# ** (60*dt)

    @ti.func
    def get_affinity(self, i:int, j:int):
        return ti.tanh(ti.math.dot(
            self.species[i % self.species_count[None], 1],
            self.species[j % self.species_count[None], 0]
        ))

    @ti.func
    def get_force(self, i:int, j:int):
        rmax = self.rmax[None]
        beta = self.beta[None]

        rel = self.pos[j] - self.pos[i]
        for d in range(2):
            if rel[d] < -1.0:
                rel[d] += 2.0
            elif rel[d] > 1.0:
                rel[d] -= 2.0

        affinity = self.get_affinity(i, j)
        dist = rel.norm()
        force = ti.Vector([0.0, 0.0])
        if dist < rmax:
            norm = dist / rmax
            magnitude = 0.0
            if norm < beta:
                magnitude = norm / beta - 1
            else:
                magnitude = affinity * (1 - abs(1 + beta - 2 * norm) / (1 - beta))

            force_mult = 1
            force = (rel / dist) * magnitude * force_mult * rmax

        return force

    @ti.kernel
    def tick(self, dt:float):
        pop = self.population[None]

        # self.apply_friction(dt, pop)

        for i in range(pop):

            self.force[i] = ti.Vector([0.0, 0.0])
            for j in range(pop):
                if i != j:
                    self.force[i] += self.get_force(i, j)

        for i in range(pop):
            self.vel[i] += self.force[i] * dt
            self.pos[i] += self.vel[i] * dt

            for d in range(2):
                if self.pos[i][d] < -1.0:
                    self.pos[i][d] += 2.0
                elif self.pos[i][d] > 1.0:
                    self.pos[i][d] -= 2.0

    @ti.kernel
    def make_screen_pos(self):
        for i in range(self.population[None]):
            self.screen_pos[i] = (self.pos[i] / 2) + 0.5

    @ti.kernel
    def make_colors(self):
        for i in range(self.population[None]):
            self.colors[i] = self.species[i % self.species_count[None], self.visualize[None]]# * 0.8 + 0.2
            if self.visualize[None] == 1:
                self.colors[i] = abs(self.colors[i])

    def get_particles(self):
        self.make_screen_pos()
        self.make_colors()
        return (self.screen_pos, self.colors)

    def save(self, radius, time_scale, bg_color):
        filepath = easygui.filesavebox(title="Save Config", default="save" + str(int(time())) + ".json", filetypes=["*.json"])
        if not filepath:
            return

        data = {
            "options": {
                "rmax": float(self.rmax[None]),
                "beta": float(self.beta[None]),
                "friction": float(self.friction[None]),
                "population": int(self.population[None]),
                "species_count": int(self.species_count[None]),
                "radius": radius,
                "time_scale": time_scale,
                "bg_color": bg_color
            },
            "species": self.species.to_numpy().tolist()
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=4)

    def load(self) -> tuple[int, float, tuple[float, float, float]]:
        filepath = easygui.fileopenbox(title="Load Config", filetypes=["*.json"])
        if not filepath:
            return

        with open(filepath, 'r') as f:
            data = json.load(f)

        options = data["options"]
        self.rmax[None] = options["rmax"]
        self.beta[None] = options["beta"]
        self.friction[None] = options["friction"]

        self.update_population(self.population[None], options["population"])
        self.species_count[None] = options["species_count"]
        self.species.from_numpy(np.array(data["species"], dtype=np.float32))

        return (options["radius"], options["time_scale"], options["bg_color"])
