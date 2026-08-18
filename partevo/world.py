import taichi as ti
from taichi.ui.gui import ScalarField

import numpy as np

ti.init(arch=ti.gpu)

@ti.data_oriented
class World:
    def __init__(self, width:int, height:int, capacity=50_000, max_species=10):
        self.width:int = width
        self.height:int = height
        self.max_species:int = max_species

        self.population:ti.MatrixField|ScalarField = ti.field(ti.i32, shape=())
        self.species_count:ti.MatrixField|ScalarField = ti.field(ti.i32, shape=())

        self.species:ti.MatrixField|ScalarField = ti.field(dtype=ti.i32, shape=capacity)
        self.species.fill(-1)
        self.pos:ti.MatrixField|ScalarField = ti.Vector.field(2, dtype=ti.f32, shape=capacity)
        self.force:ti.MatrixField|ScalarField = ti.Vector.field(2, dtype=ti.f32, shape=capacity)
        self.vel:ti.MatrixField|ScalarField = ti.Vector.field(2, dtype=ti.f32, shape=capacity)

        self.matrix = ti.field(dtype=ti.f32, shape=(max_species, max_species))

        self.rmax = ti.field(dtype=ti.f32, shape=())
        self.rmax[None] = 0.075
        self.beta = ti.field(dtype=ti.f32, shape=())
        self.beta[None] = 0.3
        self.friction = ti.field(dtype=ti.f32, shape=())
        self.friction[None] = 0.85

        self.p:ti.MatrixField|ScalarField = ti.field(ti.f32, shape=())

    @ti.kernel
    def populate(self, count:int, species_count:int):
        self.population[None] = count
        self.species_count[None] = species_count

        for i in range(self.population[None]):
            self.species[i] = i % species_count
            self.pos[i] = ti.Vector([ti.random()*2 - 1, ti.random()*2 - 1])
            self.force[i] = ti.Vector([0.0, 0.0])
            self.vel[i] = ti.Vector([0.0, 0.0])

    def randomize_matrix(self):
        count = self.species_count[None]
        matrix = np.random.uniform(-1.0, 1.0, size=(self.max_species, self.max_species)).astype(
            np.float32
        )
        self.matrix.from_numpy(matrix)

    def set_matrix_preset(self, name):
        species_count = self.species_count[None]
        if name == "chains":
            pattern = [1.0, 1.0, -1.0, -1.0, -1.0, 1.0]
            for i in range(species_count**2):
                x = i // species_count
                y = i % species_count
                self.matrix[x, y] = pattern[(i % species_count) - (i // species_count)]

    @ti.func
    def apply_friction(self, dt, pop):
        for i in range(pop):
            self.vel[i] *= self.friction[None] ** (60*dt)

    @ti.func
    def get_force(self, i, j):
        rmax = self.rmax[None]
        beta = self.beta[None]

        rel = self.pos[j] - self.pos[i]
        for d in range(2):
            if rel[d] < -1.0:
                rel[d] += 2.0
            elif rel[d] > 1.0:
                rel[d] -= 2.0

        bond = self.matrix[self.species[j], self.species[i]]
        dist = rel.norm()
        force = ti.Vector([0.0, 0.0])
        if dist < rmax:
            norm = dist / rmax
            magnitude = 0.0
            if norm < beta:
                magnitude = norm / beta - 1
            else:
                magnitude = bond * (1 - abs(1 + beta - 2 * norm) / (1 - beta))
                # self.p[None] = magnitude

            force = (rel / dist) * magnitude * rmax

        return force

    @ti.kernel
    def tick(self, dt:ti.f32):
        pop = self.population[None]

        self.apply_friction(dt, pop)

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

    def get_particles(self):
        return (self.pos.to_numpy(), self.species.to_numpy())

    def get_matrix(self):
        return self.matrix.to_numpy()
