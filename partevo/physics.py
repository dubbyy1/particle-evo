



 ##       ##      #######
#  #     #  #   #         #
#  #     #  #     ##   ##
#  #     #  #      #   #
#  #     #  #      #   #
#   #   #   #     ##   ##
 #   ^^^   #    #         #
   #######        #######







import taichi as ti
from taichi.ui.gui import ScalarField

import numpy as np

from time import time

ti.init(arch=ti.gpu, random_seed=int(time()))
print(int(time()))

@ti.data_oriented
class World:
    def __init__(self, width:int, height:int, population=50_000, species_count=5):
        self.width:int = width
        self.height:int = height

        self.population:ti.MatrixField|ScalarField = ti.field(ti.i32, shape=())
        self.population[None] = population
        self.species_count:ti.MatrixField|ScalarField = ti.field(ti.i32, shape=())
        self.species_count[None] = species_count

        self.traits:ti.MatrixField = ti.Vector.field(3, ti.f32, shape=population)
        self.receptors:ti.MatrixField = ti.Vector.field(3, ti.f32, shape=population)

        self.pos:ti.MatrixField = ti.Vector.field(2, dtype=ti.f32, shape=population)
        self.force:ti.MatrixField = ti.Vector.field(2, dtype=ti.f32, shape=population)
        self.vel:ti.MatrixField = ti.Vector.field(2, dtype=ti.f32, shape=population)

        self.rmax = ti.field(dtype=ti.f32, shape=())
        self.rmax[None] = 0.05
        self.beta = ti.field(dtype=ti.f32, shape=())
        self.beta[None] = 0.3
        self.friction = ti.field(dtype=ti.f32, shape=())
        self.friction[None] = 1

        # self.traits_data = np.array([
        #     [1.000, 0.500, 1.000],  # Particle A
        #     [0.750, 0.933, 1.000],  # Particle B
        #     [0.250, 0.933, 1.000],  # Particle C
        #     [0.000, 0.500, 1.000],  # Particle D
        #     [0.250, 0.067, 1.000],  # Particle E
        #     [0.750, 0.067, 1.000],  # Particle F
        # ], dtype=np.float32)
        # self.traits_temp = ti.Vector.field(3, ti.f32, shape=(6))
        # self.traits_temp.from_numpy(self.traits_data)

        # self.receptors_data = np.array([
        #     [1.000, 0.000, -0.500],  # Particle A
        #     [0.500, 0.866, -0.683],  # Particle B
        #     [-0.500, 0.866, -0.183],  # Particle C
        #     [-1.000, 0.000, 0.500],  # Particle D
        #     [-0.500, -0.866, 0.683],  # Particle E
        #     [0.500, -0.866, 0.183],  # Particle F
        # ], dtype=np.float32)
        # self.receptors_temp = ti.Vector.field(3, ti.f32, shape=(6))
        # self.receptors_temp.from_numpy(self.receptors_data)

        self.species = ti.Vector.field(3, ti.f32, shape=(species_count, 2))

        self.p:ti.MatrixField|ScalarField = ti.field(ti.f32, shape=())

    @ti.func
    def init_species(self):
        for i in range(self.species_count[None]):
            t1, t2, t3 = [ti.random(), ti.random(), ti.random()]
            self.species[i, 0] = ti.Vector([t1, t2, t3])
            # sum = t1 + t2 + t3
            # self.species[i, 0] = ti.Vector([t1 / sum, t2 / sum, t3 / sum])

            self.species[i, 1] = ti.Vector([(ti.random() * 2) - 1, (ti.random() * 2) - 1, (ti.random() * 2) - 1])

    @ti.kernel
    def populate(self, count:int):
        self.population[None] = count
        self.init_species()

        for i in range(count):
            self.traits[i] = self.species[i % 6, 0]
            self.receptors[i] = self.species[i % 6, 1]
            self.pos[i] = ti.Vector([ti.random()*2 - 1, ti.random()*2 - 1])
            self.force[i] = ti.Vector([0.0, 0.0])
            self.vel[i] = ti.Vector([0.0, 0.0])

    @ti.func
    def apply_friction(self, dt, pop):
        for i in range(pop):
            self.vel[i] *= self.friction[None] ** (60*dt)

    @ti.func
    def get_affinity(self, i, j):
        return ti.tanh(ti.math.dot(self.receptors[i], self.traits[j]))

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
        pop = self.population[None]
        return (self.pos.to_numpy()[:pop], self.traits.to_numpy()[:pop], self.receptors.to_numpy()[:pop])

    def get_matrix(self):
        return self.matrix.to_numpy()

    @ti.kernel
    def ti_test(self):
        self.init_species()

    def test(self):
        return self.species.to_numpy()
