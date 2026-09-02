import taichi as ti
from taichi.ui.gui import ScalarField

import numpy as np

ti.init(arch=ti.gpu)

@ti.data_oriented
class World:
    def __init__(self, width:int, height:int, capacity=50_000):
        self.width:int = width
        self.height:int = height

        self.capacity:ti.MatrixField|ScalarField = ti.field(ti.i32, shape=())
        self.capacity[None] = capacity
        self.population:ti.MatrixField|ScalarField = ti.field(ti.i32, shape=()) # not actually population, just the highest index
        self.population[None] = 0                                               # containing a living particle
        self.dead_count:ti.MatrixField|ScalarField = ti.field(ti.i32, shape=())
        self.dead_count[None] = 0
        self.dead:ti.MatrixField|ScalarField = ti.field(ti.i32, shape=capacity)
        self.dead.fill(-1)

        self.traits:ti.MatrixField = ti.Vector.field(3, ti.f32, shape=capacity)
        self.receptors:ti.MatrixField = ti.Vector.field(3, ti.f32, shape=capacity)
        self.next_traits:ti.MatrixField = ti.Vector.field(3, ti.f32, shape=capacity)
        self.next_receptors:ti.MatrixField = ti.Vector.field(3, ti.f32, shape=capacity)

        self.pos:ti.MatrixField = ti.Vector.field(2, dtype=ti.f32, shape=capacity)
        self.force:ti.MatrixField = ti.Vector.field(2, dtype=ti.f32, shape=capacity)
        self.vel:ti.MatrixField = ti.Vector.field(2, dtype=ti.f32, shape=capacity)
        self.energy:ti.MatrixField|ScalarField = ti.field(ti.f32, shape=capacity)

        self.rmax = ti.field(dtype=ti.f32, shape=())
        self.rmax[None] = 0.05
        self.beta = ti.field(dtype=ti.f32, shape=())
        self.beta[None] = 0.3
        self.friction = ti.field(dtype=ti.f32, shape=())
        self.friction[None] = 0.85

        self.p:ti.MatrixField|ScalarField = ti.field(ti.f32, shape=())

    @ti.kernel
    def populate(self, count:int):
        self.population[None] = count

        for i in range(count):
            self.traits[i] = ti.Vector([ ti.random(), ti.random(), ti.random() ])
            self.receptors[i] = ti.Vector([ ti.random()*2 - 1, ti.random()*2 - 1, ti.random()*2 - 1 ])
            self.pos[i] = ti.Vector([ti.random()*2 - 1, ti.random()*2 - 1])
            self.force[i] = ti.Vector([0.0, 0.0])
            self.vel[i] = ti.Vector([0.0, 0.0])
            self.energy[i] = 0.8 + ti.random(ti.f32) * 0.4

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
                self.combine_genome(i, j, norm / beta)
                magnitude = norm / beta - 1
            else:
                magnitude = affinity * (1 - abs(1 + beta - 2 * norm) / (1 - beta))

            force = (rel / dist) * magnitude * rmax

        return force

    @ti.func
    def combine_genome(self, i, j, dist):
        chance = ti.random(ti.f32)
        if chance < 0.1:

            t = int((chance / 0.1) * 5)
            if t < 3:
                self.next_traits[i][int(t)] = self.traits[j][int(t)]
            else:
                self.next_receptors[i][int(t) - 3] = self.receptors[j][int(t) - 3]

    @ti.func
    def drift_genome(self, i):
        pass
        for t in range(3):
            change = ti.random(ti.f32) * 0.002
            change -= 0.001
            new = self.next_traits[i][t] + change
            new = max(0.0, min(1.0, new))

            self.next_traits[i][t] = new
        for r in range(3):
            change = ti.random(ti.f32) * 0.002
            change -= 0.001
            new = self.next_receptors[i][r] + change
            new = max(-1.0, min(1.0, new))

            self.next_receptors[i][r] = new

    @ti.func
    def kill(self, i):
        if i == self.population[None] - 1:
            self.population[None] -= 1
        else:
            self.dead[self.dead_count[None]] = i
            self.dead_count[None] += 1

        self.energy[i] = -1
        self.pos[i] = ti.Vector([-2.0, -2.0])

    @ti.func
    def is_dead(self, i):
        dead = False
        for d in range(self.dead_count[None]):
            if i == self.dead[d]:
                dead = True
        return dead

    @ti.func
    def reproduce(self, i):
        pass

    @ti.kernel
    def tick(self, dt:ti.f32):
        pop = self.population[None] - self.dead_count[None]

        self.apply_friction(dt, pop)

        ti.loop_config(serialize=True)
        for i in range(pop):

            self.force[i] = ti.Vector([0.0, 0.0])
            self.next_traits[i] = self.traits[i]
            self.next_receptors[i] = self.receptors[i]
            for j in range(pop):
                if i != j:
                    self.force[i] += self.get_force(i, j)
            self.drift_genome(i)

        for i in range(pop):
            dead = self.is_dead(i)
            self.vel[i] += self.force[i] * dt
            self.pos[i] += self.vel[i] * dt

            if not dead:
                for d in range(2):
                    if self.pos[i][d] < -1.0:
                        self.pos[i][d] += 2.0
                    elif self.pos[i][d] > 1.0:
                        self.pos[i][d] -= 2.0

                self.traits[i] = self.next_traits[i]
                self.receptors[i] = self.next_receptors[i]

        for i in range(pop):
            self.energy[i] -= 0.005
            if self.energy[i] <= 0:
                self.kill(i)
            elif self.energy[i] > 2:
                self.reproduce(i)

    def get_particles(self):
        pop = self.population[None]
        return (self.pos.to_numpy()[:pop], self.traits.to_numpy()[:pop], self.receptors.to_numpy()[:pop])

    def get_matrix(self):
        return self.matrix.to_numpy()

    def get_stats(self):
        return [self.population[None], self.dead_count[None], self.dead.to_numpy()]
