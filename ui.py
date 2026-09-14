from partevo import World
import taichi as ti
import time
import sys

ti.init(arch=ti.gpu, random_seed=2)

WIDTH = 1000
HEIGHT = 1000

window = ti.ui.Window("Particle Evolution", res=(WIDTH,HEIGHT), fps_limit=120)
canvas = window.get_canvas()
gui = window.get_gui()

world = World(WIDTH, HEIGHT)
world.populate(int(sys.argv[1]), int(sys.argv[2]))

start_time = time.time()
loop_time = start_time
time_scale = 0.3

def draw_options():
    global time_scale
    gui.begin("Options", 0.02, 0.02, 0.5, 0.15)
    time_scale = gui.slider_float("Time Scale", time_scale, 0, 1)
    world.rmax[None] = gui.slider_float("Interaction Radius", world.rmax[None], 0, 1)
    world.beta[None] = gui.slider_float("Repulsion Radius", world.beta[None], 0, 1)
    world.friction[None] = 1 - gui.slider_float("Friction", 1 - world.friction[None], 0, 1)
    old_pop = world.population[None]
    new_pop = gui.slider_int("Population", old_pop, 0, 100_000)
    world.update_population(old_pop, new_pop)
    gui.end()

def draw_species():
    species_count = world.species_count[None]
    for i in range(species_count):
        gui.begin(f"Species {i}", 0.02, 0.18 + (0.14 * i), 0.5, 0.13)
        world.species[i, 0][0] = gui.slider_float("Trait 1", world.species[i, 0][0], 0, 1)
        world.species[i, 0][1] = gui.slider_float("Trait 2", world.species[i, 0][1], 0, 1)
        world.species[i, 0][2] = gui.slider_float("Trait 3", world.species[i, 0][2], 0, 1)
        world.species[i, 1][0] = gui.slider_float("Receptor 1", world.species[i, 1][0], -1, 1)
        world.species[i, 1][1] = gui.slider_float("Receptor 2", world.species[i, 1][1], -1, 1)
        world.species[i, 1][2] = gui.slider_float("Receptor 3", world.species[i, 1][2], -1, 1)
        gui.end()
    world.update_species()

def draw_ui():
    draw_options()
    draw_species()

while True:
    draw_ui()

    world.wrap = False
    world.tick(0.05 * time_scale)
    particles = world.get_particles()

    canvas.set_background_color((0,0,0))

    canvas.circles(particles[0], radius=0.002, per_vertex_color=particles[1])

    window.show()

    for event in window.get_events():
        if event.key == ti.ui.ESCAPE:
            exit()
