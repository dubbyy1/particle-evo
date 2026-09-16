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

input_pop = 100 if len(sys.argv) < 2 else int(sys.argv[1])
input_species = 5 if len(sys.argv) < 3 else int(sys.argv[2])
world.populate(input_pop, input_species)

start_time = time.time()
loop_time = start_time
time_scale = 0.3
particle_radius = 0.3
bg_color = (0, 0, 0)
show_ui = True

def draw_options():
    global time_scale, particle_radius, bg_color

    gui.begin("Options", 0.01, 0.01, 0.5, 0.175)
    time_scale = gui.slider_float("Time Scale", time_scale, 0, 1)
    world.rmax[None] = gui.slider_float("Interaction Radius", world.rmax[None], 0, 1)
    world.beta[None] = gui.slider_float("Repulsion Radius", world.beta[None], 0, 1)
    world.friction[None] = 1 - gui.slider_float("Friction", 1 - world.friction[None], 0, 1)
    old_pop = world.population[None]
    new_pop = gui.slider_int("Population", old_pop, 0, 100_000)
    world.update_population(old_pop, new_pop)
    world.species_count[None] = gui.slider_int("Species", world.species_count[None], 1, 10)
    particle_radius = gui.slider_float("Radius", particle_radius, 0, 1)

    bg_color = gui.color_edit_3("Background Color", bg_color)

    gui.end()

def draw_species():
    species_count = world.species_count[None]
    for i in range(species_count):
        gui.begin(f"Species {i}", 0.01, 0.195 + (0.075 * i), 0.5, 0.125)
        # world.species[i, 0]    = gui.color_edit_3("Traits", tuple(world.species[i, 0]))
        world.species[i, 0][0] = gui.slider_float("Trait 1", world.species[i, 0][0], 0, 1)
        world.species[i, 0][1] = gui.slider_float("Trait 2", world.species[i, 0][1], 0, 1)
        world.species[i, 0][2] = gui.slider_float("Trait 3", world.species[i, 0][2], 0, 1)
        world.species[i, 1][0] = gui.slider_float("Receptor 1", world.species[i, 1][0], -1, 1)
        world.species[i, 1][1] = gui.slider_float("Receptor 2", world.species[i, 1][1], -1, 1)
        world.species[i, 1][2] = gui.slider_float("Receptor 3", world.species[i, 1][2], -1, 1)
        gui.end()

def draw_controls():
    global particle_radius, time_scale, bg_color
    gui.begin("Controls", 0.79, 0.01, 0.2, 0.215)
    if gui.button("Full Reset"):
        world.populate(world.population[None], world.species_count[None])
    if gui.button("Randomize Positions"):
        world.random_pos()
    if gui.button("Randomize Species"):
        world.random_genome()
    if gui.button("Randomize Traits"):
        world.random_traits()
    if gui.button("Randomize Receptors"):
        world.random_receptors()
    if gui.button("Visualize " + ["Receptors", "Traits"][world.visualize[None]]):
        world.visualize[None] = 0 if world.visualize[None] == 1 else 1

    if gui.button("Save"):
        world.save(particle_radius, time_scale, bg_color)
    if gui.button("Load"):
        particle_radius, time_scale, bg_color = world.load()

    gui.end()

def draw_ui():
    draw_options()
    draw_species()
    draw_controls()


while True:
    if show_ui:
        draw_ui()


    world.wrap = False
    world.tick(0.05 * time_scale)
    particles = world.get_particles()

    canvas.set_background_color(bg_color)

    canvas.circles(particles[0], radius=0.02 * particle_radius, per_vertex_color=particles[1])

    window.show()

    for event in window.get_events(ti.ui.RELEASE):
        if event.key == ti.ui.ESCAPE:
            exit()
        if event.key == ti.ui.SPACE:
            show_ui = not show_ui
