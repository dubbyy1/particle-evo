from partevo import World
import taichi as ti
import time
import sys

ti.init(arch=ti.gpu)

WIDTH = 1000
HEIGHT = 1000

window = ti.ui.Window("Particle Evolution", res=(WIDTH,HEIGHT), fps_limit=120)
canvas = window.get_canvas()
gui = window.get_gui()

world = World(WIDTH, HEIGHT)
world.ti_test()
world.populate(int(sys.argv[1]))

start_time = time.time()
loop_time = start_time
while True:
    now = time.time()
    delta = now - loop_time
    world.tick(0.05)
    particles = world.get_particles()

    canvas.set_background_color((0,0,0))

    canvas.circles(particles[0], radius=0.002, per_vertex_color=particles[1])
    window.show()

    loop_time = time.time()

    for event in window.get_events():
        if event.key == ti.ui.ESCAPE:
            exit()
