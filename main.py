from partevo import World

import numpy as np
import pygame
import sys

def draw_particles(data:np.ndarray, screen:pygame.Surface):
    width, height = screen.get_width(), screen.get_height()
    pos, traits, receptors = data

    for i in range(len(pos)):
        x, y = (
            (float(pos[i][0] + 1)/2) * width,
            (float(pos[i][1] + 1)/2) * height
        )
        if min(x, y) < -1:
            continue

        # print(traits[i])
        fill = traits[i] * 255
        stroke = [((r + 1)/2)*255 for r in receptors[i]]

        _ = pygame.draw.circle(screen, fill, (x, y), 3)
        # _ = pygame.draw.circle(screen, stroke, (x, y), 3, 2)

click_length = 0
click_pos = (0, 0)
mouse_exit_pos = (0, 0)
mouse_down = False
def handle_inputs():
    global click_pos
    global mouse_exit_pos
    global click_length
    global mouse_down
    global matrix_rect

    res = []

    if mouse_down:
        click_length += 1

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            res.append("quit")

        # if event.type == pygame.MOUSEBUTTONDOWN:
        #     click_pos = event.pos
        #     mouse_down = True
        #     click_length = 0
        # if event.type == pygame.MOUSEMOTION and mouse_down:
        #     if matrix_rect.collidepoint((event.pos[0] - event.rel[0], event.pos[1] - event.rel[1])):
        #         pygame.event.set_grab(True)
        #         matrix_rect = matrix_rect.move(event.rel[0], event.rel[1])
        #     else:
        #         pass
        # if event.type == pygame.MOUSEBUTTONUP:
        #     mouse_down = False
        #     pygame.event.set_grab(False)
        #     if click_length < 5 and pygame.mouse.get_pos() == click_pos: # click
        #         if matrix_rect.collidepoint(event.pos):
        #             toggle_matrix()
        #     else: # drag
        #         if matrix_rect.collidepoint(event.pos):
        #             pass

    return res

def main():
    WIDTH = 1000
    HEIGHT = 1000

    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    clock = pygame.time.Clock()
    running = True

    world = World(WIDTH, HEIGHT)
    world.populate(int(sys.argv[1]))


    while running:
        delta = clock.tick(60) / 1000.0
        inputs = handle_inputs()
        if "quit" in inputs:
            running = False

        screen.fill("black") # clear screen

        world.tick(delta)
        print(world.get_stats())

        draw_particles(world.get_particles(), screen)

        pygame.display.flip()

    pygame.quit()

if __name__ == "__main__":
    main()
