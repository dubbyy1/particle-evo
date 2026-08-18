from partevo import World

import numpy as np
import pygame
import sys

species_count = 5
species_colours:list[str] = ["#FF5347", "#FE9D4E", "#F5F263", "#81D252", "#41ADC7", "#5A4DB7"]

def draw_particles(data:np.ndarray, screen:pygame.Surface):
    width, height = screen.get_width(), screen.get_height()
    pos, species = data

    for i in range(len(pos)):
        if species[i] == -1:
            break
        x, y = (
            (float(pos[i][0] + 1)/2) * width,
            (float(pos[i][1] + 1)/2) * height
        )

        _ = pygame.draw.circle(screen, species_colours[species[i]], (x, y), 1)


matrix_rect = pygame.Rect(15, 15, 0, 0)
show_matrix = True
def draw_matrix(matrix, screen):
    global matrix_rect

    CELL_SIZE = 20
    GAP_SIZE = 5
    key_size = CELL_SIZE + GAP_SIZE*2


    def draw_window():
        grid_size = CELL_SIZE * species_count + GAP_SIZE * (species_count - 1) + key_size if show_matrix else CELL_SIZE
        window_size = grid_size + GAP_SIZE*2

        matrix_rect.size = (window_size, window_size)

        _ = pygame.draw.rect(screen, (30, 30, 30), (matrix_rect.x, matrix_rect.y, window_size, window_size))
        _ = pygame.draw.rect(screen, (50, 50, 50), (matrix_rect.x, matrix_rect.y, window_size, window_size), width=2)

    def draw_grid():
        grid_pos = (matrix_rect.x + GAP_SIZE, matrix_rect.y + GAP_SIZE)

        for x in range(species_count):
            for y in range(species_count):
                if y == 0:
                    key_colour:tuple[int, int, int]|str = species_colours[x]
                    _ = pygame.draw.rect(screen, key_colour, (
                        x*25 + grid_pos[0] + key_size,
                        grid_pos[1],
                        20,
                        20)
                    )
                if x == 0:
                    key_colour:tuple[int, int, int]|str = species_colours[y]
                    _ = pygame.draw.rect(screen, key_colour, (
                        grid_pos[0],
                        y*25 + grid_pos[1] + key_size,
                        20,
                        20)
                    )


                bond = matrix[y, x]
                cell_colour = (
                    (225*bond*-1 + 30, 30, 30) if bond < 0
                    else (30, 225*bond + 30, 30) if bond > 0
                    else (30, 30, 30)
                )
                _ = pygame.draw.rect(screen, cell_colour, (
                    y*25 + grid_pos[0] + key_size,
                    x*25 + grid_pos[1] + key_size,
                    20,
                    20)
                )

    draw_window()
    if show_matrix:
        draw_grid()

def toggle_matrix():
    global show_matrix
    show_matrix = not show_matrix


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

        if event.type == pygame.MOUSEBUTTONDOWN:
            click_pos = event.pos
            mouse_down = True
            click_length = 0
        if event.type == pygame.MOUSEMOTION and mouse_down:
            if matrix_rect.collidepoint((event.pos[0] - event.rel[0], event.pos[1] - event.rel[1])):
                pygame.event.set_grab(True)
                matrix_rect = matrix_rect.move(event.rel[0], event.rel[1])
            else:
                pass
        if event.type == pygame.MOUSEBUTTONUP:
            mouse_down = False
            pygame.event.set_grab(False)
            if click_length < 5 and pygame.mouse.get_pos() == click_pos: # click
                if matrix_rect.collidepoint(event.pos):
                    toggle_matrix()
            else: # drag
                if matrix_rect.collidepoint(event.pos):
                    pass

    return res

def main():
    WIDTH = 1000
    HEIGHT = 1000

    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    clock = pygame.time.Clock()
    running = True

    world = World(WIDTH, HEIGHT)
    draw_matrix(world.get_matrix(), screen)
    world.populate(int(sys.argv[1]), species_count)
    world.randomize_matrix()
    # world.set_matrix_preset("chains")


    while running:
        delta = clock.tick(60) / 1000.0
        inputs = handle_inputs()
        if "quit" in inputs:
            running = False

        screen.fill("black") # clear screen

        world.tick(delta)

        draw_particles(world.get_particles(), screen)
        draw_matrix(world.get_matrix(), screen)

        pygame.display.flip()

    pygame.quit()

if __name__ == "__main__":
    main()
