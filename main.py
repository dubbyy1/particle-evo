from partevo import World

import pygame
import sys

def draw_particles(world:World, screen):
        for particle in list(world.particles.values()):
            pygame.draw.circle(screen, world.species[particle.species]["colour"], (particle.x, particle.y), 3)


matrix_rect = pygame.Rect(15, 15, 0, 0)
show_matrix = True
def draw_matrix(world:World, screen):
    global matrix_rect

    CELL_SIZE = 20
    GAP_SIZE = 5
    key_size = CELL_SIZE + GAP_SIZE*2
    species = list(world.species.keys())


    def draw_window():
        grid_size = CELL_SIZE * len(species) + GAP_SIZE * (len(species) - 1) + key_size if show_matrix else CELL_SIZE
        window_size = grid_size + GAP_SIZE*2

        matrix_rect.size = (window_size, window_size)

        _ = pygame.draw.rect(screen, (30, 30, 30), (matrix_rect.x, matrix_rect.y, window_size, window_size))
        _ = pygame.draw.rect(screen, (50, 50, 50), (matrix_rect.x, matrix_rect.y, window_size, window_size), width=2)

    def draw_grid():
        grid_pos = (matrix_rect.x + GAP_SIZE, matrix_rect.y + GAP_SIZE)

        for y, y_species in enumerate(species):
            for x, x_species in enumerate(species):
                if y == 0:
                    key_colour:tuple[int, int, int] = world.species[x_species]["colour"]
                    _ = pygame.draw.rect(screen, key_colour, (
                        x*25 + grid_pos[0] + key_size,
                        grid_pos[1],
                        20,
                        20)
                    )
                if x == 0:
                    key_colour = world.species[y_species]["colour"]
                    _ = pygame.draw.rect(screen, key_colour, (
                        grid_pos[0],
                        y*25 + grid_pos[1] + key_size,
                        20,
                        20)
                    )


                bond = world.get_bond((y_species, x_species))
                cell_colour = (
                    (225*bond*-1 + 30, 30, 30) if bond < 0
                    else (30, 225*bond + 30, 30) if bond > 0
                    else (30, 30, 30)
                )
                _ = pygame.draw.rect(screen, cell_colour, (
                    x*25 + grid_pos[0] + key_size,
                    y*25 + grid_pos[1] + key_size,
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

    world = World((WIDTH, HEIGHT))
    world.add_species("red", "#FF5347")
    world.add_species("orange", "#FE9D4E")
    world.add_species("yellow", "#F5F263")
    world.add_species("green", "#81D252")
    world.add_species("blue", "#41ADC7")
    world.add_species("purple", "#5A4DB7")
    world.set_matrix_preset("chains")
    # world.randomize_bonds()
    draw_matrix(world, screen)
    # world.set_bond(("blue", "blue"), 0.5)
    # world.set_bond(("blue", "red"), 1)
    # world.set_bond(("red", "blue"), -0.2)
    # world.set_bond(("red", "red"), 0.3)
    world.populate(int(sys.argv[1]))

    while running:
        delta = clock.tick(60) / 1000.0
        inputs = handle_inputs()
        if "quit" in inputs:
            running = False

        screen.fill("black") # clear screen

        world.tick(delta)

        draw_particles(world, screen)
        draw_matrix(world, screen)

        pygame.display.flip()

    pygame.quit()

if __name__ == "__main__":
    main()
