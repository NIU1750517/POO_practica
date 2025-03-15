import random
import numpy as np
import pygame


class PygameExample:
    def __init__(self, window_size=600):
        self.window_size = window_size
        self.space_radius = 1e12
        self.factor = self.window_size / 2 / self.space_radius

        self.position = np.array(
            [random.uniform(self.space_radius/4, self.space_radius/2),
             random.uniform(self.space_radius/4, self.space_radius/2)]
        )
        self.velocity = 2e-3*np.array([-self.position[1], self.position[0]])

    def animate(self, time_step, trace=False):
        pygame.init()
        self.screen = pygame.display.set_mode([self.window_size, self.window_size])
        pygame.display.set_caption('example, time step = {}'.format(time_step))
        running=True
        color_background, color_body, color_trace = (128,128,128), (0,0,0), (192,192,192)
        self.screen.fill(color_background)

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            if trace:
                self._draw(self.position, color_trace)
                self._update(time_step)
                self._draw(self.position, color_body)
                pygame.display.flip()

            else:
                self.screen.fill(color_background)
                self._draw(self.position, color_body)
                self._update(time_step)
                pygame.display.flip()
        pygame.quit()

    def _draw(self, position_space, color, size=5.):
        position_pixels = self.factor * position_space + self.window_size / 2.
        pygame.draw.circle(self.screen, color, position_pixels, size)

    def _update(self, time_step):
        self.position += self.velocity * time_step
        self.velocity = 2e-3*np.array([-self.position[1], self.position[0]])

if __name__ == '__main__':
    example = PygameExample()
    time_step = 1
    example.animate(time_step, trace=False)
