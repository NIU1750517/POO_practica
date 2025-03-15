import random
import numpy as np
import pygame

class NBodySimulator():
    def __init__(self, windowSize, universe):
        self._windowSize = windowSize
        self.space_radius = 1e12
        self.factor = self._windowSize / 2 / self.space_radius
        self.universe = universe

    def _draw(self, position_space, color, size=5.):
        position_pixels = self.factor * position_space + self._windowSize / 2.
        pygame.draw.circle(self.screen, color, position_pixels, size)
    
    def animate(self, time_step, trace=False):
        pygame.init()
        self.screen = pygame.display.set_mode([self._windowSize, self._windowSize])
        pygame.display.set_caption('example, time step = {}'.format(time_step))
        running = True
        color_background, color_body, color_trace = (128, 128, 128), (0, 0, 0), (192, 192, 192)
        self.screen.fill(color_background)

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            if trace:
                for body in self.universe.bodies:
                    self._draw(body._position, color_trace)
                self._update(time_step)
                for body in self.universe.bodies:
                    self._draw(body._position, color_body)
                pygame.display.flip()
            else:
                self.screen.fill(color_background)
                for body in self.universe.bodies:
                    self._draw(body._position, color_body)
                self._update(time_step)
                pygame.display.flip()
        pygame.quit()

    def _update(self, time_step):
        for body in self.universe.bodies:
            total_force = body.total_force(self.universe.bodies)
            body.update(total_force, time_step)



    
class Universe():
    def __init__(self, bodies):
        self.bodies = bodies

    @classmethod
    def random(cls, num_bodies):
        return cls([Body.random() for i in range(num_bodies)])
    
    @classmethod
    def from_file(cls, fname):
        bodies = []
        with open(fname,'r') as f:
            num_bodies = int(f.readline())
            radius = float(f.readline())
            for _ in range(num_bodies):
                linia = f.readline()
                m, px, py, vx, vy = [float(z) for z in linia.strip().split() if z]
                bodies.append(Body([px, py], [vx, vy], m))
        print("Universe imported succesfuly !!!")
        return cls(bodies)



class Body():
    G = 6.674e-11
    EPSILON = 1e-10  # Pequeño valor para evitar división por cero

    def __init__(self, position_i, velocity_i, mass_i):
        self._position = np.array(position_i)
        self._velocity = np.array(velocity_i)
        self._mass = mass_i

    def update(self, force, dt):
        acceleration = force / self._mass
        self._velocity += acceleration * dt
        self._position += self._velocity * dt
        print(f"Updated position: {self._position}, Updated velocity: {self._velocity}")

    def total_force(self, other_bodies):
        force = np.zeros(2)
        for body in other_bodies:
            if body is not self:
                force += self._force(body)
        print(f"Total force on body: {force}")
        return force

    def _force(self, another_body):
        distance12 = self._distance_to(another_body)
        distance12 = max(distance12, self.EPSILON)  # Evitar división por cero
        magnitude = self._mass * another_body._mass * Body.G / distance12**2
        direction = (another_body._position - self._position) / distance12
        return magnitude * direction

    def _distance(self, p, q):
        return np.sqrt(np.square(p - q).sum())

    def _distance_to(self, another_body):
        return self._distance(self._position, another_body._position)

    @staticmethod
    def random_vector(a, b, dim=1):
        return a + (b - a) * 2 * (np.random.rand(dim) - 0.5)

    @classmethod
    def random(cls):
        mass = Body.random_vector(1e10, 1e30)
        position = Body.random_vector(-1e20, +1e20, 2)
        velocity = Body.random_vector(1e15, +1e25, 2)
        return cls(position, velocity, mass)

    def move(self, force, timeStep):
        pass

#------------------------------------------------------MAIN CODE---------------------------------------------------------------------
if __name__ == '__main__':
    universe = Universe.from_file('2body.txt')
    simulator = NBodySimulator(700, universe)
    simulator.animate(1, trace=False)