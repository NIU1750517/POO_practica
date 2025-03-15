import random
import numpy as np
import pygame

class NBodySimulator():
    def __init__(self, windowSize, universe):
        self._windowSize = windowSize
        self.universe = universe
        self.space_radius = self.universe.radius
        self.factor = self._windowSize / 2 / self.space_radius
        self.star_positions = self._generate_star_field(200)  # Genera 200 estrellas

    def _generate_star_field(self, num_stars):
        """Genera una lista de posiciones para las estrellas de fondo."""
        star_positions = []
        for _ in range(num_stars):
            x = random.randint(0, self._windowSize)
            y = random.randint(0, self._windowSize)
            star_positions.append((x, y))
        return star_positions

    def _draw_star_field(self):
        """Dibuja las estrellas de fondo en la pantalla."""
        for pos in self.star_positions:
            self.screen.set_at(pos, (255, 255, 255))  # Dibuja un punto blanco

    def _draw(self, position_space, color, size=5.):
        position_pixels = self.factor * np.array(position_space) + self._windowSize / 2.
        pygame.draw.circle(self.screen, color, position_pixels.astype(int), size)


    def animate(self, time_step, trace=False):
        pygame.init()
        self.screen = pygame.display.set_mode([self._windowSize, self._windowSize])
        pygame.display.set_caption('N-Body Simulation, time step = {}'.format(time_step))
        running = True
        color_background = (21, 16, 25)
        color_body = (199, 0, 57)
        color_trace = (39, 30, 47)
        color_sun = (255, 162, 0)

        max_mass = max(body.mass for body in self.universe.bodies)
        max_mass_count = sum(1 for body in self.universe.bodies if body.mass == max_mass)


        self.screen.fill(color_background)
        self._draw_star_field()  # Dibuja las estrellas de fondo
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            if trace:
                for body in self.universe.bodies:
                    self._draw(body.position, color_trace)
                self._update(time_step)

                for body in self.universe.bodies:
                    if max_mass_count == 1 and body.mass == max_mass:
                        self._draw(body.position, color_sun, 10)
                    else:
                        self._draw(body.position, color_body)
            else:
                for body in self.universe.bodies:
                    self._draw(body.position, color_body)
                self._update(time_step)

            pygame.display.flip()
        pygame.quit()

    def _update(self, time_step):
        for body in self.universe.bodies:
            total_force = body.total_force(self.universe.bodies)
            body.update(total_force, time_step)

class Universe():
    def __init__(self, bodies, radius=1e12):
        self.bodies = bodies
        self.radius = radius
        
    @classmethod
    def random(cls, num_bodies):
        bodies=[Body([0,0],[0,0],1e32)]
        for i in range(num_bodies-1):
            bodies.append(Body.random())
        return cls(bodies)
    
    @classmethod
    def from_file(cls, fname):
        bodies = []
        with open(fname,'r') as f:
            num_bodies = int(f.readline())
            radius = float(f.readline())
            for _ in range(num_bodies):
                linia = f.readline()
                px, py, vx, vy, m = [float(z) for z in linia.strip().split() if z]
                bodies.append(Body([px, py], [vx, vy], m))
        print(f"Universe imported successfully {len(bodies)} Bodies !!!")
        return cls(bodies, radius)
    
    @classmethod
    def configured_interactive(cls):
        bodies = []
        num_bodies = int(input("¿Cuántos cuerpos querés agregar? "))
        for i in range(num_bodies):
            print(f"\nCuerpo {i+1}:")
            x = float(input("Posición X (m): "))
            y = float(input("Posición Y (m): "))
            vx = float(input("Velocidad X (m/s): "))
            vy = float(input("Velocidad Y (m/s): "))
            mass = float(input("Masa (kg): "))

            body = Body(position=[x, y], velocity=[vx, vy], mass=mass)
            bodies.append(body)

        return cls(bodies)
    

class Body():
    G = 6.67408e-11
    def __init__(self, position, velocity, mass):
        self._mass = mass
        self._position = np.array(position, dtype=np.float64)  
        self._velocity = np.array(velocity, dtype=np.float64)   


    @property
    def mass(self):
        return self._mass
    @property
    def position(self):
        return self._position
    @property
    def velocity(self):
        return self._velocity

    def _force(self, another_body):
        # Calcula la força gravitatòria exercida per un altre cos.
        distance12 = self._distance_to(another_body)
        magnitude = self._mass * another_body._mass * Body.G / distance12**2
        direction = (another_body._position - self._position) / distance12
        return magnitude * direction
    
    def _distance(self, p, q):
        return np.sqrt(np.square(p - q).sum())
    
    def _distance_to(self, another_body):
        return self._distance(self._position, another_body._position)
    
    def update(self, force, dt):
        if self._mass == 0:
            raise ValueError("La Massa NO pot ser zero")
        acceleration = force / self._mass
        self._velocity += acceleration * dt
        self._position += self._velocity * dt

    def total_force(self, other_bodies):
        force = np.zeros(2)
        for body in other_bodies:
            if body is not self:
                force += self._force(body)
        return force

    def move(self, other_bodies, dt): 
        # Actualitza la velocitat i posició del cos en funció de la força i el pas de temps.
        total_force = self.total_force(other_bodies)
        self.update(total_force, dt)
    
    @staticmethod
    def random_vector(a, b, dim=2):
        return np.random.uniform(a, b, dim)
    
    @classmethod
    def random(cls, universe_radius=1e12): 
        mass = np.random.uniform(1e22, 1e24)
        position = Body.random_vector(-universe_radius, universe_radius, 2)
        velocity = Body.random_vector(1e05, 1e04, 2)
        
        return cls(position, velocity, mass)
 


#------------------------------------------------------MAIN CODE---------------------------------------------------------------------
if __name__ == '__main__':
    #universe = Universe.random(10)
    universe = Universe.from_file('3body2.txt')
    #universe = Universe.configured_interactive()
    for body in universe.bodies:
        print(f"Body: {body._position} x, {body._velocity} v, {body._mass} m")
    simulator = NBodySimulator(800, universe)
    time_step = 5000
    simulator.animate(time_step, trace=True)