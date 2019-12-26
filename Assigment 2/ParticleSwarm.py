import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation

class Problem:
    def __init__(self, iterations, swarm_size, parameters, bounds, dimension):
        self.iterations = iterations
        self.swarm_size = swarm_size
        self.swarm = Swarm(dimension, swarm_size, bounds, parameters, self.f)

    def f(self, x):
        total = 0
        for i in range(x.shape[0]):
            for j in range(1,6):
                total += j*np.sin((j+1)*x[i]+j)
        return total

    def solve(self, convergence_plot=False, animation=False, progression_plot=False):
        all_best_costs = [self.swarm.swarm_best_cost]
        all_x = []
        new_x = []
        for i in range(self.swarm_size):
            new_x.append(self.swarm.swarm[i].x)
        all_x.append(new_x)

        for i in range(self.iterations):
            if (i+1)%(self.iterations/100) == 0:
                print("{}%".format(float(i+1)/(self.iterations/100)))
            self.swarm.update()
            all_best_costs.append(self.swarm.swarm_best_cost)
            new_x = []
            for i in range(swarm_size):
                new_x.append(self.swarm.swarm[i].x)
            all_x.append(new_x)

        self.solution = [self.swarm.swarm_best_x, self.swarm.swarm_best_cost, all_x, all_best_costs]
        
        print("Final minimum x: {}".format(self.solution[0]))
        print("Final minimum f(x): {}".format(self.solution[1]))

        if convergence_plot:
            self.convergence_plot()
        if animation:
            self.animation()
        if progression_plot:
            self.progression_plot()

        return self.solution

    def convergence_plot(self):
        plt.plot(self.solution[3])
        plt.xlabel('Iteration number')
        plt.ylabel('Best particle cost')
        plt.show()

    def contours(self):
        x = np.linspace(-2,2)
        y = np.linspace(-2,2)
        X, Y = np.meshgrid(x,y)
        Z = self.f(np.array([X,Y, np.zeros(X.shape), np.zeros(X.shape), np.zeros(X.shape)]))
        return X,Y,Z

    def animation(self):
        data = self.solution[2]
        X, Y, Z = self.contours()
        
        def animate(i, data, scatters):
            for j, point in enumerate(data[i]):
                scatters[j].set_offsets((point[0], point[1]))
            return scatters
        
        fig = plt.figure()
        ax = plt.axes(xlim=(bounds[0], bounds[1]), ylim=(bounds[0], bounds[1]))
        plt.xlabel('x1')
        plt.ylabel('x2')
        scatters = [ax.scatter(data[0][i][0], data[0][i][1], marker='x', color='black') for i in range(self.swarm_size)]
        ax.contour(X,Y,Z)
        anim = animation.FuncAnimation(fig, animate, self.iterations, fargs=(data, scatters),
                interval=50, blit=False, repeat=True)
        # writer = animation.FFMpegWriter(fps=30, codec='libx264')
        # anim.save('ParticleSwarmAnimation.gif', writer=writer)
        plt.show()
    
    def progression_plot(self):
        steps = [0, 10, 20, 50]
        data = self.solution[2]
        X, Y, Z = self.contours()
        fig, axs = plt.subplots(2, 2)

        axs[0, 0].contour(X,Y,Z, zorder=-1)
        [axs[0, 0].scatter(data[steps[0]][i][0], data[steps[0]][i][1], marker='x', color='black', zorder=1) for i in range(self.swarm_size)]
        axs[0, 0].set_title('{} steps'.format(steps[0]))

        axs[0, 1].contour(X,Y,Z, zorder=-1)
        [axs[0, 1].scatter(data[steps[1]][i][0], data[steps[1]][i][1], marker='x', color='black', zorder=1) for i in range(self.swarm_size)]
        axs[0, 1].set_title('{} steps'.format(steps[1]))

        axs[1, 0].contour(X,Y,Z, zorder=-1)
        [axs[1, 0].scatter(data[steps[2]][i][0], data[steps[2]][i][1], marker='x', color='black', zorder=1) for i in range(self.swarm_size)]
        axs[1, 0].set_title('{} steps'.format(steps[2]))

        axs[1, 1].contour(X,Y,Z, zorder=-1)
        [axs[1, 1].scatter(data[steps[3]][i][0], data[steps[3]][i][1], marker='x', color='black', zorder=1) for i in range(self.swarm_size)]
        axs[1, 1].set_title('{} steps'.format(steps[3]))
        plt.show()


class Swarm:
    def __init__(self, dimension, population_size, bounds, parameters, f):
        self.swarm_best_cost = np.inf
        self.swarm = []
        for i in range(population_size):
            self.swarm.append(Particle(dimension, bounds, parameters, f))
        
        for particle in self.swarm:
            if particle.particle_best_cost < self.swarm_best_cost:
                self.swarm_best_x = particle.particle_best_x
                self.swarm_best_cost = particle.particle_best_cost
        
    def update(self):
        for particle in self.swarm:
            self.swarm_best_x, self.swarm_best_cost = particle.update(self.swarm_best_x, self.swarm_best_cost)

class Particle:
    def __init__(self, dimension, bounds, parameters, f):
        self.bounds = bounds
        self.parameters = parameters
        self.x = np.random.uniform(low=bounds[0], high=bounds[1], size=(dimension,))
        self.cost = f(self.x)
        self.velocity = np.random.uniform(low=bounds[0], high=bounds[1], size=(dimension,))
        self.particle_best_x = self.x
        self.particle_best_cost = self.cost
    
    def update(self, swarm_best_x, swarm_best_cost):
        for i in range(self.x.size):
            r_p = np.random.rand(1)
            r_g = np.random.rand(1)
            self.velocity[i] = self.parameters[0]*self.velocity[i] + self.parameters[1]*r_p*(self.particle_best_x[i]-self.x[i]) + self.parameters[2]*r_g*(swarm_best_x[i]-self.x[i])
        self.x = self.x + self.velocity
        self.x = np.clip(self.x, self.bounds[0], self.bounds[1])
        self.cost = problem.f(self.x)

        if self.cost < self.particle_best_cost:
            self.particle_best_x = self.x
            self.particle_best_cost = self.cost

        if self.cost < swarm_best_cost:
            swarm_best_x = self.x
            swarm_best_cost = self.cost
        return swarm_best_x, swarm_best_cost

np.random.seed(1)
#Set Parameters
iterations = 200

# 0.9, 0.2, 0.6 seed=1
# 0.5, 0.01, 0.6 seed=2
omega = 0.9 #0.9  
phi_p = 0.2 #0.1
phi_g = 0.6 #0.6
swarm_size = 50
bounds = [-2,2]
dimension = 5

problem = Problem(iterations, swarm_size, [omega, phi_p, phi_g], bounds, dimension)
problem.solve(convergence_plot=True, animation=True, progression_plot=True)

