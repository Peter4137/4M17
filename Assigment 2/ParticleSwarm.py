import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation
from scipy import optimize as opt

class Problem:
    def __init__(self, iterations, swarm_size, parameters, bounds, dimension):
        self.iterations = iterations
        self.swarm_size = swarm_size
        self.swarm = Swarm(dimension, swarm_size, bounds, parameters, self.f, self)
        self.resets = []
        self.inertia = parameters[0]

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
            self.check_reset(i)
            self.swarm.update(i, self.iterations)
            all_best_costs.append(self.swarm.swarm_best_cost)
            new_x = []
            for i in range(swarm_size):
                new_x.append(self.swarm.swarm[i].x)
            all_x.append(new_x)

        self.solution = [self.swarm.swarm_best_x, self.swarm.swarm_best_cost, all_x, all_best_costs]
        
        print("Final minimum x: {}".format(self.solution[0]))
        print("Final minimum f(x): {}".format(self.solution[1]))
        print(self.resets)
        if convergence_plot:
            self.convergence_plot()
        if animation:
            self.animation()
        if progression_plot:
            self.progression_plot()

        return self.solution

    def check_reset(self, iteration):
        return
        all_dist = []
        total = 0
        epsilon = 0.01

        for particle in self.swarm.swarm:
            all_dist.append(np.linalg.norm(particle.x - self.swarm.swarm_best_x, 2))
        average = sum(all_dist)/len(all_dist)
        if average < (bounds[1]-bounds[0])*epsilon:
            self.resets.append(iteration)
            self.optimum_reset(max(all_dist))           

    def optimum_reset(self, dist):
        for particle in self.swarm.swarm:
            if self.f(particle.x + 2*(self.swarm.swarm_best_x - particle.x)) < self.f(particle.x):
                particle.x = particle.x + 2*(self.swarm.swarm_best_x - particle.x)
            # particle.x =  particle.x
            # particle.velocity = np.random.uniform(low=bounds[0], high=bounds[1], size=(dimension,))

    def convergence_plot(self):
        plt.plot(self.solution[3])
        plt.xlabel('Iteration number')
        plt.ylabel('Best particle cost')
        for xi in self.resets:
            plt.axvline(xi, ls="--", color="red")
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
        steps = [0, 5, 10, 20]
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
    def __init__(self, dimension, population_size, bounds, parameters, f, problem):
        self.swarm_best_cost = np.inf
        self.swarm = []
        for i in range(population_size):
            self.swarm.append(Particle(dimension, bounds, parameters, f, problem))
        
        for particle in self.swarm:
            
            if particle.particle_best_cost < self.swarm_best_cost:
                self.swarm_best_x = particle.particle_best_x
                self.swarm_best_cost = particle.particle_best_cost
        
    def update(self, iteration, max_iterations):
        neighborhood_size = 10
        for particle in self.swarm:
            particle.update(self.swarm_best_x, self.swarm_best_cost, iteration, max_iterations)
            for nearby_particle in self.swarm:
                if nearby_particle == particle:
                    continue
                if np.linalg.norm(nearby_particle.x - particle.x, 2) < neighborhood_size:
                    if nearby_particle.local_best_cost > particle.particle_best_cost:
                        nearby_particle.local_best_cost = particle.particle_best_cost
                        nearby_particle.local_best_x = particle.particle_best_x

        for particle in self.swarm:
            if particle.particle_best_cost < self.swarm_best_cost:
                self.swarm_best_x = particle.particle_best_x
                self.swarm_best_cost = particle.particle_best_cost

class Particle:
    def __init__(self, dimension, bounds, parameters, f, problem):
        self.bounds = bounds
        self.parameters = parameters
        self.inertia = self.parameters[0]
        self.x = np.random.uniform(low=bounds[0], high=bounds[1], size=(dimension,))
        self.cost = f(self.x)
        self.problem = problem
        self.velocity = np.random.uniform(low=bounds[0], high=bounds[1], size=(dimension,))
        self.particle_best_x = self.x
        self.particle_best_cost = self.cost
        self.local_best_x = self.x
        self.local_best_cost = self.cost
    
    def update(self, swarm_best_x, swarm_best_cost, iteration, max_iterations):
        for i in range(self.x.size):
            r_p = np.random.rand(1)
            r_g = np.random.rand(1)
            self.velocity[i] = self.parameters[0]*self.velocity[i] + self.parameters[1]*r_p*(self.particle_best_x[i]-self.x[i]) + self.parameters[2]*r_g*(self.local_best_x[i]-self.x[i])
        self.x = self.x + self.velocity
        self.x = np.clip(self.x, self.bounds[0], self.bounds[1])
        self.cost = self.problem.f(self.x)
        iteration = float(iteration)
        # self.parameters[0] = 0.95*self.parameters[0]
        # self.parameters[0] = self.inertia + 0.1*self.inertia*np.cos(3*np.pi*iteration/max_iterations)
        # self.parameters[0] = (1-iteration/max_iterations)*self.inertia
        # self.parameters[0] = np.random.uniform(0.9*self.inertia, self.inertia*1.1)
        # self.parameters[0] = self.inertia + (self.inertia/3)*np.cos(np.pi*iteration/max_iterations)

        if self.cost < self.particle_best_cost:
            self.particle_best_x = self.x
            self.particle_best_cost = self.cost        

np.random.seed(0)
#Set Parameters
iterations = 200
omega = 0.683 
phi_p = 0.605 
phi_g = 0.358
swarm_size = 50
bounds = [-2,2]
dimension = 5

def parameter_problem(u):
    good = 0
    tests = 50
    total = 0
    for i in range(tests):
        np.random.seed(i)
        problem = Problem(iterations, swarm_size, u, bounds, dimension)
        solution = problem.solve()[1]
        if solution < -73.44:
            good += 1
        total += solution
    print(100*good/tests)
    return total/tests


print(parameter_problem(np.array([omega, phi_p, phi_g])))

# result = opt.minimize(parameter_problem,np.array([0.5, 0.5, 0.5]))
# print(result.x)
# omegas = [0.5, 0.7, 0.9, 1.1]
# phi_ps = np.linspace(0, 2, num=50)
# phi_gs = np.linspace(0, 1.5, num=50)
# PP, PG = np.meshgrid(phi_ps, phi_gs)

# A0 = np.zeros((phi_gs.size, phi_ps.size))
# A1 = A0.copy()
# A2 = A0.copy()
# A3 = A0.copy()
# # print(A)
# for i, phi_g in enumerate(phi_gs):
#     for j, phi_p in enumerate(phi_ps):
#         print(" === {},{} ===".format(i,j))
#         A0[i,j] = parameter_problem(np.array([omegas[0], phi_p, phi_g]))
#         A1[i,j] = parameter_problem(np.array([omegas[1], phi_p, phi_g]))
#         A2[i,j] = parameter_problem(np.array([omegas[2], phi_p, phi_g]))
#         A3[i,j] = parameter_problem(np.array([omegas[3], phi_p, phi_g]))

# print(A0.min(), A1.min(), A2.min(), A3.min())

# # print(PP)
# # print(A)  
        
# fig, axes = plt.subplots(2, 2)
# levels = np.linspace(-75, -10, 10)
# cp = axes[0,0].contourf(PP, PG, A0, levels=levels)
# axes[0, 0].set_title("Omega = {}".format(omegas[0]))
# axes[0, 0].set_xlabel("phi_g")
# axes[0, 0].set_ylabel("phi_p")

# axes[0, 1].contourf(PP, PG, A1, levels=levels)
# axes[0, 1].set_title("Omega = {}".format(omegas[1]))
# axes[0, 1].set_xlabel("phi_g")
# axes[0, 1].set_ylabel("phi_p")

# cp = axes[1, 0].contourf(PP, PG, A2, levels=levels)
# axes[1, 0].set_title("Omega = {}".format(omegas[2]))
# axes[1, 0].set_xlabel("phi_g")
# axes[1, 0].set_ylabel("phi_p")

# cp = axes[1, 1].contourf(PP, PG, A3, levels=levels)
# axes[1, 1].set_title("Omega = {}".format(omegas[3]))
# axes[1, 1].set_xlabel("phi_g")
# axes[1, 1].set_ylabel("phi_p")
# # plt.colorbar(cp, cax=axes)
# # plt.ylabel("phi_p")
# # plt.xlabel("phi_g")
# plt.subplots_adjust(bottom=0.07, top=0.97, hspace=0.22)
# plt.show()



# problem = Problem(iterations, swarm_size, [omega, phi_p, phi_g], bounds, dimension)
# problem.solve(convergence_plot=True, animation=True, progression_plot=True)

