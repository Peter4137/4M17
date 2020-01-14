import numpy as np
from matplotlib import pyplot as plt
from scipy import optimize as opt

class Problem:
    def __init__(self, max_iterations, dimensions, bound, parameters):
        self.max_iterations = max_iterations
        self.dimensions = dimensions
        self.bound = bound
        self.parameters = parameters

        self.x = np.random.uniform(-bound, bound, dimensions)
        self.temperature = np.inf
        self.cost = self.f_penalty(self.x)
        self.d = np.diag(parameters[1]*np.ones(self.x.size))
        self.all_x = [self.x]
        self.all_costs = [self.cost]

        self.archive_length = 10

        self.archive = {
            "best_x": self.x,
            "best_cost": self.cost,
            "best_index": 0,
            "resets":[],
            "best_dissimilar_cost":[],
            "best_dissimilar_x":[],
            "all_best_costs":[]
            }

    def update_archive(self, iteration):
        if self.cost < self.archive["best_cost"]:
            self.archive["best_x"] = self.x
            self.archive["best_cost"] = self.cost
            self.archive["best_index"] = iteration
        
        if len(self.archive["best_dissimilar_cost"]) < self.archive_length:
            self.archive["best_dissimilar_cost"].append(self.cost)
            self.archive["best_dissimilar_x"].append(self.x)
        else:
            dissimilar = True
            for solution in zip(self.archive["best_dissimilar_x"], self.archive["best_dissimilar_cost"]):
                if np.linalg.norm(solution[0]-self.x, 2) < 0.5:
                    dissimilar = False
                    if self.cost < solution[1]:
                        pos = self.archive["best_dissimilar_cost"].index(solution[1])
                        self.archive["best_dissimilar_x"][pos] = self.x
                        self.archive["best_dissimilar_cost"][pos] = self.cost
        
            if dissimilar and self.cost < max(self.archive["best_dissimilar_cost"]):
                pos = self.archive["best_dissimilar_cost"].index(max(self.archive["best_dissimilar_cost"]))
                self.archive["best_dissimilar_x"][pos] = self.x
                self.archive["best_dissimilar_cost"][pos] = self.cost

    def probability_accept(self, new_cost, x_step):
        d_bar = np.sum(np.abs(x_step))
        if new_cost < self.cost:
            return 1
        else:
            return np.exp(-(new_cost-self.cost)/(self.temperature*d_bar))
        
    def initial_survey(self):
        temp_x = np.random.uniform(-bound, bound, dimensions)
        all_cost_steps = []
        for i in range(500):
            step = self.parameters[1]*np.random.uniform(-1, 1, self.x.size)
            cost_step = self.f_penalty(temp_x + step) - self.f_penalty(temp_x)
            if cost_step > 0:
                all_cost_steps.append(cost_step)
            temp_x = temp_x + step
        avg_cost_step = sum(all_cost_steps)/len(all_cost_steps)
        chi_0 = 0.8
        return -avg_cost_step/np.log(chi_0)

    def check_restart(self, restart_count):
        restart_condition = 2000
        if restart_count - self.archive["best_index"] > restart_condition:
            # print("No new best solution found in {} iterations, restarting at best known solution".format(restart_condition))
            return True
        return False

    def solve(self, plot_convergence=False, plot_path=False):
        # print("Running SA algorithm")
        self.temperature = self.initial_survey()
        # print("Initial temperature: {}".format(self.temperature))
        iterations = 0
        acceptances = 0
        restart_count = 0
        while iterations < self.max_iterations:
            iterations+=1
            restart_count+=1
            x_step = np.dot(self.d, np.random.uniform(-1, 1, self.x.size))
            new_x = self.x + x_step
            new_cost = self.f_penalty(new_x)

            if np.random.rand(1) < self.probability_accept(new_cost, x_step):
                acceptances += 1
                self.all_x.append(new_x)
                self.all_costs.append(new_cost)
                self.x = new_x
                self.cost = self.f_penalty(self.x)
                self.update_d(x_step)
                self.archive["all_best_costs"].append(self.archive["best_cost"])

            self.update_archive(iterations)

            if self.check_restart(restart_count):
                self.x = self.archive["best_x"]
                self.cost = self.archive["best_cost"]
                self.archive["resets"].append(len(self.all_x))
                restart_count = 0

            if iterations % self.parameters[2] == 0 or acceptances > self.parameters[2]:
                if acceptances == 0:
                    print("no acceptances in chain, exiting")
                    break
                self.temperature = self.parameters[0]*self.temperature
                acceptances = 0

        print("Exited after {} iterations".format(iterations))
        print("Best objective function value: {}".format(self.archive["best_cost"]))
        print("Best value found at: {}".format(self.archive["best_x"]))

        if plot_convergence:
            self.plot_convergence()

        if plot_path:
            self.plot_path()

        return self.x, self.cost, self.all_x, self.all_costs

    def plot_convergence(self):
        plt.plot(self.all_costs, label="Current cost")
        plt.xlabel('Iteration number')
        plt.ylabel('Objective function')
        for xi in self.archive["resets"]:
            plt.axvline(xi, ls="--", color="red")
        plt.plot(self.archive["all_best_costs"], color="orange", label="Best cost")
        plt.axhline(-14.83795*self.x.size, ls="--", color="green", label="Global optimum")
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_dissimilar_solutions(self):
        X,Y,Z = self.contours()
        plt.contour(X,Y,Z, zorder=-1)
        [plt.scatter(x[0], x[1], facecolors='black', edgecolors='black', zorder=1, marker='x', s=50) for x in self.archive["best_dissimilar_x"]]
        plt.xlabel("x0")
        plt.ylabel("x1")
        plt.show()

    def contours(self):
        x = np.linspace(-2,2)
        y = np.linspace(-2,2)
        X, Y = np.meshgrid(x,y)
        Z = self.f(np.array([X,Y, np.zeros(X.shape), np.zeros(X.shape), np.zeros(X.shape)]))
        return X,Y,Z

    def plot_path(self):
        plt.axes(xlim=(-self.bound, self.bound), ylim=(-self.bound, self.bound))
        X,Y,Z = self.contours()
        plt.contour(X,Y,Z, zorder=-1)
        subset_x = []
        for i, x in enumerate(self.all_x):
            if i%10 == 0:
                subset_x.append(x)
        [plt.scatter(x[0], x[1], facecolors='none', edgecolors='black', zorder=1) for x in subset_x]
        plt.xlabel("x0")
        plt.ylabel("x1")
        plt.show()

    def plot_f_adj(self):
        levels = np.linspace(-50,50,20)
        fig, axs = plt.subplots(1, 2)
        x = np.linspace(-3,3, 100)
        y = np.linspace(-3,3, 100)
        X, Y = np.meshgrid(x,y)
        Z = self.f(np.array([X,Y]))
        Z_penalty = self.g_penalty(np.array([X,Y]))
        contour0 = axs[0].contour(X,Y,Z, levels, cmap="viridis")
        axs[0].set_title("Schubert Function contours")
        axs[0].set_xlabel("x0")
        axs[0].set_ylabel("x1")
        contour1 = axs[1].contour(X,Y,Z_penalty, levels, cmap="viridis")
        axs[1].set_title("Augmented Function contours with penalty function")
        axs[1].set_xlabel("x0")
        axs[1].set_ylabel("x1")
        plt.show()

    def g_penalty(self, x):
        penalty = np.zeros(x.shape[1:])
        for i in range(penalty.shape[0]):
            for j in range(penalty.shape[1]):
                c_v = 0
                for k in range(x.shape[0]):
                    if np.abs(x[k,i,j]) > bound:
                        c_v +=  np.abs(x[k,i,j])-self.bound
                penalty[i,j] += 50*c_v
        return self.f(x) + penalty

    def f_penalty(self, x):
        w = 1000*np.ones(x.shape)
        c = np.zeros(x.shape)
        for i, x_i in enumerate(x):
            if np.abs(x_i) > bound:
                c[i] = np.abs(x_i) - bound
        return self.f(x) + np.dot(w.T, c)/self.temperature

    def f(self, x):
        total = 0
        for i in range(x.shape[0]):
            for j in range(1,6):
                total += j*np.sin((j+1)*x[i]+j)
        return total

    def update_d(self, x_step):
        # Parks 1990 method for finding new trial solutions
        alpha = 0.1
        omega = 2.1
        # Values for alpha and omega based of SA lecture notes
        self.d = (1-alpha)*self.d + alpha*omega*np.diag(np.abs(x_step))

def run_multiple(n):
    good_min_found = []
    for i in range(n):
        np.random.seed(i)
        problem = Problem(max_iterations, dimensions, bound, [alpha, neighborhood_size, chain_length])
        problem.solve()
        if problem.archive["best_cost"] < -14.5*problem.x.size:
            good_min_found.append(True)
        else:
            good_min_found.append(False)
        # all_min.append(problem.archive["best_cost"])
    return good_min_found

def parameter_problem(u):
    good = 0
    max_iterations = 10000
    dimensions = 5
    total = 0
    tests = 50
    for i in range(tests):
        np.random.seed(i)
        problem = Problem(max_iterations, dimensions, bound, u)
        problem.solve()
        total += problem.archive["best_cost"]
        if problem.archive["best_cost"] < -73.44:
            print(i)
            good+=1
    print(good)
    return total/tests

np.random.seed(1)

max_iterations = 10000
dimensions = 5
bound = 2

print(parameter_problem([0.95, 0.1, 300]))

# neighborhood_sizes = np.linspace(0.05, 0.25, 9) # np.array([0.05,0.01,  0.05, 0.1, 0.2, 0.3])
# chain_lengths  = np.linspace(100, 1000, 10) # np.array([100, 200, 500, 1000, 2000])
# ns, cl = np.meshgrid(neighborhood_sizes, chain_lengths)
# print(chain_lengths)
# print(neighborhood_sizes)
# A = np.zeros((chain_lengths.size, neighborhood_sizes.size))

# for i, neighborhood_size in enumerate(neighborhood_sizes):
#     for j, chain_length in enumerate(chain_lengths):
#         print(" === {},{} ===".format(i,j))
#         A[j,i] = parameter_problem(np.array([0.95, neighborhood_size, chain_length]))

# print(A.shape)
# print(ns.shape)
# print(cl.shape)        
# cp = plt.contourf(ns, cl, A)
# cbar = plt.colorbar(cp)
# cbar.set_label("Average best algorithm result")
# plt.ylabel("Markov Chain Length")
# plt.xlabel("Neighborhood size")
# plt.show()

# alpha = 0.95
# neighborhood_size = 0.175
# chain_length = 500
# result = opt.minimize(parameter_problem, np.array([0.95, 0.1, 1000]), options={"maxiter": 3, "disp":True})
# print(result.x)
# print(result.fun)
# result = run_multiple(20)

# print("{}% of tests produced 'good' solutions".format(100*sum(result)/len(result)))
# problem = Problem(max_iterations, dimensions, bound, [0.95, 0.1, 300])

# a = 0
# for i in range(100):
#     a += problem.initial_survey()

# print(a/100)
# problem.plot_f_adj()
# problem.solve(plot_convergence=True, plot_path=True)
# problem.plot_dissimilar_solutions()

