import numpy as np
import random
import matplotlib.pyplot as plt

random.seed(0)

class AnnealingProblem:
    def __init__(self, lowerBound, upperBound, steps, neighborRange, dimensions, x0):
        self.lowerBounds = lowerBounds
        self.upperBounds = upperBounds
        self.steps = steps
        self.neighborRange = neighborRange
        self.dimensions = dimensions
        self.x0 = x0

    def schwefel(self, x):
        sum = 0
        for i in range(x.size):
            sum += -x[i]*np.sin(np.sqrt(abs(x[i])))
        return sum

    def acceptance(self, E0, E1, temperature):
        if E1<E0:
            return 1
        else:
            return np.exp((E0-E1)/temperature)

    def annealingSchedule(self, k):
        return T0*alpha**k

    def simulatedAnnealing(self):
        functionCalls = 0
        state = self.x0
        objectiveList = np.array([self.schwefel(self.x0)])

        for k in range(steps):
            temperature = self.annealingSchedule(k)
            newState = np.zeros(self.dimensions)
            validState = False
            while not validState:
                for i in range(self.dimensions):
                    offset = random.uniform(-self.neighborRange, self.neighborRange)
                    newState[i] = state[i] + offset
                if self.isNewStateinBounds(newState):
                    validState = True
            newStateObjective = self.schwefel(newState)
            if self.acceptance(objectiveList[-1], newStateObjective, temperature) >= random.uniform(0,1):
                state = newState
                objectiveList = np.append(objectiveList, newStateObjective)

        print(state)
        print(objectiveList[-1])
        plt.plot(objectiveList)
        plt.show()

    def isNewStateinBounds(self, x):
        for i in range(len(x)):
            if x[i] < self.lowerBounds[i] or x[i] > self.upperBounds[i]:
                return False
        return True

    def run(self, T0, alpha):
        self.T0 = T0
        self.alpha = alpha
        self.simulatedAnnealing()

# problem setup
n = 5
lowerBounds = -500*np.ones(n)
upperBounds = 500*np.ones(n)
steps = 10000
neighborRange = 10

T0 = 2000
alpha = 0.9992

initialGuess = np.zeros(n)
for i in range(n):
    initialGuess[i] = random.uniform(lowerBounds[i], upperBounds[i])


problem = AnnealingProblem(lowerBounds, upperBounds, steps, neighborRange, 5, initialGuess)
problem.run(T0, alpha)


            
# Notes on choices of variables:
# currently using an exponential decay for annealing schedule
# n=2 results
# coolingRate | neighborDistance | notes
# 1000        | 10               | global max avoided, local min found
# 1000        | 100              | irratic graph, no min found (yet...)              
# 1200        | 10               | global min found
# 100         | 10               | irratic, no min found
# 2500        | 10               | local min found quickly, global min not found
# global min appears to be -837.925 for n=2
# 
# n=5 results:
# min should be 2094.9145
# global min of -1753.41 (best so far...)
#
#
#

