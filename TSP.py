# Given  - Each city needs to be visited exactly once
#Return to the original City   - Total distance is the  calcuated accordingly


#Approach
# 1. Create  a population
# 2. Determine the Fitness
# 3, Select the mating pool
# 4. Breed, Mutate
# 5. Repeat

# Definitions
# Gene = A city here 
# Individual  - A single route satisfying the given condition
# Population - A collection of possible routes
# Parents - Two routes that are combined to create a new one
# Mating Pool -  A collection of parents that are used to create our next population
# Fitness - a function that tells us - the shortest distance.
# Mutation - a way to introduce some vatitions in the population by randomly swapping two cities.



import numpy as np, random, operator , pandas as pd, matplotlib.pyplot as plt

#class City:
 ##   def distance (self,city_b):
 ##       distance = 



# Creating a Fitness Function
# Treating the fitness as inverse of the route distance.
# Need to minimize route ditance , larger fitness - socre is better.


class Fitness:
    def __init__(self, route):
        self.route = route
        self.distance = 0
        self.fitness= 0.0
    
    def routeDistance(self):
        if self.distance ==0:
            pathDistance = 0
            for i in range(0, len(self.route)):
                fromCity = self.route[i]
                toCity = None
                if i + 1 < len(self.route):
                    toCity = self.route[i + 1]
                else:
                    toCity = self.route[0]
                pathDistance += fromCity.distance(toCity)
            self.distance = pathDistance
        return self.distance
    
    def routeFitness(self):
        if self.fitness == 0:
            self.fitness = 1 / float(self.routeDistance())
        return self.fitness
    

# Randomly select the order in which we visit each city


    
# Route Generator
def createRoute(cityList):
    route = random.sample(cityList, len(cityList))
    return route


# Create First population
def initialPopulation(popSize, cityList):
    population = []

    for i in range(0, popSize):
        population.append(createRoute(cityList))
    return population

# Create GA
# Rank Individuals
# Steps to go ahead with creating a mating pool
# Step 1 : Rank the routes to determine which route to select in our SELECTION functions
# Step 2 : Need to hold the best route and then the selection function returns a list to route the ID's which can be then used to create a mating pool

def rankRoutes(population):
    fitnessResults = {}
    for i in range(0,len(population)):
        fitnessResults[i] = Fitness(population[i]).routeFitness()
    return sorted(fitnessResults.items(), key = operator.itemgetter(1), reverse = True)

# Selection function that will be used to make the list of parent routes
#

def selection(popRanked, eliteSize):
    selectionResults = []
    df = pd.DataFrame(np.array(popRanked), columns=["Index","Fitness"])
    df['cum_sum'] = df.Fitness.cumsum()
    df['cum_perc'] = 100*df.cum_sum/df.Fitness.sum()
    
    for i in range(0, eliteSize):
        selectionResults.append(popRanked[i][0])
    for i in range(0, len(popRanked) - eliteSize):
        pick = 100*random.random()
        for i in range(0, len(popRanked)):
            if pick <= df.iat[i,3]:
                selectionResults.append(popRanked[i][0])
                break
    return selectionResults


# Create mating pool - to extract a individual from our population
def matingPool(population, selectionResults):
    matingpool = []
    for i in range(0, len(selectionResults)):
        index = selectionResults[i]
        matingpool.append(population[index])
    return matingpool


# create a crossover function for two parents to create one child
# Incase of TSP - we are using the Ordered crossover 
# In Ordered Crossover - Randomly select a subset of the first parent string. - Then fill the remainder of the route with genes from the second parent in the order in which they appear without duplicating genes
def breed(parent1, parent2):
    child = []
    childP1 = []
    childP2 = []
    
    geneA = int(random.random() * len(parent1))
    geneB = int(random.random() * len(parent1))
    
    startGene = min(geneA, geneB)
    endGene = max(geneA, geneB)

    for i in range(startGene, endGene):
        childP1.append(parent1[i])
        
    childP2 = [item for item in parent2 if item not in childP1]

    child = childP1 + childP2
    return child


# create function to run crossover full maring pool
# using the breed function to fill out the rest of the next generation
def breedPopulation(matingpool, eliteSize):
    children = []
    length = len(matingpool) - eliteSize
    pool = random.sample(matingpool, len(matingpool))

    for i in range(0,eliteSize):
        children.append(matingpool[i])
    
    for i in range(0, length):
        child = breed(pool[i], pool[len(matingpool)-i-1])
        children.append(child)
    return children

# Mutate a single route
def mutate(individual, mutationRate):
    for swapped in range(len(individual)):
        if(random.random() < mutationRate):
            swapWith = int(random.random() * len(individual))
            
            city1 = individual[swapped]
            city2 = individual[swapWith]
            
            individual[swapped] = city2
            individual[swapWith] = city1
    return individual

# create a function to mutate a single route
def mutatePopulation(population, mutationRate):
    mutatedPop = []
    
    for ind in range(0, len(population)):
        mutatedInd = mutate(population[ind], mutationRate)
        mutatedPop.append(mutatedInd)
    return mutatedPop

# create a next geenration
def nextGeneration(currentGen, eliteSize, mutationRate):
    popRanked = rankRoutes(currentGen)
    selectionResults = selection(popRanked, eliteSize)
    matingpool = matingPool(currentGen, selectionResults)
    children = breedPopulation(matingpool, eliteSize)
    nextGeneration = mutatePopulation(children, mutationRate)
    return nextGeneration

# Creating the genetic algo
def geneticAlgorithm(population, popSize, eliteSize, mutationRate, generations):
    pop = initialPopulation(popSize, population)
    print("Initial distance: " + str(1 / rankRoutes(pop)[0][1]))
    
    for i in range(0, generations):
        pop = nextGeneration(pop, eliteSize, mutationRate)
    
    print("Final distance: " + str(1 / rankRoutes(pop)[0][1]))
    bestRouteIndex = rankRoutes(pop)[0][0]
    bestRoute = pop[bestRouteIndex]
    return bestRoute








