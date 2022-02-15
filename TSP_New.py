# Ashutosh Tripathi  - FH9936
#This code solves the Travelling Salesman Problem using GA

# Given  - Each city needs to be visited exactly once
#Return to the original City   - Total distance is the  calcuated accordingly

#Approach -GA
# 1. Create  a population
# 2. Determine the Fitness
# 3, Select the mating pool
# 4. Breed, Mutate
# 5. Repeat

import numpy as np
np.random.seed(42)

cities = [0, 1, 2, 3, 4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26] # Define the list of 27 cities starting from 0-26

# Mapping the city number with Actual City Name given in the Problem
map_cities= {0:"Bakersfield",1:"Barstow" ,2: "Carlsbad", 3: "Eureka", 4:"Fresno" ,5:"Lake Tahoe,So" ,6: "Las Vegas" ,7:"Long Beach" ,8:"Los Angeles",9: "Merced" ,10:"Modesto",
            11:"Monterey", 12: "Oakland",13:"Palm Springs",14:"Redding",15:"Sacramento",16:"San Benardino" ,17:"San Diego",18:"San Francisco",19:"San Jose",20:"San Luis Obispo",21:"Santa Barbara",
            22:"San Cruz",23:"Santa Rosa",24: "Sequoia Park", 25: "Stockton",26:"Yosemite"}

result =[]
# Creating a Matrix that has value of distance from CITY A to CITY B and vice versa  the values are given in the problem.
adjacency_matrix = np.asarray(
    [
        [  0.00, 129.00, 206.00, 569.00, 107.00, 360.00, 284.00, 144.00, 115.00, 162.00, 200.00, 231.00, 288.00, 226.00, 436.00, 272.00, 174.00, 231.00, 297.00, 252.00, 118.00, 146.00, 258.00, 347.00, 121.00, 227.00, 200.00 ],
        [129.00,   0.00, 153.00, 696.00, 236.00, 395.00, 155.00, 139.00, 130.00, 291.00, 329.00, 360.00, 417.00, 123.00, 565.00, 401.00,  71.00, 176.00, 426.00, 381.00, 247.00, 225.00, 387.00, 476.00, 250.00, 356.00, 329.00 ],
        [206.00, 153.00,   0.00, 777.00, 315.00, 780.00, 312.00,  82.00,  93.00, 370.00, 406.00, 428.00, 496.00, 116.00, 644.00, 480.00, 827.00,  23.00, 505.00, 460.00, 293.00, 188.00, 466.00, 565.00, 329.00, 435.00, 408.00 ],
        [569.00, 696.00, 777.00,   0.00, 462.00, 398.00, 797.00, 713.00, 694.00, 407.00, 369.00, 388.00, 291.00, 795.00, 150.00, 314.00,  43.00, 800.00, 272.00, 317.00, 504.00, 609.00, 349.00, 222.00, 544.00, 356.00, 488.00 ],
        [107.00, 236.00, 315.00, 462.00,   0.00, 388.00, 408.00, 251.00, 222.00,  55.00,  93.00, 152.00, 181.00, 333.00, 329.00, 185.00, 281.00, 338.00, 190.00, 145.00, 137.00, 242.00, 151.00, 240.00,  82.00, 120.00,  93.00 ],
        [360.00, 395.00, 780.00, 398.00, 388.00,   0.00, 466.00, 479.00, 456.00, 194.00, 156.00, 266.00, 195.00, 435.00, 249.00, 107.00, 436.00, 542.00, 192.00, 197.00, 197.00, 492.00, 229.00, 199.00, 335.00, 131.00, 133.00 ],
        [284.00, 155.00, 312.00, 797.00, 408.00, 466.00,   0.00, 314.00, 302.00, 446.00, 484.00, 504.00, 567.00, 276.00, 640.00, 587.00, 228.00, 332.00, 568.00, 524.00, 414.00, 354.00, 524.00, 610.00, 408.00, 510.00, 435.00 ],
        [144.00, 139.00,  82.00, 713.00, 251.00, 479.00, 314.00,   0.00,  29.00, 306.00, 344.00, 364.00, 432.00, 112.00, 580.00, 416.00,  68.00, 105.00, 441.00, 396.00, 229.00, 124.00, 402.00, 491.00, 265.00, 371.00, 344.00 ],
        [115.00, 130.00,  93.00, 694.00, 222.00, 456.00, 302.00,  29.00,   0.00, 277.00, 315.00, 335.00, 403.00, 111.00, 551.00, 387.00,  59.00, 116.00, 412.00, 367.00, 200.00,  95.00, 373.00, 462.00, 236.00, 342.00, 315.00 ],
        [162.00, 291.00, 370.00, 407.00,  55.00, 194.00, 446.00, 306.00, 277.00,   0.00,  37.00, 118.00, 126.00, 388.00, 274.00, 110.00, 336.00, 393.00, 135.00, 114.00, 192.00, 297.00, 118.00, 185.00, 137.00,  65.00,  81.00 ],
        [200.00, 329.00, 406.00, 369.00,  93.00, 156.00, 484.00, 344.00, 315.00,  37.00,   0.00, 153.00,  88.00, 426.00, 236.00,  72.00, 374.00, 431.00,  97.00,  82.00, 230.00, 335.00, 114.00, 147.00, 175.00,  27.00, 119.00 ],
        [231.00, 360.00, 428.00, 388.00, 152.00, 266.00, 504.00, 364.00, 335.00, 118.00, 153.00,   0.00, 111.00, 446.00, 325.00, 185.00, 394.00, 451.00, 116.00,  71.00, 135.00, 240.00,  45.00, 166.00, 234.00, 140.00, 199.00 ],
        [288.00, 417.00, 496.00, 291.00, 181.00, 195.00, 567.00, 432.00, 403.00, 126.00,  88.00, 111.00,   0.00, 514.00, 214.00,  87.00, 462.00, 519.00,   9.00,  40.00, 227.00, 332.00,  72.00,  59.00, 263.00,  75.00, 207.00 ],
        [226.00, 123.00, 116.00, 795.00, 333.00, 435.00, 276.00, 112.00, 111.00, 388.00, 426.00, 446.00, 514.00,   0.00, 682.00, 498.00,  52.00, 139.00, 523.00, 478.00, 311.00, 206.00, 484.00, 573.00, 347.00, 453.00, 426.00 ],
        [436.00, 565.00, 644.00, 150.00, 329.00, 249.00, 640.00, 580.00, 551.00, 274.00, 236.00, 325.00, 214.00, 682.00,   0.00, 164.00, 610.00, 667.00, 223.00, 254.00, 411.00, 546.00, 286.00, 251.00, 411.00, 209.00, 355.00 ],
        [272.00, 401.00, 480.00, 314.00, 185.00, 107.00, 587.00, 416.00, 387.00, 110.00,  72.00, 185.00,  87.00, 498.00, 164.00,   0.00, 446.00, 503.00,  87.00, 114.00, 301.00, 406.00, 146.00, 103.00, 247.00,  45.00, 191.00 ],
        [174.00,  71.00, 827.00,  43.00, 281.00, 436.00, 228.00,  68.00,  59.00, 336.00, 374.00, 394.00, 462.00,  52.00, 610.00, 446.00,   0.00, 105.00, 471.00, 426.00, 259.00, 254.00, 432.00, 521.00, 295.00, 401.00, 374.00 ],
        [231.00, 176.00,  23.00, 800.00, 338.00, 542.00, 332.00, 105.00, 116.00, 393.00, 431.00, 451.00, 519.00, 139.00, 667.00, 503.00, 105.00,   0.00, 528.00, 483.00, 316.00, 211.00, 489.00, 578.00, 352.00, 458.00, 431.00 ],
        [297.00, 426.00, 505.00, 272.00, 190.00, 192.00, 568.00, 441.00, 412.00, 135.00,  97.00, 116.00,   9.00, 523.00, 223.00,  87.00, 471.00, 528.00,   0.00,  45.00, 232.00, 337.00,  77.00,  50.00, 272.00,  84.00, 216.00 ],
        [252.00, 381.00, 460.00, 317.00, 145.00, 197.00, 524.00, 396.00, 367.00, 114.00,  82.00,  71.00,  40.00, 478.00, 254.00, 114.00, 426.00, 483.00,  45.00,   0.00, 187.00, 292.00,  32.00,  95.00, 227.00,  69.00, 195.00 ],
        [118.00, 247.00, 293.00, 504.00, 137.00, 197.00, 414.00, 229.00, 200.00, 192.00, 230.00, 135.00, 227.00, 311.00, 411.00, 301.00, 259.00, 316.00, 232.00, 187.00,   0.00, 105.00, 180.00, 282.00, 174.00, 256.00, 230.00 ],
        [146.00, 225.00, 188.00, 609.00, 242.00, 492.00, 354.00, 124.00,  95.00, 297.00, 335.00, 240.00, 332.00, 206.00, 546.00, 406.00, 254.00, 211.00, 337.00, 292.00, 105.00,   0.00, 285.00, 387.00, 287.00, 361.00, 335.00 ],
        [258.00, 387.00, 466.00, 349.00, 151.00, 229.00, 524.00, 402.00, 373.00, 118.00, 114.00,  45.00,  72.00, 484.00, 286.00, 146.00, 432.00, 489.00,  77.00,  32.00, 180.00, 285.00,   0.00, 127.00, 233.00, 101.00, 199.00 ],
        [347.00, 476.00, 565.00, 222.00, 240.00, 199.00, 610.00, 491.00, 462.00, 185.00, 147.00, 166.00,  59.00, 573.00, 251.00, 103.00, 521.00, 578.00,  50.00,  95.00, 282.00, 387.00, 127.00,   0.00, 322.00, 134.00, 266.00 ],
        [121.00, 250.00, 329.00, 544.00,  82.00, 335.00, 408.00, 265.00, 236.00, 137.00, 175.00, 234.00, 263.00, 347.00, 411.00, 247.00, 295.00, 352.00, 272.00, 227.00, 174.00, 287.00, 233.00, 322.00,   0.00, 202.00, 175.00 ],
        [227.00, 356.00, 435.00, 356.00, 120.00, 131.00, 510.00, 371.00, 342.00,  65.00,  27.00, 140.00,  75.00, 453.00, 209.00,  45.00, 401.00, 458.00,  84.00,  69.00, 256.00, 361.00, 101.00, 134.00, 202.00,   0.00, 146.00 ],
        [200.00, 329.00, 408.00, 488.00,  93.00, 133.00, 435.00, 344.00, 315.00,  81.00, 119.00, 199.00, 207.00, 426.00, 355.00, 191.00, 374.00, 431.00, 216.00, 195.00, 230.00, 335.00, 199.00, 266.00, 175.00, 146.00,   0.00 ],        
    ]
)
# Define a class to Represent the population - to get together all the pieces requried for a specific gen
# Here some variables  
class Population():
    def __init__(self, full_pop, adjacency_matrix):
        self.full_pop = full_pop                                # represents the full population
        self.parents = []                                       # to represent the few selected from the superior values
        self.score = 0                                          # to store the score best chromosome [here the optimal path]
        self.best = None                                        # to store the best Chromosome [optimal path itself]
        self.adjacency_matrix = adjacency_matrix                # matrix that as the distance values
   
   # the fitness function is used to determine the fitness of the chromosome 
   # In TSP we are determining the fitness by - shorter total distance    
    def fitness(self, chr):
        return sum(
        [
            self.adjacency_matrix[chr[i], chr[i + 1]]
            for i in range(len(chr) - 1)
        ]
    )
    
    # the evaluate function  - calculates the fitness of each chromosome in the full population 
    # Determining the best and storing the score and returning the probability that element in the Full population can be chosen as a Parent
    def evaluate(self):
        distances = np.asarray(
        [self.fitness(chr) for chr in self.full_pop]
        )
        self.score = np.min(distances)
        self.best = self.full_pop[distances.tolist().index(self.score)]
        self.parents.append(self.best)
        if False in (distances[0] == distances):
            distances = np.max(distances) - distances
        return distances / np.sum(distances)
    
    # Now we are select k number of parents here we are taking 10 parents as the basis of next generation 
    # a Simple method that compares the probability and a random number - if the probability is higher than add that path /Chromosome to Parents. Here do it for 10 times as K is 10 
    def select(self, k=10):
        fit = self.evaluate()
        while len(self.parents) < k:
            idx = np.random.randint(0, len(fit))
            if fit[idx] > np.random.rand():
                self.parents.append(self.full_pop[idx])
        self.parents = np.asarray(self.parents)
    
   # Main part - MUTATION
   # We are using the SWAP and CROSSOVER Mutation 
   #Just swapping is a disruptive process so we are using crossover
   # the crossover  function - grab 2 parents and slice a portion of the optimal path /chromosome and fill the rest slots with other parent without duplicates in the path
    
    def swap(self,chr):
        a, b = np.random.choice(len(chr), 2)
        chr[a], chr[b] = (
            chr[b],
            chr[a],
        )
        return chr
    
    def crossover(self, cross=0.1):
        children = []
        count, size = self.parents.shape
        for _ in range(len(self.full_pop)):
            if np.random.rand() > cross:
                children.append(
                    list(self.parents[np.random.randint(count, size=1)[0]])
                )
            else:
                parent1, parent2 = self.parents[
                    np.random.randint(count, size=2), :
                ]
                idx = np.random.choice(range(size), size=2, replace=False)
                start, end = min(idx), max(idx)
                child = [None] * size
                for i in range(start, end + 1, 1):
                    child[i] = parent1[i]
                pointer = 0
                for i in range(size):
                    if child[i] is None:
                        while parent2[pointer] in child:
                            pointer += 1
                        child[i] = parent2[pointer]
                children.append(child)
        return children
    
    # wrapping the swap and the crossover function in Mutate function - so perform mutation based on condition
    def mutate(self, cross=0.1, p_mut=0.1):
        next_full_pop = []
        children = self.crossover(cross)
        for child in children:
            if np.random.rand() < p_mut:
                swapped_child = self.swap(child)
                next_full_pop.append(swapped_child)
            else:
                next_full_pop.append(child)
        return next_full_pop
    
# this generating the first gen of population
def init_population(cities, adjacency_matrix, num_population):
        return Population(
        np.asarray([np.random.permutation(cities) for _ in range(num_population)]), 
        adjacency_matrix
    )
pop = init_population(cities, adjacency_matrix, 5)



# Main GA algo combine all the function to generate the optimal path
# Main focus - generate childern from mutation and then pass childern as full population of next generation in the Population class. [Line 189 and 190]   
def genetic_algorithm(
    cities,
    adjacency_matrix,
    num_population=5,
    num_iter=20,
    selectivity=0.15,
    cross=0.5,
    p_mut=0.1,
    print_interval=100,
    return_history=False,
    verbose=False,
):
    pop = init_population(cities, adjacency_matrix, num_population)
    best = pop.best
    score = float("inf")
    history = []
    for i in range(num_iter):
        pop.select(num_population * selectivity)
        history.append(pop.score)
        if verbose:
            print("Generation {}: {}".format(i,pop.score))
        elif i % print_interval == 0:
            print("Generation {}: {}".format(i,pop.score))
        if pop.score < score:
            best = pop.best
            score = pop.score
        children = pop.mutate(cross, p_mut)
        pop = Population(children, pop.adjacency_matrix)
    if return_history:
        return best, history
    return best
 

## calling the GA with num of population, num of iteration, crossover
best, history = genetic_algorithm(
    cities, adjacency_matrix, num_population=1000, selectivity=0.05,
    p_mut=0.05, cross=0.7, num_iter=9000, print_interval=500, verbose=False, return_history=True
)

for i in best:
    result.append(map_cities[i])
print(result)   # Printing the order of the cities visited