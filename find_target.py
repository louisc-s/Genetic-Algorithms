from random import randint , random, choices
from operator import add
import matplotlib.pyplot as plt
from functools import reduce 
import numpy as np
import math 



#create a memeber of the population
def individual (length , min , max) :
    return [randint(min, max) for x in range(length)]


def population(count, length, min, max):
#   Create a number of individuals (i.e. a population).
#   count: the number of individuals in the population
#   length: the number of values per individual
#   min: the min possible value in an individual's list of values
#   max: the max possible value in an individual's list of values
  

    return [ individual(length,min,max) for x in range(count)]

def fitness(individual, target):
    
    #Determine the fitness of an individual. Lower is better. 
    #individual: the individual to evaluate
    #target: the sum of numbers that individuals are aiming for
   
    diff = abs((target - individual[0])/target)
    return diff

def grade(pop, target):
     #Find average fitness for a population.'
     summed = reduce(add, (fitness(x, target) for x in pop), 0)
     return summed / (len(pop) * 1.0)

# SELECTION FUNCTIONS

def rankselect(pop, target, retain, random_select):
   
    #creates list with fitness of individual and individual for all individuals in population
    graded = [ (fitness(x, target), x) for x in pop]

    #creates sorted list of individuals from smallest difference between sum and 
    # target (highest fitness) to largest difference (lowest fitness) 
    graded = [ x[1] for x in sorted(graded)]
    #changes length of graded list
    retain_length = int(len(graded)*retain)
    #keeps highest perfroming indviduals within a certain proprotion of list to use as parents 
    parents = graded[:retain_length]

    #randomly add other individuals to promote genetic diversity
    for individual in graded[retain_length:]:
        if random_select > random():
            parents.append(individual)

    return parents

def rouletteselect(pop, target, retain, random_select):

    graded = [ (fitness(x, target), x) for x in pop]
    fitlist = [x[0] for x in graded] #produce ranked list of fitnesses
    graded = [ x[1] for x in sorted(graded)]
    sumfit = reduce(add,fitlist,0) #calculate sum of total fitnesses 
    fitprob = [1-(f/sumfit) for f in fitlist] #generate weighted selection probabilty for each invidual 
    retain_length = int(len(graded)*retain) #select amount of individuals to be slected with roulette wheel
    parents = choices(pop, weights = fitprob, k = retain_length) # generate set of fit parents 
   
    #randomly add other individuals to promote genetic diversity
    for individual in graded[retain_length:]:
        if random_select > random():
            parents.append(individual)

    return parents

# CROSSOVER FUNCTIONS

def one_point_crossover(parents,pop,crossover):  
#  crossover parents to create children
    parents_length = len(parents)
    desired_length = len(pop) - parents_length
    children = []
    while len(children) < desired_length:

        if crossover > random():
            #select certain value within parents list and assign male or female 
            male = randint(0, parents_length-1)
            female = randint(0, parents_length-1)
            #create child using first half of male and second half of female individuals 
            if male != female:
                male = (parents[male])
                mbin = bin(male[0])
                female = (parents[female])
                fbin = bin(female[0])
                bmale = mbin[2:]
                bfemale = fbin[2:]
                if len(bmale) > len(bfemale):
                    bfemale = bfemale.zfill(len(bmale))
                if len(bfemale) > len(bmale):
                    bmale = bmale.zfill(len(bfemale))
                crosspoint = randint(0,len(bmale)-1)
                child1 = []
                child1.append(int((bmale[:crosspoint] + bfemale[crosspoint:]),2)) #create child and convert to decimal
                children.append(child1) 
                child1.clear
                if len(children) < desired_length: #prevent too many children being added 
                    child2 = []
                    child2.append(int((bfemale[:crosspoint] + bmale[crosspoint:]),2)) #create child and convert to decimal
                    children.append(child2)
                    child2.clear

                
        else: 
            #select certain indvidual within parents list to be cloned 
            asexual = randint(0, parents_length-1)
            clone = parents[asexual]
            children.append(clone)    

    parents.extend(children) #add children to parents 
    return parents

# MUTATION FUNCTIONS

def mutation(parents, mutate):
# mutate some individuals
#for each individual if the rnadom number generator gives number 
# less than mutate value the indivual is mutated with random number 
# between the min and max values of that individuals set
    
    for individual in parents:
        if mutate > random():
            pos_to_mutate = randint(0, len(individual)-1)  
            individual[pos_to_mutate] = randint(i_min, i_max)
    return parents

def elitemutation(parents, mutate):
# mutate some individuals
#for each individual if the rnadom number generator gives number 
# less than mutate value the indivual is mutated with random number 
# between the min and max values of that individuals set
    elite = int(len(parents)*0.2) #prevents fittest 15% genes form being mutated
    for individual in parents[elite:]: 
        if mutate > random():
            pos_to_mutate = randint(0, len(individual)-1)  
            individual[pos_to_mutate] = randint(i_min, i_max)
    return parents

def proportionalmutation(parents):

    top_mutate = 0.01
    middle_mutate = 0.02
    bottom_mutate = 0.03

    graded = [ (fitness(x, target), x) for x in parents]
    graded = [ x[1] for x in sorted(graded)]

    top = int(len(parents)*0.1)
    bottom = int(len(parents)*0.6)

    for individual in graded[:top]:
        if top_mutate > random():
            pos_to_mutate = randint(0, len(individual)-1)  
            individual[pos_to_mutate] = randint(i_min, i_max)

    for individual in graded[top:bottom]:
     if middle_mutate > random():
            pos_to_mutate = randint(0, len(individual)-1)  
            individual[pos_to_mutate] = randint(i_min, i_max)

    for individual in graded[bottom:]:
     if bottom_mutate > random():
            pos_to_mutate = randint(0, len(individual)-1)  
            individual[pos_to_mutate] = randint(i_min, i_max)
    
    return graded
    


# Example usage

target = 78
p_count = 100
i_length = 1 
i_min = 0
i_max = 100
generations = 300
gen_count = 0 
mutate = 0.01
retain = 0.2
random_select=0.05
crossover = 0.9

p = population(p_count, i_length, i_min, i_max)
fitness_history = [grade(p, target),]

for i in range(generations):
    gen_count = i
    parents = rankselect(p, target, retain, random_select)
    parents = one_point_crossover(parents,p,crossover)
    p = mutation(parents, mutate)
    g = grade(p, target)
    fitness_history.append(g)

    if g == 0:
        print("converged", gen_count+1)
        break
    

for datum in fitness_history:
    print(datum)

plt.plot(fitness_history)
plt.show()

