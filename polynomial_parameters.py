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


def fitness(individual, xcoord,ycoord):
    ygen = []
    for k in xcoord:
        ygen.append(np.polyval(individual,k))
        
    
   # Determine the fitness of an individual. Lower is better. 
   #individual: the individual to evaluate
   #target: the sum of numbers that individuals are aiming for
    ygen_arr = np.array(ygen)
    ycoord_arr = np.array(ycoord)
    ydiff = np.subtract(ygen_arr,ycoord_arr)
    percerr = np.divide(ydiff,ycoord_arr+0.000001)
    mse = np.mean(np.square(percerr)) #find mean square error 
    rmse = math.sqrt(mse)# find root mean squre error 

    return rmse


def grade(pop, xcoord,ycoord):
     #Find average fitness for a population.'
     summed = reduce(add, (fitness(x, xcoord,ycoord) for x in pop), 0)
     
     score = summed / (len(pop) * 1.0)

     if score < 0.001:
        
        print("solution found", individual, i)
        #exit()
       
     return score


# SELECTION FUNCTIONS

def eliteselect(pop, xcoord,ycoord, retain, random_select):
   
    #creates list with fitness of individual and individual for all individuals in population
    graded = [ (fitness(x, xcoord,ycoord), x) for x in pop]

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
    
    #generate y coordinates from fittest polynomial coefficients 
    yfit = []
    for k in xcoord:
        yfit.append(np.polyval(graded[0],k))

    #plot ycoordinates genrated by algorithm
    plt.plot(xcoord,yfit,alpha = 0.1, color = "blue")
    
    return parents

def eliteselect(pop, xcoord,ycoord, retain, random_select):
   
    #creates list with fitness of individual and individual for all individuals in population
    graded = [ (fitness(x, xcoord, ycoord), x) for x in pop]
    #creates sorted list of individuals from smallest difference between sum and 
    # target (highest fitness) to largest difference (lowest fitness) 
    graded = [ x[1] for x in sorted(graded)]
    ranklist = [(len(graded ) - graded.index(i) )for i in graded] #create list of ranks 
    sumrank = len(graded)+ 0.0001 #total number of ranks + little extra so proability isn't 1 for first individual
    rankprob = [(r/sumrank) for r in ranklist] #generate weighted selection probabilty for each invidual 
    retain_length = int(len(pop)*retain) #select amount of individuals to be slected with roulette wheel
    parents = choices(graded, weights = rankprob, k = retain_length) # generate set of fit parents 

    #randomly add other individuals to promote genetic diversity
    for individual in graded[retain_length:]:
        if random_select > random():
            parents.append(individual)

    return parents

   
def rouletteselect(pop, xcoord,ycoord, retain, random_select):

    graded = [ (fitness(x, xcoord, ycoord), x) for x in pop]
    fitlist = [x[0] for x in sorted (graded)] #produce ranked list of fitnesses
    graded = [ x[1] for x in sorted(graded)]
    sumfit = reduce(add,fitlist,0) #calculate sum of total fitnesses 
    fitprob = [1-(f/sumfit) for f in fitlist] #generate weighted selection probabilty for each invidual 
    retain_length = int(len(graded)*retain) #select amount of individuals to be slected with roulette wheel
    parents = choices(graded, weights = fitprob, k = retain_length) # generate set of fit parents 
   
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

    while  desired_length > len(children):
        if crossover > random():
            #select certain value within parents list and assign male or female 
            male = randint(0, parents_length-1)
            female = randint(0, parents_length-1)
            #create child using first half of male and second half of female individuals 
            if male != female:
                male = parents[male]
                female = parents[female]
                crosspoint = randint(0,len(male)-1)
                child1 = male[:crosspoint] + female[crosspoint:]
                children.append(child1) 
                if desired_length > len(children):
                    child2 = female[crosspoint:] + male[:crosspoint] 
                    children.append(child2)

        else: 
            #select certain indvidual within parents list to be cloned 
            asexual = randint(0, parents_length-1)
            clone = parents[asexual]   
            children.append(clone)

    parents.extend(children) #add children to parents 
    return parents

def two_point_crossover(parents,pop,crossover):
    #  crossover parents to create children
    parents_length = len(parents)
    desired_length = len(pop) - parents_length
    children = []

    while  desired_length > len(children):
        if crossover > random():
            #select certain value within parents list and assign male or female 
            male = randint(0, parents_length-1)
            female = randint(0, parents_length-1)
             #create child using first half of male and second half of female individuals 
            if male != female:
                male = parents[male]
                female = parents[female]
                crosspoint1 = randint(0,len(male)-3)
                crosspoint2 = randint(crosspoint1+1,len(male)-2)
                child1 = male[:crosspoint1] + female[crosspoint1:crosspoint2] + male[crosspoint2:]
                children.append(child1) 
                if desired_length > len(children):
                    child2 = female[:crosspoint1] + male[crosspoint1:crosspoint2] + female[crosspoint2:]
                    children.append(child2)

        else: 
            #select certain indvidual within parents list to be cloned 
            asexual = randint(0, parents_length-1)
            clone = parents[asexual]   
            children.append(clone)

    parents.extend(children) #add children to parents 
    return parents


def uniform_crossover(parents,pop,crossover):
    #  crossover parents to create children
    parents_length = len(parents)
    desired_length = len(pop) - parents_length
    children = []

    while  desired_length > len(children):
        if crossover > random():
            #select certain value within parents list and assign male or female 
            male = randint(0, parents_length-1)
            female = randint(0, parents_length-1)
            #create child using first half of male and second half of female individuals 
            if male != female:
                male = parents[male]
                female = parents[female]
                child1 = []
                child2 = []
                for i in range(len(male)):
                    if randint(0, 1):
                        child1.append(male[i])
                        child2.append(female[i])
                    else:
                        child1.append(female[i])
                        child2.append(male[i])
                children.append(child1)
                if desired_length > len(children):
                    children.append(child2)     
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

def addmutation(parents, mutate):
#mutate some individuals
#for each individual if the random number generator gives number 
# less than mutate value the a random number is added to a gene within 
#the individual
    for individual in parents:
        if mutate > random():
            pos_to_mutate = randint(0, len(individual)-1)  
            individual[pos_to_mutate] = individual[pos_to_mutate] + randint(i_min/4, i_max/4)
    return parents

def flipmutation(parents, mutate):
#mutate some individuals
#for each individual if the random number generator gives number 
# less than mutate value the sign of a gene within 
#the individual is flipped
    for individual in parents:
        if mutate > random():
            pos_to_mutate = randint(0, len(individual)-1)  
            individual[pos_to_mutate] = -individual[pos_to_mutate]
    return parents

def elitemutation(parents, mutate):
# mutate some individuals
#for each individual if the rnadom number generator gives number 
# less than mutate value the indivual is mutated with random number 
# between the min and max values of that individuals set
    elite = int(len(parents)*0.2) #prevents fittest 20% genes form being mutated
    for individual in parents[elite:]: 
        if mutate > random():
            pos_to_mutate = randint(0, len(individual)-1)  
            individual[pos_to_mutate] = randint(i_min, i_max)
    return parents


def combimutation(parents,mutate):
    for individual in parents:
        if mutate > random():
            pos_to_mutate = randint(0, len(individual)-1)  
            p = random()
            if p < 0.7:
                #carry out addtional mutation if random number is below certain value
                 individual[pos_to_mutate] = individual[pos_to_mutate] + randint(i_min/4, i_max/4)
            else: 
                #carry out flip mutation if random number is above certain value 
                individual[pos_to_mutate] = -individual[pos_to_mutate]
            
    return parents

def proportionalmutation(parents,mutate):
# mutate some individuals
#for each individual if the random number generator gives number 
# less than mutate value the indivual is mutated with random number 
# between the min and max values of that individuals set
    
    #define proportional mutations
    top_mutate = mutate 
    middle_mutate = mutate + 0.05
    bottom_mutate = mutate + 0.2

    graded = [ (fitness(x, xcoord,ycoord), x) for x in parents]
    graded = [ x[1] for x in sorted(graded)]

    top = int(len(parents)*0.1)
    bottom = int(len(parents)*0.6)
    #if indvidual is in top percentage of population
    #mutate with standard mutation value
    for individual in graded[:top]:
        if top_mutate > random():
            pos_to_mutate = randint(0, len(individual)-1)  
            individual[pos_to_mutate] = randint(i_min, i_max)

    #if indvidual is inbetween top and bottom percentage of population
    #mutate with slightly bigger mutation value
    for individual in graded[top:bottom]:
     if middle_mutate > random():
            pos_to_mutate = randint(0, len(individual)-1)  
            individual[pos_to_mutate] = randint(i_min, i_max)

    #if indvidual is in bottom percentage of population
    #mutate with substantially bigger mutation value
    for individual in graded[bottom:]:
     if bottom_mutate > random():
            pos_to_mutate = randint(0, len(individual)-1)  
            individual[pos_to_mutate] = randint(i_min, i_max)
    
    return graded

#check if solution has been found
def findsol(pop, xcoord,ycoord):

    found = 0 
    for b in pop:
            a = fitness(b,xcoord,ycoord)
            if a == 0:
                found = 1
    return found

# Algorithm parameters

target = [25,18,31,-14,7,-19]
p_count = 300
i_length = 6
i_min = -100
i_max = 100
generations = 1000
xcoord = [] 
ycoord = [] 
gen_count = 0
retain =0.1
random_select=0.05
mutate=0.2
crossover = 0.9
results = []


for n in range(-50,50):
    xcoord.append(n)

for h in xcoord:
    ycoord.append(np.polyval(target,h))

p = population(p_count, i_length, i_min, i_max)
fitness_history = [grade(p, xcoord,ycoord),]

for i in range(generations):
    gen_count = i
    print("generation: ",gen_count)
    print("p[0]",p[0])
    #check if solution has been found and break out of generation loop
    f = findsol(p,xcoord,ycoord)
    if f == 1:
        print("solution found", gen_count+1)
        results.append(gen_count+1)
        break

    parents = eliteselect(p, xcoord,ycoord, retain, random_select)
    parents = uniform_crossover(parents,p,crossover)
    p = addmutation(parents, mutate)
    g = grade(p, xcoord, ycoord)
    fitness_history.append(g)
    print(gen_count+1)   

#plot target curve
plt.plot(xcoord,ycoord,color = 'red',alpha =1)

plt.title("Curve Fit Algorithm Performance" )
plt.show()


print(results)

