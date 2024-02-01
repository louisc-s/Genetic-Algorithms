# Genetic Algorithms
Code to implement genetic algorithms for polynomial parameter optimisation

## Overview

This project focuses on implementing two genetic algorithms for two different 
tasks. The first algorithm  is used to find a target number within a set range 
of values and the second is used to optimise a set of parameters for the curve 
of a 5th order polynomial. 

Genetic algorithms optimise a population of candidate solutions to evolve them 
towards the solution of a given problem. Candidate solutions, also known as individuals, 
are made up of genes which each represent one of the problem’s parameters. The evolution 
process consists of applying the following methods to the population at every generation: 
crossover, mutation and selection. The selection technique picks the parents, mutation 
randomly changes a parent’s genes and crossover produces a child by combining two parents’ 
genetic information. Individuals of a given generation who survive onto the next one are 
called parents and are able to have children by reporducing with other survivors.

## Project Structure 

1. find_target.py - implements first genetic algortihm to find target number within a given range
2. polynomial_parameters.py - implements second genetic algortihm to optimise a set of parameters for polynomial curve

## Author 

Louis Chapo-Saunders
