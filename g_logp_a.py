from rdkit import Chem
from rdkit.Chem import AllChem

from rdkit.Chem import Descriptors

import sascorer

#from io import StringIO
#import sys
#sio = sys.stderr = StringIO()

import numpy as np
import random
from random import randint
import matplotlib.pyplot as plt
import time

# Importing functions for SMILES, logP

import crossover as co
import mutate as mu
import scoring_functions_logp_a as sf
import GB_GA as ga
import init_pop_removal as ipr

# 1000 SMILES strings file

smiles_list = []

f = open('1000.smi', 'r')
for line in f:
  new_line = line.replace('\n', '')
  smiles_list.append(new_line)

def make_population(smiles_list):
  mol_list = []
  for smiles in smiles_list:
    mol = Chem.MolFromSmiles(smiles)
    mol_list.append(mol)
  return mol_list


target = [-1, 8]

# All my variables:

generations = 100
sigma = 2
population_size = 100
mating_pool_size = 20
mutation_rate = 0.1
runs = 10
co.average_size = 81
co.size_stdev = 10

# Genetic algorithm


print('genrations =', generations)
print('population size =', population_size)
print('mating pool size =', mating_pool_size)
print('mutation rate =', mutation_rate)
print('molecule size =', co.average_size, '+/-', co.size_stdev)
print('gaussian sigma =', sigma)

for n in range(len(target)):
  print('target =', target[n])

  all_hs_list = []
  total_counts_list = []
  success_rate_list = []

  no_mol_made = generations*population_size

  start_time = time.time()

  for run in range (runs):

    high_scores_list = []
    high_logp_scores_list = []
    high_scores_smiles_list = []

    count = 0
    count_list = []

    population = make_population(np.random.choice(ipr.remove_smiles(smiles_list, -4, 2), population_size))
    logp_scores = sf.calculate_score(population)
    scores = sf.gaussian(population, target[n], sigma) 
    fitness = sf.calculate_normalized_fitness(scores)

    for generation in range(generations):
      mating_pool = ga.make_mating_pool(population, fitness, mating_pool_size)  
      new_population, count = ga.reproduce(mating_pool,population_size,mutation_rate)
      count_list.append(count)
      new_logp_scores = sf.calculate_score(new_population)
      new_scores = sf.gaussian(new_population, target[n], sigma)
      population_tuples = list(zip(scores+new_scores, logp_scores+new_logp_scores, population+new_population))
      population_tuples = sorted(population_tuples, key=lambda x: x[0], reverse=True)[:population_size]
      population = [t[2] for t in population_tuples]
      logp_scores = [t[1] for t in population_tuples]
      scores = [t[0] for t in population_tuples]
      fitness = sf.calculate_normalized_fitness(scores)



      high_scores_list.append(scores[0])
      high_logp_scores_list.append(logp_scores[0])
      high_scores_smiles_list.append(population[0])

      if abs(target[n] - logp_scores[0]) < 0.01:
        break
    
    
    hs_tuples = list(zip(high_scores_list,high_logp_scores_list,high_scores_smiles_list))
    hs_tuples = sorted(hs_tuples, key=lambda x: x[0], reverse=True)
    hssl = [t[2] for t in hs_tuples]
    hlsl = [t[1] for t in hs_tuples]
    hsl = [t[0] for t in hs_tuples]

  
    success_rate = (no_mol_made * 100)/(no_mol_made + sum(count_list))
  
    all_hs_list.append(hsl[0])
    total_counts_list.append(sum(count_list))
    success_rate_list.append(success_rate)
  
    print ('generation =', generation, 'smiles =', Chem.MolToSmiles(hssl[0]),  'score =', hsl[0], 'logp =', hlsl[0])
  
  stop_time = time.time()

  # Getting results

  avg_score = np.mean(all_hs_list)
  std_score = np.std(all_hs_list)

  avg_errors = np.mean(total_counts_list)
  std_errors = np.std(total_counts_list)

  avg_success = np.mean(success_rate_list)
  std_success = np.std(success_rate_list)

  run_time = (stop_time - start_time)/10

  print(avg_score, std_score)
  print(no_mol_made)
  print(avg_errors, std_errors)
  print(avg_success, std_success)
  print(run_time)
