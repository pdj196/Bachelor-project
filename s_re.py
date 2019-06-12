

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
from rdkit.Chem import rdmolops 

from io import StringIO
import sys
#sio = sys.stderr = StringIO()

from rdkit import rdBase
#rdBase.DisableLog('rdApp.error')

import numpy as np
import random
from random import randint
import matplotlib.pyplot as plt
import time

# Importing functions for SMILES, TTX

import smiles_re_functions as sx
import pop_re as pr


mutation_symbols = ['C', 'N', '(', ')', 'S', '=', 'O', '1', '2', 'B', 'Br', '3', '[C@@H]', '[C@H]', 'H', '+', 'Cl', 'F', '4', '#', 'I', '-', '/', '\\', '5', '[N+]', '[O-]']


def kekulize_list(l):
  kl = []
  for smiles in l:
    mol = Chem.MolFromSmiles(smiles)
    Chem.Kekulize(mol)
    ks = Chem.MolToSmiles(mol,kekuleSmiles=True)
    kl.append(ks) 
  return kl

target_0 = ['c1ccc2ccccc2c1', 'O=S(=O)(N(C)C)c2cc1C(\c3c(Sc1cc2)cccc3)=C/CCN4CCN(C)CC4', 'O=S(=O)(c3ccc(n1nc(cc1c2ccc(cc2)C)C(F)(F)F)cc3)N']
target = kekulize_list(target_0)

# All my variables:

runs = 10
generations = 1000
population_size = 100
mating_pool_size = 200
size_stdev = 10
mutation_rate = 0.5
symbols = mutation_symbols

# Genetic algorithm

for n in range(len(target)):
  
  print('target =', target[n])

  all_hs_list = []
  total_counts_list = []
  success_rate_list = []

  no_mol_made = generations*population_size

  start_time = time.time()

  for run in range (runs):

    high_scores_list = []
    high_scores_smiles_list = []

    count = 0
    count_list = []

    best, a, b, c = pr.get_best(target[n], population_size)
    population = kekulize_list(best)
    scores = sx.calculate_score(population, target[n])
    fitness = sx.calculate_normalized_fitness(scores)
    for generation in range(generations):
     
      mating_pool = sx.make_mating_pool(population, fitness, mating_pool_size)
      new_population = sx.make_new_population(population_size, mating_pool, target[n], size_stdev, mutation_rate, symbols)

      count = new_population[len(new_population) - 1]
      count_list.append(count)

      new_population = new_population [0:len(new_population) - 2]
      new_scores = sx.calculate_score(new_population, target[n])
      population_tuples = list(zip(scores+new_scores,population+new_population))
      population_tuples = sorted(population_tuples, key=lambda x: x[0], reverse=True)[:population_size]
      population = [t[1] for t in population_tuples]
      scores = [t[0] for t in population_tuples] 
      fitness = sx.calculate_normalized_fitness(scores)


      high_scores_list.append(scores[0])
      high_scores_smiles_list.append(population[0])

      if scores[0] == 1:
        break
 

    hs_tuples = list(zip(high_scores_list,high_scores_smiles_list))
    hs_tuples = sorted(hs_tuples, key=lambda x: x[0], reverse=True)
    hssl = [t[1] for t in hs_tuples]
    hsl = [t[0] for t in hs_tuples]

    success_rate = (no_mol_made * 100)/(no_mol_made + sum(count_list))

    all_hs_list.append(hsl[0])
    total_counts_list.append(sum(count_list))
    success_rate_list.append(success_rate)

    print ('generation =', generation, 'smiles =', hssl[0], 'score =', hsl[0])


  stop_time = time.time()

  avg_score = np.mean(all_hs_list)
  std_score = np.std(all_hs_list)

  avg_errors = np.mean(total_counts_list)
  std_errors = np.std(total_counts_list)

  avg_success = np.mean(success_rate_list)
  std_success = np.std(success_rate_list)

  run_time = (stop_time - start_time)/runs

  print('average score =', avg_score, '+/-', std_score)
  print('average success rate =', avg_success, '+/-', std_success)
  print('molecules evaluated =', no_mol_made+population_size)
  print('CPU time', run_time)
  print('minimum initial =', a, ', mean initial =', b, ', maximum initial =', c)
  print(' ')

