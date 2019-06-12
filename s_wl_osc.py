import os
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
from rdkit.Chem import rdmolops

from io import StringIO
import sys
#sio = sys.stderr = StringIO()

import numpy as np
import random
from random import randint
import matplotlib.pyplot as plt
import time

import smiles_wl_functions_confs as sl
import population_generator_20 as pg

#os.environ['OMP_NUM_THREADS'] = str(1)
#os.environ['MKL_NUM_THREADS'] = str(1)

mutation_symbols = ['C', 'O', '(', '=', ')', '[C@@H]', '[C@H]', 'H', '1', 'N', '2', '3', 'F', 'S', 'Cl', '#', '+', '-', '/', '4', 'B', 'Br', '\\', '5', 'I']

target = [400, 800]

# All my variables:

runs = 10
generations = 1000
sigma = 50
population_elimination = 100
confs = 20
population_size = 20
mating_pool_size = 40
max_len_child = 40
mutation_rate = 0.15
symbols = mutation_symbols

print('generations =', generations)
print('sigma =', sigma)
print('population elimination =', population_elimination)
print('configurations =', confs)
print('population size =', population_size)
print('mating pool size =', mating_pool_size)
print('max length of child =', max_len_child)
print('mutation rate =', mutation_rate)

# Genetic algorithm

for n in range(len(target)):
  print(' ')
  print('target =', target[n])

  all_hs_list = []
  total_counts_list = []
  success_rate_list = []
  generations_list = []

  no_mol_made = generations*population_size

  start_time = time.time()

  for run in range (runs):

    high_scores_list = []
    high_wl_scores_list = []
    high_osc_scores_list = []
    high_scores_smiles_list = []

    count = 0
    count_list = []

    smiles_list, wl_list = pg.make_good_pop(pg.wld20, target[n], population_elimination, population_size)
    
    population, wl_scores, osc_scores = sl.wl_osc_scores(smiles_list, confs)
    wl_scores_0 = [wl[0] for wl in wl_scores]
    osc_scores_0 = [osc[0] for osc in osc_scores]
    scores = sl.total_scores(wl_scores_0, osc_scores_0, target[n], sigma)
    fitness = sl.calculate_normalized_fitness(scores)
    for generation in range(generations):

      mating_pool = sl.make_mating_pool(population,fitness,mating_pool_size)
      new_population_draft, count = sl.make_new_population(population_size,mating_pool,max_len_child,mutation_rate,symbols)
      count_list.append(count)
      new_population, new_wl_scores, new_osc_scores = sl.wl_osc_scores(new_population_draft, confs) 
      new_wl_scores_0 = [wl[0] for wl in new_wl_scores]
      new_osc_scores_0 = [osc[0] for osc in new_osc_scores]
      new_scores = sl.total_scores(new_wl_scores_0, new_osc_scores_0, target[n], sigma)
      population_tuples = list(zip(scores+new_scores,wl_scores_0+new_wl_scores_0,osc_scores_0+new_osc_scores_0,population+new_population))
      population_tuples = sorted(population_tuples, key=lambda x: x[0], reverse=True)[:population_size]
      population = [t[3] for t in population_tuples]
      osc_scores_0 = [t[2] for t in population_tuples]
      wl_scores_0 = [t[1] for t in population_tuples]
      scores = [t[0] for t in population_tuples]  
      fitness = sl.calculate_normalized_fitness(scores)

      high_scores_list.append(scores[0])
      high_wl_scores_list.append(wl_scores_0[0])
      high_osc_scores_list.append(osc_scores_0[0])
      high_scores_smiles_list.append(population[0])

      if scores[0] > 1.99:
        break
    
    hs_tuples = list(zip(high_scores_list,high_wl_scores_list,high_osc_scores_list,high_scores_smiles_list))
    hs_tuples = sorted(hs_tuples, key=lambda x: x[0], reverse=True)
    hssl = [t[3] for t in hs_tuples]
    hosl = [t[2] for t in hs_tuples]
    hwsl = [t[1] for t in hs_tuples]
    hsl = [t[0] for t in hs_tuples] 

    success_rate = (no_mol_made * 100)/(no_mol_made + sum(count_list))

    all_hs_list.append(hsl[0])
    total_counts_list.append(sum(count_list))
    success_rate_list.append(success_rate)
    generations_list.append(generation)

    print ('generation =', generation, 'smiles =', hssl[0], 'gaussian score =', hsl[0], 'wavelength =', hwsl[0], 'osc =', hosl[0])


  stop_time = time.time()

  avg_gen = np.mean(generations_list)
  std_gen = np.std(generations_list)

  avg_score = np.mean(all_hs_list)
  std_score = np.std(all_hs_list)

  avg_errors = np.mean(total_counts_list)
  std_errors = np.std(total_counts_list)

  avg_success = np.mean(success_rate_list)
  std_success = np.std(success_rate_list)

  run_time = (stop_time - start_time)/runs


  print(avg_gen, std_gen)
  print('average score =', avg_score, '+/-', std_score)
  print('average success rate =', avg_success, '+/-', std_success)
  print('molecules evaluated =', no_mol_made+len(smiles_list))
  print('CPU time', run_time)
