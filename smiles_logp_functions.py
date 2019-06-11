from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
from rdkit.Chem import rdmolops

import sascorer

#from io import StringIO
#import sys
#sio = sys.stderr = StringIO()

from rdkit import rdBase
rdBase.DisableLog('rdApp.error') 

import numpy as np
import random
from random import randint
import matplotlib.pyplot as plt
import time

def logP_score(s):
  m = Chem.MolFromSmiles(s)
  logp = Descriptors.MolLogP(m)
  SA_score = -sascorer.calculateScore(m)
  #cycle_list = nx.cycle_basis(nx.Graph(rdmolops.GetAdjacencyMatrix(m)))
  cycle_list = m.GetRingInfo().AtomRings() #remove networkx dependence
  #print cycle_list
  if len(cycle_list) == 0:
      cycle_length = 0
  else:
      cycle_length = max([ len(j) for j in cycle_list ])
  if cycle_length <= 6:
      cycle_length = 0
  else:
      cycle_length = cycle_length - 6
  cycle_score = -cycle_length
  #print cycle_score
  #print SA_score
  #print logp
  SA_score_norm=(SA_score-SA_mean)/SA_std
  logp_norm=(logp-logP_mean)/logP_std
  cycle_score_norm=(cycle_score-cycle_mean)/cycle_std
  score_one = SA_score_norm + logp_norm + cycle_score_norm
  
  return score_one

logP_values = np.loadtxt('logP_values.txt')
SA_scores = np.loadtxt('SA_scores.txt')
cycle_scores = np.loadtxt('cycle_scores.txt')
SA_mean =  np.mean(SA_scores)
SA_std=np.std(SA_scores)
logP_mean = np.mean(logP_values)
logP_std= np.std(logP_values)
cycle_mean = np.mean(cycle_scores)
cycle_std=np.std(cycle_scores)

def kekulize_list(l):
  kl = []
  for smiles in l:
    mol = Chem.MolFromSmiles(smiles)
    kmol = Chem.Kekulize(mol,clearAromaticFlags=True)
    ksmiles = Chem.MolToSmiles(mol)
    kl.append(ksmiles)   
  return kl

def gaussian(pool, a, sigma):
  g_list = []
  for s in pool:
    x = logP_score(s)
    g = np.exp(-0.5*((x-a)/sigma)**2)
    g_list.append(g)
  return g_list

# Functions for initialization

def calculate_score(population):
  s_list = []
  for n in range(len(population)):
    s = logP_score(population[n])
    if s < 0:
      s = 0
    s_list.append(s)
  return s_list


def calculate_normalized_fitness(scores):
  fitness = [float(i)/sum(scores) for i in scores]
  return fitness

def make_mating_pool(population, fitness, mp_size):
  p = fitness
  mating_pool = np.random.choice(population, mp_size, p=p)
  return mating_pool

def find_high_score(pool, a, sigma):
  s_list = gaussian(pool, a, sigma)
  high_score = max(s_list)
  return(high_score)

def find_high_score_smiles(pool, a, sigma):
  s_list = gaussian(pool, a, sigma)
  hs_smiles = (pool[s_list.index(max(s_list))])
  return hs_smiles

# Functions for reproduction:

# 1 - Functions for crossover

def choose_random_integers(pool):
  x = randint(0, len(pool) - 1)
  return x

def choose_parent(pool):
  number = choose_random_integers(pool)
  parent_number = pool[number]
  list_parent_number = list(parent_number)
  return list_parent_number

def midpoint(parent):
  m = randint(0, len(parent) - 1)
  return m

def make_crossover(pool): # Uden max_len bliver de tit virkelig store 
  parent_a = choose_parent(pool)
  parent_b = choose_parent(pool)
  midpoint_a = midpoint(parent_a)
  midpoint_b = midpoint(parent_b)
  a1 = parent_a[0:midpoint_a]
  b2 = parent_b[midpoint_b:len(parent_b)]
  c = a1+b2   
  return c

# 2 - Mutation

def make_mutation(child, rate, symbols):
  random_number = random.random()
  mutated_gene = randint(0, len(child) - 1)
  random_symbol_number = randint(0, len(symbols)-1)
  new_child = child
  if random_number < rate:
    new_child[mutated_gene] = symbols[random_symbol_number]
  return new_child

# Making the new population

def make_new_child(pool, rate, symbols):
  crossed_child = make_crossover(pool)
  mutated_child = make_mutation(crossed_child, rate, symbols)
  child_smiles = ''.join(mutated_child)
  return child_smiles

# Separating the smiles from the non smiles                                

def smiles2mol(smiles):
  try:
    mol = Chem.MolFromSmiles(smiles)
    return mol
  except:
    return None

def try_logP(smiles):
  try:
    logP = logP_score(smiles)
    return logP
  except:
    return None

def count_non_smiles(pool, rate, symbols):
  count = 0
  child = make_new_child(pool, rate, symbols)
  mol = smiles2mol(child)
  if mol == None:
    count +=1
  return count

def make_new_population(new_len_pop, pool, max_len_child, rate, symbols):
  new_population = []
  count = 0
  while len(new_population) < new_len_pop:
    new_child = make_new_child(pool, rate, symbols)
    mol = smiles2mol(new_child)
    if mol != None:
      logP = try_logP(new_child)
      if logP != None: 
        if len(new_child) < max_len_child:
          new_population.append(new_child)
    else:
      count += 1
  new_population.append(count)
        
  return new_population
