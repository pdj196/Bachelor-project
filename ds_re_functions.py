from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
from rdkit.Chem import rdmolops


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


from rdkit.DataStructs.cDataStructs import TanimotoSimilarity

import deepsmiles

converter = deepsmiles.Converter(rings=True, branches=True)


def kekulize_list(l):
  kl = []
  for smiles in l:
    mol = Chem.MolFromSmiles(smiles)
    kmol = Chem.Kekulize(mol,clearAromaticFlags=True)
    ksmiles = Chem.MolToSmiles(mol)
    kl.append(ksmiles)   
  return kl




def get_ECFP4(mol):
    return AllChem.GetMorganFingerprint(mol, 2)

def get_ECFP6(mol):
    return AllChem.GetMorganFingerprint(mol, 3)

def get_FCFP4(mol):
    return AllChem.GetMorganFingerprint(mol, 2, useFeatures=True)

def get_FCFP6(mol):
    return AllChem.GetMorganFingerprint(mol, 3, useFeatures=True)

def ttx_score(s, target):
  mol1 = Chem.MolFromSmiles(target)
  mol2 = Chem.MolFromSmiles(s)
  fp1 = get_ECFP4(mol1)
  fp2 = get_ECFP4(mol2)
  score = TanimotoSimilarity(fp1, fp2)
  return score


# Functions for initialization

def calculate_score(population, target):
  s_list = []
  for n in range(len(population)):
    s = ttx_score(population[n], target)
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
  return parent_number

def midpoint(parent):
  m = randint(0, len(parent) - 1)
  return m



def encoding(smiles):
  d = list(converter.encode(smiles))
  return d

def decoding(deep):
  try:
    s = ''.join(converter.decode(deep))
    return s
  except:
    return None


def make_crossover(pool): 
  parent_a = choose_parent(pool)
  parent_b = choose_parent(pool)
  deep_parent_a = encoding(parent_a)
  deep_parent_b = encoding(parent_b)
  midpoint_a = midpoint(deep_parent_a)
  midpoint_b = midpoint(deep_parent_b)
  a1 = deep_parent_a[0:midpoint_a]
  b2 = deep_parent_b[midpoint_b:len(deep_parent_b)]
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
  return mutated_child
# Separating the smiles from the non smiles                                

def smiles2mol(smiles):
  try:
    mol = Chem.MolFromSmiles(smiles)
    return mol
  except:
    return None

def try_ttx(smiles, target):
  try:
    ttx = ttx_score(smiles, target)
    return ttx
  except:
    return None

def count_non_smiles(pool, rate, symbols):
  count = 0
  child = make_new_child(pool, rate, symbols)
  mol = smiles2mol(child)
  if mol == None:
    count +=1
  return count


def is_mol_ok(m, target, size_stdev):
  mol = smiles2mol(m)
  average_size = len(target)
  try:
    Chem.SanitizeMol(mol)
    random_dev = abs(size_stdev*np.random.randn())
    max_size = average_size + random_dev
    min_size = average_size - random_dev
    if mol.GetNumAtoms() > min_size and mol.GetNumAtoms() < max_size:
      return True
    else:
      return False
  except:
    return False


def make_new_population(new_len_pop, pool, target, size_stdev, rate, symbols):
  new_population = []
  count = 0
  while len(new_population) < new_len_pop:
    new_child = make_new_child(pool, rate, symbols)
    d_child = decoding(new_child)
    if d_child != None:
      mol = smiles2mol(d_child)
      if mol != None:
        mol_ok = is_mol_ok(d_child, target, size_stdev)
        if mol_ok != False:
          ttx = try_ttx(d_child, target)
          if ttx != None: 
            new_population.append(d_child)
      else:
        count +=1
    else:
      count += 1
  new_population.append(count)
        
  return new_population 
