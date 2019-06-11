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

def logP_score(m):
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


def try_logP(mol):
  try:
    logP = logP_score(mol)
    return logP
  except:
    return None

