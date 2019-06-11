from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
from rdkit.Chem import rdmolops

from rdkit.DataStructs.cDataStructs import TanimotoSimilarity

#from io import StringIO
#import sys
#sio = sys.stderr = StringIO()

from rdkit import rdBase
#rdBase.DisableLog('rdApp.error') 

import numpy as np
import random
from random import randint
import matplotlib.pyplot as plt
import time

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
  mol2 = s
  fp1 = get_ECFP4(mol1)
  fp2 = get_ECFP4(mol2)
  score = TanimotoSimilarity(fp1, fp2)
  return score


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


def try_ttx(mol, target):
  try:
    ttx = logP_score(mol, target)
    return ttx
  except:
    return None

