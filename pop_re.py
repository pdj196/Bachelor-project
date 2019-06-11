from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
from rdkit.Chem import rdmolops
from rdkit.DataStructs.cDataStructs import TanimotoSimilarity
import numpy as np

smiles_list = []

f = open('1000.smi', 'r')
for line in f:
  new_line = line.replace('\n', '')
  smiles_list.append(new_line)


# GuacaMol article https://arxiv.org/abs/1811.09621
# adapted from https://github.com/BenevolentAI/guacamol/blob/master/guacamol/utils/fingerprints.py

def get_ECFP4(mol):
    return AllChem.GetMorganFingerprint(mol, 2)

def get_ECFP6(mol):
    return AllChem.GetMorganFingerprint(mol, 3)

def get_FCFP4(mol):
    return AllChem.GetMorganFingerprint(mol, 2, useFeatures=True)

def get_FCFP6(mol):
    return AllChem.GetMorganFingerprint(mol, 3, useFeatures=True)

def score(s, target):
  mol1 = Chem.MolFromSmiles(target)
  mol2 = Chem.MolFromSmiles(s)
  fp1 = get_FCFP6(mol1)
  fp2 = get_FCFP6(mol2)
  score = TanimotoSimilarity(fp1, fp2)
  return score

def get_best(target, size):
  scores = []
  for s in smiles_list:
    t = score(s, target)
    if t > 0.323:
      smiles_list.remove(s)
    else:
      scores.append(t)
  scores_tuples = list(zip(scores, smiles_list))
  scores_tuples = sorted(scores_tuples, key=lambda x: x[0], reverse=True)[:size]
  best = [t[1] for t in scores_tuples]
  best_scores = [t[0] for t in scores_tuples]
  a = min(best_scores)
  b = np.mean(best_scores)
  c = max(best_scores)
  return best, a, b, c
