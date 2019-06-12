from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
from rdkit.Chem import rdmolops

import subprocess

from rdkit import rdBase
rdBase.DisableLog('rdApp.error')

from io import StringIO
import sys
#sio = sys.stderr = StringIO()

import numpy as np
import random
from random import randint
import matplotlib.pyplot as plt
import time

def shell(cmd, shell=False):

    if shell:
        p = subprocess.Popen(cmd, shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    else:
        cmd = cmd.split()
        p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    output, err = p.communicate()
    return output

def write_xtb_input_file(fragment, fragment_name):
    number_of_atoms = fragment.GetNumAtoms()
    charge = Chem.GetFormalCharge(fragment)
    symbols = [a.GetSymbol() for a in fragment.GetAtoms()] 
    for i,conf in enumerate(fragment.GetConformers()):
        file_name = fragment_name+"+"+str(i)+".xyz"
        with open(file_name, "w") as file:
            file.write(str(number_of_atoms)+"\n")
            file.write("title\n")
            for atom,symbol in enumerate(symbols):
                p = conf.GetAtomPosition(atom)
                line = " ".join((symbol,str(p.x),str(p.y),str(p.z),"\n"))
                file.write(line)
            if charge !=0:
                file.write("$set\n")
                file.write("chrg "+str(charge)+"\n")
                file.write("$end")



def compute_absorbance(mol):
  write_xtb_input_file(mol, 'test')
  shell('./xtb test+0.xyz',shell=False)
  out = shell('./stda_1.6 -xtb -e 10',shell=False)
  #data = str(out).split('Rv(corr)\\n')[1].split('alpha')[0].split('\\n') # this gets all the lines
  data_list = []
  wavelength_list = []
  osc_list = []
  for x in range(10):  
    data = str(out).split('Rv(corr)\\n')[1].split('\\n')[x]
    data_list.append(data)
    wavelength, osc_strength = float(data.split()[2]), float(data.split()[3])
    wavelength_list.append(wavelength)
    osc_list.append(osc_strength)
  
  return wavelength_list, osc_list


def wl_osc(smiles, confs):
  mol = Chem.MolFromSmiles(smiles)
  mol = Chem.AddHs(mol)
  new_mol = Chem.Mol(mol)

  AllChem.EmbedMultipleConfs(mol,numConfs=confs,useExpTorsionAnglePrefs=True,useBasicKnowledge=True)
  energies = AllChem.MMFFOptimizeMoleculeConfs(mol,maxIters=2000, nonBondedThresh=100.0)

  energies_list = [e[1] for e in energies]
  min_e_index = energies_list.index(min(energies_list))

  new_mol.AddConformer(mol.GetConformer(min_e_index))

  #prop = AllChem.MMFFGetMoleculeProperties(new_mol, mmffVariant="MMFF94")
  #ff = AllChem.MMFFGetMoleculeForceField(new_mol,prop)
  #AllChem.MMFFOptimizeMolecule(new_mol,maxIters=2000)
  #new_energy = ff.CalcEnergy()
  
  #print(Chem.MolToSmiles(new_mol))
  
  wavelength, osc_strength = compute_absorbance(new_mol)
  
  return wavelength, osc_strength


def try_wl_osc(smiles, confs):
  try:
    wl, osc = wl_osc(smiles, confs)
    return wl, osc
  except:
    return None, None

def wl_osc_scores(pool, confs):
  wl_scores = []
  osc_scores = []
  for smiles in pool:
    wl, osc = try_wl_osc(smiles, confs)
    if wl != None and osc != None:
      wl_scores.append(wl)
      osc_scores.append(osc)
    else:
      pool.remove(smiles)
  return pool, wl_scores, osc_scores

def tlm(scores, threshold):
  tlm_list = []
  for score in scores:
    tlm = np.minimum(score, threshold) / threshold
    tlm_list.append(tlm)
  return tlm_list

def gaussian(wl_list, a, sigma):
  g_list = []
  for x in wl_list:
    if abs(x-a) < 400:
      g = np.exp(-0.5*((x-a)/sigma)**2)
    else:
      g = 0
    g_list.append(g)
  return g_list

def total_scores(wl_list, osc_list, a, sigma):
  total_list = []
  gw = gaussian(wl_list, a, sigma)
  to = tlm(osc_list, 0.3)
  for x, y in zip(gw, to):
    s = x + y
    total_list.append(s)
  return total_list


def calculate_normalized_fitness(scores):
  fitness = [float(i)/sum(scores) for i in scores]
  return fitness

def make_mating_pool(population,fitness,mating_pool_size):
  mating_pool = []
  for i in range(mating_pool_size):
    mating_pool.append(np.random.choice(population, p=fitness))
  
  return mating_pool

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

def make_crossover(pool): 
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
      if len(new_child) < max_len_child:
        new_population.append(new_child)
    else:
      count += 1
      
  return new_population, count
