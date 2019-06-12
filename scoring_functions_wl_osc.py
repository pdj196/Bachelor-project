from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
from rdkit.Chem import rdmolops

from collections import defaultdict
from rdkit.Chem import rdFMCS

import subprocess


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
  for x in range(1):  
    data = str(out).split('Rv(corr)\\n')[1].split('\\n')[x]
    data_list.append(data)
    wavelength, osc_strength = float(data.split()[2]), float(data.split()[3])
    wavelength_list.append(wavelength)
    osc_list.append(osc_strength)
  
  return wavelength_list, osc_list

def wl_osc(mol, confs):

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

def try_wl_osc(mol, confs):
  try:
    wl, osc = wl_osc(mol, confs)
    return wl, osc
  except:
    return None, None

def wl_osc_scores(pool, confs):
  wl_scores = []
  osc_scores = []
  for mol in pool:
    wl, osc = try_wl_osc(mol, confs)
    if wl != None and osc != None:
      wl_scores.append(wl)
      osc_scores.append(osc)
    else:
      pool.remove(mol)
  return pool, wl_scores, osc_scores

#def tlm(scores, threshold):
 # tlm_list = []
 # for score in scores:
 #   tlm = np.minimum(score, threshold) / threshold
 #   tlm_list.append(tlm)
 # return tlm_list


def tlm(scores, lower, upper):
  tlm_list = []
  for score in scores:
    if score <= lower:
      mod = 0
    if score >= upper:
      mod = 1
    if lower < score < upper:
      slope = 1/(upper - lower)
      mod = slope*(score - lower)
    tlm_list.append(mod)
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
  to = tlm(osc_list, 0.2, 0.8)
  for x, y in zip(gw, to):
    s = x + y
    total_list.append(s)
  return total_list
