import smiles_logp_a as spa


def remove_smiles(smiles_list, low, high):
  for s in smiles_list:
    l = spa.logP_score(s)
    if low<l<high:
      smiles_list.remove(s)

  return smiles_list
