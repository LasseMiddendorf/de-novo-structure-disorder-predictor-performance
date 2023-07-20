#This code was used to run batch predictions with ESMfold on Google Colab (https://colab.research.google.com/github/sokrypton/ColabFold/blob/main/ESMFold.ipynb#scrollTo=hkMp_ZwRYfAQ)
# Batch size was set to 500 sequences per batch on a Tesla T4 GPU.

#@title ##run **ESMFold**
%%time
from string import ascii_uppercase, ascii_lowercase
import hashlib, re, os
import numpy as np
from jax.tree_util import tree_map
import matplotlib.pyplot as plt
from scipy.special import softmax
from Bio import SeqIO
from google.colab import files

def parse_output(output):
  pae = (output["aligned_confidence_probs"][0] * np.arange(64)).mean(-1) * 31
  plddt = output["plddt"][0,:,1]
  
  bins = np.append(0,np.linspace(2.3125,21.6875,63))
  sm_contacts = softmax(output["distogram_logits"],-1)[0]
  sm_contacts = sm_contacts[...,bins<8].sum(-1)
  xyz = output["positions"][-1,0,:,1]
  mask = output["atom37_atom_exists"][0,:,1] == 1
  o = {"pae":pae[mask,:][:,mask],
       "plddt":plddt[mask],
       "sm_contacts":sm_contacts[mask,:][:,mask],
       "xyz":xyz[mask]}
  return o

def get_hash(x): return hashlib.sha1(x.encode()).hexdigest()
alphabet_list = list(ascii_uppercase+ascii_lowercase)

for seq in SeqIO.parse(PathToFasta, "fasta"):
  try:
    jobname = seq.id
    jobname = re.sub(r'\W+', '', jobname)[:50]

    sequence = str(seq.seq)
    sequence = re.sub("[^A-Z:]", "", sequence.replace("/",":").upper())
    sequence = re.sub(":+",":",sequence)
    sequence = re.sub("^[:]+","",sequence)
    sequence = re.sub("[:]+$","",sequence)
    copies = 1
    if copies == "" or copies <= 0: copies = 1
    sequence = ":".join([sequence] * copies)
    num_recycles = 3
    chain_linker = 25 

    ID = jobname
    seqs = sequence.split(":")
    lengths = [len(s) for s in seqs]
    length = sum(lengths)
    print("length",length)

    u_seqs = list(set(seqs))
    if len(seqs) == 1: mode = "mono"
    elif len(u_seqs) == 1: mode = "homo"
    else: mode = "hetero"

    if "model" not in dir():
      import torch
      model = torch.load("esmfold.model")
      model.eval().cuda().requires_grad_(False)

    # optimized for Tesla T4
    if length > 700:
      model.set_chunk_size(64)
    else:
      model.set_chunk_size(128)

    torch.cuda.empty_cache()
    output = model.infer(sequence,
                        num_recycles=num_recycles,
                        chain_linker="X"*chain_linker,
                        residue_index_offset=512)

    pdb_str = model.output_to_pdb(output)[0]
    output = tree_map(lambda x: x.cpu().numpy(), output)
    ptm = output["ptm"][0]
    plddt = output["plddt"][0,...,1].mean()
    O = parse_output(output)
    print(f'ptm: {ptm:.3f} plddt: {plddt:.3f}')
    os.system(f"mkdir -p {ID}")
    prefix = f"{ID}/ptm{ptm:.3f}_r{num_recycles}_default"
    np.savetxt(f"{prefix}.pae.txt",O["pae"],"%.3f")
    with open(f"{prefix}.pdb","w") as out:
      out.write(pdb_str)

    
    os.system(f"zip {ID}.zip {ID}/*")
  except:
    pass
  
os.system("mkdir results")
os.system("mv *zip results/")

!zip results.zip results/*