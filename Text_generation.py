#  !wget -nc https://raw.githubusercontent.com/lazyprogrammer/machine_learning_examples/master/hmm_class/robert_frost.txt
# !pip install transformers

from transformers import pipeline, set_seed
import textwrap
import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint

# !cat robert_frost.txt

lines = [line.rstrip() for line in open('robert_frost.txt')]
lines = [line for line in lines if len(line) > 0]

model = pipeline("text-generation")

set_seed(1234)
print(lines[0])

model(lines[0])

# This line could be used only in python interactive notebooks
pprint(_)
pprint(model(lines[0], max_length=20))
pprint(model(lines[0], num_return_sequences=3, max_length=20))

def wrap(x):
  return textwrap.fill(x, replace_whitespace=False, fix_sentence_endings=True)

