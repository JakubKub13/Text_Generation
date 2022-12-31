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

out = model(lines[0], max_length=30)
print(wrap(out[0]['generated_text']))

previous = "Two roads diverged in a yellow wood, including one that blocked the road leading to another intersection in the middle of the city"
out = model(previous + '\n' + lines[2], max_length=60)
print(wrap(out[0]['generated_text']))

# Exercise: think of ways you would apply GPT-2 and text generation in the real world and as a businnes / startup
prompt = "Neural networks with attention have been used with great success in natural language processing."
out = model(prompt, max_length=300)
print(wrap(out[0]['generated_text']))