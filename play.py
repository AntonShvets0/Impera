from gpt2generator import GPT2Generator
from pathlib import Path

import random, os

f = open("Impera/data/seed.txt", "r")
texts = f.read().split("\n")
f.close()

print("Enter context:")
context = input()

print("Enter count text:")
count = int(input())

models = [ item for item in os.listdir("Impera/models") if os.path.isdir(os.path.join("Impera/models", item)) ]
generator = GPT2Generator("Impera/models/" + models[0], generate_num=60, temperature=0.6, top_k=40, top_p=0.9, repetition_penalty=1)

f = open("result.txt", "a")
print(count)
for i in range(count):
    seed = random.choice(texts)
    f.write(seed + generator.generate(seed, context) + "\n\n")

    break

print("Results saved in result.txt")