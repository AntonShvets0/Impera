from gpt2generator import get_generator
import random, os

f = open("./data/seed.txt", "r")
texts = f.read().split("\n")
f.close()

print("Enter context:")
context = input()

print("Enter count text:")
count = int(input())

models = os.walk("./models/")
generator = get_generator(models[0], generate_num=60, temperature=0.6, top_k=40, top_p=0.9, repetition_penalty=1)

f = open("result.txt", "rw+")
for i in range(count):
    f.write(generator.generate(random.choice(texts), context) + "\n")

    break

print("Results saved in result.txt")