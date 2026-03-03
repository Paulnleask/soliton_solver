from soliton_solver import theories

for i, name in enumerate(theories.list(), start=1):
    print(f"{i}. {name}")
# print(theories.describe("Baby Skyrme model")) # if you pass keys by name you registered (see note below)