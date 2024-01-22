# %%
"""
# No idea what to eat? Just sampling!

This python script allows you to sample a food choice
    with corresponding weights.
"""

# %%
from functools import reduce
from getpass import getpass
import json
import random

max_choices = 3
print("\n# Welcome to food sampling!\n")

# %%
"""
### Food options
"""

# %%
# Read food json file.
with open("food.json", "r") as json_file:
    food_json = json.load(json_file)
    food_list = []
    for food in food_json.keys():
        if "//" not in food:
            food_list.append(food)
    food_indices = range(len(food_list))

# Turn into dictionary with food indices.
option_dict = dict(enumerate(food_list))


def print_food_table():
    print("# Table of food options:")
    for index in food_indices:
        food = option_dict[index]
        print(f"* {index} -> {food}")


print_food_table()

# %%
"""
### Computing weights and sampling

1. Compute weights from voting result.
2. Sample `num_choices` options with weights.
"""


# %%
class NumChoiceError(Exception):

    def __init__(self, num_choices):
        '''Too many choices.'''
        error_message = (f"Over maximum number of choices -> "
                         f"{num_choices} > {max_choices}")
        super().__init__(error_message)


class WrongIndexError(Exception):

    def __init__(self, index):
        '''Index does not exist.'''
        error_message = f"Check your index -> {index}"
        super().__init__(error_message)


# %%
# Rules and usage for voting.
def print_info():
    print()
    print("# Start voting, e.g., 1 1 2 or 0 8 (seperate with spaces)")
    print("# Type '-1' to stop inputing.")
    print("# Type '-t' to show the food table again.")
    print("# Type '-r' to restart voting.")
    print("# Type '-z' to undo the last voting.")
    print("# Type '-72' to choose 72.")


print_info()


def clear_line():
    print('\033[1A\033[K', end='')


# Start voting.
weights = []
ielector = 1
print(" ========= Voting Start =========")
while True:
    prompt = "(Voter " + str(ielector) + ") Vote for your choice: "
    vote = input(prompt).lower()
    clear_line()

    if vote == "-1":
        print(" ========= Voting Done  =========")
        break

    elif vote == "-t":
        print_food_table()

    elif vote == "-r":
        weights = []
        print_info()

    elif vote == "-z":
        weights = weights[:-1]

    elif vote == "-72":
        weights = [[0, 0, 0]]
        break

    else:
        try:
            vote = vote.split()
            vote = list(map(int, vote))

            if len(vote) > max_choices:
                raise NumChoiceError(len(vote))

            for index in vote:
                if index not in food_indices:
                    raise WrongIndexError(index)

        except Exception as e:
            print(f"\n@@? {e}")

        else:
            weights.append(vote)
            print(' >> ' + str(ielector) + ' voted.')
            ielector += 1

print("Total Electors: " + str(ielector - 1))
# Calculating weights.
if len(weights) == 0:
    print("\n# No weights specified -> uniform weights.")
    weights = list(food_indices)
else:
    weights = reduce(lambda x, y: x + y, weights)
print(f"\n# Total weights = {weights}\n")
for index, food in option_dict.items():
    num_votes = weights.count(index)
    if num_votes > 0:
        print(f"* {food} has {num_votes} vote(s).")

# Samples.
print("\n # Sampling result:")
candidates = []
while len(candidates) < min(3, len(set(weights))):
    sample = option_dict[random.sample(weights, k=1)[0]]
    if sample not in candidates:
        candidates.append(sample)
        print(f"* Number {len(candidates)} food candidate is {sample}")

# %%
"""
### Check the functionality of `random.sample`

Only executes when using Jupyter Notebook, i.e., `ipynb` files.
"""

# %%
try:
    # Jupyter Notebook raises "NameError", plot only when using `ipynb`.
    __file__

except NameError:
    # Import plotting packages.
    import pandas as pd
    import seaborn as sns
    sns.set_theme()

    # Randomly sampling.
    samples = [random.sample(weights, k=1)[0] for _ in range(1000)]

    # Pandas Data Frame.
    df = pd.concat([
        pd.DataFrame({
            "Source": ["Weights"] * len(weights),
            "Value": weights
        }),
        pd.DataFrame({
            "Source": ["Samples"] * len(samples),
            "Value": samples
        }),
    ])

    # Histogram of "Weights" and "Samples".
    hist = sns.histplot(
        data=df,
        x="Value",  # 0, 1, 2, 3, ... -> food_indices
        hue="Source",  # Plot histograms respect to "Weights" or "Samples".
        bins=food_indices,  # Equivalent to number of food_indices.
        multiple="dodge",  # Seperate 2 histograms instead of overlapping them.
        stat="probability",  # Normalize histograms.
        common_norm=False,  # Normalize seperately for 2 histograms.
        shrink=0.8,  # Make bins not too close together.
    )
    hist.set(title="Comparing weights and samples")

else:
    # Skip plotting when only run with `py` scripts.
    pass
