from eval_individual import *
from output_functions import write_to_excel

# Test individual at different gravities and takes the average fitness
# of 5 and writes it to an excel sheet
def fullObsTestGravity(ind, pset, path):
    gravity = [1, 2, 3, 4, 5, 6, 7, 8, 9.81, 11, 12, 13, 14, 15, 16, 17]
    for i in gravity:
        add_to_excel = []
        total = 0
        add_to_excel.append(i)

        for j in range(5):
            fit = fullObsEvalIndividual(ind, pset, i, test=False)[0]
            total += fit

        add_to_excel.append(round(total/5, 2))
        write_to_excel(add_to_excel, path)

def partObsTestGravity(ind, pset, path):
    gravity = [1, 2, 3, 4, 5, 6, 7, 8, 9.81, 11, 12, 13, 14, 15, 16, 17]
    for i in gravity:
        add_to_excel = []
        total = 0
        add_to_excel.append(i)

        for j in range(5):
            fit = partObsEvalIndividual(ind, pset, i, test=False)[0]
            total += fit

        add_to_excel.append(round(total/5, 2))
        write_to_excel(add_to_excel, path)


