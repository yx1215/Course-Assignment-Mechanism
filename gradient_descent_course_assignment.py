from scipy.optimize import minimize, LinearConstraint, Bounds
from math import log
import numpy as np
import random

random.seed(0)

STUDENTS = 20
CLASSES = 20
NUMBER_OF_P = STUDENTS * CLASSES
MAX_CLASS_PER_STUDENT = 4
MAX_STUDENT_EACH_CLASS = [10 for i in range(CLASSES)]
VALUES = np.array([random.randint(1, 10) for i in range(NUMBER_OF_P)])
p0 = np.array([0.5 for i in range(NUMBER_OF_P)])


def utility_func(p):
    u = []
    for i in range(STUDENTS):
        current = 0
        for j in range(CLASSES):
            current += p[i * CLASSES + j] * VALUES[i * CLASSES + j]
        u.append(log(current))
    return -sum(u)


# gradient
def utility_func_der(p):
    assert isinstance(p, np.ndarray)

    der = np.zeros_like(p)
    for i in range(NUMBER_OF_P):
        student_num = i // CLASSES
        der[i] = -VALUES[i] / p[student_num * CLASSES: (student_num + 1) * CLASSES].dot(VALUES[student_num * CLASSES: (student_num + 1) * CLASSES])
    return der


# hessian
def utility_func_hess(p):
    zeros = np.zeros_like(p)
    H = np.diag(zeros)
    for i in range(NUMBER_OF_P):
        student_num = i // CLASSES
        u = p[student_num * CLASSES: (student_num + 1) * CLASSES].dot(VALUES[student_num * CLASSES: (student_num + 1) * CLASSES])
        for j in range(NUMBER_OF_P):
            if j // CLASSES == student_num:
                H[i][j] = VALUES[i] * VALUES[j] / u ** 2
            else:
                H[i][j] = 0

    return H


# probability should not exceed 1
bounds = Bounds([0 for i in range(NUMBER_OF_P)], [1 for i in range(NUMBER_OF_P)])

# sum of probability for each student should not exceed 4
student_constraints_coef_matrix = [[1 if i // CLASSES == j else 0 for i in range(NUMBER_OF_P)] for j in
                                   range(CLASSES)]
student_max = [MAX_CLASS_PER_STUDENT for i in range(STUDENTS)]
student_min = [0 for i in range(STUDENTS)]
# print(student_constraints_coef_matrix)

# sum of probability for each class should not exceed Ci
class_constraints_coef_matrix = [[1 if i % STUDENTS == j else 0 for i in range(NUMBER_OF_P)] for j in range(STUDENTS)]
class_max = MAX_STUDENT_EACH_CLASS
class_min = [0 for i in range(CLASSES)]
# print(class_constraints_coef_matrix)

# total constraints
general_constraints_coef_matrix = student_constraints_coef_matrix + class_constraints_coef_matrix
general_max = student_max + class_max
general_min = student_min + class_min
# print(general_constraints_coef_matrix)
# print(general_max)
# setup linear constraints
linear_constraints = LinearConstraint(general_constraints_coef_matrix, general_min, general_max)

# minimize
res = minimize(utility_func, p0, method="trust-constr", jac=utility_func_der, hess="2-point", constraints=[linear_constraints], bounds=bounds, options={"verbose": 1})

print(res.x)

