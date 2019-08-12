import cvxpy as cp
import numpy as np
import random
import time

np.set_printoptions(suppress=True)
np.random.seed(0)
random.seed(0)
MAX_DECLARED = 20


class ProbabilityGenerator:

    def __init__(self, students, classes):
        self.students = students
        self.classes = classes
        self.max_per_student = 4
        self.max_declared_class = min(MAX_DECLARED, self.classes)
        self.values = self.generate_values()
        self.C = np.ones(self.classes) * 5
        self.p = cp.Variable((self.students, self.classes), nonneg=True)
        self.chosen_list = []
        self.cycle = []

    def generate_values(self):
        total_value = []
        for i in range(self.students):
            value = [0 for i in range(self.classes)]
            for j in range(self.max_declared_class):
                value[j] = random.randint(1, 100)
            random.shuffle(value)
            total_value.append(value)

        return np.array(total_value).T

    def generate_p(self):
        total = []
        for i in range(self.students):
            sub_p = []
            for j in range(self.classes):
                if self.values.T[i][j] != 0:
                    new = cp.Variable(nonneg=True)
                else:
                    new = cp.Constant(0)
                sub_p.append(new)
            total.append(cp.hstack(sub_p))
        return cp.vstack(total)

    def generate_constraints(self):
        dic = {i: [] for i in range(self.classes)}
        for i in range(self.students):
            class_count = 0
            for j in range(self.classes):
                if self.values.T[i][j] != 0:
                    dic[j].append(self.p[i][class_count])
        for i in range(self.classes):
            dic[i] = cp.hstack(dic[i])
        students_per_class = cp.hstack([cp.sum(v) for u, v in dic.items()])
        return students_per_class

    def values_without_zero(self):
        new = []
        for i in self.values.T:
            sub = []
            for j in i:
                if j != 0:
                    sub.append(j)
            new.append(sub)
        return np.array(new).T

    def direct_solve(self):
        self.p = self.generate_p()
        upper_bound = np.array(
            [[0.99 if self.values.T[i][j] != 0 else 0 for j in range(self.classes)] for i in range(self.students)])
        objective_func = cp.sum(cp.log(cp.diag(self.p@self.values)))
        prob = cp.Problem(cp.Maximize(objective_func), [self.p <= 0.99, cp.sum(self.p, axis=1) <= 4, cp.sum(self.p, axis=0) <= self.C])
        prob.solve(solver="SCS", max_iters=5000, verbose=True)
        # print(self.values.T)
        # print(self.p.value)
        print(self.p.value)
        print(objective_func.value)
        # print(cp.sum(self.p, axis=1).value)
        print(cp.sum(self.p, axis=1).value)
        # print(cp.sum(self.p, axis=0).value)
        # print(cp.max(self.p, axis=1).value)

    def solve_with_trick(self):
        self.p = cp.Variable((self.students, self.classes), nonneg=True)
        upper_bound = np.array([[1 if self.values.T[i][j] != 0 else 0 for j in range(self.classes)]for i in range(self.students)])
        x1 = cp.Variable(self.students)
        x2 = cp.Variable(self.students)
        y1 = cp.Variable(self.classes)
        original_func = cp.sum(cp.log(cp.diag(self.p @ self.values)))
        objective_func = cp.sum(cp.log(cp.diag(self.p@self.values))) \
                         - 10e10 * cp.sum(cp.square(self.max_per_student - cp.sum(self.p, axis=1) - x1))\
                         - 10e10 * cp.sum(cp.square(self.max_per_student - cp.sum(self.p, axis=1) + x2))\
                         - 10e10 * cp.sum(cp.square(self.C - cp.sum(self.p, axis=0) - y1))
        prob = cp.Problem(cp.Maximize(objective_func),
                           [self.p <= upper_bound, x1 <= self.max_per_student, 0 <= x1, x2 <= self.max_per_student, 0 <= x2, y1 <= self.C, 0 <= y1])
        prob.solve(solver="SCS", max_iters=5000)
        print(self.p.value)
        print(original_func.value)

    def solve_with_trick_and_constants(self):
        self.p = self.generate_p()
        x1 = cp.Variable(self.students)
        x2 = cp.Variable(self.students)
        y1 = cp.Variable(self.classes)
        original_func = cp.sum(cp.log(cp.diag(self.p@self.values)))
        objective_func = cp.sum(cp.log(cp.diag(self.p@self.values))) \
                         - 10e10 * cp.sum(cp.square(self.max_per_student - cp.sum(self.p, axis=1) - x1))\
                         - 10e10 * cp.sum(cp.square(self.max_per_student - cp.sum(self.p, axis=1) + x2))\
                         - 10e10 * cp.sum(cp.square(self.C - cp.sum(self.p, axis=0) - y1))
        prob = cp.Problem(cp.Maximize(objective_func),
                           [self.p <= 1, x1 <= self.max_per_student, 0 <= x1, x2 <= self.max_per_student, 0 <= x2, y1 <= self.C, 0 <= y1])
        prob.solve(solver="SCS", max_iters=5000)
        # print(self.values.T)
        print(self.p.value)
        print(original_func.value)

    def solve_with_trick_and_reduced_variables(self):
        self.p = cp.Variable((self.students, self.max_declared_class), nonneg=True)
        values_without_zero = self.values_without_zero()
        students_per_class = self.generate_constraints()

        x1 = cp.Variable(self.students)
        x2 = cp.Variable(self.students)
        y1 = cp.Variable(self.classes)
        original_func = cp.sum(cp.log(cp.diag(self.p@values_without_zero)))
        objective_func = cp.sum(cp.log(cp.diag(self.p@values_without_zero))) \
                         - 10e10 * cp.sum(cp.square(self.max_per_student - cp.sum(self.p, axis=1) - x1))\
                         - 10e10 * cp.sum(cp.square(self.max_per_student - cp.sum(self.p, axis=1) + x2))\
                         - 10e10 * cp.sum(cp.square(self.C - students_per_class - y1))
        prob = cp.Problem(cp.Maximize(objective_func),
                           [self.p <= 1, x1 <= self.max_per_student, 0 <= x1, x2 <= self.max_per_student, 0 <= x2, y1 <= self.C, 0 <= y1])
        prob.solve(solver="SCS", max_iters=5000)
        print(values_without_zero.T)
        print(self.p.shape)
        print(self.p.value)
        print(original_func.value)

    def fix_error(self):
        result = self.p.value
        sum_per_student = np.sum(result, axis=1)
        student_per_class = np.sum(result, axis=0)
        for i in range(self.classes):
            if student_per_class[i] > 5:
                for j in range(self.students):
                    if result[j][i] > 0.1:
                        result[j][i] -= (student_per_class[i] - self.C[i])
                        sum_per_student = np.sum(result, axis=1)

        for i in range(self.students):
            if sum_per_student[i] > 4:
                for j in range(self.classes):
                    if result[i][j] > 0.1:
                        result[i][j] -= (sum_per_student[i] - 4)
                        student_per_class = np.sum(result, axis=0)
                        break
            elif sum_per_student[i] < 4:
                for j in range(self.classes):
                    if (5 - student_per_class[j]) > 4 - sum_per_student[i]:
                        result[i][j] += (4 - sum_per_student[i])
                        break
        # print(result)
        # print(self.values)
        print(np.sum(result, axis=1))
        self.p = result

    def find_first_non_zero(self, i, j, visited, horizontal=True):
        if horizontal:
            result = None
            for idx in range(j, self.classes):
                # print("test", i, visited[i])
                if self.p[i][idx] != 0 and not visited[i][idx]:
                    visited[i][idx] = True
                    result = idx
                    break
            if result is None:
                for idx in range(j, -1, -1):
                    if self.p[i][idx] != 0 and not visited[i][idx]:
                        visited[i][idx] = True
                        result = idx
                        break
            if result is None:
                raise ValueError("Not able to find next position.")
            return i, result
        else:
            result = None
            for idx in range(i, self.students):
                if self.p[idx][j] != 0 and not visited[idx][j]:
                    visited[idx][j] = True
                    result = idx
                    break
            if result is None:
                for idx in range(i, -1, -1):
                    if self.p[idx][j] != 0 and not visited[idx][j]:
                        visited[idx][j] = True
                        result = idx
                        break
            if result is None:
                raise ValueError("Not able to find next position.")
            return result, j

    def reduce_one_path(self):

        visited = np.array([[False if (1 - self.p[m][n] > 1e-10 and self.p[m][n] > 1e-10) else True for n in range(self.classes)] for m in range(self.students)])

        student_per_class = np.sum(self.p, axis=0)

        # self.chosen_list = []

        visited_row = [False for i in range(self.students)]

        visited_column = [False for i in range(self.classes)]

        if len(self.chosen_list) == 0:
            count = 0
            for p in range(self.students):
                state = False
                for q in range(self.classes):
                    if 1 - self.p[p][q] > 1e-10 and self.p[p][q] > 1e-10:
                        state = True
                        break
                    else:
                        if self.p[p][q] < 1e-10:
                            self.p[p][q] = 0
                        elif 1 - self.p[p][q] < 1e-10:
                            self.p[p][q] = 1
                if state:
                    first = p
                    break

            i, j = initial_i, initial_j = self.find_first_non_zero(first, 0, visited)
            visited_row[i] = True
            visited_column[j] = True
            self.chosen_list.append((i, j))
        else:
            print("before", self.chosen_list)
            if len(self.chosen_list) % 2 == 0:
                count = 1
            else:
                count = 0
            i, j = self.chosen_list[-1]
            initial_i, initial_j = self.chosen_list[0]
            for u, v in self.chosen_list:
                visited_row[u] = True
                visited_column[v] = True
                visited[u, v] = True

        find_cycle = False

        while True:

            if count % 2 == 0:
                is_horizontal = True
            else:
                is_horizontal = False
            try:
                i, j = self.find_first_non_zero(i, j, visited, horizontal=is_horizontal)

                self.chosen_list.append((i, j))

                if visited_row[i] and visited_column[j]:
                    print("here")
                    if not is_horizontal:
                        for u, v in self.chosen_list[1::2]:
                            if i == u:
                                idx = self.chosen_list.index((u, v))

                                break
                    else:
                        for u, v in self.chosen_list[::2]:
                            if j == v:
                                idx = self.chosen_list.index((u, v))
                                break
                    print(idx)
                    self.cycle = self.chosen_list[idx:]
                    find_cycle = True
                    break

                visited_row[i] = True
                visited_column[j] = True
                count += 1

            except ValueError:

                if len(self.chosen_list) % 2 == 1:
                    self.chosen_list = self.chosen_list[:-1]
                break

        if not find_cycle:

            count = 0
            i, j = initial_i, initial_j
            while True:

                if count % 2 == 0:
                    is_horizontal = False
                else:
                    is_horizontal = True
                try:
                    i, j = self.find_first_non_zero(i, j, visited, horizontal=is_horizontal)

                    self.chosen_list.insert(0, (i, j))
                    print("here", self.chosen_list)

                    if visited_row[i] and visited_column[j]:

                        if not is_horizontal:
                            for u, v in self.chosen_list[2::2]:
                                if i == u:
                                    idx = self.chosen_list.index((u, v))
                                    self.cycle = self.chosen_list[:idx]

                                    break
                        else:
                            for u, v in self.chosen_list[1::2]:
                                if j == v:
                                    idx = self.chosen_list.index((u, v))
                                    self.cycle = self.chosen_list[:idx + 1]

                                    break
                        find_cycle = True
                        break

                    visited_row[i] = True
                    visited_column[j] = True
                    count += 1

                except ValueError:
                    if len(self.chosen_list) % 2 == 1:
                        self.chosen_list = self.chosen_list[1:]
                    break

        print(self.chosen_list)
        print("test", len(self.chosen_list))
        if len(self.chosen_list) == 0:
            return
        if find_cycle:
            print("cycle", self.cycle)
            print("chosen", self.chosen_list)
            odd_list = self.cycle[::2]
            even_list = self.cycle[1::2]

        else:
            odd_list = self.chosen_list[::2]
            even_list = self.chosen_list[1::2]

        min_to_give_by_circle, x1, y1 = self.find_min_to_decrease(odd_list, even_list, student_per_class)
        min_to_get_by_circle, x2, y2 = self.find_min_to_increase(odd_list, even_list, student_per_class)

        chance_to_decrease = min_to_get_by_circle / min_to_get_by_circle + min_to_give_by_circle

        x = random.random()

        if x < chance_to_decrease:
            for i, j in odd_list:
                self.p[i][j] -= min_to_give_by_circle
            for i, j in even_list:
                self.p[i][j] += min_to_give_by_circle
            idx = self.chosen_list.index((x1, y1))
            self.chosen_list = self.chosen_list[:idx]
        else:
            for i, j in odd_list:
                self.p[i][j] += min_to_get_by_circle
            for i, j in even_list:
                self.p[i][j] -= min_to_get_by_circle
            idx = self.chosen_list.index((x2, y2))
            self.chosen_list = self.chosen_list[:idx]

        try:
            u1, v1 = self.chosen_list[0]
            u2, v2 = self.chosen_list[1]
            if u1 != u2:
                self.chosen_list = self.chosen_list[1:]
        except:
            pass

        try:
            u1, v1 = self.chosen_list[-2]
            u2, v2 = self.chosen_list[-1]
            if u1 != u2:
                self.chosen_list = self.chosen_list[:-1]
        except:
            pass

        print("pre", self.chosen_list, len(self.chosen_list))

    def find_min_to_decrease(self, odd_list, even_list, student_per_class):

        current = float("inf")
        x, y = None, None
        for i, j in odd_list:
            if self.p[i][j] < current:
                current = self.p[i][j]
                x, y = i, j
        for i, j in even_list:
            if 1 - self.p[i][j] < current:
                current = 1 - self.p[i][j]
                x, y = i, j
        for i, j in even_list:
            if self.C[j] - student_per_class[j] < current:
                current = self.C[j] - student_per_class[j]
                x, y = i, j
        return current, x, y

    def find_min_to_increase(self, odd_list, even_list, student_per_class):

        current = float("inf")
        x, y = None, None
        for i, j in even_list:
            # print(current)
            if self.p[i][j] < current:
                current = self.p[i][j]
                x, y = i, j
        for i, j in odd_list:
            # print(current)
            if 1 - self.p[i][j] < current:
                current = 1 - self.p[i][j]
                x, y = i, j
        for i, j in odd_list:
            if self.C[j] - student_per_class[j] < current:
                current = self.C[j] - student_per_class[j]
                x, y = i, j
        return current, x, y

    def finished(self):
        for i in range(self.students):
            for j in range(self.classes):
                if 1 - self.p[i][j] > 1e-10 and self.p[i][j] > 1e-10:
                    return False
        return True

    def generate_allocation(self):
        count = 0
        while not self.finished():

            self.reduce_one_path()
            count += 1
            print(self.p)
            print(np.sum(self.p, axis=0))
            print(np.sum(self.p, axis=1))
            print("count", count)
            if count > 100000:
                raise RuntimeError("Exceed maximum loop.")

            if (abs(np.sum(self.p, axis=1) - np.ones(self.students) * 4) > np.ones(self.students) * 1e-10).any():
                raise ValueError("Class for each student has changed.")

            if (self.p < np.ones_like(self.p) * -1e-10).any():
                raise ValueError("Probability cannot be negative.")

            if ((np.sum(self.p, axis=0) - self.C) > np.ones_like(self.C) * 1e-10).any():

                raise ValueError("Exceed maximum class size.")


if __name__ == '__main__':
    machine = ProbabilityGenerator(20, 40)
    # machine.solve_with_trick_and_constants()
    start = time.time()
    machine.direct_solve()
    point1 = time.time()
    machine.fix_error()
    point2 = time.time()
    machine.generate_allocation()
    end = time.time()
    print("total time cost:", end - start)
    print("generating prob matrix takes:", point1 - start)
    print("fixing error takes:", point2 - point1)
    print("reducing the matrix to actual allocation takes:", end - point2)
    # 100 * 200 = 90s, 200 * 400 = 794s
    # machine.solve_with_trick()
    # machine.solve_with_trick_and_reduced_variables()
