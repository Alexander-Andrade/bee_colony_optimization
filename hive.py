import numpy as np
import random
import itertools


# формируем опорные точки случайным образом, между позицией робота и позицией цели
def control_points(start_pos, target_pos):
    n_control_points = round(random.gammavariate(3, 2))
    ctrl_points = [start_pos]

    target_x_dist = abs(target_pos[0] - start_pos[0])
    target_y_dist = abs(target_pos[1] - start_pos[1])

    for i in range(n_control_points):
        x = round(random.uniform(0, target_x_dist))
        y = round(random.uniform(0, target_y_dist))
        ctrl_points.append((x, y))

    ctrl_points.sort(key=lambda point: abs(point[0] - start_pos[0]) + abs(point[1] - start_pos[1]))
    ctrl_points.append(target_pos)

    return ctrl_points


# формируем промежуточные точки между start_pos и target_pos
def intermediate_steps(p1, p2):
    sign_x = np.sign(p2[0] - p1[0])
    sign_y = np.sign(p2[1] - p1[1])

    dist_x = abs(p2[0] - p1[0])
    dist_y = abs(p2[1] - p1[1])

    step_x = round(dist_y / dist_x) if dist_x < dist_y and dist_x > 0 else 1
    step_y = round(dist_x / dist_y) if dist_y < dist_x and dist_y > 0 else 1

    pos_x, pos_y = p1
    yield (pos_x, pos_y)
    i = 0
    while pos_x != p2[0] or pos_y != p2[1]:
        if i % step_x == 0 and pos_x != p2[0]:
            pos_x += sign_x
        if i % step_y == 0 and pos_y != p2[1]:
            pos_y += sign_y
        yield (pos_x, pos_y)
        i += 1


# формируем полный путь по начальной, промежуточный и конечной точками
def build_path(solution):
    path = []
    prev_point = solution[0]

    for point in itertools.islice(solution, 1, None):
        path += list(intermediate_steps(prev_point, point))
        prev_point = point
    return path


# рассчитываем насколько хорошо путь подходит
# учитывается расстояние, пройденное роботом (длинна пути) и 'штраф' за пересечение 'гор'
def loss_function(path, terrain):
    terrain_drop = 0
    prev_point = path[0]
    for point in itertools.islice(path, 1, None):
        terrain_drop += abs(terrain[point[1]][point[0]] - terrain[prev_point[1]][prev_point[0]])
        prev_point = point

    return terrain_drop*100 + len(path)*0.1


# модифицируем оторную точку в пути
# это может улучшить или ухудшить качество пути
def modify_point_axes(x_old, search_neighborhood, axes_size):
    dx = np.random.randint(-search_neighborhood, search_neighborhood)
    guess_x = x_old + dx
    x = guess_x
    if guess_x < 0:
        x = 0
    elif guess_x >= axes_size:
        x = axes_size - 1
    return x


class Solution:

    def __init__(self, points, terrain, search_neighborhood):
        self.points = points
        self.terrain = terrain
        self.search_neighborhood = search_neighborhood
        self.path = build_path(self.points)
        self.fitness = loss_function(self.path, self.terrain)

    # получаем новое решение, у которого немного иземены координаты опорных точек
    def modified_solution(self):
        points = self.points.copy()

        solution_len = len(self.points)

        if solution_len > 2:
            points_pos = np.random.randint(1, solution_len-1, size=solution_len // 2)
            for pos in points_pos:
                point = points[pos]
                new_x = modify_point_axes(point[0], self.search_neighborhood, self.terrain.shape[0])
                new_y = modify_point_axes(point[1], self.search_neighborhood, self.terrain.shape[1])
                points[pos] = (new_x, new_y)

        return Solution(points=points, terrain=self.terrain, search_neighborhood=self.search_neighborhood)


class Hive:
    # n_scouts - кол-во пчел-разведчиков
    # n_elite - кол-во элитных решений
    # n_bees_elite - кол-во пчел для каждого элитного решения
    # n_bees_others - кол-во пчел для остальных решений
    # search_neighborhood - кол-во ячеек, в пределах которых можно изменять опорную точку
    # terrain - ландшафт
    # pos - позиция улея
    def __init__(self, n_scouts, n_points, n_elite, n_bees_elite, n_bees_others, search_neighborhood, terrain,
                 pos):
        self.n_scouts = n_scouts
        self.n_points = n_points
        self.n_elite = n_elite
        self.n_bees_elite = n_bees_elite
        self.n_bees_others = n_bees_others
        self.search_neighborhood = search_neighborhood
        self.terrain = terrain
        self.pos = pos

    # найти наилучший путь к ячейке с позицией target_pos за n_iters
    def find(self, target_pos, n_iters):
        best = None
        for i in range(n_iters):
            solutions = self.scouts_solutions(target_pos)
            solutions.sort(key=lambda s: s.fitness)
            improved_solutions = self.local_search(solutions)
            if (best is None) or (best is not None and best.fitness > improved_solutions[0].fitness):
               best = improved_solutions[0]

        return best

    # после нахождения рандомных путей разведчиками,
    # производим локальный поиск над лучшими найденными путями,
    # возможно получится улучшить результат разведчика
    def local_search(self, solutions):
        elite_solutions = solutions[:self.n_elite]
        #  others_solutions = solutions[self.n_elite:self.n_points]

        for i in range(self.n_elite):
            current = elite_solutions[i]
            for _ in range(self.n_bees_elite):
                new_solution = current.modified_solution()
                if new_solution.fitness < current.fitness:
                    elite_solutions[i] = new_solution
                    current = new_solution

        elite_solutions.sort(key=lambda s: s.fitness)
        return elite_solutions

    # результаты пазведчиков
    # Любой путь потенциально пригодный,
    # потому что мы знаем позицию улея (робота) и позицию цели
    # и строим промежуточные точки между этими ячейками.
    # Лучший путь - это кратчайший, не 'наезжающий' на горы
    def scouts_solutions(self, target_pos):
        solutions = []
        for i in range(self.n_scouts):
            solution = Solution(points=control_points(self.pos, target_pos),
                     terrain=self.terrain,
                     search_neighborhood=self.search_neighborhood)
            solutions.append(solution)

        return solutions
