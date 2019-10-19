#!/usr/bin/env python3
"""
RANSAC Algorithm Problem
(Due date: Oct. 23, 3 P.M., 2019)
The goal of this task is to fit a line to the given points using RANSAC algorithm, and output
the names of inlier points and outlier points for the line.

Do NOT modify the code provided to you.
Do NOT use ANY API provided by opencv (cv2) and numpy (np) in your code.
Do NOT import ANY library (function, module, etc.).
You can use the library random
Hint: It is recommended to record the two initial points each time, such that you will Not 
start from this two points in next iteration.
"""
import random


def abs_val(x):
    return -x if x < 0 else x


def line_params(lp1, lp2):
    """
    returns the parameters of the equation of the line identified
    by the two given points in the form ax + by + c = 0
    """
    x1, y1, x2, y2 = lp1[0], lp1[1], lp2[0], lp2[1] 
    rise = y2 - y1
    run = x2 - x1
    if run == 0:
        return 1, 0, -x1
    slope = rise / run 
    yint = y1 - (slope * x1)

    a = -rise
    b = run
    c = -yint * run
    return a, b, c


def distance(p, lp1, lp2):
    """
    calculate distance between a point given by p (tuple) and a
    line given by two points lp1 (tuple) and lp2 (tuple):

    d = |Am + Bn + C| / sqrt(A^2 + B^2)

    where the point is given by (m, n), and
    the line is given by Ax + By + C = 0
    """
    m, n = p[0], p[1]
    a, b, c = line_params(lp1, lp2)
    return abs_val((a * m) + (b * n) + c) / (a**2 + b**2)**(1/2)


def random_sample(points, sample_size):
    rand_list = random.sample(range(len(points)), sample_size)
    return [points[r] for r in sorted(rand_list)]


def is_inlier(p, sample, t):
    dist = distance(p, sample[0]['value'], sample[1]['value'])
    return dist <= t


def min(x, y):
    return x if x < y else y


def factorial(x):
    p = 1
    num = x
    for i in range(x - 1):
        p = p * num
        num -= 1
    return p 


def remove(points, el_name):
    for i in range(len(points)):
        p = points[i]
        if p['name'] == el_name:
            del points[i]
            break


def solution(input_points, t, d, k):
    """
    :param input_points:
           t: t is the perpendicular distance threshold from a point to a line
           d: d is the number of nearby points required to assert a model fits well, you may not need this parameter
           k: k is the number of iteration times
           Note that, n for line should be 2
           (more information can be found on the page 90 of slides "Image Features and Matching")
    :return: inlier_points_name, outlier_points_name
    inlier_points_name and outlier_points_name is two list, each element of them is str type.
    For example: If 'a','b' is inlier_points and 'c' is outlier_point.
    the output should be two lists of ['a', 'b'], ['c'].
    Note that, these two lists should be non-empty.
    """
    # TODO: implement this function.
    n = 2
    inlier_points_name = [p['name'] for p in random_sample(input_points, n)]
    outlier_points_name = [p['name'] for p in input_points \
            if p['name'] not in inlier_points_name]
    points_counter = {p['name']: 0 for p in input_points}
    points_copy = input_points[:]
    p_samples = []
    num_combinations = factorial(len(input_points)) \
            // (factorial(n) * factorial(len(input_points) - n))

    for i in range(min(k, num_combinations)):
        prev_sample = True
        while prev_sample:
            sample = random_sample(points_copy, n)
            prev_sample = sample in p_samples
        p_samples.append(sample)
        for s in sample:
            points_counter[s['name']] += 1
            if points_counter[s['name']] == (num_combinations * n) \
                    // len(input_points):
                remove(points_copy, s['name'])
                del points_counter[s['name']]
        inliers_tmp = sample[:]
        for s in input_points:
            if s not in sample and is_inlier(s['value'], sample, t):
                inliers_tmp.append(s)
        if len(inliers_tmp) > d and len(inliers_tmp) > len(inlier_points_name):
            inlier_points_name = [p['name'] for p in inliers_tmp]
            outlier_points_name = [p['name'] for p in input_points \
                    if p['name'] not in inlier_points_name]
    return sorted(inlier_points_name), sorted(outlier_points_name)


if __name__ == "__main__":
    input_points = [{'name': 'a', 'value': (0.0, 1.0)}, {'name': 'b', 'value': (2.0, 1.0)},
                    {'name': 'c', 'value': (3.0, 1.0)}, {'name': 'd', 'value': (0.0, 3.0)},
                    {'name': 'e', 'value': (1.0, 2.0)}, {'name': 'f', 'value': (1.5, 1.5)},
                    {'name': 'g', 'value': (1.0, 1.0)}, {'name': 'h', 'value': (1.5, 2.0)}]
    t = 0.5
    d = 3
    k = 100
    inlier_points_name, outlier_points_name = solution(input_points, t, d, k)  # TODO
    assert len(inlier_points_name) + len(outlier_points_name) == 8  
    f = open('./results/task1_result.txt', 'w')
    f.write('inlier points: ')
    for inliers in inlier_points_name:
        f.write(inliers + ',')
    f.write('\n')
    f.write('outlier points: ')
    for outliers in outlier_points_name:
        f.write(outliers + ',')
    f.close()


