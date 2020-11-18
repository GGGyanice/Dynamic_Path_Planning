# -*- coding:utf-8 -*-
"""
Time : 2020/11/16 10:38
Author : Kexin Guan
Decs ：
"""
import copy
import json
import time
from math import floor
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import requests


class Gena_TSP(object):
    def __init__(self, data, distance, pop_size, max_iteration=1000, cross_rate=0.9, mutation_rate=0.01, select_rate=0.8):
        self.max_iteration = max_iteration  # 迭代数
        self.pop_size = pop_size  # 群体数量
        self.cross_rate = cross_rate  # 交叉概率
        self.mutation_rate = mutation_rate  # 变异概率
        self.select_rate = select_rate  # 选择概率

        self.data = data
        self.num = len(data)  # 城市个数—>染色体长度

        self.mat_distance = distance  # 距离矩阵n*n, 评判标准

        # 子代个数
        self.select_num = max(floor(self.pop_size * self.select_rate + 0.5), 2)

        # 初始化父代和子代群体
        self.parent = np.array([0] * self.pop_size * self.num).reshape(self.pop_size, self.num)
        self.child = np.array([0] * self.select_num * self.num).reshape(self.select_num, self.num)

        # 适应度—>染色体的路径总长度的倒数
        self.fitness = np.zeros(self.pop_size)

        # 每次迭代的最优距离和路径
        self.best_fitness = []
        self.best_path = []

    def init_parent(self):
        random_parent = np.array(range(self.num))
        print(random_parent)
        for i in range(self.pop_size):
            np.random.shuffle(random_parent)
            self.parent[i, :] = random_parent
            self.fitness[i] = self.calculate_fitness(random_parent)

    def calculate_fitness(self, one_path):
        res = 0
        for i in range(self.num - 1):
            res += self.mat_distance[one_path[i], one_path[i + 1]]
        res += self.mat_distance[one_path[-1], one_path[0]]  # 首尾加一遍
        return res

    def split_path(self, one_path):
        res = str(one_path[0] + 1) + '-->'
        for i in range(1, self.num):
            res += str(one_path[i] + 1) + '-->'
        res += str(one_path[0] + 1) + '\n'
        print(res)

    def select_child(self):
        fit = 1 / self.fitness  # 适应度函数
        cumsum_fit = np.cumsum(fit)
        pick = cumsum_fit[-1] / self.select_num * (np.random.rand() + np.array(range(self.select_num)))
        i, j = 0, 0
        index = []
        while i < self.pop_size and j < self.select_num:
            if cumsum_fit[i] >= pick[j]:
                index.append(i)
                j += 1
            else:
                i += 1
        self.child = self.parent[index, :]

    def cross_child(self):
        if self.select_num % 2 == 0:
            num = range(0, self.select_num, 2)
        else:
            num = range(0, self.select_num - 1, 2)
        for i in num:
            if self.cross_rate >= np.random.rand():
                self.child[i, :], self.child[i + 1, :] = self.pmx_cross(self.child[i, :], self.child[i + 1, :])

    def pmx_cross(self, child_a, child_b):  # PMX杂交操作
        r1 = np.random.randint(self.num)
        r2 = np.random.randint(self.num)
        while r2 == r1:
            r2 = np.random.randint(self.num)
        child_a1, child_b1 = child_a.copy(), child_b.copy()
        for i in range(min(r1, r2), max(r1, r2) + 1):
            child_a2, child_b2 = child_a.copy(), child_b.copy()
            child_a[i], child_b[i] = child_b1[i], child_a1[i]
            x = np.argwhere(child_a == child_a[i])
            y = np.argwhere(child_b == child_b[i])
            if len(x) == 2:
                child_a[x[x != i]] = child_a2[i]
            if len(y) == 2:
                child_b[y[y != i]] = child_b2[i]
        return child_a, child_b

    def mutation_child(self):
        for i in range(self.select_num):
            if np.random.rand() <= self.cross_rate:
                r1 = np.random.randint(self.num)
                r2 = np.random.randint(self.num)
                while r2 == r1:
                    r2 = np.random.randint(self.num)
                self.child[i, [r1, r2]] = self.child[i, [r2, r1]]

    def reverse_child(self):
        for i in range(self.select_num):
            r1 = np.random.randint(self.num)
            r2 = np.random.randint(self.num)
            while r2 == r1:
                r2 = np.random.randint(self.num)
            children = self.child[i, :].copy()

            children[min(r1, r2):max(r1, r2) + 1] = self.child[i, min(r1, r2):max(r1, r2) + 1][::-1]
            if self.calculate_fitness(children) < self.calculate_fitness(self.child[i, :]):
                self.child[i, :] = children

    def insert_child(self):
        index = np.argsort(self.fitness)[::-1]
        self.parent[index[:self.select_num], :] = self.child


def show_path(data, GA):
    fig, ax = plt.subplots()
    x, y = data[:, 0], data[:, 1]
    ax.scatter(x, y, linewidths=0.1)
    for i, txt in enumerate(range(1, len(data) + 1)):
        ax.annotate(txt, (x[i], y[i]))
    res0 = GA.parent[0]
    x0, y0 = x[res0], y[res0]
    for i in range(len(data) - 1):
        plt.quiver(x0[i], y0[i], x0[i + 1] - x0[i], y0[i + 1] - y0[i], color='r', width=0.005, angles='xy', scale=1,
                   scale_units='xy')
    plt.quiver(x0[-1], y0[-1], x0[0] - x0[-1], y0[0] - y0[-1], color='r', width=0.005, angles='xy', scale=1,
               scale_units='xy')
    plt.show()


def ga_main(data, distance):
    pop = len(data)*10
    GA = Gena_TSP(data, distance, pop)  # 生成一个遗传算法类
    GA.init_parent()  # 初始化父类

    # 绘制初始化的路径图
    # show_path(data, GA)
    # print('初始染色体的路程: ' + str(GA.fitness[0]))

    # 迭代遗传
    for i in range(GA.max_iteration):
        GA.select_child()  # 选择子代
        GA.cross_child()  # 交叉
        GA.mutation_child()  # 变异
        GA.reverse_child()  # 进化逆转
        GA.insert_child()  # 子代插入

        # 重新计算适应度
        for j in range(GA.pop_size):
            GA.fitness[j] = GA.calculate_fitness(GA.parent[j, :])

        index = GA.fitness.argmin()
        if (i + 1) % 30 == 0:
            print('第' + str(i + 1) + '步后的最短的路程: ' + str(GA.fitness[index]))
            print('第' + str(i + 1) + '步后的最优路径:')
            GA.split_path(GA.parent[index, :])  # 显示每一步的最优路径

        GA.best_fitness.append(GA.fitness[index])
        GA.best_path.append(GA.parent[index, :])
    show_path(data, GA)
    return GA


def get_resourse(k, start_x, start_y, destination_x, destination_y):  # 调用api
    api = 'https://restapi.amap.com/v3/direction/driving?parameters'
    parameters = {
        'key': k,
        'origin': '%s' % start_x + ',' + '%s' % start_y,
        'destination': '%s' % destination_x + ',' + '%s' % destination_y,
        'extensions': 'base'
    }
    r = requests.get(api, params=parameters)
    r = r.text
    json_data = json.loads(r)
    # with open("./base.json", 'w', encoding='utf-8') as json_file:
    #     json.dump(json_data, json_file, ensure_ascii=False)
    t = json_data["route"]["paths"][0]["duration"]
    d = json_data["route"]["paths"][0]["distance"]
    return t, d


def json_mat(target):  # json转矩阵
    j = json.loads(target)
    dic = j['point'][0]
    point, left, right = [], [], []
    for key in dic:
        point.append(key)
        dic[key].split(",")
        left.append(float(dic[key].split(",")[0]))
        right.append(float(dic[key].split(",")[1]))

    mat_value = np.hstack((np.array(left).reshape(-1, 1), np.array(right).reshape(-1, 1)))
    mat_col = np.array(point).reshape(-1, 1)
    return mat_col, mat_value


def main_function(key, target_json, choose):
    print("-----json转矩阵-----")
    col, data = copy.deepcopy(json_mat(target_json)[0]), copy.deepcopy(json_mat(target_json)[1])

    # 取高德上的时间、距离数据
    st = time.time()
    t = np.zeros((data.shape[0], data.shape[0]))  # 时间矩阵
    d = np.zeros((data.shape[0], data.shape[0]))  # 距离矩阵
    print("-----开始取数据-----")
    for i in range(data.shape[0]):  # 95s, 27x28次
        # print("第", i, "站点")
        for j in range(data.shape[0]-1):
            if i < j+1:
                t[i, j+1], d[i, j+1] = get_resourse(key, data[i, 0], data[i, 1], data[j+1, 0], data[j+1, 1])
                t[j+1, i], d[j+1, i] = get_resourse(key, data[j+1, 0], data[j+1, 1], data[i, 0], data[i, 1])
    # pd.DataFrame(t).to_csv("./doc/json_time.csv", index=False, header=False)
    # pd.DataFrame(d).to_csv("./doc/json_distance.csv", index=False, header=False)
    print("取数据耗时", time.time() - st)
    if choose is True:
        target_mat, second_tar = t, d  # 以路程耗时为评判标准
        target_word, second_word = "路程耗时", "路程距离"
    else:
        target_mat, second_tar = d, t  # 以路程距离为评判标准
        target_word, second_word = "路程距离", "路程耗时"

    # 开始计算
    st = time.time()
    planning = ga_main(data, target_mat)
    path_dis, path = planning.best_fitness[-1], planning.best_path[-1]
    path = np.hstack((path[int(np.argwhere(path == 0)):], path[:int(np.argwhere(path == 0))]))
    # 计算第二个指标
    sec = 0
    for i in range(len(path) - 1):
        sec += second_tar[path[i], path[i + 1]]
    sec += second_tar[path[-1], path[0]]  # 首尾加一遍
    print("-------------GA算法输出--------------")
    print(target_word, path_dis)
    print(second_word, sec)
    print("路程计划", path)
    print("计算耗时", time.time() - st)

    # 拼接输出json
    col = col[path].reshape(1, -1)[0].tolist()
    values = []
    for i in range(data.shape[0]):
        values.append(str(data[path][i, 0]) + "," + str(data[path][i, 1]))
    out_json = json.dumps({"result": [dict(zip(col, values))], str(target_word): path_dis, str(second_word): sec},
                          ensure_ascii=False)
    return out_json


if __name__ == '__main__':
    key = "your_key"
    path = "./doc/point.json"
    with open(path, 'r', encoding='gbk') as f:
        dic = json.load(f)
#
    target_json = json.dumps(dic, ensure_ascii=False)
    choose = True
    main_function(key, target_json, choose)
