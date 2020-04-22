# -*-coding:utf-8 -*-
import math
import numpy as np


# 10-->2
def encode(num, range0, length):
    n = '{:0>64b}'.format(int(num*10000) - int(range0*10000))[-length:]
    #print(list(map(int, n)))
    return list(map(int, n))


# Step1:初始化种群个体
def species_origin(num, length):
    # 10*33的二维数组
    population = np.zeros([num, length])
    for i in range(num):
        x1 = np.random.uniform(-3.0, 12.1)
        x2 = np.random.uniform(4.1, 5.8)
        # print(x1, x2)
        population[i][0:18] = encode(x1, -3.0, 18)
        population[i][18:33] = encode(x2, 4.1, 15)
    # print(population)
    return population


# Step2计算个体适应度值
# 2.1 求x1，x2
def decode(population):
    # 10*2的二维数组，存放最后结果
    domain = np.zeros([len(population), 2])
    for i in range(len(population)):
        x1 = int("".join(map(lambda x: str(int(x)), population[i][0:18])), 2)
        domain[i][0] = (x1 - 3.0 * 10000) / 10000
        x2 = int("".join(map(lambda x: str(int(x)), population[i][18:33])), 2)
        domain[i][1] = (x2 + 4.1 * 10000) / 10000
    # print(domain)
    return domain


# 2.2求适应度
def get_fitness(domain):
    fitness = np.zeros(len(domain))
    pi = math.pi
    sin = math.sin
    for i in range(len(domain)):
        fitness[i] = 21.5 + domain[i, 0] * sin(4 * pi * domain[i, 0]) + domain[i, 1] * sin(20 * pi * domain[i, 1])
    # print(fitness)
    return fitness


# Step3轮盘赌进行自然选择及交叉
# Step3.1计算个体的累计适应度
def get_cumfitness(fitness):
    # 存放累计适应度结果
    cumfitness = np.zeros(len(fitness))
    total = np.sum(fitness)
    temp = 0
    for i in range(len(fitness)):
        temp += fitness[i]
        cumfitness[i] = temp / total
    # print(cumfitness)
    return cumfitness


# Step3.2轮盘赌选择进入交叉池的个体
def first_select(cumfitness,mutated):
    new = np.zeros(np.shape(mutated))
    # 交叉变异后种群新的x1x2
    domain = decode(mutated)
    # 新的适应度
    fitness = get_fitness(domain)
    # 按适应度排序
    s = sorted(fitness, reverse=True)
    # 前3个直接进入下一代
    # 前20%个直接进入下一代
    n = int(len(mutated) * 0.3)
    for i in range(n):
        for j in range(len(mutated)):
            if fitness[j] == s[i]:
                new[i] = mutated[j]
                break
    # 新种群累计适应度
    cumfitness = get_cumfitness(fitness)
    # 进行第二次轮盘赌找后7个
    new_selected = select(cumfitness,mutated,len(mutated))
    for i in range(3, len(mutated)):
        new[i] = new_selected[i]
    # print(new)
    return new


def select(cumfitness, population, n):
    # 初始化结果数组10*33
    selected = np.zeros([n, np.shape(population)[1]])
    for i in range(n):
        r = np.random.rand()
        # 根据随机数大小所在的累计适应度区间，选择进入交叉池的个体
        for j in range(len(cumfitness)):
            if r < cumfitness[j]:
                if np.all(population[j] == 0):
                    i -= 1
                    break
                else:
                    selected[i] = population[j]
                    break
    # print(selected)
    return selected


# Step3.3 选中个体进行交叉
def crossover(selected, pc):
    # print(selected)
    for i in range(0, len(selected), 2):
        # 产生随机数，判断是否交叉
        r = np.random.rand()
        if r < pc:
            # 生成两个交叉点
            cp1 = np.random.randint(0, len(selected[0])+1)
            cp2 = np.random.randint(0, len(selected[0])+1)
            # 确保交叉点1<交叉点2
            if cp1 > cp2:
                cp1, cp2 = cp2, cp1
            # print(cp1, cp2)
            for j in range(cp1, cp2):
                #  相邻两个交换两个交叉点间的所有基因
                selected[i][j], selected[i + 1][j] = selected[i + 1][j], selected[i][j]
                x = decode(selected[i:i+2])
                if (x[0, 0] < -3.0) | (x[0, 0] > 12.1) | (x[0, 1] < 4.1) | (x[0, 1] > 5.8):
                    selected[i] = np.zeros([1, np.shape(selected)[1]])
                if (x[1, 0] < -3.0) | (x[1, 0] > 12.1) | (x[1, 1] < 4.1) | (x[1, 1] > 5.8):
                    selected[i+1] = np.zeros([1, np.shape(selected)[1]])
    return selected


# Step4 变异
def mutate(crossed, pm):
    for i in range(len(crossed)):
        for j in range(len(crossed[0])):
            # 产生随机数，判断是否进行变异
            r = np.random.rand()
            # 若发生变异，0->1,1->0
            if r < pm:
                crossed[i, j] = (crossed[i, j]+1) % 2
                x1, x2 = decode(crossed[i:i+2])[0]
                if (x1 < -3.0) | (x1 > 12.1) | (x2 < 4.1) | (x2 > 5.8):
                    crossed[i] = np.zeros([1, np.shape(crossed)[1]])
    return crossed


# Step5 选取下一代种群
def new_select(mutated):
    new = np.zeros(np.shape(mutated))
    # 新的适应度
    fitness = get_fitness(decode(mutated))
    # print(fitness)
    # 按适应度排序
    s = sorted(fitness, reverse=True)
    # 前20%个直接进入下一代
    n = int(len(mutated)*0.3)
    for i in range(n):
        for j in range(len(mutated)):
            if fitness[j] == s[i]:
                new[i] = mutated[j]
                break
    # 新累计适应度
    cumfitness = get_cumfitness(fitness)
    # 进行第二次轮盘赌
    new_selected = select(cumfitness, mutated, len(mutated)-n)
    new[n:] = new_selected
    # print(new)
    return new
