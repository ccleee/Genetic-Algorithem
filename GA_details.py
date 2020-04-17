# -*-coding:utf-8 -*-
import math
import numpy as np


# Step1:初始化种群个体
def species_origin(num, length):
    # 10*33的二维数组
    population = np.random.randint(0, 2, [num, length])
    # print(population)
    return population


# Step2计算个体适应度值
# 2.1二进制转十进制
def transition(population, length1, length2):
    # 10*2的二维数组，存放转化结果
    decimal = np.zeros([len(population), 2])
    for i in range(len(population)):
        # （0~17位）
        f = 17
        for j in range(length1):
            # print(i,j)
            # 高18位十进制结果
            decimal[i][0] += population[i][j] * (math.pow(2, f))
            f -= 1
        # （18~32位）
        f = 14
        for k in range(length1, length1 + length2):
            # 低15位数十进制结果
            decimal[i][1] += population[i][k] * (math.pow(2, f))
            f -= 1
    return decimal


# 2.2 取值范围映射（分别求出随机数对应的x1，x2的取值)
def decode(population, length1, length2):
    # 10*2的二维数组，存放最后结果
    domain = np.zeros([len(population), 2])
    decimal = transition(population, length1, length2)
    d1 = (12.1 - (-3.0)) / (math.pow(2, length1) - 1)
    d2 = (5.8 - 4.1) / (math.pow(2, length2) - 1)
    for i in range(len(population)):
        # 种群个体数值x1的取值
        domain[i][0] = decimal[i][0] * d1 - 3
        # 种群个体数值x2的取值
        domain[i][1] = decimal[i][1] * d2 + 4.8
    # print(decimal)
    # print(domain)
    return domain


# Step2.2代入适应度函数并去掉负值
def get_fitness(domain):
    fitness = np.zeros(len(domain))
    pi = math.pi
    sin = math.sin
    for i in range(len(domain)):
        # 将原函数作为适应度函数求值，越大适应度越高
        fitness[i] = 21.5 + domain[i, 0] * sin(4 * pi * domain[i, 0]) + domain[i, 1] * sin(20 * pi * domain[i, 1])
        if fitness[i] < 0:
            fitness[i] = 0.0
    #print(fitness)
    return fitness


# Step3轮盘赌进行自然选择及交叉

# Step3.1计算个体的累积适应度
def get_cumfitness(fitness):
    # 存放累计适应度结果
    cumfitness = np.zeros(len(fitness))
    total = np.sum(fitness)
    temp = 0
    for i in range(len(fitness)):
        temp += fitness[i]
        # 转化为百分比
        cumfitness[i] = temp / total
    #print(cumfitness)
    return cumfitness



# Step3.2轮盘赌选择进入交叉池的个体
def select(cumfitness, population):
    # 初始化结果数组10*33
    selected = np.zeros(np.shape(population))
    for i in range(len(population)):
        r = np.random.rand()
        # 根据随机数大小所在的累计适应度区间，选择进入交叉池的个体
        for j in range(len(cumfitness)):
            if r < cumfitness[j]:
                selected[i] = population[j]
                break
    return selected
#step3.2 保留最优个体并进行轮盘赌

# Step3.3 选中个体进行交叉
def crossover(selected, pc):
    # 进行10/5=2次循环，两两交配
    for i in range(0, len(selected), 2):
        # 产生随机数，判断是否交叉
        r = np.random.rand()
        if r < pc:
            # 生成两个交叉点
            cp1 = np.random.randint(0, len(selected[0]))
            cp2 = np.random.randint(0, len(selected[0]))
            # 确保交叉点1<交叉点2
            if cp1 > cp2:
                cp1, cp2 = cp2, cp1
            #  开始交叉
            for j in range(cp1, cp2):
                #  相邻两个染色体进行交叉，交换两个交叉点间的所有基因
                selected[i][j], selected[i + 1][j] = selected[i + 1][j], selected[i][j]
    return selected


# Step4 变异
def mutate(crossed, pm):
    for i in range(len(crossed)):
        for j in range(len(crossed[0])):
            # 产生随机数，判断是否进行变异
            r = np.random.rand()
            # 若发生变异，0->1,1->0
            if r < pm:
                crossed[i][j] = (crossed[i][j]+1) % 2
    return crossed


# Step5 选取下一代种群
def new_select(mutated, length1, length2):
    new = np.zeros(np.shape(mutated))
    # 交叉变异后种群新的x1x2
    domain = decode(mutated, length1, length2)
    # 新的适应度
    fitness = get_fitness(domain)
    # 按适应度排序
    s = sorted(fitness, reverse=True)
    # 前2个直接进入下一代
    for i in range(len(mutated)):
        if fitness[i] == s[0]:
            new[0] = mutated[i]
        elif fitness[i] == s[1]:
            new[1] = mutated[i]
    # 新种群累计适应度
    cumfitness = get_cumfitness(fitness)
    # 进行第二次轮盘赌找后8个
    new_selected = select(cumfitness, mutated)
    for i in range(2, len(mutated)):
        new[i] = new_selected[i]
    # print(new)
    return new
