# -*-coding:utf-8 -*-
import numpy as np
from GA_details import species_origin, decode, get_fitness, get_cumfitness, select, crossover, mutate, new_select

# Step0:编码方案
# 个体数目
num = 10
# 当前进化代数
t = 0
# 染色体长度1
length1 = 18
# 染色体长度2
length2 = 15
# 染色体长度：18+15
length = length1 + length2
# 交叉概率
pc = 0.6
# 变异概率
pm = 0.01
# 总进化代数
sum1 = 50

population = species_origin(num, length)  # 初始化种群

# 进化过程
for i in range(sum1):
    # x1,x2
    domain = decode(population, length1, length2)
    # 种群适应度
    fitness = get_fitness(domain)
    # 种群累积适应度概率
    cumfitness = get_cumfitness(fitness)
    # 轮盘赌选出配对个体
    selected = new_select(cumfitness,mutated, length1, length2)
    # 两点交叉
    crossed = crossover(selected, pc)
    # 变异
    mutated = mutate(crossed, pm)
    # 更新下一代
    population = new_select(cumfitness,mutated, length1, length2)
domain = decode(population, length1, length2)
fitness = get_fitness(domain)
print("最终结果：", fitness[0])
