from GA_details import species_origin, decode, get_fitness, get_cumfitness, select, crossover, mutate, new_select,first_select
# 个体数目
num = 100
# 染色体长度：18+15
length = 33
# 交叉概率
pc = 0.6
# 变异概率
pm = 0.01
# 总进化代数
sum1 = 1000
for j in range(0,20):
# 初始化种群
    population = species_origin(num, length)
# 进化过程

    for i in range(sum1):
        # 求个体累计适应度
        cumfitness = get_cumfitness(get_fitness(decode(population)))
        # 轮盘赌选10个
        selected = first_select(cumfitness,population)
        #selected = select(cumfitness, population, len(population))
        # 两点交叉
        crossed = crossover(selected, pc)
        # 变异
        mutated = mutate(crossed, pm)
        # 更新下一代
        population = new_select(mutated)
        #if i>sum1/2:
        #   pm = pm/2
    s = sorted(get_fitness(decode(population)), reverse=True)
    print("最终结果：", round(s[0],4))
    # 38.8503
