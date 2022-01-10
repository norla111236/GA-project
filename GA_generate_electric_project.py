# -*- coding: utf-8 -*-
"""
Created on Mon Jun 14 12:37:25 2021

@author: User
"""

import random
import pandas as pd
import numpy as np
from deap import base, creator, tools
import sys
import copy
import matplotlib.pyplot as plt 
import random 
from scipy.spatial import Delaunay

population_size = 100

def find_neighbors(pindex, triang):
  neighbors = list()
  for simplex in triang.vertices:
    if pindex in simplex:
      neighbors.extend([simplex[i] for i in range(len(simplex)) if simplex[i] != pindex])
  return list(set(neighbors))

def delaunay():
  x = [] 
  y = []
  #隨機產生座標
  for a in range(0,population_size): 
    i = random.randint(0,population_size) 
    j = random.randint(0,population_size) 
    x.append(i)
    y.append(j)

  points = np.array(list(zip(x,y)))

  #畫成Delaunay圖
  tri = Delaunay(points)
  plt.triplot(points[:,0], points[:,1], tri.simplices)
  plt.plot(points[:,0], points[:,1], 'o')
  plt.show()
  # print(tri.points)
  return points, tri


points, tri = delaunay()
#找鄰居的index
NB_index = []
for i in range(len(points)):
  NB_index.append(find_neighbors(i,tri))
# print(NB_index)
# print(len(points))

#%%
def delaunay_result(points, station):
  
  tri = Delaunay(points)
  plt.triplot(points[:,0], points[:,1], tri.simplices)
  charge = []
  no_charge = []
  for idx, i in enumerate(station):
     # print(points[idx])
     if i == 1:
         charge.append(points[idx].tolist())
     else:
         no_charge.append(points[idx].tolist())
  charge = np.array(charge)
  no_charge = np.array(no_charge)

  plt.plot(charge[:,0], charge[:,1], 'o', color='k')
  plt.plot(no_charge[:,0], no_charge[:,1], 'o', color='r')
  plt.show()
  # print(tri.points)
  return points, tri

def avg_curve(averge):
    plt.title("Mean Average")
    plt.xlabel("generation")
    plt.ylabel("average")
    plt.plot(averge, label = 'mean average')
    plt.show
    
#%%
initial = np.empty((population_size,7))
for idx, i in enumerate(initial):
    i[0] = random.randrange(5) #s:0-4
    i[1] = abs(int(random.gauss(mu = 10000, sigma = 3500))) #a_population
    i[2] = i[1] * (1/10) *(random.randrange(1, 1000) / 1000) #a_traffic
    i[3] = random.randrange(1, 100) #a_time
    i[4] = i[1] * 5 * random.randrange(2, 10) / 10 #a_social
    i[5] = int(i[1] * 0.0055) #cost_area(million US dollar)
    i[6] = random.randrange(100, 200) #cost_per_charger
    
initial = pd.DataFrame(initial, columns = ['s','a_population','a_traffic','a_time','a_social','cost_area(million US dollar)','cost_per_charger'])
initial['neighbor'] = NB_index

initial.to_csv(r'population.csv', index = True)




#%%
#計算fitness

def fitness_fun(population):
  global omega_p, omega_tr, omega_t, omega_s, omega_a, omega_c, config
  npindiv = np.array(population)
  result = sum(config[npindiv==1]['a_population']*omega_p + config[npindiv==1]['a_traffic']*omega_tr + config[npindiv==1]['a_time']*omega_t + config[npindiv==1]['a_social']*omega_s - config[npindiv==1]['cost_area(million US dollar)']*omega_a - config[npindiv==1]['cost_per_charger']*omega_c*config[npindiv==1]['s'])
  return result


def create_toolbox(num_bits):
    # 創建類型
    creator.create("FitnessMax", base.Fitness, weights=(1.0, ))
    creator.create("Individual", list, fitness=creator.FitnessMax)   # 定義Individual為list(FitnessMax)
    
    # Initialization
    toolbox = base.Toolbox()
    
    toolbox.register("attr_bool", random.randint, 0, 1)   # 0/1
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, num_bits)   # individual: list, [0/1]*45 次
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)   # population: list(individual)
    toolbox.register("evaluate", fitness_fun)
    toolbox.register("mate", tools.cxTwoPoint)   # 某兩點 bit 內容交換
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)   # 布林值翻轉函式
    toolbox.register("select", tools.selTournament, tournsize=3)   # 三個中選最好的 (隨機選擇, 執行k次))
    return toolbox




if __name__ == "__main__":
  config = pd.read_csv('population.csv', header=0)
  # config = config[:-1]
  
  neighbor = config['neighbor'].apply(lambda x: list(map(str, x[1:-1].split(','))))
  
  config = config.iloc[:, :-1].astype('float32')
  
  num_bits = len(config)   # n 個 0/1 隨機數 (選中那些 (P, S))
  toolbox = create_toolbox(num_bits)
  # random.seed(100)
  omega_p = 0.4
  omega_tr = 0.3
  omega_t = 0.2
  omega_s = 0.1
  omega_a = 0.5
  omega_c = 0.5

  population = toolbox.population(n=200)   # C
  CXPB, MUTPB = 0.3, 0.05   # cross prob / mutate prob
  NGEN = 20   # num of generation
  n = 1 # depth level of neighbors
  graph_num = 1   # subgraph
  print('\nEvolution process starts')

  fitnesses = list(map(toolbox.evaluate, population))
  
  print('fitnesses: ', fitnesses)
  fit_normalize = lambda x: (x-min(fitnesses)) / (max(fitnesses)-min(fitnesses)+0.0000000001)
  fitnesses = [fit_normalize(x) for x in fitnesses]
  print('fitnesses: ', fitnesses)
  for ind, fit in zip(population, fitnesses):
      ind.fitness.values = (fit, )
  print('\nEvaluated', len(population), 'individuals')
  
  avg_list = []
  for g in range(NGEN):
      print("\n- Generation", g)
      ### Select the next generation individuals
      offspring = toolbox.select(population, k=len(population)) #len(population)# 三取一, 取人口數個

      offspring = list(map(toolbox.clone, offspring)) # 複製樣品
    
      addChild = []
      for child1, child2 in zip(offspring[::2], offspring[1::2]): # 分兩群
          if random.random() < CXPB:
            g_centroid = set()
            for i in range(graph_num):
              centroid = random.randint(0,len(config)-1) #隨機產生中心點，其鄰居交換
              while centroid in g_centroid:
                centroid = random.randint(0,len(config)-1)
              g_centroid.add(centroid)
              toChange = neighbor[centroid] + [centroid]
              for i in toChange:
                i = int(i)
                temp = child1[i]
                child1[i] = child2[i]
                child2[i] = temp

          addChild.append(child1)
          addChild.append(child2)
   

      offspring = addChild
      ### Apply mutation on the offspring
      for mutant in offspring:                                # 基因突變
          if random.random() < MUTPB:
            toolbox.mutate(mutant)
            del mutant.fitness.values
      

      invalid_ind = [ind for ind in offspring if not ind.fitness.valid] #剔除無效  
      # 更新 offspring fitnesses
      fitnesses = list(map(toolbox.evaluate, invalid_ind))
      
      fitnesses = [fit_normalize(x) for x in fitnesses]
  
      for ind, fit in zip(invalid_ind, fitnesses):
          ind.fitness.values = (fit,)
      print('Evaluated', len(offspring), 'individuals')

      population[:] = offspring
      fits = [ind.fitness.values[0] for ind in population]
      # 計算平均值及標準偏差
      length = len(population) 
      mean = sum(fits) / length
      sum2 = sum(x*x for x in fits)
      std = abs(sum2 / length - mean**2)**0.5
      avg_list.append(round(mean, 2))
      print('Min =', min(fits), ', Max =', max(fits))
      print('Average =', round(mean, 2), ', Standard deviation =', round(std, 2))

  print("\n- Evolution ends")
  best_ind = tools.selBest(population, 1)[0]
  print('\nBest individual:\n', best_ind)
  print('\nNumber of ones:', sum(best_ind))
  
  delaunay_result(points, best_ind)
  avg_curve(avg_list)
  