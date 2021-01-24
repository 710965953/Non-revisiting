# -*- coding: utf-8 -*-
"""MyProblem.py"""
import numpy as np
import geatpy as ea
class MyProblem(ea.Problem): # 继承Problem父类
    def __init__(self, fn, dataset,  _decode_para,
                 name_record, single_choice_para, 
                 Dim, **kwargs):#Dim（决策变量维数）
        name = 'HPO' # 初始化name（函数名称，可以随意设置）
        M = 1 # 初始化M（目标维数）
        maxormins = [1] # 初始化目标最小最大化标记列表，1：min；-1：max
        # 初始化决策变量类型，0：连续；1：离散
        kwargs = kwargs["kwargs"]
        varTypes = kwargs.get("varTypes", [0] * Dim)
        # 决策变量下界
        lb = kwargs.get("lb", None)
        # 决策变量上界
        ub = kwargs.get("ub", None)
        # 决策变量下边界
        lbin = kwargs.get("lbin", None)
        # 决策变量上边界
        ubin = kwargs.get("ubin", None)
        
        self.name_record = name_record
        self.single_choice_para = single_choice_para
        self.dataset = dataset
        self.fn = fn
        self._decode_para = _decode_para

        # 调用父类构造方法完成实例化s
        ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, lb,
                            ub, lbin, ubin)


    def aimFunc(self, pop): # 目标函数，pop为传入的种群对象
        hpsGroup = []
        Phen = pop.Phen # 得到决策变量矩阵
        # 计算目标函数值，赋值给pop种群对象的ObjV属性
        # pop.ObjV = 4*x1 + 2*x2 + x3
        # 采用可行性法则处理约束，生成种群个体违反约束程度矩阵
        row, col = Phen.shape
        for i in range(row):
            hps = {}
            for j in range(col):
                hps[self.name_record[j]] = Phen[[i],[j]]
            hps.update(self.single_choice_para) #只有一个选项的哈批东西
            hpsGroup.append(hps)
        #每一个项都是一组超参数
        t_ObjV = []
        for hps in hpsGroup:
            #这里的每一个hps都是字典，一组超参数
            for t in hps:
                if isinstance(hps[t], list):
                    hps[t] = hps[t][0]
                elif isinstance(hps[t], np.ndarray):
                    hps[t] = hps[t].tolist()[0] #转成正常的格式
            para_for_trainer, para_for_hpo = self._decode_para(hps)
            _, perf = self.fn(self.dataset, para_for_trainer)
            t_ObjV.append([perf])
        pop.ObjV = np.array(t_ObjV)
        # pop.CV = None
