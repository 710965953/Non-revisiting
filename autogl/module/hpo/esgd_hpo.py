import random
from autogl.module.hpo.base import BaseHPOptimizer
from . import register_hpo
import numpy as np
from torch.utils.data import random_split
import geatpy as ea
import time
import math
from tqdm import tqdm
from copy import deepcopy

"""
要提供各个参数的上下界，然后还有K（种群进化的代数）、Ks(SGD steps)、Kv(每次进化代数)、m(精英数量)、pps(parent population size)、ops(offspring population size)
"""

@register_hpo("esgd")
class ESGDOptimizer(BaseHPOptimizer):
    # Get essential parameters at initialization
    def __init__(self, *args, **kwargs):
        self.max_gen = kwargs.get("max_gen", 5)
        self.sgd_steps = kwargs.get("sgd_steps", 40)
        self.pps = kwargs.get("pps", 30)
        self.ops = kwargs.get("ops", 20)
        self.elite_num = kwargs.get("elite_num", 10)
        self.subset_rate = kwargs.get("subset_rate", 10)
        self.need_split_dataset = kwargs.get("need_split_dataset", True)
        self.cata_dict = {}
        self.single_choice_para = {}


    # The most important thing you should do is completing optimization function
    def optimize(self, trainer, dataset, time_limit=None, memory_limit=None):
        # 1. Get the search space from trainer.
        space = trainer.hyper_parameter_space# + trainer.model.hyper_parameter_space
        # optional: use self._encode_para (in BaseOptimizer) to pretreat the space
        # If you use _encode_para, the NUMERICAL_LIST will be spread to DOUBLE or INTEGER, LOG scaling type will be changed to LINEAR, feasible points in CATEGORICAL will be changed to discrete numbers.
        # You should also use _decode_para to transform the types of parameters back.
        current_space = self._encode_para(space)
        #dataset继承自pytorch的dataset,简单的测试，直接使用random_split来弄
        # if self.need_split_dataset:
        #     subset_num = int((self.subset_rate / 100.0) * len(dataset))
        #     dataset, _ = random_split(dataset = dataset, lengths = [subset_num, len(dataset) - subset_num])
            #分割完成，现在就是使用数据集的一部分用来给超参数训练，而不是全部都来，太慢了

        # 2. Define your function to get the performance.
        self.best_perf = 1000
        def fn(dset, para, oripara = None):
            current_trainer = trainer.duplicate_from_hyper_parameter(para)
            current_trainer.train(dset)
            acc, self.is_higher_better = current_trainer.get_valid_score(dset)
            # For convenience, we change the score which is higher better to negative, then we should only minimize the score.
            if self.is_higher_better:
                acc = -acc
            if acc < self.best_perf:
                self.best_perf = acc
            return current_trainer, acc

        def decode_Chrom_for_bestind(bestgene):
            Phen = ea.bs2ri(np.array([bestgene]), FieldD)
            row, col = Phen.shape
            hps_ = {}
            for j in range(col):
                hps_[self.name_record[j]] = Phen[[0],[j]]
            hps_.update(self.single_choice_para) #只有一个选项的哈批东西
            #每一个项都是一组超参数
            t_ObjV = []
            for t in list(hps_.keys()):
                if isinstance(hps_[t], list):
                    hps_[t] = hps_[t][0]
                elif isinstance(hps_[t], np.ndarray):
                    hps_[t] = hps_[t].tolist()[0] #转成正常的格式
            para_for_trainer, para_for_hpo = self._decode_para(hps_)
            b_trainer = trainer.duplicate_from_hyper_parameter(para_for_trainer)
            return b_trainer, para_for_trainer
        
        def decode_Chrom(TheChrom):
            hpsGroup = []
            Phen = ea.bs2ri(TheChrom, FieldD)
            row, col = Phen.shape
            for i in range(row):
                hps = {}
                for j in range(col):
                    hps[self.name_record[j]] = Phen[[i],[j]]
                hps.update(self.single_choice_para) #只有一个选项的哈批东西
                hpsGroup.append(hps)
            #每一个项都是一组超参数
            return hpsGroup

        def get_ObjV(hpsGroup):
            t_ObjV = []
            for hps in hpsGroup:
                #这里的每一个hps都是字典，一组超参数
                for t in hps:
                    if isinstance(hps[t], list):
                        hps[t] = hps[t][0]
                    elif isinstance(hps[t], np.ndarray):
                        hps[t] = hps[t].tolist()[0] #转成正常的格式
                para_for_trainer, para_for_hpo = self._decode_para(hps)

                _, perf = fn(dataset, para_for_trainer, oripara=hps)
                t_ObjV.append([perf])
            return np.array(t_ObjV)

        VarTypes = []   #类型
        codes = [] #决策变量的编码格式，全部使用二进制格雷码
        precisions = [] #优化到小数点后6位
        scales = [] #算术刻度
        def ea_init():
            bound = []
            self.name_record = []
            # varTypes = []
            borders = []
            for para in current_space:
                if para["type"] == "DOUBLE":
                    bound.append([para["minValue"], para["maxValue"]])
                    VarTypes.append(0)
                    codes.append(1)
                    precisions.append(6)
                    scales.append(0)
                    borders.append([1,1])
                    self.name_record.append(para["parameterName"])

                elif para["type"] == "INTERGER":
                    bound.append([para["minValue"], para["maxValue"]])
                    VarTypes.append(1)
                    codes.append(1)
                    precisions.append(0)
                    scales.append(0)     
                    borders.append([1,1])               
                    self.name_record.append(para["parameterName"])

                elif para["type"] == "DISCRETE" or para["type"] == "CATEGORICAL":
                    feasible_points = para["feasiblePoints"].split(",")

                    if len(feasible_points) == 1:   #出现一个选项的情况
                        self.single_choice_para[para["parameterName"]] = feasible_points
                        continue

                    self.cata_dict[para["parameterName"]] = feasible_points
                    bound.append([0, len(feasible_points)])
                    VarTypes.append(1)
                    codes.append(1)
                    precisions.append(0)
                    scales.append(0)
                    borders.append([1,0])
                    self.name_record.append(para["parameterName"])
                    
            self.hpo_num = len(bound)
            # for i in range(self.hpo_num):
            #     print(self.name_record[i], bound[i])
            self.ranges = np.array(bound).T
            self.borders = np.array(borders).T

        ea_init()
        """==========================染色体编码设置========================="""
        Encoding = 'BG' #二进制格雷码
        VarTypes = np.array(VarTypes) #先当他都是连续型
        FieldD = ea.crtfld(Encoding, VarTypes, self.ranges, self.borders, precisions, codes, scales)
        """=========================遗传算法参数设置========================"""
        NIND = self.pps
        maxormins = np.array([1]) #表示目标函数是最小化
        selectStyle = 'etour' # 采用精英锦标赛
        recStyle = 'xovdp' # 采用两点交叉
        mutStyle = 'mutbin' # 二进制变异算子
        Lind = int(np.sum(FieldD[0, :])) # 计算染色体长度
        pc = 0.9 # 交叉概率
        pm = 1/Lind # 变异概率
        obj_trace = np.zeros((self.max_gen, 2)) # 定义目标函数值记录器
        var_trace = np.zeros((self.max_gen, Lind)) #染色体记录器，记录历代最优个体的染色体
        """=========================开始遗传算法进化========================"""
        Chrom = ea.crtpc(Encoding,NIND, FieldD) # 生成种群染色体矩阵
        ObjV = get_ObjV(decode_Chrom(Chrom)) # 先解码，再计算初始种群个体的目标函数值
        best_ind = np.argmin(ObjV) # 计算当代最优个体的序号
        # 开始进化

        best_gene, best_perf = None, None
        for gen in tqdm(range(self.max_gen)):
            FitnV = ea.ranking(maxormins * ObjV) #根据目标函数大小分配适应度值
            SelCh = Chrom[ea.selecting(selectStyle,FitnV,NIND),:] # 选择
            SelCh = ea.recombin(recStyle, SelCh, pc) # 重组
            SelCh = ea.mutate(mutStyle, Encoding, SelCh, pm) # 变异
            # 把父代精英个体与子代的染色体进行合并，得到新一代种群
            Chrom = np.vstack([Chrom[best_ind, :], SelCh])
            ObjV = get_ObjV(decode_Chrom(Chrom)) # 先解码，再计算初始种群个体的目标函数值
            # 记录
            best_ind = np.argmin(ObjV) # 计算当代最优个体的序号
            obj_trace[gen,0]=np.sum(ObjV)/ObjV.shape[0]
            #记录当代种群的目标函数均值
            obj_trace[gen,1] = ObjV[best_ind] #记录当代种群最优个体目标函数值
            # print('In gen {}, best obj = {}'.format(gen, ObjV[best_ind]))
            var_trace[gen,:] = Chrom[best_ind,:] #记录当代种群最优个体的染色体
            if not best_perf or ObjV[best_ind] < best_perf:
                best_perf = ObjV[best_ind]
                best_gene = Chrom[best_ind,:]
        print("Best_perf = ", best_perf)
        best_trainer, para_for_trainer = decode_Chrom_for_bestind(best_gene)
        best_trainer.train(dataset)
        print('ACC = ', best_trainer.get_valid_score()[0]) 
        return best_trainer, para_for_trainer