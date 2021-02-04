import random
import math
from autogl.module.hpo.base import BaseHPOptimizer
from . import register_hpo
import numpy as np
from torch.utils.data import random_split
import torch
import geatpy as ea
import requests
import time
import os
from tqdm import tqdm
from .evolution.MyProblem import MyProblem_single as MyProblem
from ..train.evaluate import Acc

"""加入自带的优化方法作为种群的初始值"""
from . import TpeAdvisorHPO
from . import MocmaesAdvisorChoco
from . import AnnealAdvisorHPO
from . import RandAdvisor
from . import HPO_DICT


def reset_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


@register_hpo("soea")
class SOEAOptimizer(BaseHPOptimizer): 
    # Get essential parameters at initialization
    def __init__(self, *args, **kwargs):
        self.max_gen = kwargs.get("max_gen", 5)
        self.pps = kwargs.get("pps", 30)
        self.subset_rate = kwargs.get("subset_rate", 10)
        self.need_split_dataset = kwargs.get("need_split_dataset", True)
        self.soea_method = kwargs.get("soea_method", None)
        self.seed = kwargs.get("seed", 2021)
        self.cata_dict = {}
        self.single_choice_para = {}
        self.best_perf = float('inf')
        self.best_trainer = None
        self.bp = None
        self.best_para = None


    # The most important thing you should do is completing optimization function
    def optimize(self, trainer, dataset, time_limit=None, memory_limit=None, seed = 2021):
        SOEA_DICT = {
            "soea_DE_best_1_bin_templet": ea.soea_DE_best_1_bin_templet,
            "soea_DE_best_1_L_templet": ea.soea_DE_best_1_L_templet,
            "soea_DE_rand_1_bin_templet": ea.soea_DE_rand_1_bin_templet,
            "soea_DE_rand_1_L_templet": ea.soea_DE_rand_1_L_templet,
            "soea_DE_targetToBest_1_bin_templet": ea.soea_DE_targetToBest_1_bin_templet,
            "soea_DE_targetToBest_1_L_templet": ea.soea_DE_targetToBest_1_L_templet,
            "soea_DE_currentToBest_1_bin_templet": ea.soea_DE_currentToBest_1_bin_templet,
            "soea_DE_currentToBest_1_L_templet": ea.soea_DE_currentToBest_1_L_templet,
            "soea_DE_currentToRand_1_templet": ea.soea_DE_currentToRand_1_templet,
            "soea_ES_1_plus_1_templet": ea.soea_ES_1_plus_1_templet,
            "soea_ES_miu_plus_lambda_templet": ea.soea_ES_miu_plus_lambda_templet,
            "soea_EGA_templet": ea.soea_EGA_templet,
            "soea_SEGA_templet": ea.soea_SEGA_templet,
            "soea_SGA_templet": ea.soea_SGA_templet,
            "soea_studGA_templet": ea.soea_studGA_templet,
        }
        def easy_decode(naiveSpace):
            res = []
            for tp in naiveSpace:
                tv = naiveSpace[tp]
                # if current_space[tp]["type"] is "double":
                #     if self._numerical_map[tp]["scalingType"] is "log":
                #         naiveSpace[tp] = math.log(tv)
                # elif self._numerical_map[tp]["type"] is "NUMERICAL_LIST":
                #     if self._numerical_map[tp]["scalingType"] is "log":
                #         naiveSpace[tp] = np.log(tv)[0]
                # elif self._numerical_map[tp]["type"] in ["CATEGORICAL", "DISCRETE"]:
                #     if self._numerical_map[tp]["type"] is "CATEGORICAL":
                #         naiveSpace[tp] = int(self._category_map[tp].index(tv))
                #     else:
                #         naiveSpace[tp] = int(self._discrete_map[tp].index(tv))
                if tp == "hidden_":
                    res.append(math.log(tv[0]))
                elif tp == "max_epoch":
                    res.append(tv)
                elif tp == "dropout":
                    res.append(tv)
                elif tp == "act":
                    res.append(int(self._category_map[tp + '_'].index(tv)))
                elif tp == "early_stopping_round":
                    res.append(tv)
                elif tp == "lr":
                    res.append(math.log(tv))
                elif tp == "weight_decay":
                    res.append(math.log(tv))
                else:
                    print("WRONG!!", tp)
            return res

        # 1. Get the search space from trainer.
        space = trainer.hyper_parameter_space
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
        def fn(dset, para):
            reset_seed(self.seed)
            current_trainer = trainer.duplicate_from_hyper_parameter(para)
            current_trainer.train(dset)
            metrics, self.is_higher_better = current_trainer.get_valid_score(return_major = False)
            # For convenience, we change the score which is higher better to negative, then we should only minimize the score.
            for i, is_higher_better in enumerate(self.is_higher_better):
                if is_higher_better:
                    metrics[i] = -metrics[i]
            acc, loss = metrics[0],  metrics[1]
            if acc < self.best_perf:
                self.best_perf = acc
                self.best_trainer = current_trainer
                self.best_para = para
            return current_trainer, acc, loss
 
        VarTypes = []   #类型
        codes = [] #决策变量的编码格式，全部使用二进制格雷码
        precisions = [] #优化到小数点后6位
        scales = [] #算术刻度
        lb = [] #决策变量下界
        ub = [] #决策变量上界
        lbin = []   #决策变量下边界
        ubin = []   #决策变量下边界
        def ea_init():
            self.name_record = []
            for para in current_space:
                if para["type"] == "DOUBLE":
                    lb.append(para["minValue"])
                    ub.append(para["maxValue"])
                    VarTypes.append(0)
                    codes.append(1)
                    precisions.append(6)
                    scales.append(0)
                    lbin.append(1)
                    ubin.append(1)
                    self.name_record.append(para["parameterName"])

                elif para["type"] == "INTERGER":
                    lb.append(para["minValue"])
                    ub.append(para["maxValue"])
                    VarTypes.append(1)
                    codes.append(1)
                    precisions.append(0)
                    scales.append(0)     
                    lbin.append(1)
                    ubin.append(1)             
                    self.name_record.append(para["parameterName"])

                elif para["type"] == "DISCRETE" or para["type"] == "CATEGORICAL":
                    feasible_points = para["feasiblePoints"].split(",")
                    if len(feasible_points) == 1:   #出现一个选项的情况
                        self.single_choice_para[para["parameterName"]] = feasible_points[0]
                        continue
                    self.cata_dict[para["parameterName"]] = feasible_points
                    lb.append(0)
                    ub.append(len(feasible_points))
                    VarTypes.append(1)
                    codes.append(1)
                    precisions.append(0)
                    scales.append(0)
                    lbin.append(1)
                    ubin.append(0)
                    self.name_record.append(para["parameterName"])
            

        
        ea_init()
        problem = MyProblem(fn = fn, dataset = dataset, _decode_para = self._decode_para,
                            name_record = self.name_record, single_choice_para = self.single_choice_para,
                            Dim = len(self.name_record), kwargs = {
                                "varTypes": VarTypes,
                                "lb": lb,
                                "ub": ub,
                                "lbin": lbin,
                                "ubin": ubin
                            })
        """==============================种群设置==========================="""
        Encoding = 'RI'
        NIND = self.pps
        Field = ea.crtfld(Encoding, problem.varTypes, problem.ranges,problem.borders) # 创建区域描述器

        """==========================先验种群训练加入========================="""
        PreHyperOptim_ParaList = [
            {"name": "tpe", "max_evals": 50},
            {"name": "anneal", "max_evals": 50},
            {"name": "random", "max_evals": 50}
        ]
        self.priori_para = []
        for hyperOptimPara in PreHyperOptim_ParaList:
            temp_hpo_module = HPO_DICT[hyperOptimPara["name"]](**hyperOptimPara)
            _, best_temp_para = temp_hpo_module.optimize(trainer, dataset, time_limit = 3600)
            for itemsname in list(best_temp_para.keys()):
                if itemsname + '_' in self.single_choice_para:    #不参与编码进化
                    best_temp_para.pop(itemsname)
            
            the_para_dict = {}
            for nameRecord in self.name_record:
                if nameRecord[-1] is '_':
                    the_para_dict[nameRecord[:-1]] = best_temp_para[nameRecord[:-1]]
                elif nameRecord[-2] is '_':
                    the_para_dict[nameRecord[:-1]] = best_temp_para[nameRecord[:-2]]
                else:
                    the_para_dict[nameRecord] = best_temp_para[nameRecord]
            the_para = easy_decode(the_para_dict)
            for it in range(5):     #根据先验随机扰动
                tp = the_para.copy()
                for i, paraitem in enumerate(tp):
                    if isinstance(paraitem, float):
                        tp[i] = random.uniform(paraitem * 0.95, paraitem * 1.05)
                self.priori_para.append(tp)
            self.priori_para.append(the_para)    #按照顺序排一下防止错误


        if not os.path.exists("./soea_pri_result"):
            os.mkdir("./soea_pri_result")
        f = open("./soea_pri_result/SOEAresult_{}.txt".format(time.strftime('%Y%m%d_%H%M')), mode='w')
        for soeaname in list(SOEA_DICT.keys()):
            reset_seed(self.seed)
            self.best_perf = float('inf')
            f.write('Method: {}\n'.format(soeaname))
            print('Method: {}'.format(soeaname))
            f.write('   Gen: {} , Pop: {}\n'.format(self.max_gen, self.pps))
            population = ea.Population(Encoding, Field, NIND) #实例化种群对象（此时种群还没被真正初始化，仅仅是生成一个种群对象）
            """===========================算法参数设置=========================="""
            myAlgorithm = SOEA_DICT[soeaname](problem, population) #实例化一个算法模板对象
            myAlgorithm.MAXGEN = self.max_gen # 最大进化代数
            # myAlgorithm.mutOper.Pm = 0.2
            # myAlgorithm.recOper.XOVR = 0.9 # 设置交叉概率

            myAlgorithm.logTras = 1 # 设置每隔多少代记录日志，若设置成0则表示不记录日志
            myAlgorithm.verbose = True # 设置是否打印输出日志信息
            myAlgorithm.drawing = 0 #设置绘图方式（0：不绘图；1：绘制结果图；2：绘制目标空间过程动画；3：绘制决策空间过程动画）
            """==========================调用算法模板进行种群进化==============="""
            tik = time.perf_counter()
            [BestIndi, population] = myAlgorithm.run() # 执行算法模板，得到最优个体以及最后一代种群
            tok = time.perf_counter()
            print("Time Cost:", tok - tik)
            predict_result = (self.best_trainer.predict_proba(dataset, mask="test").cpu().numpy())
            f.write("       Val acc: {}, Test acc: {}\n\n".format(-self.best_perf, Acc.evaluate(predict_result, dataset.data.y[dataset.data.test_mask].numpy())))
        f.close()
        url = "https://sc.ftqq.com/SCU77368T7bdf9479ae8b1f9c61e63aa56f9c481c5fef4809e7d6c.send"
        urldata = {"text": "Soea with pri训练完成，快来康康吧"}
        requests.post(url, urldata)
        return self.best_trainer, self.best_para
