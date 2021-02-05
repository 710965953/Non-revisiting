import os
import requests

current_dir = os.getcwd()
os.chdir(current_dir)
os.system("python {} >> {}".format('anneal_citeseer_acc.py','anneal_citeseer_acc.txt'))
os.system("python {} >> {}".format('anneal_citeseer_loss.py','anneal_citeseer_loss.txt'))
os.system("python {} >> {}".format('anneal_cora_acc.py','anneal_cora_acc.txt'))
os.system("python {} >> {}".format('anneal_cora_loss.py','anneal_cora_loss.txt'))
os.system("python {} >> {}".format('anneal_pubmed_acc.py','anneal_pubmed_acc.txt'))
os.system("python {} >> {}".format('anneal_pubmed_loss.py','anneal_pubmed_loss.txt'))

os.system("python {} >> {}".format('bayes_citeseer_acc.py','bayes_citeseer_acc.txt'))
os.system("python {} >> {}".format('bayes_citeseer_loss.py','bayes_citeseer_loss.txt'))
os.system("python {} >> {}".format('bayes_cora_acc.py','bayes_cora_acc.txt'))
os.system("python {} >> {}".format('bayes_cora_loss.py','bayes_cora_loss.txt'))
os.system("python {} >> {}".format('bayes_pubmed_acc.py','bayes_pubmed_acc.txt'))
os.system("python {} >> {}".format('bayes_pubmed_loss.py','bayes_pubmed_loss.txt'))

os.system("python {} >> {}".format('random_citeseer_acc.py','random_citeseer_acc.txt'))
os.system("python {} >> {}".format('random_citeseer_loss.py','random_citeseer_loss.txt'))
os.system("python {} >> {}".format('random_cora_acc.py','random_cora_acc.txt'))
os.system("python {} >> {}".format('random_cora_loss.py','random_cora_loss.txt'))
os.system("python {} >> {}".format('random_pubmed_acc.py','random_pubmed_acc.txt'))
os.system("python {} >> {}".format('random_pubmed_loss.py','random_pubmed_loss.txt'))

url = "https://sc.ftqq.com/SCU77368T7bdf9479ae8b1f9c61e63aa56f9c481c5fef4809e7d6c.send"
urldata = {"text": "gat-ori训练完成，快来康康吧"}
requests.post(url, urldata)