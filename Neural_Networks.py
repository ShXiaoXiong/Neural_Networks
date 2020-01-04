#神经网络框架
#至少3部分
#设定数量：输入层、隐藏层、输出层：
#训练：学习给定的训练集样本后，优化权重
#查询功能

import numpy
import scipy.special
import matplotlib.pyplot as plt

class Neuralnetworks:
    def __init__(self,inputnodes,hiddennodes,outputnodes,learningrate):
        #节点数量
        self.inodes=inputnodes
        self.hnodes=hiddennodes
        self.onodes=outputnodes
        #学习率
        self.lr=learningrate
        #连接权重
        self.ihw=numpy.random.normal(0.0,pow(self.hnodes,-0.5),(self.hnodes,self.inodes))
        self.how=numpy.random.normal(0.0,pow(self.onodes,-0.5),(self.onodes,self.hnodes))
        #激活函数
        self.activation_function=lambda x:scipy.special.expit(x)
     
        pass

    def train(self,inputs_list,targets_list):
        #反馈调节权重的过程/反向传播误差——告知如何优化权重
        
        #将输入转为2维数组
        inputs=numpy.array(inputs_list,ndmin=2).T
        targets=numpy.array(targets_list,ndmin=2).T

        hidden_inputs=numpy.dot(self.ihw,inputs)
        hidden_outputs=self.activation_function(hidden_inputs)

        final_inputs=numpy.dot(self.how,hidden_outputs)
        final_outputs=self.activation_function(final_inputs)

        #计算误差
        output_errors=targets-final_outputs

        #隐藏层误差
        hidden_errors=numpy.dot(self.how.T,output_errors)
        #更新how权重
        self.how += self.lr * numpy.dot((output_errors * final_outputs* (1-final_outputs)),numpy.transpose(hidden_outputs))
        #更新ihw权重
        self.ihw += self.lr * numpy.dot((hidden_errors * hidden_outputs* (1-hidden_outputs)),numpy.transpose(inputs))

        pass

    def query(self,inputs_list):
        #计算输出的过程 
        inputs=numpy.array(inputs_list,ndmin=2).T
        
        hidden_inputs=numpy.dot(self.ihw,inputs)
        hidden_outputs=self.activation_function(hidden_inputs)

        final_inputs=numpy.dot(self.how,hidden_outputs)
        final_outputs=self.activation_function(final_inputs)

        return final_outputs

n=Neuralnetworks(784,100,10,0.3)#输入为像素位的数字、隐藏层强制神经网络进行总结和归纳，代表其能力、输出为数字

#加载训练数据并转为csv
data_file=open('train.csv','r')
data_list=data_file.readlines()#转换为列表
data_file.close()




scoreboard=[]
accuracies=[]
#训练并对比
for record in data_list[1:10000]:
    all_values=record.split(',')

    #先计算，对比   
    
    correct_label=int(all_values[0])

    inputs=(numpy.asfarray(all_values[1:])/255.0 *0.99)+0.01
    networks_output=n.query(inputs)

    networks_label=numpy.argmax(networks_output)#取出最大值对于的索引值
    
    if networks_label==correct_label:
        scoreboard.append(1)
    else:
        scoreboard.append(0)

    #训练，存在重复计算
    
    targets=numpy.zeros(10)+0.01 
    targets[int(all_values[0])]=0.99

    n.train(inputs,targets)

    accuracy= sum(scoreboard)/len(scoreboard)
    accuracies.append(accuracy)

    
    
import matplotlib.pyplot as plt

y=accuracies
x=range(1,10000)
plt.plot(x,y,label='Frist line',linewidth=3,color='r',marker='o', markerfacecolor='blue',markersize=12) 
plt.show()