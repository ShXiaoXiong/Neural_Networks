#神经网络框架
#至少3部分
#设定数量：输入层、隐藏层、输出层：
#训练：学习给定的训练集样本后，优化权重
#查询功能

import numpy
import scipy.special

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
        self.how=numpy.random.rand(0.0,pow(self.onodes,-0.5),(self.onodes,self.hnodes))
        #激活函数
        self.activation_function=lambda x:scipy.special.expit(x)
     
        pass

    def train(self,inputs_list,targets_list):
        #反馈调节权重的过程/反向传播误差——告知如何优化权重
        
        #将输入转为2维数组
        inputs=numpy.array(inputs_list,ndmid=2).T
        targets=numpy.array(targets_list,ndmid=2).T

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
        inputs=numpy.array(inputs_list,ndmid=2).T
        
        hidden_inputs=numpy.dot(self.ihw,inputs)
        hidden_outputs=self.activation_function(hidden_inputs)

        final_inputs=numpy.dot(self.how,hidden_outputs)
        final_outputs=self.activation_function(final_inputs)

        return final_outputs

n=Neuralnetworks(3,3,3,0.3)
n.query([1,3,2])
