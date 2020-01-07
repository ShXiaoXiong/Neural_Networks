#神经网络框架
#至少3部分
#设定数量：输入层、隐藏层、输出层：
#训练：学习给定的训练集样本后，优化权重
#查询功能

import numpy #数组功能
import scipy.special #激活函数
import matplotlib.pyplot as plt #可视化

class Neuralnetworks:
    def __init__(self,inputnodes,hiddennodes,outputnodes,learningrate):
        #待传递节点数量
        self.inodes=inputnodes#根据input的量决定
        self.hnodes=hiddennodes#隐藏层强制神经网络进行总结和归纳，代表其能力
        self.onodes=outputnodes#这个案例中，输出的是具体的数字识别
        #待传递学习率
        self.lr=learningrate #过低的学习率限制了步长、限制了梯度下降发生的速度，对性能造成了损害。过高的学习率会导致在梯度下降过程中超调及来回跳动
        #设定初始连接权重：使用正态概率分布采样权重，也可以使用其他更为复杂的方法
        self.ihw=numpy.random.normal(0.0,pow(self.hnodes,-0.5),(self.hnodes,self.inodes))
        self.how=numpy.random.normal(0.0,pow(self.onodes,-0.5),(self.onodes,self.hnodes))
        #设定激活函数:使用sigmoid函数，一个常用的非线性激活函数，接受任何数值，输出0到1之间的某个值，但不包含0和1
        #Sigmoid的output不是0均值（即zero-centered），导致反向传递时可能梯度爆炸或者梯度消失
        # 深度学习往往需要大量时间来处理大量数据，模型的收敛速度是尤为重要的。
        # 所以，总体上来讲，训练深度学习网络尽量使用zero-centered数据 (可以经过数据预处理实现) 和zero-centered输出。所以要尽量选择输出具有zero-centered特点的激活函数以加快模型的收敛速度。
        self.activation_function=lambda x:scipy.special.expit(x)
     
        pass

    def query(self,inputs_list):
        #计算输出的过程 
        inputs=numpy.array(inputs_list,ndmin=2).T#传递列表，转换为二维数组，转置
        
        hidden_inputs=numpy.dot(self.ihw,inputs)#点乘
        hidden_outputs=self.activation_function(hidden_inputs)#使用激活函数

        final_inputs=numpy.dot(self.how,hidden_outputs)#点乘
        final_outputs=self.activation_function(final_inputs)#使用激活函数

        return final_outputs#如果不写return，会返回一个None对象

    def train(self,inputs_list,targets_list):
        #反馈调节权重的过程/反向传播误差——告知如何优化权重
        
        #完全相同的计算，因此在循环中要重写
        inputs=numpy.array(inputs_list,ndmin=2).T#传递列表，转换为二维数组，转置
        
        hidden_inputs=numpy.dot(self.ihw,inputs)#点乘
        hidden_outputs=self.activation_function(hidden_inputs)#使用激活函数

        final_inputs=numpy.dot(self.how,hidden_outputs)#点乘
        final_outputs=self.activation_function(final_inputs)#使用激活函数

        targets=numpy.array(targets_list,ndmin=2).T#传递列表，转换为二维数组
        
        output_errors=targets-final_outputs#计算误差

        #隐藏层误差
        hidden_errors=numpy.dot(self.how.T,output_errors)#点乘
        #反向传递，更新how权重
        self.how += self.lr * numpy.dot((output_errors * final_outputs* (1-final_outputs)),numpy.transpose(hidden_outputs))#点乘
        #反向传递，更新ihw权重
        self.ihw += self.lr * numpy.dot((hidden_errors * hidden_outputs* (1-hidden_outputs)),numpy.transpose(inputs))#点乘
        pass


#构建神经网络实例
n=Neuralnetworks(784,100,10,0.3)

#加载训练数据并转为列表
data_file=open('train.csv','r')#每个新行表示一个新的数据库行，每个数据库行由一个或多个以逗号分隔的字段组成
data_list=data_file.readlines()#转换为列表
data_file.close()


scoreboard=[]
accuracies=[]
#训练并对比
epoch=2#增加世代
for e in range(epoch):
    for record in data_list[1:1000]:
        #######收敛数据
        all_values=record.split(',')#指定分隔符‘，’，对字符串进行切片，返回一个列表   
        inputs=(numpy.asfarray(all_values[1:])/255.0 *0.99)+0.01#将输入值进行了预先处理：收敛：将输入值收缩到(0,1)之间。asfarray转换为浮点型数组
    

        #计算过程
        inputs=numpy.array(inputs,ndmin=2).T#传递列表，转换为二维数组，转置

        hidden_inputs=numpy.dot(n.ihw,inputs)#点乘
        hidden_outputs=n.activation_function(hidden_inputs)#使用激活函数

        final_inputs=numpy.dot(n.how,hidden_outputs)#点乘
        final_outputs=n.activation_function(final_inputs)#使用激活函数


        #对比
        networks_label=numpy.argmax(final_outputs)#取出最大值对应的索引值
        correct_label=int(all_values[0])#答案是第一个值
        #记录结果1
        if networks_label==correct_label:
            scoreboard.append(1)
        else:
            scoreboard.append(0)
        #记录结果2
        accuracy= sum(scoreboard)/len(scoreboard)
        accuracies.append(accuracy)
    
        #人为设定一个targets值
        targets=numpy.zeros(10)+0.01 
        targets[int(all_values[0])]=0.99

        #训练过程
   
        targets=numpy.array(targets,ndmin=2).T#传递列表，转换为二维数组
        
        output_errors=targets-final_outputs#计算误差

        #隐藏层误差
        hidden_errors=numpy.dot(n.how.T,output_errors)#点乘
        #反向传递，更新how权重
        n.how += n.lr * numpy.dot((output_errors * final_outputs* (1-final_outputs)),numpy.transpose(hidden_outputs))#点乘
        #反向传递，更新ihw权重
        n.ihw += n.lr * numpy.dot((hidden_errors * hidden_outputs* (1-hidden_outputs)),numpy.transpose(inputs))#点乘
        pass


#最终结果可视化        
import matplotlib.pyplot as plt

y=accuracies
x=range(len(accuracies))
plt.plot(x,y,label='Frist line',linewidth=3,color='r',marker='o', markerfacecolor='blue',markersize=12) 
plt.show()