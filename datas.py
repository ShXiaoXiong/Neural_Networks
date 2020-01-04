import numpy
import matplotlib.pyplot as plt

data_file=open('train.csv','r')
data_list=data_file.readlines()#转换为列表
data_file.close()

all_values=data_list[245].split(',')

image_array=numpy.asfarray(all_values[1:]).reshape((28,28))#原图就是28*28
plt.imshow(image_array,cmap='Blues',interpolation='None')
plt.show()

