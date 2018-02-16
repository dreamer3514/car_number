import matplotlib.pyplot as plt
import numpy as np
from numpy import exp, array, random, dot
import cv2
import sys
from os import listdir
#两层神经网络 暂时能识别数字
class TwoLayerNeuralNetwork(object):
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes
        np.random.seed(1)

        #随机的第一层权重
        self.weights_0_1 = np.random.normal(0.0, self.hidden_nodes**-0.5, (self.input_nodes, self.hidden_nodes))
        # n * 2

        #随机的第二层权重
        self.weights_1_2 = np.random.normal(0.0, self.output_nodes**-0.5, (self.hidden_nodes, self.output_nodes))
        # 2 * 1
        #学习率
        self.lr = learning_rate

        self.activation_function = self.__sigmoid
    def __sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    #导数
    def __sigmoid_derivative(self, x):
        return x * (1 - x)

    #学习过程
    def train(self, inputs_list, targets_list):
        inputs = np.array(inputs_list,ndmin=2)
        # 1 * n
        layer_0 = inputs
        targets = np.array(targets_list,ndmin=2)

        #正向传播过程
        layer_1 = self.activation_function(layer_0.dot(self.weights_0_1))
        # 1 * 2
        layer_2 = self.activation_function(layer_1.dot(self.weights_1_2))

        # print(target)
        #计算误差
        layer_2_error = targets - layer_2
        layer_2_delta = layer_2_error * self.__sigmoid_derivative(layer_2)
        #第一层误差与第二层有关
        layer_1_error = layer_2_delta.dot(self.weights_1_2.T)
        layer_1_delta = layer_1_error * self.__sigmoid_derivative(layer_1)

        #反向传播调整权重
        self.weights_1_2 += self.lr * layer_1.T.dot(layer_2_delta)
        self.weights_0_1 += self.lr * layer_0.T.dot(layer_1_delta)

    #预测
    def run(self, inputs_list):

        inputs = np.array(inputs_list,ndmin=2) ####
        #训练后将参数保存至文件，之后的预测直接从文件读取参数
        self.weights_0_1=np.loadtxt("F://weights_0_1.txt")
        self.weights_1_2=np.loadtxt("F://weights_1_2.txt")
        layer_1 = self.activation_function(inputs.dot(self.weights_0_1)) # 1 * 2
        layer_2 = self.activation_function(layer_1.dot(self.weights_1_2)) # 1 * 1
        return layer_2

#均方差
def MSE(y, Y):
    return np.mean((y-Y)**2)

fileList = listdir("F://charSamples")
m = len(fileList)
print(m)
result = [[0,0,0,0],[0,0,0,1],[0,0,1,0],[0,0,1,1],[0,1,0,0],[0,1,0,1],[0,1,1,0],[0,1,1,1],[1,0,0,0],[1,0,0,1]]
# print(type(result[0]))
output=[]
inputs=[]

#输入训练数据
for index in range(10):
    # print(index)
    list = listdir("F://charSamples/"+str(index))
    # print(len(list))
    for each_img in list:
        if each_img == "Thumbs.db":
            break
        img = cv2.imread("F://charSamples/"+str(index)+"/"+each_img,0)
        img = cv2.resize(img,(10,20))
        # print(img.shape[1])
        temp=[]
        m=0
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                m=m+1
                if img[i][j]>=127:
                    temp.append(1)
                else:temp.append(0)
        # print("temp"+str(len((temp))))
        # print("M="+str(m))
        inputs.append(temp)
        # print(temp)
        output.append(result[index])
        # print(output)
training_set_inputs = np.array(inputs)
training_set_outputs = np.array(output)

#循环学习次数
epochs = 20000
learning_rate = 1
hidden_nodes = 200
output_nodes = 4
N_i = 200
network = TwoLayerNeuralNetwork(N_i, hidden_nodes, output_nodes, learning_rate)
# print(output)
losses = {'train': []}
for e in range(epochs):
    for record, target in zip(training_set_inputs, training_set_outputs):
        network.train(record, target)
        # train_loss = MSE(network.run(training_set_inputs),training_set_outputs)
        sys.stdout.write("\rProgress: " + str(100 * e / float(epochs))[:4] +"%"                                                                       "" )
        # losses['train'].append(train_loss)

#输出参数
# print(" ")
# print("After train,layer_0_1: ")
# print(network.weights_0_1)
# print("After train,layer_1_2: ")
# print(network.weights_1_2)

# 保存调整后的参数
np.savetxt("F://weights_0_1.txt",network.weights_0_1)
# print(network.weights_0_1)
np.savetxt("F://weights_1_2.txt",network.weights_1_2)
# print(network.weights_1_2)

#测试数据 测试数据来自训练数据
test=[]
test_img = cv2.imread("F://number.jpg",0)
test_img = cv2.resize(test_img,(10,20))
for i in range(test_img.shape[0]):
    for j in range(test_img.shape[1]):
        if test_img[i][j] >= 127:
            test.append(1)
        else:
            test.append(0)

str=0
a=8
result_num=network.run(test)
for each in result_num[0]:
    if each>0.5:
        str+=a
        a/=2
    else:
        str+=0
        a/= 2
print('')
print(int(str))
# img = cv2.imread("F://charSamples/0/1_0.822474_gray_13564_5332_step5_recog_2_0_0.891312_0.733081.png", 0)
# cv2.imshow("s",img)
