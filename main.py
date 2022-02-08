import matplotlib.pyplot as plt
import numpy as np
import random

class neuron:
    
    def __init__(self, clear_function, noised_train, a, b):
        self.alpha = 1
        self.n = 0.4
        self.e0 = 0.1
        self.a_coefitient_count = a
        self.b_coefitient_count = b
        self.step = 1
        self.start_weight = 1 / (self.a_coefitient_count + self.b_coefitient_count)
        
        self.clean = clear_function
        self.arr_in = noised_train
        
        self.weights = [[self.start_weight] * self.a_coefitient_count, 
                        [self.start_weight] * self.b_coefitient_count]
        
        self.input = [self.arr_in[1:self.a_coefitient_count + 1],
                      self.arr_in[:self.b_coefitient_count]]
        
        self.output = [self.get_output()]
        
        
    def learn(self):
        
        while self.step < len(self.arr_in):
            
            self.set_input()
            self.output.append(self.get_output()) 
            self.set_new_w()

            self.step += 1
            self.n = 1/self.step
        
        
    def set_input(self):
        self.input = [[self.arr_in[self.step]] + self.input[0][:-1], 
                      [self.get_output()] + self.input[1][:-1]]
            
    def get_output(self):
        return sum([self.weights[0][i] * self.input[0][i] for i in range(self.a_coefitient_count)]) +\
                sum([self.weights[1][i] * self.input[1][i] for i in range(self.b_coefitient_count)])
    
    def set_new_w(self):
        for i in range(self.a_coefitient_count):
            self.weights[0][i] -= self.n * (self.get_output() - self.clean[self.step]) * self.input[0][i]
        
        for i in range(self.b_coefitient_count):
            self.weights[1][i] -= self.n * (self.get_output() - self.clean[self.step]) * self.input[1][i]
            
            
            
            
    def work(self, noised_test): 

        
        self.arr_in = noised_test
        self.step = 1
        self.input = [self.arr_in[1:self.a_coefitient_count + 1],
                      self.arr_in[:self.b_coefitient_count]]
        
        self.output = [self.get_output()]
        
        while self.step < len(self.arr_in):
            
            self.set_input()
            self.output.append(self.get_output()) 

            self.step += 1
            self.n = 1/self.step
                

f = open("signal_clear.txt", "r")
clear = []
for i in f:
    clear.append(float(i))
f.close()

f = open("signal_noised_train.txt", "r")
noised_train = []
for i in f:
    noised_train.append(float(i))
f.close()

substract = (max(noised_train) + min(noised_train)) / 2
scale = (max(noised_train) - min(noised_train)) * 2

noised_train = [(i - substract) / scale for i in noised_train]
clear = [(i - substract) / scale for i in clear]

noised_test = noised_train

Neuron = neuron(clear, noised_train, 3, 9)
epochs = 1

for i in range(epochs):
    Neuron.output = [Neuron.get_output()]
    Neuron.step = 1
    Neuron.learn()

noised_train = [i * scale + substract for i in noised_train]
Neuron.output = [i * scale + substract for i in Neuron.output]
clear = [i * scale + substract for i in clear]

fig = plt.figure(figsize=(16, 9))
fig.subplots_adjust(top=0.8)
ax1 = fig.add_subplot(111)
ax1.plot(range(len(noised_train)), noised_train, linewidth=1)#, s=1, c=['#ff0000'])
ax1.plot(range(len(Neuron.output)), Neuron.output, linewidth=2)
# ax1.scatter(range(500), noised_train[:500], s=1, c=['#ff0000'])
# ax1.plot(range(500), Neuron.output[:500], linewidth=2)
ax1.grid(2)
ax1.set_xlabel('time')
ax1.plot(range(len(clear)), clear)
plt.savefig("sample_plot.jpg", dpi=300)
plt.show()

f = open('output.txt', 'w')
for i in Neuron.output:
    f.write(f'{str(i)}\n')
f.close()