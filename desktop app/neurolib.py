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
                