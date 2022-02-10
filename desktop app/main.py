from doctest import OutputChecker
from turtle import clear
from PyQt5.QtWidgets import*
from PyQt5.uic import loadUi

from matplotlib.backends.backend_qt5agg  import (NavigationToolbar2QT as NavigationToolbar)

import numpy as np
import random
import neurolib
     
class MatplotlibWidget(QMainWindow):
    
    def __init__(self):
        
        QMainWindow.__init__(self)

        loadUi("design.ui",self)

        self.setWindowTitle("Neural filtering")

        self.pushButton.clicked.connect(self.update_graph)

        self.addToolBar(NavigationToolbar(self.MplWidget.canvas, self))


    def update_graph(self):

        t = np.linspace(0,1,1000)
        
        clean = []
        f = open("signal_clear.txt", "r")
        for i in f:
            clean.append(float(i))
        f.close()

        noised_train = []
        f = open("signal_noised_train.txt", "r")
        for i in f:
            noised_train.append(float(i))
        f.close()

        substract = (max(noised_train) + min(noised_train)) / 2
        scale = (max(noised_train) - min(noised_train)) * 2

        noised_train = [(i - substract) / scale for i in noised_train]
        clear = [(i - substract) / scale for i in clean]

        Neuron = neurolib.neuron(clear, noised_train, self.spinBox_a.value(), self.spinBox_b.value())

        epochs = self.spinBox_epochs.value()

        for i in range(epochs):
            self.progressBar1.setValue(int(100/epochs*(i+1)))
            Neuron.output = [Neuron.clean[0]]
            Neuron.step = 1
            Neuron.learn()
            Neuron.arr_in = Neuron.output
            Neuron.input = [[Neuron.arr_in[0]]*(Neuron.a_coefitient_count),
                            [Neuron.arr_in[0]]*(Neuron.b_coefitient_count)]


        clear = [i * scale + substract for i in clear]
        noised_train = [i * scale + substract for i in noised_train]
        Neuron.output = [i * scale + substract for i in Neuron.output]

        self.plainTextEdit.setPlainText('')
        self.plainTextEdit.appendPlainText(f'1:    {abs(np.mean(Neuron.output[:500])-np.mean(Neuron.output[500:1000])) /6/np.std(Neuron.output[:500]):.2f}')

        for i in range(3):
            self.plainTextEdit.appendPlainText(f'{i+2}:    {abs(np.mean(Neuron.output[100+500*i:500*(i+1)])-np.mean(Neuron.output[100+500*(i+1):500*(i+2)])) /6/np.std(Neuron.output[100+500*(i+1):500*(i+2)]):.2f}')

        self.MplWidget.canvas.axes.clear()
        self.MplWidget.canvas.axes.plot(range(len(clean)), clean)
        self.MplWidget.canvas.axes.plot(range(len(noised_train)), noised_train)
        self.MplWidget.canvas.axes.plot(range(len(Neuron.output)), Neuron.output)
        self.MplWidget.canvas.axes.legend(('clean', 'train', 'output'),loc='upper right')
        self.MplWidget.canvas.axes.set_title('Neural filtering')
        self.MplWidget.canvas.draw()
        

app = QApplication([])
window = MatplotlibWidget()
window.show()
app.exec()