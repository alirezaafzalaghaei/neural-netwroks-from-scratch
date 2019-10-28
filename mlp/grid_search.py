import csv
from mlp.mlp import MLP


class MLPGridSearch:
    def __init__(self, task, hidden_layers, activations, batch_sizes, epochs, mus,betas,etas, alphas):
        self.task = task
        self.hidden_layers = hidden_layers
        self.activations = activations
        self.batch_sizes = batch_sizes
        self.epochs = epochs
        self.mus = mus
        self.betas = betas
        self.etas = etas
        self.alphas = alphas
        self.csv = open('result.csv', 'w')
        self.writer = csv.writer(self.csv, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        self.writer.writerow(['hidden_layer','activation','epoch','eta','beta','alpha','mu','batch_size'])

    def run(self,x, y):
        histories = []
        for hidden_layer in self.hidden_layers:
            for activation in self.activations:
                for batch_size in self.batch_sizes:
                    for epoch in self.epochs:
                        for mu in self.mus:
                            for beta in self.betas:
                                for eta in self.etas:
                                    for alpha in self.alphas:
                                        mlp = MLP(hidden_layer,activation,epoch,eta,beta,alpha,mu,batch_size,False,self.task)
                                        histories.append(mlp.fit(x, y))
                                        self.writer.writerow([hidden_layer,activation,epoch,eta,beta,alpha,mu,batch_size])
        self.csv.close()
        return histories

    def _save_csv(self):
        pass
