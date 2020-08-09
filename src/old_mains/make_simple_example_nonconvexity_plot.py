import numpy as np
import matplotlib.pyplot as plt
import time

class OneDimSimpleExample:

    def __init__(self, a, b, theta, s=10):
        self.a = a
        self.b = b
        self.theta = theta
        self.s = s

    def get_log_prob(self, data, labels):
        probs = self.sigmoid(self.a * self.get_features(data) + self.b)
        log_prob_per_point = labels * np.log(probs) + (1 - labels) * np.log(1 -  probs)
        print('features', self.get_features(data) + self.b)
        print(probs)
        print(log_prob_per_point)
        return np.sum(log_prob_per_point)

    def get_feat_diff(self, data, labels):
        features = self.get_features(data)
        pos_features_avg = features[labels == 1]
        neg_features_avg = features[labels == 0]
        return np.mean(pos_features_avg) - np.mean(neg_features_avg)
    
    def get_neg_regularization(self, data, labels):
        features = self.get_features(data)
        neg_features = features[labels == 0]
        return np.mean(neg_features)
    
    def get_probs(self, data, labels):
        return self.sigmoid(self.a * self.get_features(data) + self.b)

    def get_acc(self, data, labels):
        preds = [1 if prob > .5 else 0 for prob in self.get_probs(data, labels)]
        return np.mean(preds == labels)
    
    def sigmoid(self, np_arr):
        return 1/(1 + np.exp(-1 * np_arr))

    def get_features(self, data):
        return self.sigmoid(self.s * (data - self.theta))
    
class OneDimExamplePlotterVaryingTheta:

    def __init__(self, data, labels, a, b, theta_grid):
        self.data = data
        self.labels = labels
        self.a = a
        self.b = b
        self.theta_grid = theta_grid

    def plot_loss_grid_varying_theta(self,  axis):
        self.losses = []
        for theta in theta_grid:
            self.losses.append(-1 * OneDimSimpleExample(self.a, self.b, theta).get_log_prob(self.data, self.labels))
        axis.plot(theta_grid, self.losses, label='negative log loss')
        axis.set_xlabel('Cut position')
        axis.set_ylabel('Negative Log-Likelihood')

    def plot_accuracies_varying_theta(self,  axis):
        self.accs = []
        for theta in theta_grid:
            self.accs.append(OneDimSimpleExample(self.a, self.b, theta).get_acc(self.data, self.labels))
        axis.plot(theta_grid, self.accs, label='accuracy')
        axis.set_xlabel('Cut position')
        axis.set_ylabel('Accuracy')

    # TODO: update this to the correct functional form in utils/bayes_gate.py
    def plot_feat_diff_varying_theta(self, axis):
        self.feat_diffs = []
        for theta in theta_grid:
            self.feat_diffs.append(-1 * OneDimSimpleExample(self.a, self.b, theta).get_feat_diff(self.data, self.labels))
        axis.plot(theta_grid, self.feat_diffs, label='average feature differences')

    def plot_neg_regularization_varying_theta(self, axis):
        self.neg_regs = []
        for theta in theta_grid:
            self.neg_regs.append(OneDimSimpleExample(self.a, self.b, theta).get_neg_regularization(self.data, self.labels))
        axis.plot(theta_grid, self.neg_regs, label='negative feature penalty')
        


def main_varying_theta(data_mean, data_variance, num_points, a, b, theta_grid, figlength=7):
    data = np.random.normal(loc=data_mean, scale=data_mean,  size=num_points)
    labels = np.random.randint(0, 2, num_points)
    
    plotter = OneDimExamplePlotterVaryingTheta(data, labels, a, b, theta_grid)
    #fig, axes = plt.subplots(2, 1, figsize=(figlength * 1, figlength * 2))
    plotter.plot_loss_grid_varying_theta(plt.gca())
    #plotter.plot_accuracies_varying_theta(plt.gca())
    data_height = 0
    pos_data = data[labels == 1]
    neg_data = data[labels == 0]
    plt.scatter(pos_data, data_height * np.ones(len(labels[labels==1])), color='r', label='positive 1d data point')
    plt.scatter(neg_data, data_height * np.ones(len(labels[labels==0])), color='b', label='negative 1d data point')
    plt.legend()
    plt.savefig('loss_versus_theta_simple_example.png')

if __name__ == '__main__':
    mean = 3
    var = 2
    num_points = 3
    a = 1
    b = 0
    seed = 103
    np.random.seed(seed)
    theta_grid = np.linspace(mean - 4*var, mean + 4*var, 1000) 

    main_varying_theta(mean, var, num_points, a, b, theta_grid)
        
