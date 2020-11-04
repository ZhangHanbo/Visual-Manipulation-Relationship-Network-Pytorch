from sklearn.neighbors import KernelDensity as kde
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt

class gaussian_kde(object):
    def __init__(self, data, bandwidth=0.03):
        self.training_data = data
        self.bandwidth = bandwidth
        self.kde = kde(kernel='gaussian', bandwidth=self.bandwidth).fit(self.training_data)

    def update(self, new_data):
        self.training_data = np.concatenate([self.training_data, new_data], axis = 0)
        self.kde.fit(self.training_data)
        return self

    def comp_prob(self, x):
        if isinstance(x, (float, np.float, np.float32, np.float64)):
            x = np.array([[x]])
        elif isinstance(x, (list, np.ndarray)):
            x = np.expand_dims(np.array(x), axis=-1)
        x = np.exp(self.kde.score_samples(x))
        return x.squeeze()

class object_belief(object):
    def __init__(self):
        self.belief = np.array([0.5, 0.5])

    def update(self, score, kde):
        neg_prob = kde[0].comp_prob(score)
        pos_prob = kde[1].comp_prob(score)
        self.belief *= [neg_prob, pos_prob]
        self.belief /= self.belief.sum()
        return self.belief

    def reset(self):
        self.belief = np.array([0.5, 0.5])

if __name__=="__main__":
    with open("../../density_esti_train_data.pkl") as f:
        data = pkl.load(f)
    data = data["ground"]

    pos_data = []
    neg_data = []
    for d in data:
        for i, score in enumerate(d["scores"]):
            if str(i) in d["gt"]:
                pos_data.append(score)
            else:
                neg_data.append(score)
    pos_data = np.expand_dims(np.array(pos_data), axis=-1)
    pos_data = np.sort(pos_data, axis=0)[5:-5]
    neg_data = np.expand_dims(np.array(neg_data), axis=-1)
    neg_data = np.sort(neg_data, axis=0)[5:-5]

    kde_pos = gaussian_kde(pos_data)
    kde_neg = gaussian_kde(neg_data)

    x = (np.arange(100).astype(np.float32) / 100 - 0.5) * 2
    y = np.array([kde_pos.comp_prob(x[i]) for i in range(len(x))])
    plt.plot(x, y, ls="-", lw=2, label="positive")
    y = np.array([kde_neg.comp_prob(x[i]) for i in range(len(x))])
    plt.plot(x, y, ls="-", lw=2, label="negative")
    plt.legend()
    plt.show()

    obj1 = object_belief()
    kdes = [kde_neg, kde_pos]
    obj1.update(0.38, kdes)
    obj1.update(0.3, kdes)
    obj1.update(0.22, kdes)
    obj1.update(0.35, kdes)
    obj1.update(0.49, kdes)

