import numpy as np
from sklearn.mixture import BayesianGaussianMixture
from scipy.stats import chi2

class GMBDT:
    def __init__(self, max_iter: int=1000):
        self.list_model = []
        self.list_cluster = []
        self.max_iter = max_iter
        self.n_features = 1
            
    def recursive_BGMM(self, array_value: np.array):
        if array_value.shape[0] < 2: return array_value

        model = BayesianGaussianMixture(n_components=2,
                                        covariance_type="full",
                                        weight_concentration_prior_type="dirichlet_process",
                                        weight_concentration_prior=0.5,
                                        init_params="random",
                                        max_iter=self.max_iter).fit(array_value)
        self.n_features = model.n_features_in_
        array_pred = model.predict(array_value)
        array_values = np.concatenate((array_value, array_pred.reshape(-1, 1)), axis=1)
        unique, counts = np.unique(array_pred, return_counts=True)
        if len(unique) < 2: return array_value
        
        array_medians = np.array([]).reshape(-1, self.n_features)
        # for idx in unique:
        #     array_median = np.median(array_values[array_values[:, -1] == idx, :-1], axis=0).reshape(1, -1)
        #     array_medians = np.concatenate((array_medians, array_median), axis=0)
        # array_cluster = np.concatenate((unique.reshape(-1, 1), counts.reshape(-1, 1), array_medians), axis=1)
        array_cluster = np.concatenate((unique.reshape(-1, 1), counts.reshape(-1, 1), model.means_), axis=1)
        array_cluster = array_cluster[array_cluster[:, 1].argsort()][::-1]
        if sum(array_cluster[:, 1]) > 20 and array_cluster[1, 1] / array_cluster[0, 1] > 0.05: return array_value
        
        self.list_model.append(model)
        self.list_cluster.append(array_cluster)

        array_value = array_values[array_values[:, -1] == array_cluster[0, 0], :-1]
        return self.recursive_BGMM(array_value)

    def reset(self):
        self.list_model = []
        self.list_cluster = []
        self.n_features = 1

    def fit(self, array_value: np.array):
        self.reset()
        self.recursive_BGMM(array_value)

    def predict(self, array_value: np.array, weight_type: str="linear", criteria: float=0.5):
        assert weight_type.lower() == "linear" or weight_type.lower() == "exp", "weight_type is not specific!\n Choose 'linear' or 'exp' please."

        depth = len(self.list_cluster)
        if depth < 1: return np.zeros(array_value.shape[0])

        array_pred = np.array([]).reshape(array_value.shape[0], -1)
        for model, array_cluster in zip(self.list_model, self.list_cluster):
            LLMean = model.score_samples(array_cluster[0, -self.n_features:].reshape(-1, self.n_features))
            LLValues = model.score_samples(array_value)
            LR = 2 * np.abs(LLMean - LLValues)
            p_values = chi2.sf(LR, 1)
            prob_values = model.predict_proba(array_value)[:, int(array_cluster[0, 0])]
            array_p = np.concatenate((p_values.reshape(-1, 1), prob_values.reshape(-1, 1)), axis=1)

            array_score = np.zeros_like(p_values)
            array_score[(array_p[:, 0] < 0.05) & (array_p[:, 1] < 0.025)] = 1
            array_pred = np.concatenate((array_pred, array_score.reshape(-1, 1)), axis=1)

        if weight_type == "linear":
            array_weights = np.linspace(1, depth, num=depth)
        elif weight_type == "exp":
            array_weights = np.exp(np.linspace(1, depth, num=depth))
        array_return = np.dot(array_pred, array_weights[::-1] / depth)
        array_return = np.array(list(map(lambda x: 1 if x >= criteria else 0, array_return)))

        return array_return
        
    def fit_predict(self, array_value: np.array, weight_type: str="linear", criteria: float=0.5):
        self.reset()
        self.fit(array_value.copy())
        return self.predict(array_value.copy(), weight_type=weight_type, criteria=criteria)

