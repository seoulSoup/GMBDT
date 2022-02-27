import numpy as np
from sklearn.mixture import BayesianGaussianMixture
from scipy.stats import chi2

class GMBDT:
    def __init__(self, max_iter: int=1000, min_sample: int=20, ratio_criteria: float=0.05, LLT_type: str="median"):
        """
        Initialize queues and hyperparameters.

        Parameters
        ----------
        max_iter : int
            The number of EM iterations to perform in BayesianGaussianMixture Model.
        min_sample : int
            Minimum size of cluster that can be treat as a single cluster.
        ratio_criteria : float
            When two clusters are detected by BGMM model, the ratio of smaller one to bigger one.
            If ratio is bigger than ratio_criteria, two clusters are treat as single cluster.
            It's for preventing overfitting.
        LLT_type: str
            There are two choices: median or mean.
            This decides value for Log Likelihood Ratio Test.
            If anomalys have a random pattern or values, median is better.
            But, in simple case mean is might be a better choice.
        
        Returns
        -------
        None

        See Also
        --------
        "list_model" : The queue of models at each tree depth.
        "list_cluster" : The queue of predicted clusters informations at each tree depth.
        "n_features" : Feature dimension of input array.
        
        Notes
        -----
        Optimize hyperparameter "min_sample" and "ratio_criteria" to your dataset please.
        
        Examples
        --------
        bgmm = GMBDT(max_iter=100, min_sample=100, ratio_criteria=0.1, LLT_type="mean")
        """
        assert LLT_type.lower() == "median" or LLT_type.lower() == "med" or LLT_type.lower() == "mean", "LLT_type is not specific!\n Choose 'median' or 'med' or 'mean' please."
        self.list_model = []
        self.list_cluster = []
        self.max_iter = max_iter
        self.n_features = 1
        self.min_sample = min_sample
        self.ratio_criteria = ratio_criteria
        self.LLT_type = LLT_type.lower()
            
    def recursive_BGMM(self, array_value: np.array):
        """
        Recursive Function to make Binary Decision Tree using BayesianGaussianMixture Model.

        Parameters
        ----------
        array_value : np.array
            Train set for BayesianGaussianMixture Model.
        
        Returns
        -------
        array_value : np.array
            This is a cluster that predicted as "normal" by BayesianGaussianMixture Model
            or can't be seperate to two clusters.
            Return to next stack of recursive function, so its value is not actually important.

        See Also
        --------
        "list_model" : The queue of models at each tree depth.
        "list_cluster" : The queue of predicted clusters informations at each tree depth.
        "n_features" : Feature dimension of input array.
        
        Notes
        -----
        Not recommended use as independent method. 
        Use fit(), predict() or fit_predict() method please.
        """
        
        if array_value.shape[0] < 2: return array_value
        if len(array_value.shape) < 2: array_value = array_value.reshape(-1, 1)

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
        if len(unique) < 2:
            if len(self.list_cluster) >= 1: return array_value
        
        if self.LLT_type == "median" or self.LLT_type == "med":
            # using median for Log Likelihood Ratio Test in predict()
             array_medians = np.array([]).reshape(-1, self.n_features)
            for idx in unique:
                array_median = np.median(array_values[array_values[:, -1] == idx, :-1], axis=0).reshape(1, -1)
                array_medians = np.concatenate((array_medians, array_median), axis=0)
            array_cluster = np.concatenate((unique.reshape(-1, 1), counts.reshape(-1, 1), array_medians), axis=1)
        elif self.LLT_type == "mean":
            # using mean for Log Likelihood Ratio Test in predict()
            array_cluster = np.concatenate((unique.reshape(-1, 1), counts.reshape(-1, 1), model.means_), axis=1)
        array_cluster = array_cluster[array_cluster[:, 1].argsort()][::-1]
        
        if sum(array_cluster[:, 1]) > self.min_sample and array_cluster[1, 1] / array_cluster[0, 1] > self.ratio_criteria: return array_value
        
        self.list_model.append(model)
        self.list_cluster.append(array_cluster)

        array_value = array_values[array_values[:, -1] == array_cluster[0, 0], :-1]
        return self.recursive_BGMM(array_value)

    def reset(self):
        """
        Initialize attributes of GMBDT object.

        Parameters
        ----------
        None
        
        Returns
        -------
        None

        See Also
        --------
        "list_model" : The queue of models at each tree depth.
        "list_cluster" : The queue of predicted clusters informations at each tree depth.
        "n_features" : Feature dimension of input array.
        
        Notes
        -----
        Initialize queues and feature dimension information.

        Examples
        --------
        def fit(self, array_value: np.array):
            self.reset()
            self.recursive_BGMM(array_value)
        """
        self.list_model = []
        self.list_cluster = []
        self.n_features = 1

    def fit(self, array_value: np.array):
        """
        Fit data to GaussianMixtureBinaryDecisionTree.

        Parameters
        ----------
        array_value : np.array
            Train set for BayesianGaussianMixture Model.
        
        Returns
        -------
        None

        See Also
        --------
        "list_model" : The queue of models at each tree depth.
        "list_cluster" : The queue of predicted clusters informations at each tree depth.

        Notes
        -----
        Important informations are saved in attributes "list_model" and "list_cluster".
        They're used when call predict() method.

        Examples
        --------
        bgmm = GMBDT()
        bgmm.fit(array_train)
        pred = bgmm.predict(array_test)
        """
        self.reset()
        self.recursive_BGMM(array_value)

    def predict(self, array_value: np.array, weight_type: str="linear", criteria: float=0.5):
        """
        Predict from trained GaussianMixtureBinaryDecisionTree in fit() method.

        Parameters
        ----------
        array_value : np.array
            Test set for BayesianGaussianMixture Model.
        weight_type : str
            Matrix for calculation of an anomaly score.
            More weight to earlier detected anomalys.
            "linear" weight or "exp" weight.
        criteria : float
            Anomalys are predicted only when anomaly score is greater than criteria.
        
        Returns
        -------
        array_return : np.array
            Result of prediction. {0: normal, 1: anomaly}

        See Also
        --------
        "LLMean" : Log Likelihood of median or mean saved in fit() method.
        "LLValues" : Log Likelihood of Test set values.

        Notes
        -----
        Calculate anomaly score using trained model and information saved in fit() method.
        Prediction is decided by p-value of Log Likelihood Ratio Test and probability of normal cluster.

        Examples
        --------
        bgmm = GMBDT()
        bgmm.fit(array_train)
        pred = bgmm.predict(array_test)
        """
        assert weight_type.lower() == "linear" or weight_type.lower() == "exp", "weight_type is not specific!\n Choose 'linear' or 'exp' please."
        if len(array_value.shape) < 2: array_value = array_value.reshape(-1, 1)
        
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
        # print(np.concatenate((array_value, array_pred), axis=1))
        array_return = np.dot(array_pred, array_weights[::-1] / depth)
        array_return = np.array(list(map(lambda x: 1 if x >= criteria else 0, array_return)))
        
        return array_return
        
    def fit_predict(self, array_value: np.array, weight_type: str="linear", criteria: float=0.5):
        """
        If your dataset can't be seperated to train/test set or not necessary to do, 
        there's no need to use fit() and predict() method seperately.

        Parameters
        ----------
        array_value : np.array
            Test set for BayesianGaussianMixture Model.
        weight_type : str
            Matrix for calculation of an anomaly score.
            More weight to earlier detected anomalys.
            "linear" weight or "exp" weight.
        criteria : float
            Anomalys are predicted only when anomaly score is greater than criteria.
        
        Returns
        -------
        array_return : np.array
            Result of prediction. {0: normal, 1: anomaly}

        Notes
        -----
        Actually, it's exactly same to using fit() and predict() method sequentially. :D

        Examples
        --------
        bgmm = GMBDT()
        pred = bgmm.fit_predict(array_eval)
        """
        self.reset()
        self.fit(array_value.copy())
        return self.predict(array_value.copy(), weight_type=weight_type, criteria=criteria)

