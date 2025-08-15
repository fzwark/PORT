import numpy as np
from scipy.optimize import minimize

class Router:
    def __init__(self, ann, base_data, models, B, alpha, eps, length):
        self.ann = ann
        self.base_data = base_data
        self.models = models
        self.M = len(models)
        self.B = np.asarray(B, dtype=np.float32)
        self.alpha = np.asarray(alpha, dtype=np.float32)
        self.eps = eps
        self.length = length
        self.hatd = np.zeros((self.M, length), dtype=np.float32)
        self.hatg = np.zeros((self.M, length), dtype=np.float32)
        self.index = 0
        self.optimal = False
        self.gamma = np.ones(self.M, dtype=np.float32) / self.M
        self.learn = True
        self.learn_limit = int(np.ceil(self.eps * self.length))
        self._d_cols = np.vstack([np.asarray(base_data[m]) for m in models])  # [M, N]
        self._g_cols = np.vstack([np.asarray(base_data[f"{m}|total_cost"]) for m in models])

    def _estimate_batch(self, queries):
        idxs, dists = self.ann.search(queries)  
        d, g = self._calc_batch(idxs)
        n = d.shape[1]
        self.hatd[:, self.index:self.index+n] = d
        self.hatg[:, self.index:self.index+n] = g
        self.index += n
        return d, g

    def _calc_batch(self, idxs):
        d = self._d_cols[:, idxs].mean(axis=2)
        g = self._g_cols[:, idxs].mean(axis=2)
        return d, g

    def _optimize_gamma(self):
        Hd = self.hatd[:, :self.index] 
        Hg = self.hatg[:, :self.index]

        def F(gamma):
            term1 = self.eps * np.dot(gamma, self.B)
            scores = Hd.T * self.alpha - Hg.T * gamma
            term2 = np.max(scores, axis=1).sum()
            return term1 + term2

        x0 = np.full(self.M, 1.0/self.M, dtype=np.float32)
        bounds = [(0.0, 1.0)] * self.M
        res = minimize(F, x0, method='L-BFGS-B', bounds=bounds)
        self.gamma = res.x.astype(np.float32)
        self.optimal = True

    def routing_batch(self, queries):
        n = len(queries)
        out = np.empty(n, dtype=np.int32)

        remaining_learn = max(0, self.learn_limit - self.index)
        n_learn = min(remaining_learn, n)

        if n_learn > 0:
            out[:n_learn] = np.random.randint(0, self.M + 1, size=n_learn)
            self._estimate_batch(queries[:n_learn])  

        n_exploit = n - n_learn
        if n_exploit > 0:
            if not self.optimal:
                self._optimize_gamma()   
            d, g = self._estimate_batch(queries[n_learn:])  
            scores = d * self.alpha.reshape(-1, 1) - g * self.gamma.reshape(-1, 1)
            out[n_learn:] = scores.argmax(axis=0)

        return out