import numpy as np


class ICA():
    def __init__(self, X, nSig, var_floor=0.01):
        self.X = X  # X: mixtures
        (nMic, nTime) = self.X.shape
        self.XX = np.einsum('mt,kt->tmk', self.X, self.X)
        self.W = np.eye(nSig)  # W: separation matrix
        self.E = np.eye(nMic)  # E: identity matrix
        self.losshist = []  # loss history
        self.Whist = []  # separation matrix history
        self.var_floor = var_floor

    def optimization(self, nLoop):
        for lo in range(nLoop):
            self.Whist.append(self.W.copy())
            self.calcY2()
            self.normalize_scale()
            self.updateW()
            self.calc_loss()

    def calcY2(self):
        Y = self.W.dot(self.X)
        self.Y2 = np.real(Y * Y)  # Y2: power spectrum of separated signals

    def normalize_scale(self):
        ''' normalize the scales of W and Y2 '''
        scale2 = self.Y2.mean()
        self.W /= np.sqrt(scale2)
        self.Y2 /= scale2

    def updateW(self):
        self.Var = self.Y2 + self.var_floor  # Var: estimated variances
        (nSig, nMic) = self.W.shape
        # Q: weighted covariance matrices with shape (nSig, nMic, nMic)
        Q = np.einsum('st,tmk->smk', 1 / self.Var, self.XX) / self.XX.shape[0]
        for s in range(nSig):
            self._updateW_row(s, Q[s])

    def _updateW_row(self, s, Qs):
        WQ = self.W.dot(Qs)
        Es = self.E[:, s]  # shape (nMic)
        w = np.linalg.solve(WQ, Es)  # shape (nFreq, nMic)
        wQw = w.dot(Qs).dot(w)
        self.W[s] = w / np.sqrt(wQw)

    def calc_loss(self):
        (nMic, nTime) = self.X.shape
        det = np.linalg.det(self.W)
        lossW = -2 * nTime * np.log(np.abs(det))  # loss regarding W
        lossY = self.Y2 / self.Var + np.log(self.Var)  # loss regarding Y
        total_loss = lossY.sum() + lossW.sum()
        self.losshist.append(total_loss)
