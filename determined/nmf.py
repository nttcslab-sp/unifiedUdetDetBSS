import numpy as np


class NMF():
    ''' Nonnegative Matrix Factorization '''
    eps = np.spacing(1)  # eps: 2.22e-16

    def __init__(self, X2, nBasis=5):
        super().__init__()
        self.X2 = X2  # X2: power spectrum
        (nFreq, nTime) = self.X2.shape
        self.T = np.random.rand(nFreq, nBasis)
        self.V = np.random.rand(nTime, nBasis)
        self.Var = np.dot(self.T, self.V.T) + NMF.eps
        self.losshist = []  # loss history

    def optimization(self, nLoop):
        for lo in range(nLoop):
            self.updateVar()
            self.calc_loss()

    def updateVar(self):
        self._updateT()
        self.Var = np.dot(self.T, self.V.T) + NMF.eps
        self._updateV()
        self.Var = np.dot(self.T, self.V.T) + NMF.eps

    def _updateT(self):
        Varinv = 1. / self.Var
        X2ViVi = self.X2 * Varinv * Varinv
        Tu = np.dot(X2ViVi, self.V)
        Tl = np.dot(Varinv, self.V) + NMF.eps
        self.T = self.T * np.sqrt(Tu / Tl)

    def _updateV(self):
        Varinv = 1. / self.Var
        X2ViVi = self.X2 * Varinv * Varinv
        Vu = np.dot(X2ViVi.T, self.T)
        Vl = np.dot(Varinv.T, self.T) + NMF.eps
        self.V = self.V * np.sqrt(Vu / Vl)

    def calc_loss(self):
        (nFreq, nTime) = self.X2.shape
        loss = self.X2 / self.Var + np.log(self.Var)  # loss regarding Y
        self.losshist.append(loss.sum())
