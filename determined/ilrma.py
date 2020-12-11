import numpy as np


class ILRMA():
    ''' Independent Low-Rank Matrix Analysis '''
    eps = np.spacing(1)  # eps: 2.22e-16

    def __init__(self, X, nSig, nBasisPerSig=5):
        self.X = X.transpose(1, 0, 2)  # X: mixtures
        (nFreq, nMic, nTime) = self.X.shape
        self.XX = np.einsum('fmt,fkt->ftmk', self.X, self.X.conj())
        self.Y2 = np.empty((nSig, nFreq, nTime))  # Y2: power spectrum of separated signals
        self.Var = np.empty((nSig, nFreq, nTime))  # Var: estimated variances
        self.W = np.tile(np.eye(nSig, dtype=complex)[None, :, :], (nFreq, 1, 1))  # W: separation matrix
        self.E = np.eye(nMic)  # E: identity matrix
        self.losshist = []  # loss history
        self.T = np.random.rand(nSig, nFreq, nBasisPerSig)
        self.V = np.random.rand(nSig, nTime, nBasisPerSig)
        for s in range(nSig):
            self.Var[s] = np.dot(self.T[s], self.V[s].T) + ILRMA.eps

    def optimization(self, nLoop):
        for lo in range(nLoop):
            self.calcY2()
            self.normalize_global_scale()
            self.updateVar()
            self.updateW()
            self.calc_loss()

    def calcY2(self):
        Y = np.einsum('fsm,fmt->sft', self.W, self.X)
        self.Y2 = np.real(Y * Y.conj())

    def normalize_global_scale(self):
        ''' normalize the global scales of W and Y2 for source s '''
        scale2 = self.Y2.mean(axis=2).mean(axis=1)
        self.W /= np.sqrt(scale2)[None, :, None]
        self.Y2 /= scale2[:, None, None]

    def updateVar(self):
        nSig = self.W.shape[1]
        for s in range(nSig):
            self._updateT(s)
            self.Var[s] = np.dot(self.T[s], self.V[s].T) + ILRMA.eps
            self._updateV(s)
            self.Var[s] = np.dot(self.T[s], self.V[s].T) + ILRMA.eps

    def _updateT(self, s):
        Varinv = 1. / self.Var[s]
        Y2ViVi = self.Y2[s] * Varinv * Varinv
        Tu = np.dot(Y2ViVi, self.V[s])
        Tl = np.dot(Varinv, self.V[s]) + ILRMA.eps
        self.T[s] = self.T[s] * np.sqrt(Tu / Tl)

    def _updateV(self, s):
        Varinv = 1. / self.Var[s]
        Y2ViVi = self.Y2[s] * Varinv * Varinv
        Vu = np.dot(Y2ViVi.T, self.T[s])
        Vl = np.dot(Varinv.T, self.T[s]) + ILRMA.eps
        self.V[s] = self.V[s] * np.sqrt(Vu / Vl)

    def updateW(self):
        (nFreq, nSig, nMic) = self.W.shape
        var = self.Var  # shape (nSig, nFreq, nTime)
        # Q: weighted covariance matrices with shape (nSig, nFreq, nMic, nMic)
        Q = np.einsum('sft,ftmk->sfmk', 1 / var, self.XX) / self.XX.shape[1]
        for lo in range(3):  # iterate a few times for convergence
            for s in range(nSig):
                self._updateW_row(s, Q[s])

    def _updateW_row(self, s, Qs):
        WQ = np.einsum('fsm,fmk->fsk', self.W, Qs)
        Es = self.E[:, s][None, :]  # shape (1, nMic)
        w = np.linalg.solve(WQ, Es)  # shape (nFreq, nMic)
        wc = w.conj()  # shape (nFreq, nMic)
        wcU = np.einsum('fk,fkm->fm', wc, Qs)  # shape (nFreq, nMic)
        wQw = np.einsum('fm,fm->f', wcU, w)  # shape (nFreq,)
        self.W[:, s, :] = wc / np.sqrt(wQw[:, None])

    def calc_loss(self):
        (nFreq, nMic, nTime) = self.X.shape
        det = np.linalg.det(self.W)
        lossW = -2 * nTime * np.log(np.abs(det))  # loss regarding W
        lossY = self.Y2 / self.Var + np.log(self.Var)  # loss regarding Y
        total_loss = lossY.sum() + lossW.sum()
        self.losshist.append(total_loss)

    def estimate_mixing_matrix(self):
        ''' A: estimated mixing matrix with shape (nFreq, nMic, nSig) '''
        A = np.linalg.inv(self.W)
        (nFreq, nSig, nMic) = self.W.shape
        return A[:, :, :nSig]

    def scaled_separated_signals(self):
        ''' YA: scaled separated signals with shape (nSig, nMic, nFreq, nTime) '''
        A = self.estimate_mixing_matrix()
        Y = np.einsum('fsm,fmt->sft', self.W, self.X)
        YA = np.einsum('fms,sft->smft', A, Y)
        return YA
