% X: mixtures (complex STFT); an array of size nMic x nFrame x nFreq
%    (nFreq: number of frequency bins up to the Nyquist frequency)
% nSrc: number of sources
% Z: separated signals (complex STFT); an array of size 
%    nMic x nFrame x nFreq x nSrc
function Z = FastFCA_EM(X,nSrc)
    param.rate = 16000; % sampling frequency in Hz
    param.freqRange = [0 8000]; % frequency range
    param.fftsize = 2 .^ [10 9]; % FFT size
    iternum=20; % number of iterations
    [nMic,nFrame,nFreq]=size(X);
    reg=10^-3;
    reg2=0;
    % allocation
    Mu=zeros(nMic,nFrame,nFreq,nSrc);
    V=zeros(nFrame,nFreq,nSrc);
    R=zeros(nMic,nMic,nFreq,nSrc);
    P=zeros(nMic,nMic,nFreq);% joint diagonalization matrix
    LambdaDiag=zeros(nMic,nSrc,nFreq);% generalized eigenvalue matrix
    Y=zeros(size(X));% basis-transformed data
    Z=zeros(size(Mu));
    % initialization 
    Gamma = initGamma(X,nSrc,20,param); 
    for j=1:nSrc
        for f=1:nFreq
            R(:,:,f,j)=(repmat(Gamma(j,:,f),[nMic 1]).*X(:,:,f))*X(:,:,f)';
            R(:,:,f,j)=R(:,:,f,j)/trace(R(:,:,f,j));
            R(:,:,f,j)=(R(:,:,f,j)+R(:,:,f,j)')/2;
            R(:,:,f,j)=R(:,:,f,j)+reg2*mean(diag(R(:,:,f,j)))*eye(nMic);
        end
    end
    for t=1:nFrame
        for f=1:nFreq
            for j=1:nSrc
                V(t,f,j)=Gamma(j,t,f)*real(trace(X(:,t,f)'/R(:,:,f,j)*X(:,t,f)))/nMic;
            end
        end
    end
    V = max(V,reg*max(max(max(V))));
    for f=1:nFreq
        [P(:,:,f),~]=eig(R(:,:,f,2),R(:,:,f,1));
    end
    for f=1:nFreq
        Y(:,:,f)=P(:,:,f)'*X(:,:,f);
    end
    for f=1:nFreq
        for j=1:nSrc
            LambdaDiag(:,j,f)=max(real(diag(P(:,:,f)'*R(:,:,f,j)*P(:,:,f))),0);
            LambdaDiag(:,j,f)=LambdaDiag(:,j,f)+reg2*mean(LambdaDiag(:,j,f))*ones(nMic,1,1);
        end
    end 
    % EM iterations
    if(iternum==0)
        vlambda = repmat(permute(V,[4 1 2 3]),[nMic 1 1 1]) .* ...
                    repmat(permute(LambdaDiag,[1 4 3 2]),[1 nFrame 1 1]);
        WF = vlambda ./ repmat(sum(vlambda,4),[1 1 1 nSrc]); 
        Mu = WF.*repmat(Y,[1 1 1 nSrc]);
    end
    for l = 1:iternum
        vlambda = repmat(permute(V,[4 1 2 3]),[nMic 1 1 1]) .* ...
                    repmat(permute(LambdaDiag,[1 4 3 2]),[1 nFrame 1 1]);
        WF = vlambda ./ repmat(sum(vlambda,4),[1 1 1 nSrc]); 
        Mu = WF.*repmat(Y,[1 1 1 nSrc]);
        if(l==iternum)
            break;
        end
        PsiDiag=real((1-WF).*vlambda);
        PsiDiag=max(PsiDiag,reg2*max(max(max(max(PsiDiag)))));
        V = real(squeeze(sum((abs(Mu).^2+PsiDiag)./repmat(permute(LambdaDiag,[1 4 3 2]),[1 nFrame 1 1]),1)/nMic));
        V = max(V,reg*max(max(max(V))));
        LambdaDiag = real(permute(mean((abs(Mu).^2+PsiDiag)./repmat(permute(V,[4 1 2 3]),[nMic 1 1 1]),2),[1 4 3 2]));
        LambdaDiag = max(LambdaDiag,reg2*max(max(max(LambdaDiag))));
        vlambdasum = sum(repmat(permute(V,[4 1 2 3]),[nMic 1 1 1]) .* ...
                    repmat(permute(LambdaDiag,[1 4 3 2]),[1 nFrame 1 1]),4);
        for f=1:nFreq
            Q = eye(nMic) / P(:,:,f)';
            for m = 1:nMic
                T = (X(:,:,f)./repmat(vlambdasum(m,:,f),[nMic 1]))*X(:,:,f)'/nFrame;
                P(:,m,f) = T \ Q(:,m);
                P(:,m,f) = P(:,m,f) / sqrt(P(:,m,f)'*T*P(:,m,f));
            end
            Y(:,:,f) = P(:,:,f)' * X(:,:,f);
        end
    end
    for f=1:nFreq
        for j=1:nSrc
            Z(:,:,f,j)=P(:,:,f)'\Mu(:,:,f,j);
        end
    end
end