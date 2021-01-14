function Gamma = initGamma(X,nCl,loops,param)
    [nMic,nFrame,nFreq] = size(X);
    Z = zeros(size(X)); 
    for f = 1:nFreq
        % normalization
        x = X(:,:,f);
        x2 = sum(x .* conj(x));
        x = x ./ repmat(sqrt(x2), [nMic 1]);
        % prewhitening
        [EPhi,DPhi] = eig(x * x' / nFrame);
        z = sqrt(inv(DPhi)) * EPhi' * x;
        % normalization
        z2 = sum(z .* conj(z));
        z = z ./ repmat(sqrt(z2), [nMic 1]);
        Z(:,:,f) = z;
    end
    Alpha = ones(nCl,nFrame,nFreq) / nCl;
    A=randn(nMic,nCl,nFreq)+1i*randn(nMic,nCl,nFreq);
    A=A./repmat(sqrt(sum(abs(A).^2,1)),[nMic 1 1]);
    Kappa = ones(nCl,nFreq)*20;
    for l = 1:loops
        % E step
        Gamma = updateGamma(Z,A,Kappa,Alpha);
        % M step
        Alpha = repmat(sum(Gamma,2),[1 nFrame 1])/nFrame;    
        [A,Kappa] = updateAKappa(Z,Gamma);
    end
    [allpermu,~] = permuYchar(Gamma,5,[],2,param);
    Gamma = permutation(Gamma,allpermu,param);
end

function [A,Kappa] = updateAKappa(Z,Gamma)
    nCl = size(Gamma,1);
    [nMic,~,nFreq] = size(Z);
    R = zeros(nCl,nFreq,nMic,nMic);
    Lambda = zeros(nCl,nFreq);
    A = zeros(nMic,nCl,nFreq);
    for k = 1:nCl
        for f = 1:nFreq
            R(k,f,:,:) = (repmat(Gamma(k,:,f), [nMic 1]) .* Z(:,:,f)) * Z(:,:,f)' / sum(Gamma(k,:,f),2);
            [E0,D0] = eig(squeeze(R(k,f,:,:)));
            [Lambda(k,f),ix] = max(real(diag(D0)));
            A(:,k,f) = E0(:,ix);
        end
    end
    Kappa = (nMic * Lambda - 1) ./ Lambda ./ (1 - Lambda) / 2....
        .* (1 + sqrt(1 + 4 * (nMic + 1) / (nMic - 1) * Lambda .* (1 - Lambda)));
    Kappa = min(max(Kappa,0),100);
end

function Gamma = updateGamma(Z,A,Kappa,Alpha)
    nCl = size(A,2);
    [nMic,nFrame,nFreq] = size(Z);
    hyptmp = zeros(1,nFreq);
    P = zeros(nCl,nFrame,nFreq);
    for k = 1:nCl
        tmp1 = conj(repmat(A(:,k,:), [1 nFrame 1])) .* Z;
        tmp2 = squeeze(abs(sum(tmp1)).^2) .* repmat(Kappa(k,:),[nFrame 1]);
        for f = 1:nFreq
            hyptmp(1,f) = kummer(1,nMic,Kappa(k,f));
        end
        P(k,:,:) = exp(tmp2) ./ repmat(hyptmp,[nFrame 1]);
    end
    Gamma = Alpha .* P;
    Gamma = Gamma ./ repmat(sum(Gamma,1), [nCl 1 1]);
end

function f = kummer(a,b,x)
    y=x*a/b; f=1+y;
    for n=1:100
        y=x.*y*(a+n)/(b+n)/(n+1); f=f+y;
    end
end

