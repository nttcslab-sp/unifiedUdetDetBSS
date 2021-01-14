function underdetBSS
    %% experimental conditions
    nSrc = 4;                   % number of sources
    param.rate = 16000;         % sampling rate
    param.awinsel = 'sqrthann'; % window function      
    param.fftsize = 2 .^ [10 9];% frame length/shift
    
    %% read mixtures
    mixWav = {wavread('mixture.wav')};
    
    %% STFT
    [X,param] = stftAnaly(mixWav,param);
    
    %% FastFCA
    estSTFT = FastFCA_EM(X,nSrc);
    %estSTFT = FastFCA_MM(X,nSrc);
    
    %% ISTFT
    estwav = stftSynth(estSTFT,param);
    
    %% plot estimated sources
    estwav = permute(cell2mat(reshape(estwav,[1 1 nSrc])),[3 1 2]);
    figure
    title('estimated sources')
    for n=1:nSrc
        subplot(nSrc,1,n);plot(estwav(n,:,1));
    end
    
    %% output estimated sources
    for n=1:nSrc
        wavwrite(estwav(n,:,1)/max(max(abs(estwav(n,:,1))))*0.95,param.rate,['est', int2str(n), '.wav']);
    end
end