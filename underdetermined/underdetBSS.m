function underdetBSS
    %% experimental conditions
    nSrc = 3;                   % number of sources
    param.rate = 16000;         % sampling rate
    param.awinsel = 'sqrthann'; % window function      
    param.fftsize = 2 .^ [10 9];% frame length/shift
    
    %% read mixtures
    mixWav = {wavread('mixture3sources1_16k.wav')};
    
    %% STFT
    [X,param] = stft_analy(mixWav,param);
    
    %% FastFCA
    estSTFT = FastFCA_EM(X,nSrc);
    %estSTFT = FastFCA_MM(X,nSrc);
    
    %% ISTFT
    estwav = stft_synth(estSTFT,param);
    
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