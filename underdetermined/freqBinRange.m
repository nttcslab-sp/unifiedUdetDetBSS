function [range, freqStep] = freqBinRange(param)

% Calculate the range of frequency bins from param

freqRange = param.freqRange;
frameSize = param.fftsize(1);
freqStep = param.rate / frameSize;

initF = ceil( freqRange(1)/freqStep )+1;
lastF = floor( freqRange(2)/freqStep )+1;
range = [initF:lastF];
