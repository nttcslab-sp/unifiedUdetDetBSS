function [X, param] = stftAnaly(x, param)

% stftAnaly: STFT analysis for x
%
% [X, param] = stftAnaly(x, param)
%
%   x: cell, x{i}: array, sigLen x nMic
%   param: parameters
%       param.awinsel: analysis window, 'hann' or 'sqrthann' (default)
%   X: matrix of nMic * nFrame * nFreq * length(x(:)), nFreq = nfft/2 +1

if iscell(x)==0, error('x should be a cell'); end

nfft = param.fftsize(1);
shift = param.fftsize(2);

if isfield(param, 'awinsel')
  awinsel = param.awinsel;
else 
  awinsel = 'sqrthann';
end

switch awinsel
 case 'hann',
  awin = hanning(nfft, 'periodic');
 case 'sqrthann',
  awin = sqrt(hanning(nfft, 'periodic'));
end

param.awin = awin;
[param.siglen, nMic] = size(x{1});

for i=length(x(:)):-1:1
  for j=1:nMic
    X(j,:,:,i) = local_stft(x{i}(:,j), param.awin, shift).';
  end
end

%------------------------------------------------------------------
%    Local functions
%------------------------------------------------------------------

function S = local_stft(s, awin, shift)

% short-time Fourier transform
%   s: column vector
%   S: matrix of nFreq * nFrame, nFreq = length(awin)/2 +1

frameSize = length(awin);
nFreq = frameSize/2 +1; % freq: 0 - fs/2

sLength = length(s);
nFrame = ceil( (sLength-frameSize)/shift )+1;
nFrame = max(nFrame, 1); % for the case sLength < frameSize
newlen = frameSize + (nFrame-1)*shift;

s = [s; zeros(newlen-sLength,1)];
S = zeros(nFreq, nFrame);

range = [1:frameSize];
begin = 0;
for i=1:nFrame
  winsig = s(range+begin) .* awin;
  fftS = fft(winsig);
  S(:,i) = fftS(1:nFreq); % discard the conjugate part
  begin = begin + shift;
end
