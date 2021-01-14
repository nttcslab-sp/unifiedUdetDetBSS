function [y, swin] = stftSynth(Y, param)

% inverse operation of stftAnaly
% synthesize time-domain signals from time-frequency representations
% made by short-time Fourier transform 
%
% [y, swin] = stftSynth(Y, param)
%
%   Y: time-frequency representation
%   param: parameters
%      param.swinsel: synthesis window, 'rect' or 'dual' (default)
%   y: time domain signal
%   swin: used synthesis window


nfft = param.fftsize(1);
shift = param.fftsize(2);

[nMic, nFrame, nFreq, nOut] = size(Y);

swinsel = 'dual';
if isfield(param, 'swinsel')
  swinsel = param.swinsel;
end

if isfield(param, 'siglen')
  siglen = param.siglen;
else
  siglen = [];
end

switch swinsel
 case 'rect', 
  mult = nfft / (2*shift);
  swin = ones(nfft,1) / mult;
 case 'dual'
  swin = local_synthwin(param, 1, 2);
end

for i=1:nOut
  for j=1:nMic
    Yji = reshape(Y(j,:,:,i), [nFrame, nFreq]);
    y{i,1}(:,j) = local_istft(Yji.', swin, shift, siglen);
  end
end

%------------------------------------------------------------------
%    Local functions
%------------------------------------------------------------------

function y = local_istft(Y, swin, shift, siglen)

% istft: inverse short-time Fourier transform
%   Y: matrix of nFreq * nFrame
%   y: column vector
%   swin: synthesis window
%   shift: shift amount of the window

frameSize = length(swin);
[nFreq, nFrame] = size(Y); % nFreq: frameSize/2 +1

y = zeros( shift*(nFrame-1)+frameSize, 1 );

range = 1:frameSize;
begin = 0;
for i=1:nFrame,
  fftS = [ Y(:,i); conj( Y(end-1:-1:2,i) ) ];
  winsig = real(ifft(fftS)) .* swin;
  y(begin+range) = y(begin+range) + winsig;
  begin = begin + shift;
end;
if ~isempty(siglen)
  y = y(1:siglen);
end

%--------------------------------------------

function [swin,awin,A,B] = local_synthwin(param, center, Nt)

%DUET_SDW - Standard dual (biorthogonal) window for a frame.
%
% [swin,awin,A,B]=duet_sdw(kmax,timestep,numfreq,win,center,Nt)
%

kmax = 1;
win = param.awin;
timestep = param.fftsize(2);
numfreq = param.fftsize(1);

% KMAX * NUMFREQ is the size of returned dual window.
% TIMESTEP is the # of samples between adjacent windows in time.
% NUMFREQ is the # of frequency components per timestep.
% WIN is the frame window.
% CENTER [0], if 1, centers the input win for use as awin.
% NT [256] is the number of t points in the (0,1) Zak domain.
%
% [*] denotes optional argument default value.
%
% SWIN is the (truncated) standard dual.
% AWIN is win plus zero-padding to have length(swin).
% A is the lower frame bound.
% B is the upper frame bound.
      
% 24 July 2000 R. Balan & S. Rickard
% (c) Siemens Corporate Research

win = win(:);
  
q = numfreq/timestep; % redundancy factor

if q~=round(q),
  error('q = numfreq/timestep must be an integer');
end
if length(win) > (kmax*numfreq),
  error('Input window is too large; Must be smaller than kmax*numfreq.');
end
if nargin<2
  center = 0;
end
if nargin<3
  Nt = 256;
end

M = numfreq;
N = timestep;

if center
  numz = ((kmax*numfreq)-length(win))/2;
  g = [zeros(floor(numz),1); win ;zeros(ceil(numz),1); zeros(M*Nt-kmax*numfreq,1)];
else
 g = [win ; zeros(M*Nt-length(win),1)];
end

% zak
G = sqrt(M)*Nt*ifft(reshape(g,M,Nt).');


% rescale the zak to create the standard dual zak
cind = 1:N:M;
Gd = zeros(Nt,M);
A = Inf; B = 0;
for i = 1:N
 Gfactor = sum(abs(G(:,i+cind-1)).^2,2);
 A = min(A,min(Gfactor));
 B = max(B,max(Gfactor));
 % If we get divide by zero warnings, the window and
 % parameters do not form a frame. Nevertheless, we
 % might want to put the below line back in to avoid
 % the warnings. For now, leave it out.
 % Gfactor(Gfactor<eps) = 1;
 for j= 0:(q-1)
   Gd(:,j*N+i) = G(:,j*N+i)./Gfactor;
  end
end

% izak and take the first kmax points as the synthesis window.
swin = real(fft(Gd)); 
swin = reshape(swin(1:kmax,:).',kmax*M,1);
swin = swin/(sqrt(M)*Nt);

swin = numfreq * swin;

% just to check the izak process...
%awin = real(fft(G)); 
%awin = reshape(awin(1:kmax,:).',kmax*M,1);
%awin = awin/(sqrt(M)*Nt);

awin = g(1:(kmax*numfreq));
