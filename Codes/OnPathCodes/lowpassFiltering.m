function [signal] = lowpassFilter(signal,lowPass,samplingInterval)
sampRate = 1/((samplingInterval)*(1/10^3));
low = lowPass/sampRate; 
[b,a] = butter(2,low,'low');
signal = filtfilt(b,a,signal);
end