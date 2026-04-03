% testing firing rate

clear
clc

subject = 'Sub16_2'; %Sub16_2 Sub15 Sub17 Sub18 Sub19 Sub19_2 Sub20
electrode = '195';
unit = 'SU5';

event = 'allAppearingAsteroidTimes'; % allAppearingAsteroidTimes allFirstHitTimes missMeanTimes allDisappearingAsteroidTimes
comparison = 'allFirstHitTimes';
comparison2 = 'missMeanTimes';

binWidth = 50;
rateSlidingWindow = 200;
windowPreEvent = 2200;
windowPostEvent = +4000;

%rootDir = 'M:\My Drive\DamisahLab\MATLABData\spaceship';
rootDir = '/Users/mauricio/Library/CloudStorage/GoogleDrive-mm4325@yale.edu/My Drive/DamisahLab/MATLABData/spaceship';
%%
cd (rootDir)
cd (subject)

load allAppearingAsteroidTimes
load allDisappearingAsteroidTimes
load allFirstHitTimes
load missMeanTimes
load ttlTimes

cd ('spikesSemiAuto')
cd (electrode)
cd (unit)

load spikeTimes

eventTimes = eval(event);
spikeTimes = spikeTimes - ttlTimes(1,1);

allSpikesAtEvent =[];
allTrialsForRaster = [];
for i=1:length(eventTimes)
    
    trialsForRaster =[];
    
    spikesAtEventIndex = find(spikeTimes >= (eventTimes(i) - windowPreEvent) & spikeTimes < (eventTimes(i) + windowPostEvent));
    if length(spikesAtEventIndex) < 1
        continue
    end
    
    spikesAtEvent = spikeTimes(spikesAtEventIndex) - eventTimes(i);
    trialsForRaster(1:length(spikesAtEvent),1) = i;
    allSpikesAtEvent = [allSpikesAtEvent; spikesAtEvent];
    allTrialsForRaster = [allTrialsForRaster; trialsForRaster];
    
    h = histogram(allSpikesAtEvent,'BinLimits',[-2000,4000],'BinWidth',binWidth);
    counts = h.Values;
    edges = h.BinEdges';
    
    spikesPerTrial = [(counts/length(eventTimes))'; counts(end)/length(eventTimes)];
    spikesPerTrial = spikesPerTrial * 20;
    
       
end
close
%%
time = (-2000:1:4000)';
j=0;
for j=1:length(time)
    
spikesAtWindow = find(allSpikesAtEvent > (time(j) - rateSlidingWindow) & allSpikesAtEvent <= time(j));
rate(j) = length(spikesAtWindow);

clearvars spikesAtWindow
end

rate = (rate .* 10)';
rate=rate/length(eventTimes);

%% THIS IS THE COMPARISON

compareTimes = eval(comparison);

allSpikesAtCompare =[];
allTrialsForRasterCompare = [];
for i=1:length(compareTimes)
    
    trialsForRasterCompare =[];
    
    spikesAtCompareIndex = find(spikeTimes >= (compareTimes(i) - windowPreEvent) & spikeTimes < (compareTimes(i) + windowPostEvent));
    if length(spikesAtCompareIndex) < 1
        continue
    end
    
    spikesAtCompare = spikeTimes(spikesAtCompareIndex) - compareTimes(i);
    trialsForRasterCompare(1:length(spikesAtCompare),1) = i + allTrialsForRaster(end);
    allSpikesAtCompare = [allSpikesAtCompare; spikesAtCompare];
    allTrialsForRasterCompare = [allTrialsForRasterCompare; trialsForRasterCompare];
    
    hC = histogram(allSpikesAtCompare,'BinLimits',[-2000,4000],'BinWidth',binWidth);
    countsC = hC.Values;
    edgesC = hC.BinEdges';
    
    spikesPerTrialCompare = [(countsC/length(compareTimes))'; countsC(end)/length(compareTimes)];
    spikesPerTrialCompare = spikesPerTrialCompare * 20;
    
       
end
close
%%

j=0;
for j=1:length(time)
    
spikesAtWindowCompare = find(allSpikesAtCompare > (time(j) - rateSlidingWindow) & allSpikesAtCompare <= time(j));
rateCompare(j) = length(spikesAtWindowCompare);

clearvars spikesAtWindowCompare
end

rateCompare = (rateCompare .* 10)';
rateCompare=rateCompare/length(compareTimes);


%% THIS IS THE COMPARISON2

compareTimes2 = eval(comparison2);

allSpikesAtCompare2 =[];
allTrialsForRasterCompare2 = [];
i=0;
for i=1:length(compareTimes2)
    
    trialsForRasterCompare2 =[];
    
    spikesAtCompare2Index = find(spikeTimes >= (compareTimes2(i) - windowPreEvent) & spikeTimes < (compareTimes2(i) + windowPostEvent));
    if length(spikesAtCompare2Index) < 1
        continue
    end
    
    spikesAtCompare2 = spikeTimes(spikesAtCompare2Index) - compareTimes2(i);
    try trialsForRasterCompare2(1:length(spikesAtCompare2),1) = i + allTrialsForRasterCompare(end);
        catch
            continue
    end
    
    allSpikesAtCompare2 = [allSpikesAtCompare2; spikesAtCompare2];
    allTrialsForRasterCompare2 = [allTrialsForRasterCompare2; trialsForRasterCompare2];
    
    hC2 = histogram(allSpikesAtCompare2,'BinLimits',[-2000,4000],'BinWidth',binWidth);
    countsC2 = hC2.Values;
    edgesC2 = hC2.BinEdges';
    
    spikesPerTrialCompare2 = [(countsC2/length(compareTimes2))'; countsC2(end)/length(compareTimes2)];
    spikesPerTrialCompare2 = spikesPerTrialCompare2 * 20;
    
       
end
close
%%

j=0;
for j=1:length(time)
    
spikesAtWindowCompare2 = find(allSpikesAtCompare2 > (time(j) - rateSlidingWindow) & allSpikesAtCompare2 <= time(j));
rateCompare2(j) = length(spikesAtWindowCompare2);

clearvars spikesAtWindowCompare
end

rateCompare2 = (rateCompare2 .* 10)';
rateCompare2=rateCompare2/length(compareTimes2);

%%
f = figure(1);
subplot(2,2,[1,3])
plot(allSpikesAtEvent, allTrialsForRaster, '.k')
hold on
plot(allSpikesAtCompare, allTrialsForRasterCompare, '.r')
plot(allSpikesAtCompare2, allTrialsForRasterCompare2, '.b')
title('Raster')

subplot(2,2,2)
stairs(edges,spikesPerTrial,'k')
hold on
stairs(edges,spikesPerTrialCompare,'r')
stairs(edges,spikesPerTrialCompare2,'b')
title(['PSTH' ' ' 'BinWidth:' mat2str(binWidth) 'ms'])

subplot(2,2,4)
plot(time,rate,'k')
hold on
plot(time,rateCompare,'r')
plot(time,rateCompare2,'b')
title(['Rate' ' ' 'Smoothed:' mat2str(rateSlidingWindow) 'ms'])

cd (rootDir)
