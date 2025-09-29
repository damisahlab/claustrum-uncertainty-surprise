% Plotting PSTHs based on permutation tests responders

clear
clc
close all

parentDir = '/Users/mauricio/Library/CloudStorage/OneDrive-YaleUniversity/DamisahLab/MATLABData/spaceship';
event = 'allAppearingAsteroidTimes'; % allAppearingAsteroidTimes allFirstHitTimes missMeanTimes
region = 'claustrum';
binWidth = 50;
baseline=[-2000;-1500];


cd (parentDir)
cd ('permutationTests10k')
cd (region)

load permMatFDR
permMat = permMatFDR;
clear permMatFDR

%% Now select which neurons are responsive

responderIndex = [];
for i=1:size(permMat,1)
    pApp = permMat{i,4};
    pHit = permMat{i,5};
    pMiss = permMat{i,6};
    if pApp < 0.05 || pHit < 0.05 || pMiss < 0.05
    responderIndex= [responderIndex; i];
    end
end

pathsResp = string(permMat(responderIndex,1:3));

%% Now navigate through neurons

ALLspikesPerTrialAppear = [];
ALLspikesPerTrialEvent = [];
ALLtrialsEvent = [];

for neuron = 1:size(pathsResp,1)

    cd (parentDir)
    cd (pathsResp(neuron,1))
    load allAppearingAsteroidTimes %all event times are in ms. First TTLbnc is 0
    load allFirstHitTimes
    load missMeanTimes
    load appearingForHitTimes
    load appearingForMissTimes
    load ttlTimes % in ms
    
    eventTimes = eval(event);
    numTrials = length(eventTimes);

    baseEventTimes = allAppearingAsteroidTimes;
    baseNumTrials = length(baseEventTimes);
    
    cd ('spikesSemiAuto')
    cd (pathsResp(neuron,2))
    cd (pathsResp(neuron,3))
    
    load spikeTimes
    spikeTimes = spikeTimes - ttlTimes(1,1);
    
    cellEventSpikeTimesAppear = [];
        
    j=0;
    for j = 1:length(baseEventTimes)
        spikesAtEventIndexAppear = find(spikeTimes >= (baseEventTimes(j)-2000) & spikeTimes < (baseEventTimes(j)+4000));
        if length(spikesAtEventIndexAppear) < 1
            continue
        end
        spikeTimesAtAppearEvent = spikeTimes(spikesAtEventIndexAppear)-baseEventTimes(j);
        cellEventSpikeTimesAppear = [cellEventSpikeTimesAppear; spikeTimesAtAppearEvent(:)];
    end
    
    hB = histogram(cellEventSpikeTimesAppear,500,'BinLimits',[-2000,4000],'BinWidth',binWidth);
    countsAppear = hB.Values;
    edgesAppear = hB.BinEdges';

    totalTrialsAppear = sum(baseNumTrials);
    spikesPerTrialAppear = [(countsAppear/totalTrialsAppear)'; countsAppear(end)/totalTrialsAppear];

    spikesPerTrialAppear = (1000.*spikesPerTrialAppear)./binWidth;
            
    ALLspikesPerTrialAppear = [ALLspikesPerTrialAppear spikesPerTrialAppear];
    
    
    cellEventSpikeTimes = [];
        
    s=0;
    for s = 1:length(eventTimes)
        spikesAtEventIndex = find(spikeTimes >= (eventTimes(s)-2000) & spikeTimes < (eventTimes(s)+4000));
        if length(spikesAtEventIndex) < 1
            continue
        end
        spikeTimesAtEvent = spikeTimes(spikesAtEventIndex)-eventTimes(s);
        cellEventSpikeTimes = [cellEventSpikeTimes; spikeTimesAtEvent(:)];
    end
    
    h = histogram(cellEventSpikeTimes,500,'BinLimits',[-2000,4000],'BinWidth',binWidth);
    counts = h.Values;
    edges = h.BinEdges;

    totalTrials = sum(numTrials);
    spikesPerTrialEvent = [(counts/totalTrials)'; counts(end)/totalTrials];

    spikesPerTrialEvent = (1000.*spikesPerTrialEvent)./binWidth;
            
    ALLspikesPerTrialEvent = [ALLspikesPerTrialEvent spikesPerTrialEvent];
    
    ALLtrialsEvent = [ALLtrialsEvent; totalTrials];
     
    clearvars -except baseline parentDir pathsResp event binWidth edges neuron ALLspikesPerTrialEvent ALLspikesPerTrialAppear ALLtrialsEvent

end
    
%% check the neurons
baseEdges = find(edges==baseline);
eventEdges = find(edges>0 & edges<300);
for neuron = 1:size(pathsResp,1)
normFR(neuron,:) = (ALLspikesPerTrialEvent(:,neuron) - min(ALLspikesPerTrialEvent(:,neuron)))./(max(ALLspikesPerTrialEvent(:,neuron)) - min(ALLspikesPerTrialEvent(:,neuron))) ;
end

%% arrange them based on the time of their response

baselineMeanResp = mean(normFR(:,baseEdges),2);
postEventMeanResp = mean(normFR(:,eventEdges),2);
postEventMeanResp(:,2)= 1:size(pathsResp,1);
deltaResp = postEventMeanResp(:,1) - baselineMeanResp;
deltaResp(:,2)=1:size(pathsResp,1);

indexMeanPostEvent=sortrows(postEventMeanResp,1,'descend');
indexDeltaResp = sortrows(deltaResp,1,'descend');
heatMapIgor = normFR(indexMeanPostEvent(:,2),:)';

for i = 1: size(pathsResp,1)
    toIgorNumber(i,1:size(edges,2))=i;
end

%% categorize neurons by suppressors and bursters, and plot the z-score
spikesBaseline = ALLspikesPerTrialEvent(baseEdges(1):baseEdges(2),:);
meanSpikesBaseline = mean(spikesBaseline);
stdSpikesBaseline = std(spikesBaseline);

allZScores =[];
for i=1:size(pathsResp,1)
    zScore = (ALLspikesPerTrialEvent(:,i) - meanSpikesBaseline(1,i))./(stdSpikesBaseline(1,i));
    allZScores = [zScore allZScores];
    clearvars zScore
end

%% max change in FR
% first calculate an absolute delta from a mean of baseline
% then normalize that to 0 or 1, so that the biggest change is 1
% then plot them on heatmap to see when they respond, to appear and to
% event


baseIndexFR = find(edges>=-2000 & edges <-1500);
meanBaselineFR = mean(ALLspikesPerTrialAppear(baseIndexFR,:));

allAbsDeltas =[];
for i=1:size(pathsResp,1)
    absDelta = abs((ALLspikesPerTrialEvent(:,i) - meanBaselineFR(1,i)));
    allAbsDeltas = [allAbsDeltas absDelta];
    clearvars absDelta
end

% now normalize those changes in firing rate
for neuron=1:size(pathsResp,1)
    normDelta(:,neuron) = (allAbsDeltas(:,neuron) - min(allAbsDeltas(:,neuron)))./(max(allAbsDeltas(:,neuron)) - min(allAbsDeltas(:,neuron))) ;
end

%% sort responses using time, as early as possible bottom
% average per 2 bins
% check during the first 1000ms

binsToCheck = find(edges>=0 & edges <3000);
deltaCheck = normDelta(binsToCheck,:);

for neuron=1:size(pathsResp,1)
    x = find(deltaCheck(:,neuron) == max(deltaCheck(:,neuron)));
    maxBinPerNeuron(neuron,1) = x(1,1);
    maxBinPerNeuron(neuron,2) = neuron;
    clearvars x
end


indexSortedPerBin = sortrows(maxBinPerNeuron,1);
heatmapNormDeltaIgor = normDelta(:,indexSortedPerBin(:,2));

%% for positive or negative check at 0-300 if the delta increased or not
baseDelta = find(edges>=-300 & edges <0);

surf(edges,toIgorNumber,heatmapNormDeltaIgor')
