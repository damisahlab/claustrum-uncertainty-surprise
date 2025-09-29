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
respondersMatrix = permMat;

%% Now select which neurons are responsive

appRespIndx = [];
for i=1:size(permMat,1)
    pApp = permMat{i,4};
    pHit = permMat{i,5};
    pMiss = permMat{i,6};

    if pHit < 0.05 || pMiss < 0.05
        continue
    end

    if  pApp < 0.05
    appRespIndx= [appRespIndx; i];
    end
end

hitRespIndx = [];
for i=1:size(permMat,1)
    pApp = permMat{i,4};
    pHit = permMat{i,5};
    pMiss = permMat{i,6};
    
    if pApp < 0.05 || pMiss < 0.05
        continue
    end

    if  pHit < 0.05
    hitRespIndx= [hitRespIndx; i];
    end
end

missRespIndx = [];
for i=1:size(permMat,1)
    pApp = permMat{i,4};
    pHit = permMat{i,5};
    pMiss = permMat{i,6};
    
    if pApp < 0.05 || pHit < 0.05
        continue
    end

    if  pMiss < 0.05
    missRespIndx= [missRespIndx; i];
    end
end

inespRespIndx = [];
for i=1:size(permMat,1)
    pApp = permMat{i,4};
    pHit = permMat{i,5};
    pMiss = permMat{i,6};
    
    if pApp < 0.05
        continue
    end

    if  pMiss < 0.05 && pHit < 0.05
    inespRespIndx= [inespRespIndx; i];
    end
end

appEventIndx =[];
for i=1:size(permMat,1)
    pApp = permMat{i,4};
    pHit = permMat{i,5};
    pMiss = permMat{i,6};
    
    if  pApp <0.05 && pMiss < 0.05 || pApp <0.05 && pHit < 0.05
    appEventIndx= [appEventIndx; i];
    end
end

taskRespIndx = [];
for i=1:size(permMat,1)
    pApp = permMat{i,4};
    pHit = permMat{i,5};
    pMiss = permMat{i,6};
    if  pApp <0.05 || pHit < 0.05 || pMiss<0.05
    taskRespIndx= [taskRespIndx; i];
    end
end

resp=[appEventIndx];
nonResp = find(~ismember(1:105,resp))';
responderIndex = resp;

pathsResp = string(respondersMatrix(responderIndex,1:3));
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
    load missMinTimes
    load yPosMoveTimes
    load yPosNoMoveTimes

    
    eventTimes = allAppearingAsteroidTimes; % allFirstHitTimes missMeanTimes 
    numTrials = length(eventTimes);

    baseEventTimes = allAppearingAsteroidTimes; % appearingForHitTimes appearingForMissTimes
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
     
    clearvars -except responderIndex region baseline parentDir pathsResp event binWidth edges neuron ALLspikesPerTrialEvent ALLspikesPerTrialAppear ALLtrialsEvent ALLrawFRMeans
    
end

close all
%% clean neurons with < 0.5 of FR
eventSpikeMeans = mean(ALLspikesPerTrialEvent(:,:));
eraseEvent = find(eventSpikeMeans<0.01);
ALLspikesPerTrialAppear(:,eraseEvent)=[];
ALLspikesPerTrialEvent(:,eraseEvent)=[];
ALLtrialsEvent(eraseEvent)=[];
pathsResp(eraseEvent,:)=[];

%% check the neurons
baseEdges = find(edges==baseline);
eventEdges = find(edges>0 & edges<=300);
for neuron = 1:size(pathsResp,1)
normFR(neuron,:) = (ALLspikesPerTrialEvent(:,neuron) - min(ALLspikesPerTrialEvent(:,neuron)))./(max(ALLspikesPerTrialEvent(:,neuron)) - min(ALLspikesPerTrialEvent(:,neuron))) ;
end

%% arrange them based on the time of their response

baselineMeanResp = mean(normFR(:,baseEdges),2);
postEventMeanResp = mean(normFR(:,eventEdges),2);
postEventMeanResp(:,2)= 1:size(pathsResp,1);
deltaResp = postEventMeanResp(:,1) - baselineMeanResp;
deltaResp(:,2)=1:size(pathsResp,1);

indexMeanPostEvent=sortrows(postEventMeanResp,1,'ascend');
indexDeltaResp = sortrows(deltaResp,1,'ascend');
heatMapIgor = normFR(indexMeanPostEvent(:,2),:)';

for i = 1: size(pathsResp,1)
    toIgorNumber(i,1:size(edges,2))=i;
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


%% to know which ones are bursters vs pausers, compare deltas before and after event

edgesBaseline = find(edges>=-2000 & edges <-1500);
edgesAfterAppear = find(edges>=0 & edges <=500);
meanBeforeEvent = mean(ALLspikesPerTrialEvent(edgesBaseline,:));
meanPostEvent = mean(ALLspikesPerTrialEvent(edgesAfterAppear,:));
deltas(:,1) = meanBeforeEvent;
deltas(:,2) = meanPostEvent ; 
deltas(:,3) = meanPostEvent - meanBeforeEvent;


bursterIndex=[];
pauserIndex = [];
for i=1:size(pathsResp,1)
    delta = meanPostEvent(i) - meanBeforeEvent(i);
    if delta > 0
        bursterIndex = [bursterIndex;i];
    else
        pauserIndex = [pauserIndex;i];
    end
end

%% check for sanity
% idxPLot= bursterIndex;
% for i=1:size(idxPLot,1)
%     plot(edges,ALLspikesPerTrialEvent(:,idxPLot(i)))
%     pause
% end


%% take the zScores

cd (parentDir)
cd (region)
cd ('spikes')
load appearBurstersIdx
load appearPausersIdx
load appSpecBurstersIdx
load appSpecPausersIdx
load appEventBursters 
load appEventPausers
% 
bursterIndex = appSpecBurstersIdx;
pauserIndex = appSpecPausersIdx ;

ALLZScoresEvent = [];
ALLZScoresAppear = [];
for j=1:size(pathsResp)
    baselineSpikes = ALLspikesPerTrialAppear(baseIndexFR,j);
    meanBaseline = mean(baselineSpikes);
    if meanBaseline == 0
        meanBaseline = mean(ALLspikesPerTrialAppear(:,j));
    end
    stdBaseline = std(baselineSpikes);
    if stdBaseline == 0
        stdBaseline = std(ALLspikesPerTrialAppear(:,j));
    end
    zScoreNeuron = (ALLspikesPerTrialEvent(:,j) - meanBaseline) ./ stdBaseline;
    zScoreAppear = (ALLspikesPerTrialAppear(:,j) - meanBaseline) ./ stdBaseline;
    ALLZScoresEvent = [ALLZScoresEvent zScoreNeuron];
    ALLZScoresAppear = [ALLZScoresAppear zScoreAppear];
    clear baselineSpikes meanBaseline stdBaseline zScoreNeuron zScoreAppear
end

zScoresBursters = ALLZScoresEvent(:,bursterIndex);
zScoresPausers = ALLZScoresEvent(:,pauserIndex);
zScoresBurstersApp = ALLZScoresAppear(:,bursterIndex);
zScoresPausersApp = ALLZScoresAppear(:,pauserIndex);


%%

% smooth it with a moving average (4bins = 200-ms)
movAvgDeltasBursters = movmean(zScoresBursters,4);
movAvgDeltasBurstApp = movmean(zScoresBurstersApp,4);
movAvgDeltasPausers = movmean(zScoresPausers,4);
movAvgDeltasPauseApp = movmean(zScoresPausersApp,4);

toIgorEvent(:,1) = mean(movAvgDeltasBursters,2);
toIgorEvent(:,2) = std(movAvgDeltasBursters')./sqrt(size(bursterIndex,1));
toIgorEvent(:,3) = mean(movAvgDeltasPausers,2);
toIgorEvent(:,4) = std(movAvgDeltasPausers')./sqrt(size(pauserIndex,1));

toIgorAppear(:,1) = mean(movAvgDeltasBurstApp,2);
toIgorAppear(:,2) = std(movAvgDeltasBurstApp')./sqrt(size(bursterIndex,1));
toIgorAppear(:,3) = mean(movAvgDeltasPauseApp,2);
toIgorAppear(:,4) = std(movAvgDeltasPauseApp')./sqrt(size(pauserIndex,1));

movAvgDeltasAll = movmean(abs(ALLZScoresAppear),4);
toIgorAppearAll(:,1)=mean(movAvgDeltasAll,2);
toIgorAppearAll(:,2)=std(movAvgDeltasAll')./sqrt(size(ALLZScoresAppear,2));
%% want to calculate the abs deltas from spikes, for dotplots

edgesAfterAppear = find(edges>=0  & edges <=500);
edgesBeforeAppear = find(edges>=-500 & edges<=-50);
peakHitClaus = find(edges>=1000 & edges <=1500);

load ALLspikesHitApp
load ALLspikesMissApp
load ALLspikesAppear
load ALLspikesHit
load ALLspikesMiss

spikesRespBase = mean(ALLspikesAppear(baseIndexFR,responderIndex));
spikesPreApp = mean(ALLspikesAppear(edgesBeforeAppear,responderIndex));
spikesRespApp = mean(ALLspikesAppear(edgesAfterAppear,responderIndex));

deltaRespPreApp = abs(spikesPreApp - spikesRespBase)';
deltaRespApp = abs(spikesRespApp - spikesRespBase)';

igorLinesDeltaAppear(1,:) = deltaRespPreApp';
igorLinesDeltaAppear(2,:) = deltaRespApp';

[pDeltasApp] = signrank(deltaRespPreApp,deltaRespApp)


%% 

edgesAfterEvent = find(edges>=0 & edges <=500);
peakHitClaus = find(edges>=650 & edges <=1150);

spikesRespBaseHit = mean(ALLspikesHitApp(baseIndexFR,responderIndex));
spikesPreAppHit = mean(ALLspikesHitApp(edgesBeforeAppear,responderIndex));
spikesRespHit = mean(ALLspikesHit(edgesAfterEvent,responderIndex));

spikesRespBaseMiss = mean(ALLspikesMissApp(baseIndexFR,responderIndex));
spikesPreAppMiss = mean(ALLspikesMissApp(edgesBeforeAppear,responderIndex));
spikesRespMiss = mean(ALLspikesMiss(edgesAfterEvent,responderIndex));

deltaRespPreAppHit = abs(spikesPreAppHit - spikesRespBaseHit)';
deltaRespPreAppMiss = abs(spikesPreAppMiss - spikesRespBaseMiss)';
deltaRespHit = abs(spikesRespHit - spikesRespBaseHit)';
deltaRespMiss = abs(spikesRespMiss - spikesRespBaseMiss)';

igorLinesDeltaEvent(1,:) = deltaRespMiss';
igorLinesDeltaEvent(2,:) = deltaRespHit';

[pDeltasHitMiss] = signrank(deltaRespHit,deltaRespMiss)
pDeltasEvAppHit = signrank(deltaRespPreAppHit,deltaRespHit)
pDeltasEvAppMiss = signrank(deltaRespPreAppMiss,deltaRespMiss)
