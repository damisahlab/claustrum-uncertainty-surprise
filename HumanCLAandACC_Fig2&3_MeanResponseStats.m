    % Plotting PSTHs based on permutation tests responders

clear
clc
close all

parentDir = '/Users/mauricio/Library/CloudStorage/OneDrive-YaleUniversity/DamisahLab/MATLABData/spaceship';
event = 'allAppearingAsteroidTimes'; % allAppearingAsteroidTimes allFirstHitTimes missMeanTimes
region = 'ACC';
binWidth = 50;
baseline=[-2000;-1500];

cd (parentDir)
cd ('permutationTests10k')
cd (region)

load permMatFDR
permMat=permMatFDR;
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

appMissIndx = [];
for i=1:size(permMat,1)
    pApp = permMat{i,4};
    pHit = permMat{i,5};
    pMiss = permMat{i,6};
    
    if pHit<0.05
        continue
    end

    if  pApp <0.05 && pMiss < 0.05
    appMissIndx= [appMissIndx; i];
    end
end

appHitIndx = [];
for i=1:size(permMat,1)
    pApp = permMat{i,4};
    pHit = permMat{i,5};
    pMiss = permMat{i,6};
    
    if pMiss<0.05
        continue
    end

    if  pApp <0.05 && pHit < 0.05
    appHitIndx= [appHitIndx; i];
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

allAppear = [appRespIndx;appEventIndx];
eventNeuronsIndx=[hitRespIndx;missRespIndx;inespRespIndx];
missGroup = [appMissIndx;missRespIndx;inespRespIndx];
hitGroup = [appHitIndx;hitRespIndx;inespRespIndx];

resp=appEventIndx;
nonResp = find(~ismember(1:105,resp))';
responderIndex = resp;

pathsResp = string(respondersMatrix(responderIndex,1:6));

%% load the spikes per neuron

cd (parentDir)
cd (region)
cd ('spikes')
load ALLspikesHitApp
load ALLspikesMissApp
load ALLspikesAppear
load ALLspikesHit
load ALLspikesMiss
load edges

baseWindow = [-2000;-1500];
eventWindow = [0;500];

%% clean neurons with < 0.05 of FR throuhout recording

eraseApp = find(mean(ALLspikesAppear(:,appRespIndx))<0.05);
eraseAppIdx = appRespIndx(eraseApp,1);
eraseHit = find(mean(ALLspikesHit(:,hitRespIndx))<0.05);
eraseHitIdx = hitRespIndx(eraseHit,1);
eraseMiss = find(mean(ALLspikesMiss(:,missRespIndx))<0.05);
eraseMissIdx = missRespIndx(eraseMiss,1);

eraseInesH = find(mean(ALLspikesHit(:,inespRespIndx))<0.05);
eraseInesM = find(mean(ALLspikesMiss(:,inespRespIndx))<0.05);
eraseInes = unique([eraseInesH eraseInesM]);
eraseInesIdx = unique(inespRespIndx([eraseInesH eraseInesM]));

appRespIndx(eraseApp)=[];
hitRespIndx(eraseHit)=[];
missRespIndx(eraseMiss)=[];
inespRespIndx(eraseInesH)=[];
%% take the zScores

baseIndx = find(edges>=baseWindow(1,1) & edges<=baseWindow(2,1));

ALLZScoresHit = [];
ALLZScoresHitApp =[];

for j=1:size(permMat,1)
    baselineSpikes = ALLspikesHitApp(baseIndx,j);
    meanBaseline = mean(baselineSpikes);
    if meanBaseline == 0
        meanBaseline = mean(ALLspikesHitApp(:,j));
    end
    stdBaseline = std(baselineSpikes);
    if stdBaseline == 0
        stdBaseline = std(ALLspikesHitApp(:,j));
    end
    zScoreNeuron = (ALLspikesHit(:,j) - meanBaseline) ./ stdBaseline;
    zScoreAppear = (ALLspikesHitApp(:,j) - meanBaseline) ./ stdBaseline;
    ALLZScoresHit = [ALLZScoresHit zScoreNeuron];
    ALLZScoresHitApp = [ALLZScoresHitApp zScoreAppear];
    clear baselineSpikes meanBaseline stdBaseline zScoreNeuron zScoreAppear
end

ALLZScoresMiss = [];
ALLZScoresMissApp = [];
j=0;
for j=1:size(permMat,1)
    baselineSpikes = ALLspikesMissApp(baseIndx,j);
    meanBaseline = mean(baselineSpikes);
    if meanBaseline == 0
        meanBaseline = mean(ALLspikesMissApp(:,j));
    end
    stdBaseline = std(baselineSpikes);
    if stdBaseline == 0
        stdBaseline = std(ALLspikesMissApp(:,j));
    end
    zScoreNeuron = (ALLspikesMiss(:,j) - meanBaseline) ./ stdBaseline;
    zScoreAppear = (ALLspikesMissApp(:,j) - meanBaseline) ./ stdBaseline;
    ALLZScoresMiss = [ALLZScoresMiss zScoreNeuron];
    ALLZScoresMissApp = [ALLZScoresMissApp zScoreAppear];
    clear baselineSpikes meanBaseline stdBaseline zScoreNeuron zScoreAppear
end

%% smooth it with a moving average (4bins = 200-ms)

eventNeuZScoresHit = abs(ALLZScoresHit(:,responderIndex));
eventNeuZScoresMiss = abs(ALLZScoresMiss(:,responderIndex));

movAvgHit = movmean(eventNeuZScoresHit,4);
movAvgMiss = movmean(eventNeuZScoresMiss,4);

toIgorHitMiss(:,1) = mean(movAvgHit,2);
toIgorHitMiss(:,2) = std(movAvgHit')./sqrt(size(responderIndex,1));
toIgorHitMiss(:,3) = mean(movAvgMiss,2);
toIgorHitMiss(:,4) = std(movAvgMiss')./sqrt(size(responderIndex,1));

figure(1)
plot(edges,toIgorHitMiss(:,1),'r')
hold on
plot(edges,toIgorHitMiss(:,3),'b')
%% want to calculate the abs deltas from spikes, for dotplots
% For Appear claus bursters and pausers

edgesAfterAppear = find(edges>=0  & edges <=500);
edgesBeforeAppear = find(edges>=-500 & edges<=-50);

spikesRespBase = mean(ALLspikesAppear(baseIndx,responderIndex));
spikesPreApp = mean(ALLspikesAppear(edgesBeforeAppear,responderIndex));
spikesRespApp = mean(ALLspikesAppear(edgesAfterAppear,responderIndex));

deltaRespPreApp = abs(spikesPreApp - spikesRespBase)';
deltaRespApp = abs(spikesRespApp - spikesRespBase)';

igorLinesDeltaAppear(1,:) = deltaRespPreApp';
igorLinesDeltaAppear(2,:) = deltaRespApp';

[pDeltasApp] = signrank(deltaRespPreApp,deltaRespApp)


%% Now lets do it for hit vs miss in appear neurons

edgesAfterEvent = find(edges>=0 & edges <=500);

spikesRespBaseHit = mean(ALLspikesHitApp(baseIndx,responderIndex));
spikesPreAppHit = mean(ALLspikesHitApp(edgesBeforeAppear,responderIndex));
spikesRespHit = mean(ALLspikesHit(edgesAfterEvent,responderIndex));

spikesRespBaseMiss = mean(ALLspikesMissApp(baseIndx,responderIndex));
spikesPreAppMiss = mean(ALLspikesMissApp(edgesBeforeAppear,responderIndex));
spikesRespMiss = mean(ALLspikesMiss(edgesAfterEvent,responderIndex));

deltaRespPreAppHit = abs(spikesPreAppHit - spikesRespBaseHit)';
deltaRespPreAppMiss = abs(spikesPreAppMiss - spikesRespBaseMiss)';
deltaRespHit = abs(spikesRespHit - spikesRespBaseHit)';
deltaRespMiss = abs(spikesRespMiss - spikesRespBaseMiss)';

igorLinesDeltaEvent(1,:) = deltaRespMiss';
igorLinesDeltaEvent(2,:) = deltaRespHit';

[pDeltasEv] = signrank(deltaRespHit,deltaRespMiss)
pDeltasEvAppHit = signrank(deltaRespPreAppHit,deltaRespHit);
pDeltasEvAppMiss = signrank(deltaRespPreAppMiss,deltaRespMiss);
