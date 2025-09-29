%% getting neurons with perm test

clear
close all
clc

rootDir = '/Users/mauricio/Library/CloudStorage/OneDrive-YaleUniversity/DamisahLab/MATLABData/spaceship';

cd (rootDir)

counterPermMat = 0;

binWidth = 50;
subjects = {'Sub15';'Sub16';'Sub16_2';'Sub17';'Sub18';'Sub19';'Sub19_2';'Sub20'; 'Sub23';'Sub24'}; 
region = 'claustrum'; % claustrum, insulaAnt, insulaPost, ACC, amygdala, hippocampus, orbitalis, OFC, SMA, calcarine, pulvinar
baseline=[-2000;-1500];
event = 'missMeanTimes'; % allAppearingAsteroidTimes missMeanTimes allFirstHitTimes 

%%

for sub = 1:length(subjects)

cd (rootDir)
cd (subjects{sub})

try
    electrodes = load (region);
catch
    continue
end

    load allAppearingAsteroidTimes %all event times are in ms. First TTLbnc is 0
    load allFirstHitTimes
    load missMeanTimes
    load appearingForHitTimes
    load appearingForMissTimes
    load ttlTimes % in ms

eventTimes = eval(event);
baseEventTimes = appearingForMissTimes;


electrodes = struct2array(electrodes);
electrodes = string(electrodes);

cd ('spikesSemiAuto')
spikesDir = pwd;
%%

for m=1:length(electrodes)

cd (spikesDir)

cd (electrodes(m))

electrodeDir = pwd;

foldersPerElectrode = struct2cell(dir(pwd));
nameFolders = foldersPerElectrode(1,:);
unitFolderNames = nameFolders(contains(nameFolders,'SU'));

for unit=1:length(unitFolderNames)

cd (electrodeDir)
cd (unitFolderNames{unit})

load spikeTimes
spikeTimes = spikeTimes - ttlTimes(1,1);

for s=1:length(eventTimes)
    
    spikesAtEventIndex = find(spikeTimes >= (eventTimes(s) - 2000) & spikeTimes < (eventTimes(s) + 2000));

    spikeTimesAlignedToEvent = spikeTimes(spikesAtEventIndex)-eventTimes(s);
    
    [countsPerTrial,edgesPerTrial] = histcounts(spikeTimesAlignedToEvent,'BinLimits', [-2000,2000],'BinWidth',50);

    spikesPerSingleTrial = [countsPerTrial countsPerTrial(end)];
    spikesPerTrial(s,1:length(spikesPerSingleTrial))= spikesPerSingleTrial;
    
clearvars spikesPerSingleTrial
end

for q=1:length(baseEventTimes)
    
    spikesAtEventIndexBase = find(spikeTimes >= (baseEventTimes(q) - 2000) & spikeTimes < (baseEventTimes(q) + 2000));

    spikeTimesAlignedToBase = spikeTimes(spikesAtEventIndexBase)-baseEventTimes(q);
    
    [countsPerTrialBase,edgesPerTrialBase] = histcounts(spikeTimesAlignedToBase,'BinLimits', [-2000,2000],'BinWidth',50);

    spikesPerSingleTrialBase = [countsPerTrialBase countsPerTrialBase(end)];
    spikesPerTrialBase(q,1:length(spikesPerSingleTrialBase))= spikesPerSingleTrialBase;
    
clearvars spikesPerSingleTrialBase
end

baselinePerm = find(edgesPerTrialBase>=baseline(1,1) & edgesPerTrialBase<=baseline(2,1));
eventPerm = find(edgesPerTrial >=0 & edgesPerTrial <= 500);

spikesBaselinePerm = mean(spikesPerTrialBase(:,baselinePerm),2);
spikesEventPerm = mean(spikesPerTrial(:,eventPerm),2);

[p, observeddifference, effectsize] = permutationTest(spikesEventPerm, spikesBaselinePerm, 10000);


counterPermMat = counterPermMat + 1;

permMat{counterPermMat,1} = subjects(sub);
permMat{counterPermMat,2} = electrodes(m);
permMat{counterPermMat,3} = unitFolderNames{unit};
permMat{counterPermMat,4} = p;

clearvars p spikesEvent spikesBaseline spikeTimes spikesAtEventIndex spikeTimesAlignedToEvent
clearvars hTrial countsPerTrial edgesPerTrial spikesPerTrial
clearvars hBase countsPerTrialBase edgesPerTrialBase spikesPerTrialBase

end

end

%%
clearvars -except permMat sub rootDir counterPermMat binWidth subjects region baseline event              

cd (rootDir)

end