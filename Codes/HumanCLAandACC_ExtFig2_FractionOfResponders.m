% selecting fraction of responders per 200 ms, with perm test

clear
clc

parentDir = '/Users/mauricio/Library/CloudStorage/OneDrive-YaleUniversity/DamisahLab/MATLABData/spaceship';

region = 'ACC';

cd (parentDir)
cd ('permutationTests10k')
cd (region)
load permMatFDR
permMat = permMatFDR;
load edges

paths = string(permMat(:,1:3));

baseWindow = [-2000;-1500];
bin = 50;

% then calculate the baseline
% then calculate a p for each 200-ms window, for each neuron
% then calculate the fraction of neurons that have significance from
% baseline


%% we are gonna perform permutations every 200ms
% first calculate firing rate for every 50ms bin, for each trial and each
% neuron, all aligned to appear

baseIndx = find(edges>=baseWindow(1,1) & edges<=baseWindow(2,1));

for n=1:size(paths)
cd (parentDir)
cd (paths(n,1))

load allAppearingAsteroidTimes
load ttlTimes

cd ('spikesSemiAuto')
cd (paths(n,2))
cd (paths(n,3))

load spikeTimes

spikeTimes = spikeTimes - ttlTimes(1,1);

% calculate fr for appearing

for j = 1:length(allAppearingAsteroidTimes)
    spikesAtIndxApp = find(spikeTimes >= (allAppearingAsteroidTimes(j)-2000) & spikeTimes < (allAppearingAsteroidTimes(j)+4000));
    if length(spikesAtIndxApp) < 1
        continue
    end
    
    spikeTimesAlignedAppear = spikeTimes(spikesAtIndxApp)-allAppearingAsteroidTimes(j);
    
    hApp=histcounts(spikeTimesAlignedAppear, 'BinLimits', [-2000,4000],'BinWidth',50);
    hApp = [hApp hApp(end)];

    %now take the mean spikes at baseline and at every 200ms, starting
    %from -2 seconds
    spikesBaseline = mean(hApp(baseIndx));
    
     % populate the matrix per cell for perm test
    neuronAppSpikes(j,1) = spikesBaseline;

    % make a for loop for all the 200-ms bins that you are using, it should
    % be a matrix of 6000/200=30
    
    totBins = 6000/bin;
    matCount = 2:(6000/bin);
    
    for m=1:size(matCount,2)
    
    window = find(edges>=(-2000 + bin*m) & edges<=(-2000 + bin*m+bin));

    spikesAtWindow = mean(hApp(window));
   
    neuronAppSpikes(j,matCount(m)) = spikesAtWindow;

    clearvars spikesAtWindow window
    end

    clearvars spikeTimesAlignedAppear hApp spikesBaseline
end



%populate the cell, with spikes per neuron
spikesNeuron{1,n}=neuronAppSpikes;

clearvars neuronAppSpikes 

end

%% Now run the permutations

permutationCell = permMat(:,1:3);

for q=1:size(permutationCell,1)
dataNeuron = spikesNeuron{1,q};
baseline = dataNeuron(:,1);

for w = 2:size(dataNeuron,2)
    spks = dataNeuron(:,w);
    [p, observeddifference, effectsize] = permutationTest(baseline, spks, 1000);
    permutationCell{q,w+2}=p;
    clearvars spks p
end

clearvars baseline dataNeuron

end

%% Correct for multiple comparisons
permCellFDR = permutationCell(:,1:3);
for i=1:size(permutationCell,1)
    pValues = cell2mat(permutationCell(i,4:end));
    q_values = mafdr(pValues, 'BHFDR', true);
    for p = 4:size(permutationCell,2)
        permCellFDR{i,p}=q_values(p-3);
    end
    clearvars q_values pValues
end

%% now fraction of responders per 200-ms bins
fraction = [];
for f=4:size(permCellFDR,2)
data = cell2mat(permCellFDR(:,f));
resp = sum(data<0.05);
fraction=[fraction; resp];
clearvars data resp
end

fraction = [fraction(1,1); fraction; fraction(end)];
fraction = fraction/size(permMat,1);
toIgor = movmean(fraction,4);