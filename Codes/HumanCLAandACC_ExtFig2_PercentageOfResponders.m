% calculating % responders per session
% for each session you need at least 5 neurons

clear;clc;close all

rootDir='/Users/mauricio/Library/CloudStorage/OneDrive-YaleUniversity/DamisahLab/MATLABData/spaceship';

%% claustrum
cd (rootDir)
cd ('permutationTests10k')
cd ('claustrum')
load permMatFDR
permMat = permMatFDR;

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

eventIndx = sort([hitRespIndx;missRespIndx;inespRespIndx]);

s1Idx=1:34; s2Idx=35:71; s3Idx=72:87; s4Idx=89:105;
 % we'll avoid Sub19_2 since it only has one neuron
sessions={s1Idx s2Idx s3Idx s4Idx};

for i=1:4
    claPercent(i,1)=sum(ismember(appRespIndx,sessions{i}))./sessions{i}(end);
    claPercent(i,2)=sum(ismember(appEventIndx,sessions{i}))./sessions{i}(end);
    claPercent(i,3)=sum(ismember(eventIndx,sessions{i}))./sessions{i}(end);
end

claPercent = claPercent*100; %appSpecific app+Eve Event
%claPercent(claPercent == 0) = NaN;
p=nan(1,3);
for i=1:3
[p(i),~,~]=signrank(claPercent(:,i))
end

%% ACC

clearvars permMat appRespIndx appEventIndx eventIndx hitRespIndx missRespIndx inespRespIndx

cd (rootDir)
cd ('permutationTests10k')
cd ('ACC')
load permMatFDR
permMat = permMatFDR;


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

eventIndx = sort([hitRespIndx;missRespIndx;inespRespIndx]);

s1=1:21; s2=22:35; s3=36:49; s4=50:61; s5=62:67; s6=70:76;
 % we'll avoid Sub23 since it only has 2 neurons
sessionsACC={s1 s2 s3 s4 s5 s6};


for i=1:6
    accPercent(i,1)=sum(ismember(appRespIndx,sessionsACC{i}))./sessionsACC{i}(end);
    accPercent(i,2)=sum(ismember(appEventIndx,sessionsACC{i}))./sessionsACC{i}(end);
    accPercent(i,3)=sum(ismember(eventIndx,sessionsACC{i}))./sessionsACC{i}(end);
end

accPercent = accPercent*100; %appSpecific app+Eve Event
%accPercent(accPercent == 0) = NaN;
pACC=nan(1,3);
for i=1:3
[pACC(i),~,~]=signrank(accPercent(:,i)-5);
end

[~,pA]=ttest2(accPercent(:,1),claPercent(:,1))
pAE=ranksum(accPercent(:,2),claPercent(:,2))
pE=ranksum(accPercent(:,3),claPercent(:,3))

pAcc=nan(1,3);
for i=1:3
[pAcc(i),~,~]=signrank(accPercent(:,i))
end

p = kruskalwallis(claPercent)
p = kruskalwallis(accPercent)