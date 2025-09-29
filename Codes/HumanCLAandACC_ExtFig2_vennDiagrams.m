% Creat venn diagrams using venn function  venn(A, I)
%For three-circle venn diagrams, A is a three element vector [c1 c2 c3], 
%and I is a four element vector [i12 i13 i23 i123], specifiying the 
%two-circle intersection areas i12, i13, i23, and the three-circle
%intersection i123.

clear
clc

% rootDir = 'C:\Users\Dr. Mauricio Pizarro\OneDrive - Yale University\DamisahLab\MATLABData\spaceship';
rootDir = '/Users/mauricio/Library/CloudStorage/OneDrive-YaleUniversity/DamisahLab/MATLABData/spaceship';
region= 'ACC';

cd (rootDir)
cd ('permutationTests10k')
cd (region)
load permMatFDR
permMat = permMatFDR;


appSpecIndx = [];
for i=1:size(permMat,1)
    pApp = permMat{i,4};
    pHit = permMat{i,5};
    pMiss = permMat{i,6};

    if pHit < 0.05 || pMiss < 0.05
        continue
    end

    if  pApp < 0.05
    appSpecIndx= [appSpecIndx; i];
    end
end

hitSpecIndx = [];
for i=1:size(permMat,1)
    pApp = permMat{i,4};
    pHit = permMat{i,5};
    pMiss = permMat{i,6};
    
    if pApp < 0.05 || pMiss < 0.05
        continue
    end

    if  pHit < 0.05
    hitSpecIndx= [hitSpecIndx; i];
    end
end

missSpecIndx = [];
for i=1:size(permMat,1)
    pApp = permMat{i,4};
    pHit = permMat{i,5};
    pMiss = permMat{i,6};
    
    if pApp < 0.05 || pHit < 0.05
        continue
    end

    if  pMiss < 0.05
    missSpecIndx= [missSpecIndx; i];
    end
end

appHitIdx=[]; %i12
for i=1:size(permMat,1)
    pApp = permMat{i,4};
    pHit = permMat{i,5};
    pMiss = permMat{i,6};

    if pMiss < 0.05
        continue
    end

    if  pApp < 0.05 && pHit <0.05
    appHitIdx= [appHitIdx; i];
    end
end

appMissIdx=[];%i13
for i=1:size(permMat,1)
    pApp=permMat{i,4};
    pHit=permMat{i,5};
    pMiss=permMat{i,6};

    if pHit<0.05
        continue
    end

    if pApp<0.05 && pMiss<0.05
        appMissIdx=[appMissIdx;i];
    end
end

hitMissIdx=[];%i23
for i=1:size(permMat,1)
    pApp=permMat{i,4};
    pHit=permMat{i,5};
    pMiss=permMat{i,6};

    if pApp<0.05
        continue
    end

    if pHit<0.05 && pMiss<0.05
        hitMissIdx=[hitMissIdx;i];
    end
end

appHitMissIdx=[];%i123
for i=1:size(permMat,1)
    pApp=permMat{i,4};
    pHit=permMat{i,5};
    pMiss=permMat{i,6};

    if pApp<0.05 && pHit<0.05 && pMiss<0.05
        appHitMissIdx=[appHitMissIdx;i];
    end
end

%[i12 i13 i23 i123]
%1 is app
%2 is hit
%3 is miss

Z=[size(appSpecIndx,1) size(hitSpecIndx,1) size(missSpecIndx,1) size(appHitIdx,1) size(appMissIdx,1) size(hitMissIdx,1) size(appHitMissIdx,1)];
fig=figure();

[H, S] =venn(Z,'FaceColor',...
                            {[0 0.370 0.8410],[0.4940 0.1840 0.5560],[0.4660 0.6740 0.1880]},'FaceAlpha',...
                            {0.2,0.4,0.4},'EdgeColor',[1 1 1]);
