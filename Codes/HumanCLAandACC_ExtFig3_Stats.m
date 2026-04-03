%stats for Extended Fig. 3

clear; clc; close all

rootDir = '/Users/mauricio/Library/CloudStorage/OneDrive-YaleUniversity/DamisahLab/MATLABData/spaceship';
cd(rootDir)

cd ('stats')
cd ('ExtFig3')
cd ('ACC')

load edges
load appEventAppearZScores
load appEventYMoveZScores
load appSpecAppearZScores
load appSpecYMoveZScores
load nonRespZScoresMiss
load nonRespZScoresHit

beforeIndx = find(edges>=-2000 & edges <-1500)';
afterIndx = find(edges>=0 & edges<=2000)';

appEvNeurons(:,1) = mean(appEventAppearZScores(beforeIndx,:))';
appEvNeurons(:,2) = mean(appEventAppearZScores(afterIndx,:))';
appEvNeurons(:,3) = mean(appEventYMoveZScores(beforeIndx,:))';
appEvNeurons(:,4) = mean(appEventYMoveZScores(afterIndx,:))';

appSpecNeurons(:,1) = mean(appSpecAppearZScores(beforeIndx,:))';
appSpecNeurons(:,2) = mean(appSpecAppearZScores(afterIndx,:))';
appSpecNeurons(:,3) = mean(appSpecYMoveZScores(beforeIndx,:))';
appSpecNeurons(:,4) = mean(appSpecYMoveZScores(afterIndx,:))';

pSpecApp = signrank(appSpecNeurons(:,1),appSpecNeurons(:,2));
pSpecY = signrank(appSpecNeurons(:,3),appSpecNeurons(:,4));

pEvApp = signrank(appEvNeurons(:,1),appEvNeurons(:,2));
pEvY = signrank(appEvNeurons(:,3),appEvNeurons(:,4));

[P,H,STATS]=signrank(appEvNeurons(:,4))

mSpec = mean(appSpecAppearZScores,2);
mSpec(:,2)=mean(appSpecYMoveZScores,2);
mEvent = mean(appEventAppearZScores,2);
mEvent(:,2)=mean(appEventYMoveZScores,2);

%%
noRespHit = mean(nonRespZScoresHit(afterIndx,:))';
noRespMiss = mean(nonRespZScoresMiss(afterIndx,:))';

[Pn,Hn,STATSnoResp]=signrank(noRespHit);
