%-----------------------------------------------------------------------------------------
% Script for behavioral analysis of the spaceship task per participant
%
% Human claustrum neurons encode uncertainty and prediction errors during aversive learning
% Figure   1e
% Author:  Mauricio Medina
% License: 
%-----------------------------------------------------------------------------------------

% clean workspace
clear; clc; close all;

% load data
data_path = fullfile(fileparts(fileparts(pwd)), 'Data'); 
load(fullfile(data_path, 'Fig1e_TaskExample.mat'))
addpath(fullfile(fileparts(pwd), 'OnPathCodes')) % add path to helper tools 

% populate data per trial
aSafePerTrial = [];
for i=1:length(dataForEveryTrial) 
    trialASafe = dataForEveryTrial{1,i}(1,7);
    aSafePerTrial = [aSafePerTrial; trialASafe];
end

bSafePerTrial = [];
for i=1:length(dataForEveryTrial)
    trialBSafe = dataForEveryTrial{1,i}(1,8);
    bSafePerTrial = [bSafePerTrial; trialBSafe];
end

% clean the safe zones into NaN

aSafePerTrial(aSafePerTrial==-999)=NaN;
bSafePerTrial(bSafePerTrial==-999)=NaN;

A=aSafePerTrial;
B=bSafePerTrial;

holeAUp = ((A + 1) * 25 + 10)./600;
holeADown = ((A - 4) * 25 + 10)./600;
holeBDown = ((B - 1) * 25 + 10)./600;
holeBUp = ((B + 4) * 25 + 10)./600;

trialNumber = 1:length(dataForEveryTrial);
for i=1:size(trialNumber,2)
    yPosFig(i,1) = mean(dataForEveryTrial{1,i}(:,6)); %trial samples asteroids health time yPos aSafe bSafe
    healthFig(i,1) = mean(dataForEveryTrial{1,i}(:,4));
end

healthFig = lowpassFiltering(healthFig,100,2);
yPosFig = lowpassFiltering(yPosFig,100,2)./600;

f = figure(1);
clf
f.Position = [0 600 600 350];
fontname('Arial')
plot(trialNumber,healthFig,'b')
hold on
plot(trialNumber,yPosFig,'k')
plot(trialNumber,holeAUp/600+1, '.m')
plot(trialNumber,B/600, '.r')
xlim([1 350])
ylim([-0.1 1.3])
fontsize(15,'pixels')
xlabel('Trial','FontSize',15)
ylabel('Health','FontSize',15)
sgt= sgtitle('Figure 1e. Example of Behavioral Responses');
sgt.FontSize = 12;
lgd = legend("Health","Y Position","Safety A", "Safety B");
lgd.Location = "north";
lgd.Orientation = "horizontal";
