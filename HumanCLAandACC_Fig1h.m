%-----------------------------------------------------------------------------------------
% Script for behavioral analysis of the position from safety for each task
% event
%
% Human claustrum neurons encode uncertainty and prediction errors during aversive learning
% Figure   1h
% Author:  Mauricio Medina
% License: 
%-----------------------------------------------------------------------------------------

% clean workspace
clear; clc; close all;

% Navigate to ...GitHub/Codes/FigureCodes

% load data
data_path = fullfile(fileparts(fileparts(pwd)), 'Data'); 
load(fullfile(data_path, 'Fig1h_DistanceToSafety.mat'))
addpath(fullfile(fileparts(pwd), 'OnPathCodes')) % add path to helper tools 

allDistToSafetyHit = Fig1h_DistanceToSafety{1,2};
allDistToSafetyMiss = Fig1h_DistanceToSafety{2,2};



%plot distributions
f = figure(1);
clf
f.Position = [0 600 600 350];
fontname('Arial')
hHit=histogram(allDistToSafetyHit,'BinLimits',[0,600], 'Binwidth', 25, 'FaceColor',[0.6 0.2 0.8]);
hold on
hMiss=histogram(allDistToSafetyMiss,'BinLimits',[0,600], 'Binwidth', 25,'FaceColor',[0 0.8 0]);
xlim([0 600])
ylim([0 600])
fontsize(12,'pixels')
xlabel('Y position from safety','FontSize',12)
ylabel('Number of trials','FontSize',12)
sgt= sgtitle('Figure 1h. Distance from safety for each event');
sgt.FontSize = 13;
lgd = legend("Crash","Avoidance",'FontSize',10);
lgd.Location = "northeast";