%-----------------------------------------------------------------------------------------
% Script for behavioral analysis of spaceship displacement after event
%
% Human claustrum neurons encode uncertainty and prediction errors during aversive learning
% Figure   1i
% Author:  Mauricio Medina
% License: 
%-----------------------------------------------------------------------------------------

% clean workspace
clear; clc; close all;

% Navigate to ...GitHub/Codes/FigureCodes

% load data
data_path = fullfile(fileparts(fileparts(pwd)), 'Data'); 
load(fullfile(data_path, 'Fig1i_DistanceAfterEvent.mat'))
addpath(fullfile(fileparts(pwd), 'OnPathCodes')) % add path to helper tools 

allDistanceHit = Fig1i_DistanceAfterEvent{1,2};
allDistanceMiss = Fig1i_DistanceAfterEvent{2,2};

%plot responses
f = figure(1);
clf
f.Position = [0 600 900 350];
fontname('Arial')
% subplot(1,2,1)
violinplot(1,allDistanceMiss,'FaceColor',[0 0.8 0])
hold on
violinplot(2.5, allDistanceHit,'FaceColor',[0.6 0.2 0.8])
xlim([0 3.5])
ylim([-500 2000])
fontsize(12,'pixels')
xNames={'Avoid';'Crash'};
xticks([1 2.5])
xticklabels({'Avoidance','Crash'})
ylabel('Displacement after outcome (pixels)','FontSize',12)
sgt= sgtitle('Figure 1i. Y-position displacement after outcome');
sgt.FontSize = 13;

[p,H,STATS] = ranksum(allDistanceHit,allDistanceMiss);
display = ['Wilcoxon signed rank test, p=',num2str(p)];
disp(display)

