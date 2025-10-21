%-----------------------------------------------------------------------------------------
% Script for analysis of pupil change after asteroid appearance 
%
% Human claustrum neurons encode uncertainty and prediction errors during aversive learning
% Figure   3o
% Author:  Mauricio Medina
% License: 
%-----------------------------------------------------------------------------------------

% clean workspace
clear; clc; close all;

% Navigate to ...GitHub/Codes/FigureCodes

% load data
data_path = fullfile(fileparts(fileparts(pwd)), 'Data'); 
load(fullfile(data_path, 'Fig3ko.mat'))
addpath(fullfile(fileparts(pwd), 'OnPathCodes')) % add path to helper tools 


% load data
acc = Fig3ko.acc;
accAllTask = acc{1,2};
accHit = acc{1,4};
accMiss = acc{1,6};


%%

%plot responses
f = figure(1);
clf
f.Position = [0 600 900 600];
fontname('Arial')

plot([0.5 1.5],[accAllTask(:,1), accAllTask(:,2)],'Color',[0.9 0.9 0.9])
hold on
plot(0.5,accAllTask(:,1),'v','MarkerEdgeColor',[0 0.8 0],'MarkerFaceColor',[1 1 1])
plot(1.5,accAllTask(:,2),'^','MarkerEdgeColor',[0.6 0.2 0.8],'MarkerFaceColor',[1 1 1])

plot([3 4],[accHit(:,1), accHit(:,2)],'Color',[0.9 0.9 0.9])
plot(3,accHit(:,1),'v','MarkerEdgeColor',[0 0.8 0],'MarkerFaceColor',[1 1 1])
plot(4,accHit(:,2),'^','MarkerEdgeColor',[0.6 0.2 0.8],'MarkerFaceColor',[1 1 1])

plot([5.5 6.5],[accMiss(:,1), accMiss(:,2)],'Color',[0.9 0.9 0.9])
plot(5.5,accMiss(:,1),'v','MarkerEdgeColor',[0 0.8 0],'MarkerFaceColor',[1 1 1])
plot(6.5,accMiss(:,2),'^','MarkerEdgeColor',[0.6 0.2 0.8],'MarkerFaceColor',[1 1 1])

xlim([0 7])
ylim([-0.5 8])
xticks([1 3.5 6])
xticklabels({'All-Task Responsive','Crash-Responsive','Avoid-Responsive'})
ylabel('mean |D| rate', 'FontSize',12)

sgt= sgtitle('Figure 3o. ACC Neurons Responses');
sgt.FontSize = 13;

%STATS
[pAll, hAll, statsAll] = signrank(accAllTask(:,1),accAllTask(:,2));
[pCrash, hCrash, statsCrash] = signrank(accHit(:,1),accHit(:,2));
[pAvoid, hAvoid, statsAvoid] = signrank(accMiss(:,1),accMiss(:,2));

display1 = ['Wilcoxon rank test for All task responsive, p=',num2str(pAll)];
display2 = ['Wilcoxon rank test for Crash-responsive, p=', num2str(pCrash)];
display3 = ['Wilcoxon rank test for Avoid-responsive, p=', num2str(pAvoid)];
disp(display1)
disp(display2)
disp(display3)