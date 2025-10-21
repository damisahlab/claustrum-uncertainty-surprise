%-----------------------------------------------------------------------------------------
% Script for analysis of pupil change after asteroid appearance 
%
% Human claustrum neurons encode uncertainty and prediction errors during aversive learning
% Figure   3n
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
cla = Fig3ko.cla;
claAllTask = cla{1,2};
claHit = cla{1,4};
claMiss = cla{1,6};


%%

%plot responses
f = figure(1);
clf
f.Position = [0 600 900 600];
fontname('Arial')

plot([0.5 1.5],[claAllTask(:,1), claAllTask(:,2)],'Color',[0.9 0.9 0.9])
hold on
plot(0.5,claAllTask(:,1),'v','MarkerEdgeColor',[0 0.8 0],'MarkerFaceColor',[1 1 1])
plot(1.5,claAllTask(:,2),'^','MarkerEdgeColor',[0.6 0.2 0.8],'MarkerFaceColor',[1 1 1])

plot([3 4],[claHit(:,1), claHit(:,2)],'Color',[0.9 0.9 0.9])
plot(3,claHit(:,1),'v','MarkerEdgeColor',[0 0.8 0],'MarkerFaceColor',[1 1 1])
plot(4,claHit(:,2),'^','MarkerEdgeColor',[0.6 0.2 0.8],'MarkerFaceColor',[1 1 1])

plot([5.5 6.5],[claMiss(:,1), claMiss(:,2)],'Color',[0.9 0.9 0.9])
plot(5.5,claMiss(:,1),'v','MarkerEdgeColor',[0 0.8 0],'MarkerFaceColor',[1 1 1])
plot(6.5,claMiss(:,2),'^','MarkerEdgeColor',[0.6 0.2 0.8],'MarkerFaceColor',[1 1 1])

xlim([0 7])
ylim([-0.5 8])
xticks([1 3.5 6])
xticklabels({'All-Task Responsive','Crash-Responsive','Avoid-Responsive'})
ylabel('mean |D| rate', 'FontSize',12)

sgt= sgtitle('Figure 3n. CLA Neurons Responses');
sgt.FontSize = 13;

%STATS
[pAll, hAll, statsAll] = signrank(claAllTask(:,1),claAllTask(:,2));
[pCrash, hCrash, statsCrash] = signrank(claHit(:,1),claHit(:,2));
[pAvoid, hAvoid, statsAvoid] = signrank(claMiss(:,1),claMiss(:,2));

display1 = ['Wilcoxon rank test for all task responsive, p=',num2str(pAll)];
display2 = ['Wilcoxon rank test for crash-responsive, p=', num2str(pCrash)];
display3 = ['Wilcoxon rank test for avoid-responsive, p=', num2str(pAvoid)];
disp(display1)
disp(display2)
disp(display3)