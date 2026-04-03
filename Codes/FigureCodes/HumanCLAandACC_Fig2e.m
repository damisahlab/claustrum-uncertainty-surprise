%-----------------------------------------------------------------------------------------
% Script for fraction of responding CLA and ACC neurons

% Human claustrum neurons encode uncertainty and prediction errors during aversive learning
% Figure   2cd
% License: 
%-----------------------------------------------------------------------------------------

% clean workspace
clear; clc; close all;

% Navigate to ...GitHub/Codes/FigureCodes

% load data
data_path = fullfile(fileparts(fileparts(pwd)), 'Data'); 
load(fullfile(data_path, 'Fig2e_percentageOfNeurons'))
addpath(fullfile(fileparts(pwd), 'OnPathCodes')) % add path to helper tools 
claustrum = Fig2e_percentageOfNeurons{1,2};
acc = Fig2e_percentageOfNeurons{2,2};
groups=1:3;

%plot distributions
f = figure(1);
clf
f.Position = [0 600 800 350];
fontsize(15,'pixels')
fontname('Arial')

subplot(1,2,1)
boxplot(claustrum,'Labels',{'appear-specific','appear-outcome','outcome-specific'},'Colors','k')
hold on
scatter(groups,claustrum,15, 'MarkerFaceColor', [1 0.6 0.2], 'MarkerEdgeColor', [1 0.6 0.2],'MarkerFaceAlpha', 0.7) 
xlim([0 4])
ylim([-5 50])
%xlabel('Time from appear (s)','FontSize',15)
ylabel('Modulated neurons (%)','FontSize',15)
lgd = legend("claustrum");
lgd.Location = "northwest";

subplot(1,2,2)
boxplot(acc,'Labels',{'appear-specific','appear-outcome','outcome-specific'},'Colors','k')
hold on
scatter(groups,acc,15, 'MarkerFaceColor', [0 0.3 0.4], 'MarkerEdgeColor', [0 0.3 0.4],'MarkerFaceAlpha', 0.7) 
xlim([0 4])
ylim([-5 50])
%xlabel('Time from appear (s)','FontSize',15)
ylabel('Modulated neurons (%)','FontSize',15)
lgd = legend("ACC");
lgd.Location = "northwest";

sgt= sgtitle('Figure 2e. Percentage of responding neurons per session');
sgt.FontSize = 12;

% use Kwallis because the n is very small
[pCla, tblCla, statsCla] = kruskalwallis(claustrum,groups,'off');
[p,tbl,stats] = kruskalwallis(acc,groups,'off');
