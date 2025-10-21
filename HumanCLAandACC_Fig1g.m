%-----------------------------------------------------------------------------------------
% Script for behavioral analysis of the proportion of collisions among
% subjects
%
% Human claustrum neurons encode uncertainty and prediction errors during aversive learning
% Figure   1g
% Author:  Mauricio Medina
% License: 
%-----------------------------------------------------------------------------------------

% clean workspace
clear; clc; close all;

% Navigate to ...GitHub/Codes/FigureCodes

% load data
data_path = fullfile(fileparts(fileparts(pwd)), 'Data'); 
load(fullfile(data_path, 'Fig1g_PercentOfCrash.mat'))
addpath(fullfile(fileparts(pwd), 'OnPathCodes')) % add path to helper tools 

avg=mean(allMeanScores);
sem=std(allMeanScores)./sqrt(6);
avg=avg';
sem=sem';
percent = (5:5:100)';


%plot distributions
f = figure(1);
clf
f.Position = [0 600 600 350];
fontname('Arial')
errorbar(percent,avg,sem)
xlim([0 105])
ylim([40 90])
fontsize(14,'pixels')
xlabel('% of completed trials','FontSize',15)
ylabel('Percentage of crash (%)','FontSize',15)
sgt= sgtitle('Figure 1g. Proportion of Collisions');
sgt.FontSize = 14;

%% now make stats of 1st quarter vs 4th quarter
firstQ = allMeanScores(:,1:5);
firstQ = firstQ(:);
lastQ = allMeanScores(:,16:20);
lastQ = lastQ(:);
%% both are not normally distributed, should be signed rank since it is the same subject

[p,H,STATS] = signrank(firstQ,lastQ);
display = ['Wilcoxon signed rank test, p=',num2str(p)];
disp(display)