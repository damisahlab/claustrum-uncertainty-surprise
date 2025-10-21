%-----------------------------------------------------------------------------------------
% Script for analysis of pupil change after asteroid appearance 
%
% Human claustrum neurons encode uncertainty and prediction errors during aversive learning
% Figure   3g,h
% Author:  Mauricio Medina
% License: 
%-----------------------------------------------------------------------------------------

% clean workspace
clear; clc; close all;

% Navigate to ...GitHub/Codes/FigureCodes

% load data
data_path = fullfile(fileparts(fileparts(pwd)), 'Data'); 
load(fullfile(data_path, 'Fig3gh_ACC_AppearOutcomeNeurons.mat'))
addpath(fullfile(fileparts(pwd), 'OnPathCodes')) % add path to helper tools 

% load data
rateData = Fig3gh_ACC_AppearOutcomeNeurons{1,1};

% line data
edges = rateData(:,9);
edges=edges./1000;
crashBursters = rateData(:,1);
semCrashBursters = rateData(:,2);
avoidBurtsers = rateData(:,3);
semAvoidBursters = rateData(:,4);
crashPausers = rateData(:,5);
semCrashPausers = rateData(:,6);
avoidPausers = rateData(:,7);
semAvoidPausers = rateData(:,8);

%stats data
statsData = Fig3gh_ACC_AppearOutcomeNeurons {1,2};
preApp = statsData(:,1);
app = statsData(:,2);
avoid = statsData(:,3);
crash = statsData(:,4);
%%

%plot responses
f = figure(1);
clf
f.Position = [0 600 450 600];
fontname('Arial')

subplot(2,1,1)
shadedErrorBar(edges,crashBursters,semCrashBursters,'lineProps',{'Color',[0.6 0.2 0.8],'LineWidth',2})
hold on
shadedErrorBar(edges,avoidBurtsers,semAvoidBursters,'lineProps',{'Color',[0 0.8 0],'LineWidth',2})
shadedErrorBar(edges,crashPausers,semCrashPausers,'lineProps',{'--','Color',[0.6 0.2 0.8],'LineWidth',1})
shadedErrorBar(edges,avoidPausers,semAvoidPausers,'lineProps',{'--','Color',[0 0.8 0],'LineWidth',1})
xlabel('Time to appear (s)','FontSize',12)
ylabel('Rate (z-scored)','FontSize',12)
xlim([-2 4])
ylim([-6 8])
legend('crash','avoidance')
hold off

subplot(2,1,2)
plot([1 2],[preApp, app],'Color',[0.9 0.9 0.9])
hold on
plot(1,preApp,'vk','MarkerFaceColor',[1 1 1])
plot(2,app,'^b','MarkerFaceColor',[1 1 1])
plot([3.5 4.5],[avoid, crash],'Color',[0.9 0.9 0.9])
plot(3.5,avoid,'v','MarkerEdgeColor',[0 0.8 0],'MarkerFaceColor',[1 1 1])
plot(4.5,crash,'^','MarkerEdgeColor',[0.6 0.2 0.8],'MarkerFaceColor',[1 1 1])
xlim([0 5.5])
ylim([-0.5 8])
xticks([1 2 3.4 4.5])
xticklabels({'Pre','Post','Avoid','Crash'})
ylabel('mean |D| rate', 'FontSize',12)

sgt= sgtitle('Figure 3g and 3h. Appear & Outcome ACC Neurons');
sgt.FontSize = 13;

%STATS
[pApp, hApp, statsApp] = signrank(preApp,app);
[pOut, hOut, statsOut] = signrank(avoid,crash);

display1 = ['Wilcoxon rank test for appearance, p=',num2str(pApp)];
display2 = ['Wilcoxon rank test for outcome, p=', num2str(pOut)];
disp(display1)
disp(display2)