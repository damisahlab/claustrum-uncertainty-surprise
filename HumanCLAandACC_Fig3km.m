%-----------------------------------------------------------------------------------------
% Script for analysis of pupil change after asteroid appearance 
%
% Human claustrum neurons encode uncertainty and prediction errors during aversive learning
% Figure   3k,m
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
claAllTask = cla{1,1};
claHit = cla{1,3};
claMiss = cla{1,5};
edges = cla{1,7};
edges=edges./1000;
acc = Fig3ko.acc;
accAllTask = acc{1,1};
accHit = acc{1,3};
accMiss = acc{1,5};

%%

%plot responses
f = figure(1);
clf
f.Position = [0 600 700 800];
fontname('Arial')

subplot(3,2,1)
shadedErrorBar(edges,claAllTask(:,1),claAllTask(:,2),'lineProps',{'Color',[0.6 0.2 0.8],'LineWidth',2})
hold on
shadedErrorBar(edges,claAllTask(:,3),claAllTask(:,4),'lineProps',{'Color',[0 0.8 0],'LineWidth',2})
plot([0 0],[-100 100],'--r')
ylabel('Rate (z-scored)','FontSize',12)
xlim([-2 4])
ylim([0 6])
title('CLA All-Task Responsive');

subplot(3,2,2)
shadedErrorBar(edges,accAllTask(:,1),accAllTask(:,2),'lineProps',{'Color',[0.6 0.2 0.8],'LineWidth',2})
hold on
shadedErrorBar(edges,accAllTask(:,3),accAllTask(:,4),'lineProps',{'Color',[0 0.8 0],'LineWidth',2})
plot([0 0],[-100 100],'--r')
ylabel('Rate (z-scored)','FontSize',12)
xlim([-2 4])
ylim([0 6])
legend('crash','avoidance')
title('ACC All-Task Responsive');

subplot(3,2,3)
shadedErrorBar(edges,claHit(:,1),claHit(:,2),'lineProps',{'Color',[0.6 0.2 0.8],'LineWidth',2})
hold on
shadedErrorBar(edges,claHit(:,3),claHit(:,4),'lineProps',{'Color',[0 0.8 0],'LineWidth',2})
plot([0 0],[-100 100],'--r')
ylabel('Rate (z-scored)','FontSize',12)
xlim([-2 4])
ylim([0 6])
title('CLA Crash-Responsive');

subplot(3,2,4)
shadedErrorBar(edges,accHit(:,1),accHit(:,2),'lineProps',{'Color',[0.6 0.2 0.8],'LineWidth',2})
hold on
shadedErrorBar(edges,accHit(:,3),accHit(:,4),'lineProps',{'Color',[0 0.8 0],'LineWidth',2})
plot([0 0],[-100 100],'--r')
ylabel('Rate (z-scored)','FontSize',12)
xlim([-2 4])
ylim([0 6])
title('ACC Crash-Responsive');

subplot(3,2,5)
shadedErrorBar(edges,claMiss(:,1),claMiss(:,2),'lineProps',{'Color',[0.6 0.2 0.8],'LineWidth',2})
hold on
shadedErrorBar(edges,claMiss(:,3),claMiss(:,4),'lineProps',{'Color',[0 0.8 0],'LineWidth',2})
plot([0 0],[-100 100],'--r')
xlabel('Time to outcome (s)','FontSize',12)
ylabel('Rate (z-scored)','FontSize',12)
xlim([-2 4])
ylim([0 12])
title('CLA Avoidance-Responsive');

subplot(3,2,6)
shadedErrorBar(edges,accMiss(:,1),accMiss(:,2),'lineProps',{'Color',[0.6 0.2 0.8],'LineWidth',2})
hold on
shadedErrorBar(edges,accMiss(:,3),accMiss(:,4),'lineProps',{'Color',[0 0.8 0],'LineWidth',2})
plot([0 0],[-100 100],'--r')
xlabel('Time to outcome (s)','FontSize',12)
ylabel('Rate (z-scored)','FontSize',12)
xlim([-2 4])
ylim([0 6])
title('ACC Avoidance-Responsive');

sgt= sgtitle('Figure 3k-m. CLA and ACC Responses');
sgt.FontSize = 13;
