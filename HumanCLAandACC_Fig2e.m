%-----------------------------------------------------------------------------------------
% Script for analysis of pupil change after asteroid appearance 
%
% Human claustrum neurons encode uncertainty and prediction errors during aversive learning
% Figure   2e
% Author:  Mauricio Medina
% License: 
%-----------------------------------------------------------------------------------------

% clean workspace
clear; clc; close all;

% Navigate to ...GitHub/Codes/FigureCodes

% load data
data_path = fullfile(fileparts(fileparts(pwd)), 'Data'); 
load(fullfile(data_path, 'Fig2e_CLA_AppearNeurons.mat'))
addpath(fullfile(fileparts(pwd), 'OnPathCodes')) % add path to helper tools 

% load data
edges = Fig2e_CLA_AppearNeurons{1,1};
edges=edges./1000;
neuronNumber = Fig2e_CLA_AppearNeurons{1,2};
normFR = Fig2e_CLA_AppearNeurons{1,3};
bursters = Fig2e_CLA_AppearNeurons{1,4}(:,1);
pausers = Fig2e_CLA_AppearNeurons{1,4}(:,2);
semBursters = Fig2e_CLA_AppearNeurons{1,5}(:,1);
semPausers = Fig2e_CLA_AppearNeurons{1,5}(:,2);
%%

%plot responses
f = figure(1);
clf
f.Position = [0 600 500 600];
fontname('Arial')
subplot(2,1,1)
imagesc(edges,neuronNumber(:,1),normFR)
xlabel('Time to appear (s)','FontSize',12)
ylabel('Claustrum neuron # (sorted)','FontSize',12)
colormap(hot)
colorbar
xlim([-2 4])
ylim([1 58])
yticks([1 58])
yticklabels({'58','1'})
ylabel('Claustrum neuron number (sorted)','FontSize',12)
subplot(2,1,2)
shadedErrorBar(edges,bursters,semBursters,'lineProps',{'Color',[0 0 1],'LineWidth',2})
hold on
shadedErrorBar(edges,pausers,semPausers,'lineProps',{'--','Color',[0 0 1],'LineWidth',2})
plot([0 0],[-100 100],'--r','LineWidth',1)
xlabel('Time to appear (s)','FontSize',12)
ylabel('Rate (z-scored)','FontSize',12)
xlim([-2 4])
ylim([-4 12])

sgt= sgtitle('Figure 2e. Appear-Modulated Claustrum Neurons');
sgt.FontSize = 13;

%STATS
% [H,P,CI,STATS]=ttest(meanAppBaseline,meanAfterZero);
% display = ['Student T test, p=',num2str(P)];
% disp(display)
