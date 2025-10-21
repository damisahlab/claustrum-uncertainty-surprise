%-----------------------------------------------------------------------------------------
% Script for analysis of pupil change after asteroid appearance 
%
% Human claustrum neurons encode uncertainty and prediction errors during aversive learning
% Figure   3d
% Author:  Mauricio Medina
% License: 
%-----------------------------------------------------------------------------------------

% clean workspace
clear; clc; close all;

% Navigate to ...GitHub/Codes/FigureCodes

% load data
data_path = fullfile(fileparts(fileparts(pwd)), 'Data'); 
load(fullfile(data_path, 'Fig3d_ACC_AppearNeurons.mat'))
addpath(fullfile(fileparts(pwd), 'OnPathCodes')) % add path to helper tools 

% load data
edges = Fig3d_ACC_AppearNeurons{1,1};
neuronNumber = Fig3d_ACC_AppearNeurons{1,2};
normFR = Fig3d_ACC_AppearNeurons{1,3};
bursters = Fig3d_ACC_AppearNeurons{1,4}(:,1);
pausers = Fig3d_ACC_AppearNeurons{1,4}(:,3);
semBursters = Fig3d_ACC_AppearNeurons{1,4}(:,2);
semPausers = Fig3d_ACC_AppearNeurons{1,4}(:,4);
%%

%plot responses
f = figure(1);
clf
f.Position = [0 600 500 600];
fontname('Arial')
subplot(2,1,1)
imagesc(edges,neuronNumber,normFR)
xlabel('Time to appear (s)','FontSize',12)
ylabel('Claustrum neuron # (sorted)','FontSize',12)
colormap(hot)
colorbar
xlim([-2 4])
ylim([1 39])
yticks([1 39])
yticklabels({'39','1'})
ylabel('ACC neuron number (sorted)','FontSize',12)

subplot(2,1,2)
shadedErrorBar(edges,bursters,semBursters,'lineProps',{'Color',[0 0 1],'LineWidth',2})
hold on
shadedErrorBar(edges,pausers,semPausers,'lineProps',{'--','Color',[0 0 1],'LineWidth',2})
plot([0 0],[-100 100],'--r','LineWidth',1)
xlabel('Time to appear (s)','FontSize',12)
ylabel('Rate (z-scored)','FontSize',12)
xlim([-2 4])
ylim([-6 6])
legend('Bursters (26)','Pausers (13)')
sgt= sgtitle('Figure 3d. Appear-Modulated ACC Neurons');
sgt.FontSize = 13;

%STATS
% [H,P,CI,STATS]=ttest(meanAppBaseline,meanAfterZero);
% display = ['Student T test, p=',num2str(P)];
% disp(display)
