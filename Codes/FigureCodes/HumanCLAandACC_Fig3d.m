%-----------------------------------------------------------------------------------------
% Script for heatmap of claustrum neurons
%
% Human claustrum neurons encode uncertainty and prediction errors during aversive learning
% Figure   3d
% License: 
%-----------------------------------------------------------------------------------------

% clean workspace
clear; clc; close all;

% Navigate to ...GitHub/Codes/FigureCodes

% load data
data_path = fullfile(fileparts(fileparts(pwd)), 'Data'); 
load(fullfile(data_path, 'Fig3d_CLAheatmap.mat'))
addpath(fullfile(fileparts(pwd), 'OnPathCodes')) % add path to helper tools 


% load data
edges = Fig3d_CLAheatmap{1,1};
neuronNumber = Fig3d_CLAheatmap{1,2};
normFR = Fig3d_CLAheatmap{1,3};

% plot responses
f = figure(1);
clf
f.Position = [0 600 900 500];
fontname('Arial')
imagesc(edges./1000,sort(neuronNumber(:,1),'descend'),normFR')
xlabel('Time to appear (s)','FontSize',12)
ylabel('Claustrum neuron # (sorted)','FontSize',12)
colormap(hot)
colorbar
xlim([-2 4])
ylim([1 78])
yticks([1 78])
yticklabels({'78','1'})
ylabel('Claustrum neuron number (sorted)','FontSize',12)
sgt= sgtitle('Figure 3d. All task-modulated Claustrum Neurons');
sgt.FontSize = 13;