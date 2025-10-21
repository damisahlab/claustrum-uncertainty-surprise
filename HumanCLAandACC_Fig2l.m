%-----------------------------------------------------------------------------------------
% Script for analysis of pupil change after asteroid appearance 
%
% Human claustrum neurons encode uncertainty and prediction errors during aversive learning
% Figure   2l
% Author:  Mauricio Medina
% License: 
%-----------------------------------------------------------------------------------------

% clean workspace
clear; clc; close all;

% Navigate to ...GitHub/Codes/FigureCodes

% load data
data_path = fullfile(fileparts(fileparts(pwd)), 'Data'); 
load(fullfile(data_path, 'Fig2l_CLA_OutcomeNeurons.mat'))
addpath(fullfile(fileparts(pwd), 'OnPathCodes')) % add path to helper tools 

% load data
hit = Fig2l_CLA_OutcomeNeurons{1,1};
inesp = Fig2l_CLA_OutcomeNeurons{1,2};
miss = Fig2l_CLA_OutcomeNeurons{1,3};
unityLine = Fig2l_CLA_OutcomeNeurons{1,4};


%%

%plot responses
f = figure(1);
clf
f.Position = [0 600 500 500];
fontname('Arial')

plot(hit(:,1),hit(:,2),'^','MarkerSize',5,'MarkerEdgeColor',[1 0.2 0.8],'MarkerFaceColor',[1 0.2 0.8])
hold on
plot(miss(:,1),miss(:,2),'v','MarkerSize',5,'MarkerEdgeColor',[0 1 0.8],'MarkerFaceColor',[0 1 0.8])
plot(inesp(:,1),inesp(:,2),'o','MarkerSize',5,'MarkerEdgeColor',[0 0 0],'MarkerFaceColor',[0.8 0.8 0.8])
plot([-1 10],[-1 10],'--r','LineWidth',1)
legend('crash-specific(9)','avoid-specific(1)','non-specific(7)')
xlim([-0.5 8])
ylim([-0.5 8])
hold off

sgt= sgtitle('Figure 2l. Outcome-specific Claustrum Neurons');
sgt.FontSize = 13;
