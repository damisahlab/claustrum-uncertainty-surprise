%-----------------------------------------------------------------------------------------
% Script for analysis of pupil change after asteroid appearance 
%
% Human claustrum neurons encode uncertainty and prediction errors during aversive learning
% Figure   2,a-c
% Author:  Mauricio Medina
% License: 
%-----------------------------------------------------------------------------------------

% clean workspace
clear; clc; close all;

% Navigate to ...GitHub/Codes/FigureCodes

% load data
data_path = fullfile(fileparts(fileparts(pwd)), 'Data'); 
load(fullfile(data_path, 'Fig2abc_CLAExampleNeurons.mat'))
addpath(fullfile(fileparts(pwd), 'OnPathCodes')) % add path to helper tools 

% select panel to plot: 'a', 'b' or 'c'
panel = 'a';

%% load data
if panel == 'a'
rasterAppear = Fig2abc_CLAExampleNeurons{2,2};
rasterCrash = Fig2abc_CLAExampleNeurons{2,3};
rasterAvoid = Fig2abc_CLAExampleNeurons{2,4};
rate = Fig2abc_CLAExampleNeurons{2,5};
elseif panel == 'b'
rasterAppear = Fig2abc_CLAExampleNeurons{3,2};
rasterCrash = Fig2abc_CLAExampleNeurons{3,3};
rasterAvoid = Fig2abc_CLAExampleNeurons{3,4};
rate = Fig2abc_CLAExampleNeurons{3,5};
elseif panel == 'c'
rasterAppear = Fig2abc_CLAExampleNeurons{4,2};
rasterCrash = Fig2abc_CLAExampleNeurons{4,3};
rasterAvoid = Fig2abc_CLAExampleNeurons{4,4};
rate = Fig2abc_CLAExampleNeurons{4,5};
end

% plot responses
f = figure(1);
clf
f.Position = [0 600 900 500];
fontname('Arial')
subplot(2,2,1)
plot(rasterAppear(:,1)./1000, rasterAppear(:,2), '.k', 'MarkerSize',2)
hold on
plot([0 0],[-1 350],'-r','LineWidth',1)
xlim([-2 4])
if panel == 'a'
    ylim([0 320])
else
    ylim([0 270])
end
xlabel('Time to appear (ms)','FontSize',12)
ylabel('Trial #','FontSize',12)

subplot(2,2,2)
plot(rate(:,1)./1000,rate(:,2),'k','LineWidth',1)
hold on
plot([0 0],[-1 350],'--r','LineWidth',1)
xlim([-2 4])
if panel == 'a'
    ylim([0 8])
else
    ylim([0 24])
end
xlabel('Time to appear (s)','FontSize',12)
ylabel('rate (spikes/s)','FontSize',12)

subplot(2,2,3)
plot(rasterCrash(:,1)./1000, rasterCrash(:,2), '.', 'MarkerSize',2, 'Color',[0.6 0.2 0.8])
hold on
plot(rasterAvoid(:,1)./1000, rasterAvoid(:,2), '.', 'MarkerSize',2, 'Color', [0 0.8 0])
plot([0 0],[-1 350],'-r','LineWidth',1)
xlim([-2 4])
if panel == 'a'
    ylim([0 320])
else
    ylim([0 270])
end
xlabel('Time to outcome (ms)','FontSize',12)
ylabel('Trial #','FontSize',12)

subplot(2,2,4)
plot(rate(:,1)./1000,rate(:,3),'Color',[0.6 0.2 0.8],'LineWidth',1)
hold on
plot(rate(:,1)./1000, rate(:,4), 'Color',[0 0.8 0],'LineWidth',1)
plot([0 0],[-1 350],'--r','LineWidth',1)
xlim([-2 4])
if panel == 'a'
    ylim([0 8])
else
    ylim([0 24])
end

xlabel('Time to outcome (s)','FontSize',12)
ylabel('rate (spikes/s)','FontSize',12)

if panel == 'a'
    tit = 'Figure 2a. Appear-specific Clausutrum Neuron';
elseif panel == 'b'
    tit = 'Figure 2b. Appear & Outcome Modulated Claustrum Neuron';
elseif panel =='c'
    tit = 'Figure 2c. Outcome-specific Claustrum Neuron';
end
sgt= sgtitle(tit);
sgt.FontSize = 13;
