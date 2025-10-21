%-----------------------------------------------------------------------------------------
% Script for analysis of pupil change after asteroid appearance 
%
% Human claustrum neurons encode uncertainty and prediction errors during aversive learning
% Figure   3,a-c
% Author:  Mauricio Medina
% License: 
%-----------------------------------------------------------------------------------------

% clean workspace
clear; clc; close all;

% Navigate to ...GitHub/Codes/FigureCodes

% load data
data_path = fullfile(fileparts(fileparts(pwd)), 'Data'); 
load(fullfile(data_path, 'Fig3abc_ACCExampleNeurons.mat'))
addpath(fullfile(fileparts(pwd), 'OnPathCodes')) % add path to helper tools 

% select panel to plot: 'a', 'b' or 'c'
panel = 'c';

%% load data
if panel == 'a'
rasterAppear = Fig3abc_ACCExampleNeurons{2,2};
rasterCrash = Fig3abc_ACCExampleNeurons{2,3};
rasterAvoid = Fig3abc_ACCExampleNeurons{2,4};
rate = Fig3abc_ACCExampleNeurons{2,5};
elseif panel == 'b'
rasterAppear = Fig3abc_ACCExampleNeurons{3,2};
rasterCrash = Fig3abc_ACCExampleNeurons{3,3};
rasterAvoid = Fig3abc_ACCExampleNeurons{3,4};
rate = Fig3abc_ACCExampleNeurons{3,5};
elseif panel == 'c'
rasterAppear = Fig3abc_ACCExampleNeurons{4,2};
rasterCrash = Fig3abc_ACCExampleNeurons{4,3};
rasterAvoid = Fig3abc_ACCExampleNeurons{4,4};
rate = Fig3abc_ACCExampleNeurons{4,5};
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
    ylim([0 130])
end
xlabel('Time to appear (ms)','FontSize',12)
ylabel('Trial #','FontSize',12)

subplot(2,2,2)
plot(rate(:,1)./1000,rate(:,2),'k','LineWidth',1)
hold on
plot([0 0],[-1 350],'--r','LineWidth',1)
xlim([-2 4])
if panel == 'a'
    ylim([0 12])
else
    ylim([0 20])
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
    ylim([0 130])
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
    ylim([0 12])
else
    ylim([0 25])
end

xlabel('Time to outcome (s)','FontSize',12)
ylabel('rate (spikes/s)','FontSize',12)

if panel == 'a'
    tit = 'Figure 3a. Appear-modulated ACC Neuron';
elseif panel == 'b'
    tit = 'Figure 3b. Appear & Outcome Modulated ACC Neuron';
elseif panel =='c'
    tit = 'Figure 3c. Outcome-specific ACC Neuron';
end
sgt= sgtitle(tit);
sgt.FontSize = 13;
