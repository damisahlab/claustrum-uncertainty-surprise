%-----------------------------------------------------------------------------------------
% Script for fraction of responding CLA neurons

% Human claustrum neurons encode uncertainty and prediction errors during aversive learning
% Figure   2cd
% License: 
%-----------------------------------------------------------------------------------------

% clean workspace
clear; clc; close all;

% Navigate to ...GitHub/Codes/FigureCodes

% load data
data_path = fullfile(fileparts(fileparts(pwd)), 'Data'); 
load(fullfile(data_path, 'Fig2cd_fractionOfNeurons'))
addpath(fullfile(fileparts(pwd), 'OnPathCodes')) % add path to helper tools 
claustrum = Fig2cd_fractionOfNeurons{1,2};
acc = Fig2cd_fractionOfNeurons{2,2};
time=-2:0.050:4;

%plot distributions
f = figure(1);
clf
f.Position = [0 600 800 350];
fontsize(15,'pixels')
fontname('Arial')
subplot(1,2,1)
bar(time,claustrum,'b')
xlim([-2 4])
ylim([0 0.45])
xlabel('Time from appear (s)','FontSize',15)
ylabel('Responsive neurons (%)','FontSize',15)
lgd = legend("claustrum");
lgd.Location = "northwest";

subplot(1,2,2)
bar(time,acc,'r')
hold on
stairs(time,claustrum,'b')
xlim([-2 4])
ylim([0 0.45])
xlabel('Time from appear (s)','FontSize',15)
ylabel('Responsive neurons (%)','FontSize',15)

sgt= sgtitle('Figure 2cd. Percentage of responding neurons');
sgt.FontSize = 12;
lgd = legend("ACC","claustrum");
lgd.Location = "northwest";
