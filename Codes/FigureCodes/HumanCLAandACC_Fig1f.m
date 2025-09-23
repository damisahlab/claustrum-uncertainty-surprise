%-----------------------------------------------------------------------------------------
% Script for time distributions of events of the spaceship task
%
% Single-Neuron Responses of CLA and ACC to Salient Information
% Figure   1f
% Author:  Mauricio Medina
% License: 
%-----------------------------------------------------------------------------------------

% clean workspace
clear; clc; close all;

% Navigate to GitHub/Codes/FigureCodes
%cd('/Users/mauricio/Library/CloudStorage/OneDrive-YaleUniversity/claustrumPaper/claustrumPaperNature/GitHub/Codes/FigureCodes')

% load data
data_path = fullfile(fileparts(fileparts(pwd)), 'Data'); 
load(fullfile(data_path, 'Fig1f_TimeDistribution.mat'))
allApp2Hit = (Fig1f_TimeDistribution.AppeartoHit)./1000;
allApp2Disapp = (Fig1f_TimeDistribution.AppeartoDisappear)./1000;

%plot distributions
f = figure(1);
clf
f.Position = [0 600 600 350];
fontname('Arial')
hHit = histogram(allApp2Hit,'BinLimits',[0,1.500],'BinWidth',.050,'FaceColor',[0 0 0]);
    countsApp2Hit = hHit.Values';
    edges = hHit.BinEdges;
hold on
hDisap = histogram(allApp2Disapp,'BinLimits',[0,1.500],'BinWidth',0.050,'FaceColor',[0.5 0.5 0.5]);
    countsApp2Disap = hDisap.Values';
xlim([0 1.5])
ylim([1 1200])
fontsize(15,'pixels')
xlabel('Time between events (s)','FontSize',15)
ylabel('Number of trials','FontSize',15)
sgt= sgtitle('Figure 1f. Time distribution of Events');
sgt.FontSize = 12;
lgd = legend("Appear to hit","Appear to disappear");
lgd.Location = "northwest";
