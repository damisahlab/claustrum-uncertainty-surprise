%-----------------------------------------------------------------------------------------
% Script for analysis of pupil change after asteroid appearance 
%
% Single-Neuron Responses of CLA and ACC to Salience Events
% Figure   1j
% Author:  Mauricio Medina
% License: 
%-----------------------------------------------------------------------------------------

% clean workspace
clear; clc; close all;

% Navigate to ...GitHub/Codes/FigureCodes

% load data
data_path = fullfile(fileparts(fileparts(pwd)), 'Data'); 
load(fullfile(data_path, 'Fig1j_PupilChangeToAsteroids.mat'))
addpath(fullfile(fileparts(pwd), 'OnPathCodes')) % add path to helper tools 

timeForPupil = Fig1j_PupilChangeToAsteroids{1,2};
allAppear = Fig1j_PupilChangeToAsteroids{2,2};
allHit = Fig1j_PupilChangeToAsteroids{3,2};
allMiss = Fig1j_PupilChangeToAsteroids{4,2};

bsApp = []; %bs baseline subtracted
for i=1:size(allAppear,1)
    trial = allAppear(i,:);
    trial = trial - mean(trial(1:94));
    bsApp = [bsApp; trial];
    clear trial
end

bsHit = [];
for i=1:size(allHit,1)
    trial = allHit(i,:);
    trial = trial - mean(trial(1:94));
    bsHit = [bsHit; trial];
    clear trial
end

bsMiss = [];
for i=1:size(allMiss,1)
    trial = allMiss(i,:);
    trial = trial - mean(trial(1:94));
    bsMiss = [bsMiss; trial];
    clear trial
end


smoothApp = smoothdata(mean(bsApp),'gaussian',10);
smoothHit = smoothdata(mean(bsHit),'gaussian',10);
smoothMiss = smoothdata(mean(bsMiss),'gaussian',10);

semApp = std(bsApp)/sqrt(size(bsApp,1));
semApp = semApp';
semHit = std(bsHit)/sqrt(size(bsHit,1));
semHit = semHit';
semMiss = std(bsMiss)/sqrt(size(bsMiss,1));
semMiss = semMiss';

appBaseline = allAppear(:,1:94);
meanAppBaseline = mean(appBaseline,2);

%plot responses
f = figure(1);
clf
f.Position = [0 600 900 350];
fontname('Arial')
subplot(1,2,1)
p = fill(timeForPupil, smoothApp + semApp', [0 0 1]);
p.EdgeColor = [1 1 1];
hold on
f=fill(timeForPupil, smoothApp - semApp', [1 1 1]);
f.EdgeColor = [1 1 1];
plot(timeForPupil, smoothApp,'b','LineWidth',2)
plot([0 0],[-10 10],'--r')
hold off
alpha(p,0.2)
xlim([-2 6])
ylim([-0.01 0.4])
fontsize(12,'pixels')
xlabel('Time from appear (s)','FontSize',12)
ylabel('Pupil diameter change (mm)','FontSize',12)
sgt= sgtitle('Figure 1j. Pupil responses to asteroid appearance');
sgt.FontSize = 13;

