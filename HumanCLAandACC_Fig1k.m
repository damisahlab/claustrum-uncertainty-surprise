%-----------------------------------------------------------------------------------------
% Script for analysis of pupil change after asteroid appearance 
%
% Human claustrum neurons encode uncertainty and prediction errors during aversive learning
% Figure   1k
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

meanHitBox = mean(allHit(:,514:745),2); %263:328
meanMissBox = mean(allMiss(:,514:745),2);

%plot responses
f = figure(1);
clf
f.Position = [0 600 900 350];
fontname('Arial')
subplot(1,2,1)
p = fill(timeForPupil, smoothHit + semHit', [0.6 0.2 0.8]);
p.EdgeColor = [1 1 1];
alpha(p,0.2)
hold on
f=fill(timeForPupil, smoothHit - semHit', [1 1 1]);
f.EdgeColor = [1 1 1];

pM = fill(timeForPupil, smoothMiss + semMiss', [0 0.8 0]);
pM.EdgeColor = [1 1 1];
alpha(pM,0.2)
fM=fill(timeForPupil,smoothMiss - semMiss',[1 1 1]);
fM.EdgeColor = [1 1 1];

plot(timeForPupil, smoothHit,'Color',[0.6 0.2 0.8],'LineWidth',2)
plot(timeForPupil, smoothMiss, 'Color', [0 0.8 0], 'LineWidth',2)
plot([0 0],[-10 10],'--r')
xlim([-2 6])
ylim([-0.1 0.4])
fontsize(12,'pixels')
xlabel('Time from appear (s)','FontSize',12)
ylabel('Pupil diameter change (mm)','FontSize',12)
hold off

subplot(1,2,2)
plot([1 2.5],[meanHitBox, meanMissBox],'Color',[0.9 0.9 0.9])
hold on
plot(1,meanHitBox,'v','MarkerEdgeColor',[0.6 0.2 0.8],'MarkerFaceColor',[0.6 0.2 0.8])
plot(2.5,meanMissBox,'^','MarkerEdgeColor',[0 0.8 0],'MarkerFaceColor',[0 0.8 0])
%alpha(p,0.2)
xlim([0 3.5])
ylim([0 6])
xticks([1 2.5])
xticklabels({'Crash','Avoidance'})
ylabel('Absolute pupil diameter (mm)','FontSize',12)
sgt= sgtitle('Figure 1k. Pupil responses to outcomes');
sgt.FontSize = 13;

%STATS
[H,P,CI,STATS]=ttest(meanHitBox,meanMissBox);

display = ['Student T test, p=',num2str(P)];
disp(display)
