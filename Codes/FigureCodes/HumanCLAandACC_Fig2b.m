%-----------------------------------------------------------------------------------------
% Script for venn diagram analysis

% Human claustrum neurons encode uncertainty and prediction errors during aversive learning
% Figure   2b
% License: 
%-----------------------------------------------------------------------------------------

% Creat venn diagrams using venn function  venn(A, I)
%For three-circle venn diagrams, A is a three element vector [c1 c2 c3], 
%and I is a four element vector [i12 i13 i23 i123], specifiying the 
%two-circle intersection areas i12, i13, i23, and the three-circle
%intersection i123.

% clean workspace
clear; clc; close all;

% Navigate to ...GitHub/Codes/FigureCodes

% load data
data_path = fullfile(fileparts(fileparts(pwd)), 'Data'); 
load(fullfile(data_path, 'Fig2b_Venn.mat'))
addpath(fullfile(fileparts(pwd), 'OnPathCodes')) % add path to helper tools 

%[i12 i13 i23 i123]
%1 is app
%2 is hit
%3 is miss
appSpecIndx=Fig2b_Venn{1,2};
hitSpecIndx=Fig2b_Venn{2,2};
missSpecIndx=Fig2b_Venn{3,2};
appHitIdx=Fig2b_Venn{4,2};
appMissIdx=Fig2b_Venn{5,2};
hitMissIdx=Fig2b_Venn{6,2};
appHitMissIdx=Fig2b_Venn{7,2};

Z=[size(appSpecIndx,1) size(hitSpecIndx,1) size(missSpecIndx,1) size(appHitIdx,1) size(appMissIdx,1) size(hitMissIdx,1) size(appHitMissIdx,1)];

fig=figure();
f = figure(1);
clf
f.Position = [0 600 450 350];
fontname('Arial')

[H, S] =venn(Z,'FaceColor',...
                            {[0 0.370 0.8410],[0.4940 0.1840 0.5560],[0.4660 0.6740 0.1880]},'FaceAlpha',...
                            {0.2,0.4,0.4},'EdgeColor',[1 1 1]);


fontsize(12,'pixels')
sgt= sgtitle('Figure 2b. Distribution of ACC neurons');
sgt.FontSize = 13;

