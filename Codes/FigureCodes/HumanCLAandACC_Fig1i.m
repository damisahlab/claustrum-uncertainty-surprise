%-----------------------------------------------------------------------------------------
% Script for behavioral analysis of spaceship displacement after event
%
% Single-Neuron Responses of CLA and ACC to Salience Events
% Figure   1i
% Author:  Mauricio Medina
% License: 
%-----------------------------------------------------------------------------------------

% clean workspace
clear; clc; close all;

% Navigate to ...GitHub/Codes/FigureCodes

% load data
data_path = fullfile(fileparts(fileparts(pwd)), 'Data'); 
load(fullfile(data_path, 'Fig1i_DistanceAfterEvent.mat'))
addpath(fullfile(fileparts(pwd), 'OnPathCodes')) % add path to helper tools 

allDistanceHit = Fig1i_DistanceAfterEvent{1,2};
allDistanceMiss = Fig1i_DistanceAfterEvent{2,2};

violinplot(1, allDistanceHit)