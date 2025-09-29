%-----------------------------------------------------------------------------------------
% Script for behavioral analysis of the proportion of collisions among
% subjects
%
% Single-Neuron Responses of CLA and ACC to Salience Events
% Figure   1g
% Author:  Mauricio Medina
% License: 
%-----------------------------------------------------------------------------------------

% clean workspace
clear; clc; close all;

% Navigate to GitHub/Codes/FigureCodes
%cd('/Users/mauricio/Library/CloudStorage/OneDrive-YaleUniversity/claustrumPaper/claustrumPaperNature/GitHub/Codes/FigureCodes')
cd ('C:\Users\Dr. Mauricio Pizarro\OneDrive - Yale University\claustrumPaper\claustrumPaperNature\GitHub\Codes\FigureCodes')

% load data
data_path = fullfile(fileparts(fileparts(pwd)), 'Data'); 
load(fullfile(data_path, 'Fig1g_PercentageOfCrash.mat'))
addpath(fullfile(fileparts(pwd), 'OnPathCodes')) % add path to helper tools 

