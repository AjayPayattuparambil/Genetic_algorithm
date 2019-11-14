%% Prepare workspace
% clear;
clc;
close all;
warning off all;
%% import data
% import_data;
d=zeros(length(data),14);
for i=1:length(data)
datai=data{i,:};
s=strsplit(datai,',');
s1=zeros(1,length(s));
for j=1:length(s)
    s1(j)=str2double(s{j});
end
d(i,:)=double(s1);
end
%% input for classifier
Y = d(:,1:end-1);
Y(isnan(Y))=0;
id=d(:,end)+1;
%% feature selection
options = optimoptions('ga', 'CreationFcn', {@popfcn,Y,id},...
                     'PopulationSize',100,...
                     'Generations',50,...
                     'Display', 'iter');
nVars = 10;                          % set the number of desired features
FitnessFcn = {@biogafit,Y,id};       % set the fitness function
feat = ga(FitnessFcn,nVars,options); % call the Genetic Algorithm
% feat = biogafit(FitnessFcn,nVars,options); % call the Genetic Algorithm
feat = round(abs(feat));
out=selectFeat(Y,feat);
Y=Y(:,out);
%% classifier
Mdl = fitcknn(Y,id,'NumNeighbors',5,'Standardize',1);
save knnmodel Mdl out