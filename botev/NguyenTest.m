
clear all,clc,

d=5;
l=-ones(d,1)/2;
u=ones(d,1);
Sig=0.5*eye(d)+.5*ones(d,d);

est= mvncdf(l,u,Sig,10^2) % output of our method

% Executing Matlab's toolbox\stats\stats\mvncdf.m
% with n=10^7 below is slow and inaccurate
cd(matlabroot) % change to Matlab default path
options=optimset('TolFun',0,'MaxFunEvals',10^7,'Display','iter');
[prob,err]=mvncdf(l,u,zeros(d,1),Sig,options)