clearvars
% close all
clc

%%% Brief description:
%---------------------
% This code converts the output data of the Wireless InSite stored as
% '.p2m' files into MATLAB data stored as '.mat' files

%%% Instructions:
%-----------------
% 1) Move this '.m' file to the main directory
% 2) Create a new folder named "RayTracing Scnearios" in the main directory
% 3) Create two subfolders inside the "RayTracing Scnearios" folder, one is
% named with the name of the scenario you will read and the other is named
% with the scenario name you will write
% 4) move all the Wireless InSite '.p2m' files of the scenario you will
% read inside its corresponding subfolder
% N.B.: main '.p2m' files needed are 'dod', 'doa', 'cir', and 'pl' files
% 5) Change only the values of the input parameters in this '.m' file
% 6) Run this '.m' file!
% 7) Fetch the the output '.mat' files of the scenario you wrote from its
% corresponding subfolder

fprintf('This message is posted at date and time %s\n', datestr(now,'dd-mmm-yyyy HH:MM:SS.FFF'))
%% Input parameters

%--- Wireless InSite scenario name mapping parameters ---%
scenario_read = 'Boston5G_3p5'; %Wireless InSite input scenario name
scenario_write = 'Boston5G_3p5'; %Output scenario name

carrier_freq = (3.5)*1e9; %Hz
transmit_power = 0; %dBm

%--- Scenario parameters ---%
%%% Transmitter info
TX_ID_BS = [3]; % Transmitter ID number matching order (Base Stations for O1 scenarios)
TX_ID_BS_Output = [1]; % Transmitter ID number to be called in DeepMIMO (we want to call it as BS-1 to BS-18)
TX_ID= TX_ID_BS;
num_TX_BS =numel(TX_ID_BS); % Number of BS %%%%%%%Exported directly to DeepMIMO
num_BS=numel(TX_ID_BS); % Number of BS %%%%%%%Exported directly to DeepMIMO

%%% Receiver info
RX_grids_ID_user = [59]; % user ID number matching order
RX_grids_ID_BS = [3]; % BS receiver ID number matching order
user_grids=[1,622,541]; % User Grid Info %%%%%%%Exported directly to DeepMIMO

RX_grids_ID = [RX_grids_ID_user,RX_grids_ID_BS]; % Receiver ID number matching order
%BS_RX_row_ID = [(5204:1:(5204+numel(RX_grids_ID_BS)-1));(5204:1:(5204+numel(RX_grids_ID_BS)-1));ones(1,numel(RX_grids_ID_BS))];
BS_RX_row_ID = [(1:1:(1+numel(RX_grids_ID_BS)-1));(1:1:(1+numel(RX_grids_ID_BS)-1));ones(1,numel(RX_grids_ID_BS))];
BS_grids=BS_RX_row_ID.'; % BS Grid Info %%%%%%%Exported directly to DeepMIMO
RX_grids=[user_grids;BS_grids]; % ALL RX Grid Info

% Each row is the information associated with each of the 3 user grids:
% from user row R#"first element" to user row R#"second element" with "third element" users per row. 

%%

TX_ID_str = number2stringTX(TX_ID);
TX_ID_BS_str = number2stringTX(TX_ID_BS);
TX_ID_BS2_str = number2stringRX(TX_ID_BS);

% num_TX=numel(TX_ID); % Number of transmitters
RX_grids_ID_str = number2stringRX(RX_grids_ID);
RX_grids_ID_user_str = number2stringRX(RX_grids_ID_user);
RX_grids_ID_BS_str = number2stringRX(RX_grids_ID_BS);

num_RX_grids=numel(RX_grids_ID); % Number of RX grids
num_user_grids=numel(RX_grids_ID_user); % Number of user RX grids
num_BS_grids = numel(RX_grids_ID_BS); % Number of BS RX grids

%% Read the path loss files and Write the pathloss (PL) files for the user receivers (User receivers - DeepMIMO V2)

for ii=1:1:num_BS % For each BS
    disp(['Reading and writing the LR files for BS# ' num2str(ii) ' out of ' num2str(num_BS) ' BSs (BS-user) ...'])
    PL_array_full=[];
    for jj=1:1:num_user_grids % For each user grid
        filename_PL=strcat('Raytracing_scenarios/',scenario_read,'/',scenario_read,'.pl.t001_',TX_ID_BS_str{ii},'.r',RX_grids_ID_user_str{jj},'.p2m');
        PL_array=importdata(filename_PL);
        PL_array_full=[PL_array_full; PL_array.data(:,[5 6])];
    end    
    % Write the PL output file
    sfile_PL=strcat('Raytracing_scenarios/',scenario_write,'/',scenario_write,'.',num2str(TX_ID_BS_Output(ii)),'.PL.mat');
    save(sfile_PL,'PL_array_full');
end

%% Read the path loss files and Write the pathloss (PL) files for the BS receivers (BS receivers - DeepMIMO V2)

for ii=1:1:num_TX_BS % For each BS transmitter
    disp(['Reading and writing the LR files for BS# ' num2str(ii) ' out of ' num2str(num_TX_BS) ' BSs (BS-BS) ...'])    
    BSBS_PL_array_full=[];
    for jj=1:1:num_BS_grids % For each BS grid
        filename_PL=strcat('Raytracing_scenarios/',scenario_read,'/',scenario_read,'.pl.t001_',TX_ID_BS_str{ii},'.r',RX_grids_ID_BS_str{jj},'.p2m');
        PL_array=importdata(filename_PL);
        BSBS_PL_array_full=[BSBS_PL_array_full; PL_array.data(:,[5 6])];
    end
    % Write the PL output file
    sfile_PL=strcat('Raytracing_scenarios/',scenario_write,'/',scenario_write,'.',num2str(TX_ID_BS_Output(ii)),'.PL.BSBS.mat');
    save(sfile_PL,'BSBS_PL_array_full');
end
disp(' ')
disp('done!')
disp(' ')
fprintf('This message is posted at date and time %s\n', datestr(now,'dd-mmm-yyyy HH:MM:SS.FFF'))

%% Local functions
function [stringarrayTX] = number2stringTX(numberarrayTX)
%number2stringTX converts the BS ID number to a string with prefix of appended zeros
stringarrayTX = cell(numel(numberarrayTX),1);
for tt=1:1:numel(numberarrayTX)
    if numberarrayTX(tt)<10
        stringarrayTX{tt} = strcat('0',num2str(numberarrayTX(tt)));
    else
        stringarrayTX{tt} = num2str(numberarrayTX(tt));
    end
end
end

function [stringarrayRX] = number2stringRX(numberarrayRX)
%number2stringRX converts the user grid ID number to a string with prefix of appended zeros
stringarrayRX = cell(numel(numberarrayRX),1);
for rr=1:1:numel(numberarrayRX)
    if numberarrayRX(rr)<10
        stringarrayRX{rr} = strcat('00',num2str(numberarrayRX(rr)));
    else
        stringarrayRX{rr} = strcat('0',num2str(numberarrayRX(rr)));
    end
end
end
