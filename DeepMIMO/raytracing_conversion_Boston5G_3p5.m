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

%% Read and Write the Direction of Departures (DoDs) and Complex Impulse Responses (CIRs) for every BS

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

for ii=1:1:num_BS % For each BS 
    disp(['Reading and writing the DoD, DOA, CIR files for BS# ' num2str(ii) ' out of ' num2str(num_BS) ' BSs (BS-user) ...'])
    DoD_array_full=[];
    DoA_array_full=[];
    CIR_array_full=[];
    num_points=0;
    for jj=1:1:num_user_grids % For each user grid
        
       % Read DoD files
       filename_DoD=strcat('Raytracing_scenarios/',scenario_read,'/',scenario_read,'.dod.t001_',TX_ID_BS_str{ii},'.r',RX_grids_ID_user_str{jj},'.p2m');
       DoD_data=importdata(filename_DoD);
       DoD_array=reshape(DoD_data.data',1,size(DoD_data.data,1)*size(DoD_data.data,2));
       % Read DoA files
       filename_DoA=strcat('Raytracing_scenarios/',scenario_read,'/',scenario_read,'.doa.t001_',TX_ID_BS_str{ii},'.r',RX_grids_ID_user_str{jj},'.p2m');
       DoA_data=importdata(filename_DoA);
       DoA_array=reshape(DoA_data.data',1,size(DoA_data.data,1)*size(DoA_data.data,2));      
       % Read CIR files
       filename_CIR=strcat('Raytracing_scenarios/',scenario_read,'/',scenario_read,'.cir.t001_',TX_ID_BS_str{ii},'.r',RX_grids_ID_user_str{jj},'.p2m');
       CIR_data=importdata(filename_CIR);
       CIR_array=reshape(CIR_data.data',1,size(CIR_data.data,1)*size(CIR_data.data,2));
    
       num_points=num_points+DoD_data.data(1);

       DoD_array(1:2)=[];
       DoA_array(1:2)=[];
       CIR_array(1:2)=[];
       %Concatenate DoD and CIR arrays
       DoD_array_full=[DoD_array_full DoD_array];
       DoA_array_full=[DoA_array_full DoA_array];
       CIR_array_full=[CIR_array_full CIR_array];
    end
    % Name of the output DoD & CIR files
    sfile_DoD=strcat('Raytracing_scenarios/',scenario_write,'/',scenario_write,'.',num2str(TX_ID_BS_Output(ii)),'.DoD.mat');
    sfile_DoA=strcat('Raytracing_scenarios/',scenario_write,'/',scenario_write,'.',num2str(TX_ID_BS_Output(ii)),'.DoA.mat');
    sfile_CIR=strcat('Raytracing_scenarios/',scenario_write,'/',scenario_write,'.',num2str(TX_ID_BS_Output(ii)),'.CIR.mat'); 
    % Concatenate the number of points to the final output arrays
    DoD_array_full=[num_points DoD_array_full];
    DoA_array_full=[num_points DoA_array_full];
    CIR_array_full=[num_points CIR_array_full];
    % Write the DoD and CIR output files
    save(sfile_DoD,'DoD_array_full');
    save(sfile_DoA,'DoA_array_full');
    save(sfile_CIR,'CIR_array_full');
end

for ii=1:1:num_TX_BS % For each BS transmitter
    disp(['Reading and writing the DoD, DOA, CIR files for BS# ' num2str(ii) ' out of ' num2str(num_TX_BS) ' BSs (BS-BS) ...'])
    BB_DoD_array_full=[];
    BB_DoA_array_full=[];
    BB_CIR_array_full=[];
    BB_num_points=0;
    for jj=1:1:num_BS_grids % For each BS receiver
        
       % Read DoD files
       filename_DoD=strcat('Raytracing_scenarios/',scenario_read,'/',scenario_read,'.dod.t001_',TX_ID_BS_str{ii},'.r',RX_grids_ID_BS_str{jj},'.p2m');
       DoD_data=importdata(filename_DoD);
       DoD_array=reshape(DoD_data.data',1,size(DoD_data.data,1)*size(DoD_data.data,2));
       % Read DoA files
       filename_DoA=strcat('Raytracing_scenarios/',scenario_read,'/',scenario_read,'.doa.t001_',TX_ID_BS_str{ii},'.r',RX_grids_ID_BS_str{jj},'.p2m');
       DoA_data=importdata(filename_DoA);
       DoA_array=reshape(DoA_data.data',1,size(DoA_data.data,1)*size(DoA_data.data,2));      
       % Read CIR files
       filename_CIR=strcat('Raytracing_scenarios/',scenario_read,'/',scenario_read,'.cir.t001_',TX_ID_BS_str{ii},'.r',RX_grids_ID_BS_str{jj},'.p2m');
       CIR_data=importdata(filename_CIR);
       CIR_array=reshape(CIR_data.data',1,size(CIR_data.data,1)*size(CIR_data.data,2));
    
       BB_num_points=BB_num_points+DoD_data.data(1);

       DoD_array(1:2)=[];
       DoA_array(1:2)=[];
       CIR_array(1:2)=[];
       %Concatenate DoD and CIR arrays
       BB_DoD_array_full=[BB_DoD_array_full DoD_array];
       BB_DoA_array_full=[BB_DoA_array_full DoA_array];
       BB_CIR_array_full=[BB_CIR_array_full CIR_array];
    end
    % Name of the output DoD & CIR files
    sfile_DoD=strcat('Raytracing_scenarios/',scenario_write,'/',scenario_write,'.',num2str(TX_ID_BS_Output(ii)),'.DoD.BSBS.mat');
    sfile_DoA=strcat('Raytracing_scenarios/',scenario_write,'/',scenario_write,'.',num2str(TX_ID_BS_Output(ii)),'.DoA.BSBS.mat');
    sfile_CIR=strcat('Raytracing_scenarios/',scenario_write,'/',scenario_write,'.',num2str(TX_ID_BS_Output(ii)),'.CIR.BSBS.mat'); 
    % Concatenate the number of points to the final output arrays
    BB_DoD_array_full=[BB_num_points BB_DoD_array_full];
    BB_DoA_array_full=[BB_num_points BB_DoA_array_full];
    BB_CIR_array_full=[BB_num_points BB_CIR_array_full];
    % Write the DoD and CIR output files
    save(sfile_DoD,'BB_DoD_array_full');
    save(sfile_DoA,'BB_DoA_array_full');
    save(sfile_CIR,'BB_CIR_array_full');
end

%% Read the path loss/gain files and Write the Location (Loc) files for the users (DeepMIMO V1)
%Reading files from one BS is enough

Loc_array_full=[];
disp('Reading and writing the Loc files for the users (DeepMIMOv1) ...')
% Read one Loc file from one BS
for jj=1:1:num_user_grids % For each user grid
   filename_Loc=strcat('Raytracing_scenarios/',scenario_read,'/',scenario_read,'.pl.t001_',TX_ID_BS_str{1},'.r',RX_grids_ID_user_str{jj},'.p2m');
   Loc_array=importdata(filename_Loc);
   Loc_array_full=[Loc_array_full; Loc_array.data];   
end

% Write the Loc output file
sfile_Loc=strcat('Raytracing_scenarios/',scenario_write,'/',scenario_write,'.Loc.mat');
save(sfile_Loc,'Loc_array_full');

%% Read the path loss/gain files and Write the Location (Loc) files for the user receivers (User receivers - DeepMIMO V2)
%Reading files from one BS is enough

RX_Loc_array_full=[];
disp('Reading and writing the Loc files for all the receivers (BS-user) ...')
% Read one Loc file from one BS
for jj=1:1:num_user_grids % For each user grid
   filename_Loc=strcat('Raytracing_scenarios/',scenario_read,'/',scenario_read,'.pl.t001_',TX_ID_BS_str{1},'.r',RX_grids_ID_user_str{jj},'.p2m');
   Loc_array=importdata(filename_Loc);
   RX_Loc_array_full=[RX_Loc_array_full; Loc_array.data];   
end

% Write the Loc output file
sfile_RX_Loc=strcat('Raytracing_scenarios/',scenario_write,'/',scenario_write,'.RX_Loc.mat');
save(sfile_RX_Loc,'RX_Loc_array_full');

%% Read the path loss/gain files and Write the Location (Loc) files for all the BS transmitters (User receivers - DeepMIMO V2)
%Reading files from one BS is enough

TX_Loc_array_full=[];
disp('Reading and writing the Loc files for all the transmitters (BS-user) ...')
% Read one Loc file from one BS
for jj=1:1:num_TX_BS % For each user grid
   filename_Loc=strcat('Raytracing_scenarios/',scenario_read,'/',scenario_read,'.pl.t001_',TX_ID_BS_str{1},'.r',TX_ID_BS2_str{jj},'.p2m');
   Loc_array=importdata(filename_Loc);
   TX_Loc_array_full=[TX_Loc_array_full; Loc_array.data];   
end

% Write the Loc output file
sfile_TX_Loc=strcat('Raytracing_scenarios/',scenario_write,'/',scenario_write,'.TX_Loc.mat');
save(sfile_TX_Loc,'TX_Loc_array_full');

%% Read the path loss/gain files and Write the Location (Loc) files for the BS receivers (BS receivers - DeepMIMO V2)
%Reading files from one BS is enough

BSBS_RX_Loc_array_full=[];
disp('Reading and writing the Loc files for all the receivers (BS-BS) ...')
% Read one Loc file from one BS
for jj=1:1:num_BS_grids % For each user grid
   filename_Loc=strcat('Raytracing_scenarios/',scenario_read,'/',scenario_read,'.pl.t001_',TX_ID_BS_str{1},'.r',TX_ID_BS2_str{jj},'.p2m');
   Loc_array=importdata(filename_Loc);
   BSBS_RX_Loc_array_full=[BSBS_RX_Loc_array_full; Loc_array.data];   
end

% Write the Loc output file
sfile_RX_Loc=strcat('Raytracing_scenarios/',scenario_write,'/',scenario_write,'.BSBS.RX_Loc.mat');
save(sfile_RX_Loc,'BSBS_RX_Loc_array_full');

%% Read the path loss/gain files and Write the Location (Loc) files for the BS transmitters (BS receivers - DeepMIMO V2)
%Reading files from one BS is enough

BSBS_TX_Loc_array_full=[];
disp('Reading and writing the Loc files for all the transmitters (BS-BS) ...')
% Read one Loc file from one BS
for jj=1:1:num_TX_BS % For each user grid
   filename_Loc=strcat('Raytracing_scenarios/',scenario_read,'/',scenario_read,'.pl.t001_',TX_ID_BS_str{1},'.r',TX_ID_BS2_str{jj},'.p2m');
   Loc_array=importdata(filename_Loc);
   BSBS_TX_Loc_array_full=[BSBS_TX_Loc_array_full; Loc_array.data];   
end

% Write the Loc output file
sfile_TX_Loc=strcat('Raytracing_scenarios/',scenario_write,'/',scenario_write,'.BSBS.TX_Loc.mat');
save(sfile_TX_Loc,'BSBS_TX_Loc_array_full');

%% Write the scenario parameters
disp('Reading and writing the params files for BS-user data ...')
sfile_params=strcat('Raytracing_scenarios/',scenario_write,'/',scenario_write,'.params.mat');
save(sfile_params,'num_BS','user_grids','carrier_freq','transmit_power')
disp('Reading and writing the params files for BS-BS data ...')
sfile_params=strcat('Raytracing_scenarios/',scenario_write,'/',scenario_write,'.BSBS.params.mat');
save(sfile_params,'num_TX_BS','BS_grids')

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
