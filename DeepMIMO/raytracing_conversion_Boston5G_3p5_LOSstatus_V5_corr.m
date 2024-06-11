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

num_points = sum((user_grids(:,2)-user_grids(:,1)+1).*user_grids(:,3));

BB_num_points = sum((BS_grids(:,2)-BS_grids(:,1)+1).*BS_grids(:,3));

%% Read and write the LoS tags from every BS to every user

for ii= 1:1:num_BS % For each BS
    disp(['Reading and writing the LoS tag files for BS# ' num2str(ii) ' out of ' num2str(num_BS) ' BSs (BS-user) ...'])
    fprintf('This message is posted at time %s\n', datestr(now,'HH:MM:SS.FFF'))
    for jj= 1:1:num_user_grids % For each user grid
        disp('  ');
        disp(['User grid ' num2str(jj)  ' :       ']);
        % Read PATHS files
        filename_PATHS=strcat('Raytracing_scenarios/',scenario_read,'/',scenario_read,'.paths.t001_',TX_ID_BS_str{ii},'.r',RX_grids_ID_user_str{jj},'.p2m');
        PATHS_data=importdata(filename_PATHS,' ', 1e8);
        No_lines = length(PATHS_data);
        Starting_line_idx = 23; %Skip 22 lines of headers %first line of [No. of RX_point path_num]
        %size of LOS_tag_array is the number of users/BS per user/BS grid
        No_users_per_currentgrid = (user_grids(jj,2)-user_grids(jj,1)+1)*user_grids(jj,3);
        assert( str2num(PATHS_data{Starting_line_idx-1}) == No_users_per_currentgrid, 'the number of receiver per user grid does not match!')
        No_users_per_previousgrids = 0;
        if jj>1
            No_users_per_previousgrids = sum((user_grids(1:1:(jj-1),2)-user_grids(1:1:(jj-1),1)+1).*user_grids(1:1:(jj-1),3));
        end
        count_update=0;
        reverseStr=0;
        ll = Starting_line_idx;
        while ll <=No_lines %Starting_line_idx:1:No_lines %increase line index for each loop
            if(length(str2num(PATHS_data{ll})) == 2) %find the line with [RX point no , num_paths]
                current_user_data = str2num(PATHS_data{ll}); % Read the info on the current user ==> [RX point no , num_paths]
                current_user_ID = current_user_data(1)+No_users_per_previousgrids;
                if(current_user_data(2) == 0) % Check if the number of paths is 0 ==> receiver is completely blocked
                    LOS_tag_array_user(current_user_ID) = -1;  %Status: "complete blockage". Nothing is received. %%%%%%%%%%%%%%%%%%
                    ll = ll +1;
                    %continue;
                else
                    Each_path_status = zeros(current_user_data(2),1);
                    for pp=1:1:current_user_data(2) %current_user(2) is the number of paths for the current user
                        if pp==1
                            Disp = 2; %Pointer displacement from the "ll" pointer
                        end
                        % Read the info on the current path
                        current_path_data = str2num(PATHS_data{ll + Disp});
                        if  current_path_data(2) == 0 % Number of reflections is zero
                            Each_path_status(pp) = 1; %Status: "LoS exists". LoS and NLoS path(/s) are received.
                        else
                            path_int_summary = split( PATHS_data{ll + Disp+1}, '-' );
                            path_int_summary([1 end]) =[];
                            int_char = {'R','D','T','DS'};
                            inter_type_1 = zeros(current_path_data(2),numel(int_char));
                            for qq = 1:4 % Path must have R, T, or D to be NLOS
                                inter_type_1(:,qq) = cellfun(@(y) strcmp(y, int_char{qq}), path_int_summary);
                            end
                            if any(inter_type_1(:))
                                Each_path_status(pp) = 0; %Status: "LoS blockage". Only NLoS path(/s) is(/are) received.
                            else
                                Each_path_status(pp) = 1; %Status: "LoS exists". LoS and NLoS path(/s) are received.
                            end
                        end
                        Disp = Disp + 4 + current_path_data(2);
                    end
                    if any(Each_path_status)
                        LOS_tag_array_user(current_user_ID) = 1;  %Status: "LoS exists". LoS and NLoS path(/s) are received. %%%%%%%%%%%%%%%%%%
                    else
                        LOS_tag_array_user(current_user_ID) = 0; %Status: "LoS blockage". Only NLoS path(/s) is(/are) received. %%%%%%%%%%%%%%%%%%
                    end
                    ll = ll + Disp;
                end
            else
                warning('Pointer calculation logical error!');
            end
            count_update = count_update + 1;
            perc_update = 100 * count_update /(No_users_per_currentgrid);
            msg = sprintf('- Percent done: %3.1f', perc_update); %Don't forget this semicolon
            fprintf([reverseStr, msg]);
            reverseStr = repmat(sprintf('\b'), 1, length(msg));
        end
    end
    disp('          ');
    fprintf('This message is posted at time %s\n', datestr(now,'HH:MM:SS.FFF'))
    disp('          ');
    
    % Name of the output LoS files
    sfile_LoS_tag_user=strcat('Raytracing_scenarios/',scenario_write,'/',scenario_write,'.',num2str(TX_ID_BS_Output(ii)),'.LoS.mat');
    % Concatenate the number of points to the final output array
    LOS_tag_array_full=[num_points LOS_tag_array_user];
    % Write the LoS output files
    save(sfile_LoS_tag_user,'LOS_tag_array_full');
end

%% Read and write the LoS tags from every BS to every BS

for ii= 1:1:num_TX_BS % For each BS
    disp(['Reading and writing the LoS tag files for BS# ' num2str(ii) ' out of ' num2str(num_TX_BS) ' BSs (BS-BS) ...'])
    fprintf('This message is posted at time %s\n', datestr(now,'HH:MM:SS.FFF'))
    for jj=1:1:num_BS_grids % For each user grid
        disp('  ');
        disp(['BS grid ' num2str(jj)  ' :       ']);
        % Read PATHS files
        filename_PATHS=strcat('Raytracing_scenarios/',scenario_read,'/',scenario_read,'.paths.t001_',TX_ID_BS_str{ii},'.r',RX_grids_ID_BS_str{jj},'.p2m');
        PATHS_data=importdata(filename_PATHS,' ', 1e8);
        No_lines = length(PATHS_data);
        Starting_line_idx = 23; %Skip 22 lines of headers %first line of [No. of RX_point path_num]
        %size of LOS_tag_array is the number of users/BS per user/BS grid
        No_BSs_per_currentgrid = (BS_grids(jj,2)-BS_grids(jj,1)+1)*BS_grids(jj,3);
        assert( str2num(PATHS_data{Starting_line_idx-1}) == No_BSs_per_currentgrid, 'the number of receiver per BS grid does not match!')
        No_BSs_per_previousgrids = 0;
        if jj>1
            No_BSs_per_previousgrids = sum((BS_grids(1:1:(jj-1),2)-BS_grids(1:1:(jj-1),1)+1).*BS_grids(1:1:(jj-1),3));
        end
        count_update=0;
        reverseStr=0;
        ll = Starting_line_idx;
        while ll <=No_lines %Starting_line_idx:1:No_lines %increase line index for each loop
            if(length(str2num(PATHS_data{ll})) == 2) %find the line with [RX point no , num_paths]
                current_BS_data = str2num(PATHS_data{ll}); % Read the info on the current user ==> [RX point no , num_paths]
                current_BS_ID = current_BS_data(1)+No_BSs_per_previousgrids;
                if(current_BS_data(2) == 0) % Check if the number of paths is 0 ==> receiver is completely blocked
                    LOS_tag_array_BS(current_BS_ID) = -1;  %Status: "complete blockage". Nothing is received. %%%%%%%%%%%%%%%%%%
                    ll = ll +1;
                    %continue;
                else
                    Each_path_status = zeros(current_BS_data(2),1);
                    for pp=1:1:current_BS_data(2) %current_user(2) is the number of paths for the current user
                        if pp==1
                            Disp = 2; %Pointer displacement from the "ll" pointer
                        end
                        % Read the info on the current path
                        current_path_data = str2num(PATHS_data{ll + Disp});
                        if  current_path_data(2) == 0 % Number of reflections is zero
                            Each_path_status(pp) = 1; %Status: "LoS exists". LoS and NLoS path(/s) are received.
                        else
                            path_int_summary = split( PATHS_data{ll + Disp+1}, '-' );
                            path_int_summary([1 end]) =[];
                            int_char = {'R','D','T','DS'};
                            inter_type_1 = zeros(current_path_data(2),numel(int_char));
                            for qq = 1:4 % Path must have R, T, or D to be NLOS
                                inter_type_1(:,qq) = cellfun(@(y) strcmp(y, int_char{qq}), path_int_summary);
                            end
                            if any(inter_type_1(:))
                                Each_path_status(pp) = 0; %Status: "LoS blockage". Only NLoS path(/s) is(/are) received.
                            else
                                Each_path_status(pp) = 1; %Status: "LoS exists". LoS and NLoS path(/s) are received.
                            end
                        end
                        Disp = Disp + 4 + current_path_data(2);
                    end
                    if any(Each_path_status)
                        LOS_tag_array_BS(current_BS_ID) = 1;  %Status: "LoS exists". LoS and NLoS path(/s) are received. %%%%%%%%%%%%%%%%%%
                    else
                        LOS_tag_array_BS(current_BS_ID) = 0; %Status: "LoS blockage". Only NLoS path(/s) is(/are) received. %%%%%%%%%%%%%%%%%%
                    end
                    ll = ll + Disp;
                end
            else
                warning('Pointer calculation logical error!');
            end
            count_update = count_update + 1;
            perc_update = 100 * count_update /(No_BSs_per_currentgrid);
            msg = sprintf('- Percent done: %3.1f', perc_update); %Don't forget this semicolon
            fprintf([reverseStr, msg]);
            reverseStr = repmat(sprintf('\b'), 1, length(msg));
        end
    end
    disp('          ');
    fprintf('This message is posted at time %s\n', datestr(now,'HH:MM:SS.FFF'))
    disp('          ');
    
    % Name of the output LoS files
    sfile_LoS_tag_BS=strcat('Raytracing_scenarios/',scenario_write,'/',scenario_write,'.',num2str(TX_ID_BS_Output(ii)),'.LoS.BSBS.mat');
    % Concatenate the number of points to the final output array
    LOS_tag_array_full=[BB_num_points LOS_tag_array_BS];
    % Write the LoS output files
    save(sfile_LoS_tag_BS,'LOS_tag_array_full');
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
