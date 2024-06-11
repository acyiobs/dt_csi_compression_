files = dir('**/*.mat');
% folder = strsplit(files(1).folder, '\')
% The following command will create a single precision of copy 
% the all mat files under this directory (including subfolders)
% The copies will be put under the current subfolder names
% with a '_single' postfix.
% To make a single copy of the folder in the previous directory, comment out +1 below
save_subdir_id = length(strsplit(pwd, '\')); %+1; 


for i=1:length(files)
    folder = files(i).folder;
    save_folder = get_save_folder(folder, save_subdir_id);
    
    load_file = [files(i).folder, '\', files(i).name];
    disp(files(i).name)
    x = load(load_file);
    variable_name = fieldnames(x);
    var = getfield(x, variable_name{1});
    if ~strcmp(class(var), 'double')
        disp(['not double - passing ' load_file])
        continue;
    else
        eval([variable_name{1} ' = single(getfield(x, variable_name{1}));']);
    end
    if numel(x)>1
        error('There might be a problem!')
    end
    if ~exist(save_folder, 'dir')
        mkdir(save_folder);
    end
    save([save_folder '\' files(i).name], variable_name{1});
end

function save_folder = get_save_folder(folder, save_subdir_id)
    folder = strsplit(folder, '\');
    folder{save_subdir_id} = [folder{save_subdir_id} '_single'];
    save_folder = strjoin(folder, '\');
end