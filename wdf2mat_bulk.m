% Add toolbox path
addpath('toolbox_path\Matlab');  
% Add target path to save the result files
targetpath = "target_file_path\MATLAB_Data programe\wdf2mat\Matlab\mat_files"; 
% Add WDF files path
file_path = "WDF_files_path"; 

% Get every WDF files
file_list = dir(fullfile(file_path, '*.wdf'));
for i = 1:length(file_list)
    % Read the data and get number
    current_file = file_list(i).name;
    full_file_path = fullfile(file_path, current_file);
    wdf = WdfReader(full_file_path, 'rb'); 
    raman_shift = wdf.GetXList();  
    spectrum = wdf.GetSpectra(1, 1);  

    % Storage the files with same name
    [~, name, ~] = fileparts(current_file);  % Find file names
    output_mat_name = fullfile(targetpath, [name '.mat']); % Form complete path
    Data = [raman_shift; spectrum];
    save(output_mat_name, 'Data');        % Save
    targetpath/file_name.mat

    % 5. Close the file
    wdf.Close();
end
