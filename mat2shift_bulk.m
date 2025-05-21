% Clear workspace variables.
clear;

% Add the file path of the toolbox
addpath('toolbox_file_path\PeakFit-main')
% Add the file path of MAT files
file_path = "mat_files_file_path\mat_files"; 

% getting every .mat files
file_list = dir(fullfile(file_path, '*.mat'));
file_names = {file_list.name};
% Get numbers in the file names
num = cellfun(@(x) sscanf(x, 'Gr-%d.mat'), file_names);  % get number after Gr-
[~, idx] = sort(num);  % Get sort numbers
file_list = file_list(idx);  % list again according to numbers


% Build an empty table for the result
results = table('Size', [length(file_list), 2], ...
    'VariableNames', {'FileName', 'PeakPosition'}, ...
    'VariableTypes', {'string', 'double'});

% Pepare the average value list
group_num = length(file_list) / 5;
average_list = zeros(group_num, 1);  % initialize the average value list

for i = 1:length(file_list)
    current_file = file_list(i).name;
    full_file_path = fullfile(file_path, current_file);
    % loading data
    S = load(full_file_path);
    % Starting fitting the peek
    S.Fit = PeakFit(S.Data, 'Window', [1000, 2000], 'PeakShape', 'Lorentzian', ...
    'CenterLow', [1000], ...
    'CenterUp', [2000], ...
    'WidthUp', [50], ...
    'BaselinePolyOrder', 1);
    
    % record results
    results.FileName(i) = current_file;
    results.PeakPosition(i) = S.Fit.Center(1);

    % calculate average value for each 5 files
    if mod(i, 5) == 0  
        group_idx = i/5;  % index
        start_idx = i-4;  
        average_list(group_idx) = mean(results.PeakPosition(start_idx:i));
    end

end

% output_excel = fullfile(mat_files_path, 'peak_results.xlsx');
% writetable(results, output_excel);
disp(results);
% Show the average_list
disp('===== Average Value =====');
for k = 1:length(average_list)
    fprintf('Average peek position (Raman shift): %.2f cm⁻¹\n', k, average_list(k));
end
% save(fullfile(file_path, 'peak_averages.mat'), 'average_list');
