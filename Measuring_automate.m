%% Connect dev (visadev)
try
    dev = visadev('USB0::0xF4EC::0x1102::SDG2XCAD5R2983::0::INSTR');
    dev.Timeout = 10; 
    disp('Successfully connecte!!');
catch ME
    error('Connection failed！Error information：%s', ME.message);
end

%% Basic parameters of rectangular function
try
    writeline(dev, 'C1:BSWV WVTP,SQUARE');    % Setting Rectangular wave
    writeline(dev, 'C1:BSWV AMP,0.4');         % Setting amplitude
    writeline(dev, 'C1:BSWV OFST,0');         % Setting offset
    writeline(dev, 'C1:BSWV DUTY,50');        % Setting duty-cycle
    writeline(dev, 'C1:OUTP LOAD,50');        % Setting 50 ou load
    writeline(dev, 'C1:OUTP ON');             % Open output
    disp('Finish setting rectangular wave!');
catch ME
    clear dev;
    error('Failed to set the rectangular wave！Error information：%s', ME.message);
end

%% Setting frequency list
freq_list = [1e3, 10e3, 100e3, 1e6];    % Frequency list
tolerance = 50;                          % Allowing error: 50Hz
max_retries = 2;                         % Maximum Retry

%% Initial mouse setting
try
    mouse = java.awt.Robot;
    measure_time = 6;   % measure time per running
    num_clicks = 5;     % 5 times per frequency
catch
    clear dev;
    error('mouse initial setting failed!');
end

run_button_pos = [500, 1080-1000];  % Setting the click position

%% main loop: frequency
for i = 1:length(freq_list)
    target_freq = freq_list(i);
    success = false;

    for retry = 1:max_retries
        % Send setting commandation
        writeline(dev, sprintf('C1:BSWV FRQ,%g', target_freq));
        
        % Ask real frequency
        writeline(dev, 'C1:BSWV?');
        resp = readline(dev);
        freq_pattern = 'FRQ,([\d\.]+)HZ';
        tokens = regexp(resp, freq_pattern, 'tokens'); 
        actual_freq = str2double(tokens{1}{1}); 
        
        
        % 检查频率误差
        if abs(actual_freq - target_freq) <= tolerance
            fprintf('[Succss] frequency set: target=%.3f kHz, real frequency=%.3f kHz\n', target_freq/1e3, actual_freq/1e3);
            success = true;
            break;
        else
            fprintf('[Retry%d/%d] Frequency error: Target=%.3f kHz, Real=%.3f kHz\n', retry, max_retries, target_freq/1e3, actual_freq/1e3);
        end
    end

    if success
        fprintf('>>> Ready to run（Frequency=%.3f kHz）\n', actual_freq/1e3);
        for click_count = 1:num_clicks
            % Mouse click
            mouse.mouseMove(run_button_pos(1), run_button_pos(2));
            mouse.mousePress(java.awt.event.InputEvent.BUTTON1_MASK);
            mouse.mouseRelease(java.awt.event.InputEvent.BUTTON1_MASK);
            fprintf('  %dtimes begin @ %s\n', click_count, datetime("now"));
            pause(measure_time);  % Waiting for measurement
        end
        fprintf('Frequency %.3f kHz Measurement done\n\n', actual_freq/1e3);
    else
        clear dev;
        error('Failed to set frequncy: Target=%.3f kHz, Real=%.3f kHz', target_freq/1e3, actual_freq/1e3);
    end
end

%% Close connection
try
    writeline(dev, 'C1:OUTP OFF');   % clost channel 1
    clear dev;
    disp('Device is closed。');
catch
    warning('warning');
end
