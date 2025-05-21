%% Setting power
power_list = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50];    % 功率列表mW

%% Initial set the mouse
try
    mouse = java.awt.Robot;
    num_clicks = 5;     % 5 times measurement for each power
catch
    clear dev;
    error('Failed to set mouse!');
end

run_button_pos_run = [500, 1080-1000];  % the run botton position
run_button_pos_power = [1314, 1080-864]; % the power botton position

%% Main loop: power
for i = 1:length(power_list)
    target_power = power_list(i);

    % Click the power place
    mouse.mouseMove(run_button_pos_power(1), run_button_pos_power(2));
    mouse.mousePress(java.awt.event.InputEvent.BUTTON1_MASK);
    mouse.mouseRelease(java.awt.event.InputEvent.BUTTON1_MASK);
    mouse.mousePress(java.awt.event.InputEvent.BUTTON1_MASK);
    mouse.mouseRelease(java.awt.event.InputEvent.BUTTON1_MASK);
    pause(0.1); 

    % Input power value
    power_str = num2str(target_power); % change the power value into string
    for j = 1:length(power_str)
        % deal with float dot
        if power_str(j) == '.'
            key_code = java.awt.event.KeyEvent.VK_PERIOD;
        else
            key_code = java.awt.event.KeyEvent.VK_0 + str2double(power_str(j));
        end
        % Press and release
        mouse.keyPress(key_code);
        mouse.keyRelease(key_code);
        pause(0.1); % A short pause
    end
    
    % Conform by enter
    mouse.keyPress(java.awt.event.KeyEvent.VK_ENTER);
    mouse.keyRelease(java.awt.event.KeyEvent.VK_ENTER);
    pause(0.1);

    fprintf('>>> Begin to run（Power=%.3f W）\n', target_power);

    for click_count = 1:num_clicks
        % Mouse click run bottun
        mouse.mouseMove(run_button_pos_run(1), run_button_pos_run(2));
        mouse.mousePress(java.awt.event.InputEvent.BUTTON1_MASK);
        mouse.mouseRelease(java.awt.event.InputEvent.BUTTON1_MASK);
        fprintf('  %d times begin @ %s\n', click_count, datetime("now"));
        pause(measure_time);  % Waiting for masurement
    end

    fprintf('Power: %.3f kHz Finish measurement\n\n', target_power);
end
