function read_EEG(filename, band, notch, l, h, n, trim)
global data;
global trigger;
EEG = pop_biosig(filename);
disp(filename)
EEG = pop_reref(EEG, [97 98]);
%滤波
if band==1
    %EEG = pop_eegfiltnew(EEG, [], 55, 3862, 1, [], 0);
    %EEG = pop_eegfiltnew(EEG, [], 95, 762, 0, [], 0);
    EEG = pop_eegfiltnew(EEG, 'locutoff',l,'hicutoff',h);%带通
end
if notch==1
    EEG = pop_eegfiltnew(EEG, 49, 51, 13518, 1, [], 0);%陷波
end
%通道选取
data = EEG.data(1:96, :);
trigger = EEG.event;
trigger = struct2cell(trigger);
trigger = cell2mat(trigger);
trigger = trigger(2, :, :);
trigger = trigger(2:end);
if trim==1
    trigger = trigger(:, :, 1:n);
end
trigger = reshape(trigger, [1, n]);
