function generate(save_root, txt, fs,endtime ,down_ratio, n, video)
global data;
global trigger;
if ~exist(save_root, 'dir')  
  mkdir(save_root);  
end 
data = data(1:96, :)';
channel = 96;

% 参数定义
pre_stimulus_time = 0.2; % 刺激前0.2秒
post_stimulus_time = endtime; % 刺激后2.0秒
channels = size(data, 2);
samples = round((pre_stimulus_time + post_stimulus_time) * fs);

EEG = zeros(n, round(post_stimulus_time*fs/down_ratio),channel);%6144，96
for i = 1:length(trigger)
    % 提取事件对应的数据段
    EEGtmp = reshape(data(trigger(i)-round(pre_stimulus_time*fs):trigger(i)+round(post_stimulus_time*fs)-1, :), [1, samples, channels]);
    eeg = reshape(EEGtmp, [samples, channels]);
    % 基线矫正
    baseline_idx = round(1:pre_stimulus_time*fs);
    baseline_eeg = eeg(baseline_idx, :);
    baseline = mean(baseline_eeg, 1);
    tmpeeg = eeg - baseline;
    eeg = tmpeeg(pre_stimulus_time*fs:end, :);
    %降采样
    eeg = eeg(1:down_ratio:end, :);
    % 保存处理后的数据
       EEG(i, :, :) = reshape(eeg, [1, size(eeg, 1), size(eeg, 2)]);
end
%归一化
samples=round(post_stimulus_time*fs/down_ratio);
EEG = reshape(EEG, [n*samples, channel]);
ave = mean(EEG, 1);
STD = std(EEG, 1);
for i = 1:channel
   EEG(:, i) = (EEG(:, i)-ave(i))/STD(i);
end
EEG = reshape(EEG, [n, samples, channel]);
fid = fopen(txt);
tline = fgetl(fid);
i = 1;
while ischar(tline)
    if video==0
        tline = tline(1:end-5);
    end
    tline = [tline, '.mat'];
    eeg = reshape(EEG(i, :, :), [samples, channel]);
    name = [save_root, tline];
    save(name, 'eeg');
    tline = fgetl(fid);
    i = i+1;
end


