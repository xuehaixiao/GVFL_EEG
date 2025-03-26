%addpath ~/pkg/eeglab14_1_1b

dir = '/home/ubuntu/Desktop/2tdisk/tsunami/CVPR2021-02785/data/imagenet40-1000-';%datapath
stimname = '/home/ubuntu/Desktop/2tdisk/tsunami/CVPR2021-02785/design/run-';%run.txt-path
name = '/home/ubuntu/Desktop/16tdisk/tsunami_data/';%savepath

eeglab;

band = 1;
notch = 1;

for subject = 1:1
    %band = 1;
    %notch = 0;
    % for run = 0:99
    % 
    % 
    %     bdf = sprintf('%s%d-%02d.bdf', dir, subject, run);
    %     disp(bdf)
    % 
    %     out = sprintf('%s/delta1_4hz/', name);
    %     stim = sprintf('%s%02d.txt',stimname, run);
    %     trim = run==14;
    %     disp('a')
    %     read_EEG(bdf, band, notch, 1,4,400, trim);%重参考、滤波、获取事件
    %     disp('b')
    %     generate(out, stim, 4096,2.0, 16, 400, 0);%通道选取，基线矫正，降采样（400，1024，96）
    %     clear bdf;
    %     clear stim;
    %     clear stim;
    %     clear eeg;
    % end
    % for run = 47:99
    % 
    % 
    %     bdf = sprintf('%s%d-%02d.bdf', dir, subject, run);
    %     disp(bdf)
    % 
    %     out = sprintf('%s/theta_4-7hz/', name);
    %     stim = sprintf('%s%02d.txt',stimname, run);
    %     trim = run==14;
    %     disp('a')
    %     read_EEG(bdf, band, notch,4,7, 400, trim);%重参考、滤波、获取事件
    %     disp('b')
    %     generate(out, stim, 4096,2.0, 16, 400, 0);%通道选取，基线矫正，降采样（400，1024，96）
    %     clear bdf;
    %     clear stim;
    %     clear stim;
    %     clear eeg;
    % end
    % for run = 0:99
    % 
    % 
    %     bdf = sprintf('%s%d-%02d.bdf', dir, subject, run);
    %     disp(bdf)
    % 
    %     out = sprintf('%s/alpha_7-12hz/', name);
    %     stim = sprintf('%s%02d.txt',stimname, run);
    %     trim = run==14;
    %     disp('a')
    %     read_EEG(bdf, band, notch,7,12, 400, trim);%重参考、滤波、获取事件
    %     disp('b')
    %     generate(out, stim, 4096,2.0, 16, 400, 0);%通道选取，基线矫正，降采样（400，1024，96）
    %     clear bdf;
    %     clear stim;
    %     clear stim;
    %     clear eeg;
    % end
    % for run = 0:99
    % 
    % 
    %     bdf = sprintf('%s%d-%02d.bdf', dir, subject, run);
    %     disp(bdf)
    % 
    %     out = sprintf('%s/gamma_25-45hz/', name);
    %     stim = sprintf('%s%02d.txt',stimname, run);
    %     trim = run==14;
    %     disp('a')
    %     read_EEG(bdf, band, notch,25,45, 400, trim);%重参考、滤波、获取事件
    %     disp('b')
    %     generate(out, stim, 4096, 1.0, 8, 400, 0);%通道选取，基线矫正，降采样（400，1024，96）
    %     clear bdf;
    %     clear stim;
    %     clear stim;
    %     clear eeg;
    % end

    % band = 1;
    % notch = 1;
    % for run = 0:99
    % 
    % 
    %     bdf = sprintf('%s%d-%02d.bdf', dir, subject, run);
    %     disp(bdf)
    % 
    %     out = sprintf('%s/1.25s/', name);
    %     stim = sprintf('%s%02d.txt',stimname, run);
    %     trim = run==14;
    %     disp('a')
    %     read_EEG(bdf, band, notch,1,100, 400, trim);%重参考、滤波、获取事件
    %     disp('b')
    %     generate(out, stim, 4096,1.25, 4, 400, 0);%通道选取，基线矫正，降采样（400，1024，96）
    %     clear bdf;
    %     clear stim;
    %     clear stim;
    %     clear eeg;
    % end
    % band = 1;
    % notch = 1;
    % for run = 0:99
    % 
    % 
    %     bdf = sprintf('%s%d-%02d.bdf', dir, subject, run);
    %     disp(bdf)
    % 
    %     out = sprintf('%s/0.25s/', name);
    %     stim = sprintf('%s%02d.txt',stimname, run);
    %     trim = run==14;
    %     disp('a')
    %     read_EEG(bdf, band, notch,1,100, 400, trim);%重参考、滤波、获取事件
    %     disp('b')
    %     generate(out, stim, 4096,0.25, 2, 400, 0);%通道选取，基线矫正，降采样（400，1024，96）
    %     clear bdf;
    %     clear stim;
    %     clear stim;
    %     clear eeg;
    % end
    % band = 1;
    % notch = 1;
    % for run = 0:99
    % 
    % 
    %     bdf = sprintf('%s%d-%02d.bdf', dir, subject, run);
    %     disp(bdf)
    % 
    %     out = sprintf('%s/0.5s/', name);
    %     stim = sprintf('%s%02d.txt',stimname, run);
    %     trim = run==14;
    %     disp('a')
    %     read_EEG(bdf, band, notch,1,100, 400, trim);%重参考、滤波、获取事件
    %     disp('b')
    %     generate(out, stim, 4096,0.5, 4, 400, 0);%通道选取，基线矫正，降采样（400，1024，96）
    %     clear bdf;
    %     clear stim;
    %     clear stim;
    %     clear eeg;
    % end
    % band=1;
    % notch=1;
    % for run = 0:99
    % 
    %     bdf = sprintf('%s%d-%02d.bdf', dir, subject, run);
    %     disp(bdf)
    % 
    %     out = sprintf('%s/0.75s/', name);
    %     stim = sprintf('%s%02d.txt',stimname, run);
    %     trim = run==14;
    %     disp('a')
    %     read_EEG(bdf, band, notch,1,100, 400, trim);%重参考、滤波、获取事件
    %     disp('b')
    %     generate(out, stim, 4096,0.75, 6, 400, 0);%通道选取，基线矫正，降采样（400，1024，96）
    %     clear bdf;
    %     clear stim;
    %     clear stim;
    %     clear eeg;
    % end
    band=1;
    notch=1;
    for run = 0:99

        bdf = sprintf('%s%d-%02d.bdf', dir, subject, run);
        disp(bdf)

        out = sprintf('%s/0.1s/', name);
        stim = sprintf('%s%02d.txt',stimname, run);
        trim = run==14;
        disp('a')
        read_EEG(bdf, band, notch,1,100, 400, trim);%重参考、滤波、获取事件
        disp('b')
        generate(out, stim, 4096,0.1, 1, 400, 0);%通道选取，基线矫正，降采样（400，1024，96）
        clear bdf;
        clear stim;
        clear stim;
        clear eeg;
    end
    % band=1;
    % notch=1;
    % for run = 0:99
    % 
    %     bdf = sprintf('%s%d-%02d.bdf', dir, subject, run);
    %     disp(bdf)
    % 
    %     out = sprintf('%s/1.75s/', name);
    %     stim = sprintf('%s%02d.txt',stimname, run);
    %     trim = run==14;
    %     disp('a')
    %     read_EEG(bdf, band, notch,1,100, 400, trim);%重参考、滤波、获取事件
    %     disp('b')
    %     generate(out, stim, 4096,1.75, 14, 400, 0);%通道选取，基线矫正，降采样（400，1024，96）
    %     clear bdf;
    %     clear stim;
    %     clear stim;
    %     clear eeg;
    % end
    % for run = 33:99
    % 
    %     bdf = sprintf('%s%d-%02d.bdf', dir, subject, run);
    %     disp(bdf)
    % 
    %     out = sprintf('%s/0.75s/', name);
    %     stim = sprintf('%s%02d.txt',stimname, run);
    %     trim = run==14;
    %     disp('a')
    %     read_EEG(bdf, band, notch,1,100, 400, trim);%重参考、滤波、获取事件
    %     disp('b')
    %     generate(out, stim, 4096,1.25, 10, 400, 0);%通道选取，基线矫正，降采样（400，1024，96）
    %     clear bdf;
    %     clear stim;
    %     clear stim;
    %     clear eeg;
    % end
   
end



exit

