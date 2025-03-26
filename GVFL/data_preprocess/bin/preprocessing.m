addpath ~/pkg/eeglab14_1_1b

dir = '/fs/aux/qobi/eeg-datasets/imagenet40-1000/eeg/imagenet40-1000-';
name = '/tmp/imagenet40-1000/imagenet40-1000-';

eeglab;

band = 0;
notch = 0;

for subject = 1:1
    for run = 0:99
        bdf = sprintf('%s%d-%02d.bdf', dir, subject, run);
        out = sprintf('%s%d/', name, subject);
        stim = sprintf('../design/run-%02d.txt', run);
        trim = run==14;
        read_EEG(bdf, band, notch, 400, trim);
        generate(out, stim, 4096*0.5, 4, 400, 0);
    end
end

exit
