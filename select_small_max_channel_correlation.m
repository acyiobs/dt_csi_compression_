clear;
clc;

synth_channel = load('DeepMIMO\DeepMIMO_datasets\Boston5G_3p5_small_notree\channel_ad_clip.mat').all_channel_ad_clip;
real_channel = load('DeepMIMO\DeepMIMO_datasets\Boston5G_3p5_1\channel_ad_clip.mat').all_channel_ad_clip;
synth_pos = load('DeepMIMO\DeepMIMO_datasets\Boston5G_3p5_small_notree\all_pos.mat').all_pos;
real_pos = load('DeepMIMO\DeepMIMO_datasets\Boston5G_3p5_1\all_pos.mat').all_pos;
load('result3\select_data_idx_synth.mat');
load('result3\select_data_idx_real.mat');

rng("default");

synth_channel_flat = reshape(synth_channel, size(synth_channel,1), []);
real_channel_flat = reshape(real_channel, size(real_channel,1), []);

synth_channel_flat_norm = zeros(size(synth_channel_flat));
for i=1:size(synth_channel_flat,2)
    synth_channel_flat_norm(i,:) = synth_channel_flat(i,:) / norm(synth_channel_flat(i,:));
end

real_channel_flat_norm = zeros(size(real_channel_flat));
for i=1:size(real_channel_flat,2)
    real_channel_flat_norm(i,:) = real_channel_flat(i,:) / norm(real_channel_flat(i,:));
end

synth_real_channel_correlation = zeros(size(real_channel,1),1);
parfor i=1:size(real_channel,1)
    correlation = abs(real_channel_flat_norm(i,:) * synth_channel_flat_norm');
    synth_real_channel_correlation(i) = max(correlation);
end
figure;
cdfplot(synth_real_channel_correlation);
[synth_real_channel_correlation_sort, sort_idx] = sort(synth_real_channel_correlation, 'ascend');
sort_idx = int64(sort_idx);
%%
num_low_correlation_samples = 1000;
select_real_idx = sort_idx(1:num_low_correlation_samples);

real_idx_correlation_sort = ori_real_idx_(sort_idx)';
correlation_data = [real_idx_correlation_sort-1, synth_real_channel_correlation_sort]; % the idx starts from 0 to match python
csvwrite('DeepMIMO\DeepMIMO_datasets\real_correlatin_idx_sort2.csv', correlation_data);

figure
scatter(real_pos(:,1), real_pos(:,2));
hold on;
scatter(real_pos(select_real_idx,1), real_pos(select_real_idx,2));
grid on;
legend('All real data points', 'Low correlation real data points');
xlabel('x-coordinate (meter)');
ylabel('y-coordinate (meter)');


