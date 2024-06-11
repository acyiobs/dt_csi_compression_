clear;
clc;

% synth_channel = load('DeepMIMO\DeepMIMO_datasets\old\Boston5G_3p5_small_notree\channel_ad_clip.mat').all_channel_ad_clip;
% real_channel = load('DeepMIMO\DeepMIMO_datasets\old\Boston5G_3p5_1\channel_ad_clip.mat').all_channel_ad_clip;
% synth_pos = load('DeepMIMO\DeepMIMO_datasets\old\Boston5G_3p5_small_notree\all_pos.mat').all_pos;
% real_pos = load('DeepMIMO\DeepMIMO_datasets\old\Boston5G_3p5_1\all_pos.mat').all_pos;


synth_channel = matfile('DeepMIMO\DeepMIMO_datasets\Boston5G_3p5_notree\channel_ad_clip.mat').all_channel_ad_clip;
real_channel = matfile('DeepMIMO\DeepMIMO_datasets\Boston5G_3p5_real\channel_ad_clip.mat').all_channel_ad_clip;
synth_pos = matfile('DeepMIMO\DeepMIMO_datasets\Boston5G_3p5_notree\dataset.mat').all_pos;
real_pos = matfile('DeepMIMO\DeepMIMO_datasets\Boston5G_3p5_real\dataset.mat').all_pos;

% load('result3\select_data_idx_synth.mat');
% load('result3\select_data_idx_real.mat');

rng("default");

ori_synth_idx = 1:size(synth_channel,1);
ori_real_idx = 1:size(real_channel,1);

[~, idx_synth] = ismember(synth_pos, real_pos, 'rows');

% Find non-zero indices (matching rows)
matching_sample_synth = find(idx_synth > 0);
matching_sample_real = idx_synth(matching_sample_synth);

synth_channel_ = synth_channel(matching_sample_synth,:,:,:);
real_channel_ = real_channel(matching_sample_real,:,:,:);

synth_pos_ = synth_pos(matching_sample_synth,:);
real_pos_ = real_pos(matching_sample_real,:);

ori_synth_idx_ = ori_synth_idx(matching_sample_synth);
ori_real_idx_ = ori_real_idx(matching_sample_real);


synth_real_channel_correlation = zeros(size(real_channel_,1),1);
for i=1:size(real_channel_,1)
    synth_channel_flat = reshape(synth_channel_(i,:,:,:), 1, []);
    real_channel_flat = reshape(real_channel_(i,:,:,:), 1, []);


    synth_real_channel_correlation(i) = abs(synth_channel_flat*real_channel_flat') /norm(synth_channel_flat)/norm(real_channel_flat);
end
figure;
cdfplot(synth_real_channel_correlation);
xlim([0, 1]);
title('');
xlabel('Normalized channel correlation');
ylabel('CDF');
grid on;

[synth_real_channel_correlation_sort, sort_idx] = sort(synth_real_channel_correlation, 'ascend');
sort_idx = int64(sort_idx);
%%
num_low_correlation_samples = 4000;
select_synth_idx = ori_synth_idx_(sort_idx(1:num_low_correlation_samples));
select_real_idx = ori_real_idx_(sort_idx(1:num_low_correlation_samples));

real_idx_correlation_sort = ori_real_idx_(sort_idx)';
correlation_data = [real_idx_correlation_sort-1, synth_real_channel_correlation_sort]; % the idx starts from 0 to match python
% csvwrite('DeepMIMO\DeepMIMO_datasets\real_correlatin_idx_sort.csv', correlation_data);

figure
scatter(synth_pos(:,1), synth_pos(:,2));
hold on;
scatter(synth_pos(select_synth_idx,1), synth_pos(select_synth_idx,2));
grid on;
legend('All synth data points', 'Low correlation synth data points');
xlabel('x-coordinate (meter)');
ylabel('y-coordinate (meter)');

figure
scatter(real_pos(:,1), real_pos(:,2));
hold on;
scatter(real_pos(select_real_idx,1), real_pos(select_real_idx,2));
grid on;
legend('All real data points', 'Low correlation real data points');
xlabel('x-coordinate (meter)');
ylabel('y-coordinate (meter)');


