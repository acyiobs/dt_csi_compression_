clear;
clc;

synth_channel = matfile('DeepMIMO\DeepMIMO_datasets\Boston5G_3p5_notree\channel_ad_clip.mat').all_channel_ad_clip;
real_channel = matfile('DeepMIMO\DeepMIMO_datasets\Boston5G_3p5_real\channel_ad_clip.mat').all_channel_ad_clip;
synth_pos = matfile('DeepMIMO\DeepMIMO_datasets\Boston5G_3p5_notree\dataset.mat').all_pos;
real_pos = matfile('DeepMIMO\DeepMIMO_datasets\Boston5G_3p5_real\dataset.mat').all_pos;
% load('result3\select_data_idx_synth.mat');
% load('result3\select_data_idx_real.mat');

rng("default");

% tmp_idx1 = find(synth_pos(:,1)>200);
% tmp_idx2 = find(synth_pos(:,2)<100);
% tmp_idx = intersect(tmp_idx1, tmp_idx2);
% synth_pos = synth_pos(tmp_idx,:);
% synth_channel = synth_channel(tmp_idx,:);
% 
% tmp_idx1 = find(real_pos(:,1)>200);
% tmp_idx2 = find(real_pos(:,2)<100);
% tmp_idx = intersect(tmp_idx1, tmp_idx2);
% real_pos = real_pos(tmp_idx,:);
% real_channel = real_channel(tmp_idx,:);

figure;
scatter(synth_pos(:,1), synth_pos(:,2));
figure
scatter(real_pos(:,1), real_pos(:,2));

[~, idx_synth] = ismember(synth_pos, real_pos, 'rows');

% Find non-zero indices (matching rows)
matching_sample_synth = find(idx_synth > 0);
matching_sample_real = idx_synth(matching_sample_synth);

synth_channel_ = synth_channel(matching_sample_synth,:,:,:);
real_channel_ = real_channel(matching_sample_real,:,:,:);

tmp=(synth_pos(matching_sample_synth,:) - real_pos(matching_sample_real,:));
figure
scatter(synth_pos(matching_sample_synth,1), synth_pos(matching_sample_synth,2));
figure
scatter(real_pos(matching_sample_real,1), real_pos(matching_sample_real,2));

synth_channel_ = reshape(synth_channel_, size(synth_channel_,1),[]);
real_channel_ = reshape(real_channel_, size(real_channel_,1),[]);

plot_sample_idx = randperm(numel(matching_sample_synth), 50);

channel_concat = [synth_channel_(plot_sample_idx, :); real_channel_(plot_sample_idx, :)];

channel_concat_normed = zeros(size(channel_concat));
for i=1:size(channel_concat, 1)
    channel_concat_normed(i, :) = channel_concat(i, :) / norm(channel_concat(i, :));
end

channel_correlation = abs(channel_concat_normed * channel_concat_normed');
figure
imagesc(channel_correlation);
xlabel('Datat sample index');
ylabel('Datat sample index');
% colormap gray;

synth_real_channel_correlation = zeros(size(real_channel_,1),1);
for i=1:size(real_channel_,1)
    synth_real_channel_correlation(i) = abs(synth_channel_(i,:)*real_channel_(i,:)') /norm(synth_channel_(i,:))/norm(real_channel_(i,:));
end

