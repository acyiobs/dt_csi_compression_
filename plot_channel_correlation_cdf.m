clear;
clc;
rng("default");

synth_test_nmse = matfile('result_new_data_1\select_data_synth.mat').test_nmse_synth;
synth_test_idx = matfile('result_new_data_1\select_data_synth.mat').select_data_idx_synth;

real_test_nmse = matfile('result_new_data_1\select_data_real.mat').test_nmse_real;
real_test_idx = matfile('result_new_data_1\select_data_real.mat').select_data_idx_real;


synth_channel = matfile('DeepMIMO\DeepMIMO_datasets\Boston5G_3p5_notree\channel_ad_clip.mat').all_channel_ad_clip;
real_channel = matfile('DeepMIMO\DeepMIMO_datasets\Boston5G_3p5_real\channel_ad_clip.mat').all_channel_ad_clip;
synth_pos = matfile('DeepMIMO\DeepMIMO_datasets\Boston5G_3p5_notree\dataset.mat').all_pos;
real_pos = matfile('DeepMIMO\DeepMIMO_datasets\Boston5G_3p5_real\dataset.mat').all_pos;


num_data = 5120;
% figure;
% cdfplot(10*log10(real_test_nmse(1:num_data)));
% hold on;
% cdfplot(10*log10(synth_test_nmse(1:num_data)));
% 
% title('');
% xlabel('Test NMSE');
% ylabel('CDF');
% legend('Test on real', 'Test on synth')
% grid on;

% normalize the synth channel
synth_channel_flat = reshape(synth_channel, size(synth_channel,1), []);
for i=1:size(synth_channel_flat, 1)
    channel_norm = norm(synth_channel_flat(i, :));
    synth_channel_flat(i,:) = synth_channel_flat(i,:) / channel_norm;
end
% normalize the real channel
real_channel_flat = reshape(real_channel, size(real_channel,1), []);
for i=1:size(real_channel_flat, 1)
    channel_norm = norm(real_channel_flat(i, :));
    real_channel_flat(i,:) = real_channel_flat(i,:) / channel_norm;
end

%% real-synth channel correlation: selecetd real data with high NMSE
real_synth_channel_correlation = abs(real_channel_flat * synth_channel_flat');
real_synth_channel_correlation = max(real_synth_channel_correlation, [], 2);

select_idx = real_test_idx(1:num_data);
correlation_select = real_synth_channel_correlation(select_idx);

random_idx = randperm(size(real_test_idx, 2), num_data);
random_idx = real_test_idx(random_idx);
correlation_random = real_synth_channel_correlation(random_idx);

real_synth_channel_correlation_sort = sort(real_synth_channel_correlation, 'ascend');

figure;
cdfplot(correlation_select(1:100));
hold on;
cdfplot(correlation_select(1:1000));
hold on;
cdfplot(real_synth_channel_correlation)


xlim([0.3, 1]);
legend('Top-100 real data points with high test NMSE', 'Top-1000 real data points with high test NMSE', 'All 5120 real training data points');
xlabel('Normalized real-synth channel correlation');
ylabel('Empirical CDF');
title('');

figure;
tmp = [];
for i=1:100
    tmp = [tmp; real_synth_channel_correlation(randperm(size(real_synth_channel_correlation, 1), 100))];
end
cdfplot(tmp);
hold on;
cdfplot(correlation_select(1:100));
grid on;
xlim([0.3, 1]);
legend('Randomly selected 100 target data points (Averaged)', 'Top-100 target data points with high test NMSE');
xlabel('Normalized real-synth channel correlation');
ylabel('Empirical CDF');
title('');

figure;
for i=1:1000
    h1 = cdfplot(real_synth_channel_correlation(randperm(size(real_synth_channel_correlation, 1), 100)));
    set(h1, 'LineStyle', '--', 'Color', '#A2142F', 'LineWidth', 0.5);
    hold on;
end
h2 = cdfplot(real_synth_channel_correlation);
set(h2, 'LineStyle', '-', 'Color','#0072BD', 'LineWidth', 2);

h3 = cdfplot(correlation_select(1:100));
set(h3, 'LineStyle', '-', 'Color','#D95319', 'LineWidth', 2);

h4 = cdfplot(correlation_select(1:1000));
set(h4, 'LineStyle', '-', 'Color','#7E2F8E', 'LineWidth', 2);

grid on;
xlim([0.3, 1]);
handlevec = [h1 h2, h3, h4];
legend(handlevec,'Random 100 real training data points', 'All 5120 real training data points', 'Top-100 real data points with high test NMSE', 'Top-1000 real data points with high test NMSE');

figure;
histogram(real_synth_channel_correlation, 'Normalization', 'pdf', 'BinEdges', 0.3:0.1:1)
hold on
histogram(correlation_select(1:50), 'Normalization', 'pdf', 'BinEdges', 0.3:0.1:1)






