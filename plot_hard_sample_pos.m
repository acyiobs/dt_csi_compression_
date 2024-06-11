synth_pos = load('DeepMIMO\DeepMIMO_datasets\Boston5G_3p5_small_notree\all_pos.mat').all_pos;
real_pos = load('DeepMIMO\DeepMIMO_datasets\Boston5G_3p5_1\all_pos.mat').all_pos;
load('result3\select_data_idx_synth.mat');
load('result3\select_data_idx_real.mat');

synth_hard_pos = synth_pos(select_data_idx_synth+1,:);
real_hard_pos = real_pos(select_data_idx_real+1,:);

figure;
scatter(synth_hard_pos(:,1), synth_hard_pos(:,2));
hold on;
scatter(real_hard_pos(:,1), real_hard_pos(:,2));
grid on;
legend('Syntn hard data points', 'Real hard data points');
xlabel('x-coordinate (meter)');
ylabel('y-coordinate (meter)');

figure;
scatter(synth_pos(:,1), synth_pos(:,2));
grid on;
legend('Synth data points');
xlabel('x-coordinate (meter)');
ylabel('y-coordinate (meter)');

figure;
scatter(real_pos(:,1), real_pos(:,2));
grid on;
legend('Real data points');
xlabel('x-coordinate (meter)');
ylabel('y-coordinate (meter)');
