num_user = size(DeepMIMO_dataset{1, 1}.user,2);
num_paths = zeros(1,num_user);
for u =1:num_user
num_paths(u) = DeepMIMO_dataset{1, 1}.user{1,u}.path_params.num_paths;
end

% Compute bin edges
max_path=25;
binCenters = 1:max_path;
halfBinWidth = 0.5;
binEdges = [binCenters - halfBinWidth, binCenters(end) + halfBinWidth];
figure;
histogram(num_paths(num_paths>0), binEdges);
grid on;
xlabel('Number of paths');
ylabel('Number of data points');