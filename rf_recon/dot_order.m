ntrials = 60;
n_dots = 40;

dots_order = [];

rng(1234)
for i = 1:ntrials
    random_list = randperm(numel(1:n_dots));
    dots_order = [dots_order random_list];
end