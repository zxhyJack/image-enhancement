clc;
clear all;
close all;

% Add subfolder with images.
path = pwd;
index_dir = strfind(pwd,'\');
addpath(strcat(path(1:index_dir(end)-1),'\images'), '-end');
addpath(strcat(pwd,'\sources'), '-end');

%Reading image
Img   = imread('pout.tif');
% Img   = imresize(Img,[256 256]); 
% I     = rgb2gray(I);
I = im2gray(Img);
[m, n] = size(Img);

swarmSize   = 30;    %swarm size
maxIter    = 20;    %number of iterations
localSize = 3;
dim = 4;
wmax = 0.9; % Maximal inertia weight.
wmin = 0.4; % Minimal inertia weight.
c1 = 2.4; % Cognitive acceleration coefficient.
c2 = 1.7; % Social acceleration coefficient.
r1      = rand;
r2      = rand;
P_best  = [];       %matrix for storing pbest values
G_best  = [];       %matrix for storing gbest values
pbest   = [];

figure;
imshow(I);
title('Grayscaled image', 'fontsize', 10);

for i = 1:swarmSize
    x(i,1) = (1.5).*rand(1,1); % [0, 1.5]
    x(i,2) = (0.5).*rand(1,1); % [0, 0.5]
    x(i,3) = rand(1,1); % [0, 1]
    x(i,4) = 0.5 + 1.*rand(1,1); % [0.5, 1.5]
    fitnesses(i) = fitnessFunction(...
        enhanceGsclImage(I, localSize, x(i,1), x(i,2), x(i,3), x(i,4)), m ,n);
end

[value, argmax] = max(fitnesses);
gbest   = x(argmax,:);
v = rand(swarmSize, dim); % Updating particle velocity.
P_best = x;

count = 1;
for iter = 1:maxIter
    w = wmax -(wmax - wmin) * iter / maxIter;
    for i = 1:swarmSize
        img_out = enhanceGsclImage(I, localSize, x(i,1), x(i,2), x(i,3), x(i,4));
        imwrite(img_out, ['../result/pout/figure',num2str(count),'.png']);
        count = count+1;
        % Calculating fitness value.
        fitness = fitnessFunction(img_out, m, n);
        if (fitness > fitnesses(i))
            pbest = x(i,:);
            fitnesses(i) = fitness;
            P_best(i,:) = pbest;
        end
        if (fitness > max(fitnesses))
            gbest = x(i,:);
        end
    end
    
    G_best = [G_best, gbest];
    
    for i = 1:swarmSize
        v(i,:) = w.*v(i,:) + c1.*r1.*(P_best(i,:) - x(i,:)) + c2.*r2.*(gbest - x(i,:));
        x(i,:) = x(i,:) + v(i,:); % Updating particle position.
    end
end

% Comparing images
% fprintf('Sharpness of Original image: %5.3f \n', ...
%         mean(getImageSharpness(I)));
% fprintf('Brightness of Original image: %5.3f \n', mean2(I));
% fprintf('Contrast of Original image: %5.3f \n\n', max(I(:)) - min(I(:)));
% 
% fprintf('Sharpness of Enhanced image: %5.3f \n', ...
%         mean(getImageSharpness(img_out)));
% fprintf('Brightness of Enhanced image: %5.3f \n', mean2(img_out));
% fprintf('Contrast of Enhanced image: %5d \n\n', max(img_out(:)) - min(img_out(:)));
% 
figure;
imshow(img_out);
title('Enhanced image', 'fontsize', 10);


