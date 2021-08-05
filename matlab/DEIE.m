clc;
clear all;
close all;

addpath(strcat(pwd,'\images'), '-end');
addpath(strcat(pwd,'\sources'), '-end');

% paramters
swarmSize = 30;
maxIter = 20;
localSize = 3;
dim = 4;
Fmax = 1.9;
Fmin = 0.4;
CR = 0.5;

img = imread('rice.png');
% img = im2gray(img);
[m, n] = size(img);

figure;
imshow(img);
title('Grayscaled image', 'fontsize', 10);

% initial population
for i = 1:swarmSize
    x(i,1) = 1.5*rand; % [0,1.5]
    x(i,2) = 0.5*rand; % [0,0.5]
    x(i,3) = rand; % [0,1]
    x(i,4) = 0.5 + rand; % [0.5,1.5]
    fitnesses(i) = fitnessFunction(...
              enhanceGsclImage(img, localSize, x(i,1),x(i,2),x(i,3),x(i,4)), m, n);
end

[value,argmax] = max(fitnesses);
best = x(argmax,:);

count = 1;
for iter = 1:maxIter
    F = (Fmax-Fmin) * (maxIter-iter) / maxIter;
    for i = 1:swarmSize
        img_out = enhanceGsclImage(img, localSize, x(i,1), x(i,2), x(i,3), x(i,4));
        imwrite(img_out,['./rice/figure', num2str(count), '.png']);
        count = count + 1;
        fitness = fitnessFunction(img_out, m,n);
        if(fitness > fitnesses(i))
            best = x(i,:);
        end

        % mutation
        % r1 r2 r3 random and different
        r1 = randi(swarmSize);
        while r1 == i
            r1 = randi(swarmSize);
        end 
        r2 = randi(swarmSize);
        while r2 == i || r2 == r1
            r2 = randi(swarmSize);
        end
        r3 = randi(swarmSize);
        while r3 == i || r3 == r1 || r3 == r2
            r3 = randi(swarmSize);
        end

        for j = 1:dim
            v(i,j) = x(r1,j) + F * (x(r2,j) - x(r3,j));
        end

        % crossover
        for j = 1:dim
            if (rand <= CR || j == randi(dim)) 
                u(i,j) = v(i,j); 
            else 
                u(i,j) = x(i,j); 
            end
        end

        % select
        uFitness = fitnessFunction(...
            enhanceGsclImage(img, localSize, u(i,1), u(i,2), u(i,3), u(i,4)),m, n);
        if uFitness > fitnesses(i)
            x(i,:) = u(i,:);
        end
    end
    bests(i,:) = best;
end

% Comparing images
fprintf('Sharpness of Original image: %5.3f \n', ...
        mean(getImageSharpness(img)));
fprintf('Brightness of Original image: %5.3f \n', mean2(img));
fprintf('Contrast of Original image: %5.3f \n\n', max(img(:)) - min(img(:)));

fprintf('Sharpness of Enhanced image: %5.3f \n', ...
        mean(getImageSharpness(img_out)));
fprintf('Brightness of Enhanced image: %5.3f \n', mean2(img_out));
fprintf('Contrast of Enhanced image: %5d \n\n', max(img_out(:)) - min(img_out(:)));

figure;
imshow(img_out);
title('Enhanced image', 'fontsize', 10);