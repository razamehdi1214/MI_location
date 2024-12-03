clear
% close
clc

elems = importdata('elems.txt');
elems = elems(:,2:end);
nodes = importdata('nodes.txt');
nodes = nodes(:,2:end);

% calculating centroids
A = [];
for i=1:length(elems)
ctd1 = elems(i,:);
ctd2x = sum(nodes(ctd1,1))/10;
ctd2y = sum(nodes(ctd1,2))/10;
ctd2z = sum(nodes(ctd1,3))/10;
A = [A;ctd2x ctd2y ctd2z];
end

%% Obtain elements of each slice from the whole slices file
for j=0:5
close all
fileName = sprintf('segmentElemSet_%d.txt', j);
lines = textread(fileName, '%s', 'delimiter', '\n', 'whitespace', '');

% find the lines that define slices 1-5
slice1_lines = find(contains(lines, '*ELSET, ELSET=slice1'));
slice2_lines = find(contains(lines, '*ELSET, ELSET=slice2'));
slice3_lines = find(contains(lines, '*ELSET, ELSET=slice3'));
slice4_lines = find(contains(lines, '*ELSET, ELSET=slice4'));
slice5_lines = find(contains(lines, '*ELSET, ELSET=slice5'));

slice{1} = slice_elem(lines,slice1_lines+1,slice2_lines-1);
slice{2} = slice_elem(lines,slice2_lines+1,slice3_lines-1);
slice{3} = slice_elem(lines,slice3_lines+1,slice4_lines-1);
slice{4} = slice_elem(lines,slice4_lines+1,slice5_lines-1);
slice{5} = slice_elem(lines,slice5_lines+1,numel(lines)-1);

%% load strains obtained from dat file at each element
strainName = sprintf('ES_CRLFNS_%d.xlsx', j);
Es_strains = importdata(strainName);

Es_strains = Es_strains.data;
strains_slice{1} = Es_strains.Slice1;
strains_slice{2} = Es_strains.Slice2;
strains_slice{3} = Es_strains.Slice3;
strains_slice{4} = Es_strains.Slice4;
strains_slice{5} = Es_strains.Slice5;
%% Interpolate strains from elements to obtain at nodes
for i = 1:4   
    slice_node_def_C = strain_itp(A, slice{i}, strains_slice{i}, nodes, elems, 1);
    slice_node_def_R = strain_itp(A, slice{i}, strains_slice{i}, nodes, elems, 2);
    slice_node_def_L = strain_itp(A, slice{i}, strains_slice{i}, nodes, elems, 3);
    
    CRL_strains{i} = [slice_node_def_C, slice_node_def_R, slice_node_def_L];
end


%% Plotting strains for inputs of ML
for i=1:4
x_i = []; y_i = []; u_i = [];
node_file{i} = [nodes(elems(slice{i},:),1),nodes(elems(slice{i},:),3)]; 
if i==1
    max_r = 1.0; min_r = 0.75;
elseif i==2
    max_r = 0.75; min_r = 0.5;
elseif i==3
    max_r = 0.5; min_r = 0.25;
else
    max_r = 0.25; min_r = 0.0;
end
[x_i, y_i, u_i] = polar_interpolation(node_file{i}, CRL_strains{i}(:,1), max_r, min_r);  % change CRL_strains{i}(:,1) column number for different strains
plot_strain(x_i,y_i,u_i)
hold on
end
set(gca, 'Units', 'normalized', 'Position', [0, 0, 1, 1]); % Set axis position to fill the figure
fileName = sprintf('circum_%d.png', j);
saveas(gcf, fileName, 'png');

end

%% Functions
function slice = slice_elem(lines,slice1_lines,slice2_lines)
% extract the elements of slice1
slice1_str = lines(slice1_lines:slice2_lines);
slice = [];
for i = 1:numel(slice1_str)
    slice_row = str2double(strsplit(slice1_str{i}, ','));
    slice = horzcat(slice, slice_row);
end
slice(isnan(slice)) = [];
slice = slice';
slice = sort(slice);    % in dat file, strains are given in ascending order of elements
end

function node_strains = strain_itp(A,slice,strain_slice, nodes, elem, num)

original_nodes = [A(slice,1),A(slice,3)]; % Original node positions
original_deformations = strain_slice(:, num); % Original node deformations
new_nodes = [nodes(elem(slice,:),1),nodes(elem(slice,:),3)]; % Original node positions

new_deformations = scatteredInterpolant(original_nodes, original_deformations,'linear','nearest');
new_deformations.Values = new_deformations(new_nodes);

node_strains = new_deformations.Values;
end

function [x_i, y_i, u_i] = polar_interpolation(nodes_file, val_file, max_r, min_r)
%POLAR_INTERPOLATION performs polar interpolation of a set of points.

    nodes = nodes_file;
    val_array = val_file;
    
    x_array = nodes(:,1)*100;
    y_array = nodes(:,2)*100;

    xc = mean(x_array);
    yc = mean(y_array);

    polar_array = zeros(length(x_array), 2);
    polar_array(:, 1) = (((x_array - xc).^2 + (y_array - yc).^2).^0.5) * max_r;
    polar_array(:, 2) = deg2rad(360) + atan2(y_array - yc, x_array - xc);
    polar_array(:, 1) = round(polar_array(:, 1), 4);
    polar_array(:, 2) = round(polar_array(:, 2), 4);

    t_array_one = polar_array((find(polar_array(:, 2) < deg2rad(380))), 2);
    t_array_two = polar_array((find(polar_array(:, 2) > deg2rad(380))), 2);

    theta = deg2rad(180);
    r = [];
    t = [];
    u = [];
    r_array = [0 0 0];
    t_list = [];

    while theta <= deg2rad(540)
        t_list = [t_list theta];
        for ind = 1:length(polar_array(:,2))
            if abs((theta - polar_array(ind,2))/theta) < 0.01
                r = [r polar_array(ind,1)]; % Radius of each point in real plane
                t = [t polar_array(ind,2)]; % Theta of each point in real plane
                u = [u val_array(ind)]; % Strain value for each point
            end
        end

        for ind = 1:length(r)
            if r(ind) == min(r)
                r_array = [r_array; [min_r t(ind) - 0 u(ind)]];
            elseif r(ind) == max(r)
                r_array = [r_array; [max_r t(ind) - 0 u(ind)]];
            else
                r_array = [r_array; [min_r + (((max_r - min_r)/(max(r) - min(r)) * (r(ind) - min(r)))) t(ind) - 0 u(ind)]];
            end
        end

        r = [];
        t = [];
        u = [];
        theta = theta + pi/180;
    end

    r_array = unique(r_array, 'rows'); % Collects all unique points in (r, theta)

    [r_i, t_i] = deal(r_array(2:end,1), r_array(2:end,2)); % Points in ideal plane (polar co-ordinates)
    u_i = r_array(2:end,3); % Strain values

    [x_i, y_i] = deal([], []); % Points in ideal plane (cartesian co-ordinates)

    x_i = r_i .* cos(t_i);
    y_i = r_i .* sin(t_i);

end

function plot_strain(x_i,y_i,u_i)
x = x_i;
y = y_i;
e = u_i;

r_array = sqrt(x.^2 + y.^2);
max_r = max(r_array);
min_r = min(r_array);

xi = linspace(min(x),max(x),1600);
yi = linspace(min(y),max(y),1600);

[Xi, Yi] = meshgrid(xi, yi);
interpolator = scatteredInterpolant(x, y, e);
ei = interpolator(Xi, Yi);

r2 = max_r; r1 = min_r; 
L1 = (abs(Xi.^2 + Yi.^2) < r1^2);  
L2 = (abs(Xi.^2 + Yi.^2) > r2^2);

L = L1 | L2;

Xi(L) = nan;  Yi(L) = nan; ei(L) = nan;


figure(1)
ax = pcolor(Xi, Yi, ei);
shading interp
colormap parula
% colormap gray
hold on

colorbar
axis off
% title('Longitudinal strains')
colorbar off

set(gca,'xdir','reverse')
end
