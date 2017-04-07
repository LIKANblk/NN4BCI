% generate source data for geodetic curves calculation

% test function: generate test surface description 
% (the real function will generate description in the same format)
% N --- number of vertices in square grid
% return values: vertices coordinates and faces descriptions (triples of vertices)
function [V,F] = load_surface_test_flat(N=4)
  src_grid = repmat([1:N],N,1);
  V = [reshape(src_grid,N^2,1) reshape(src_grid',N^2,1) zeros(N^2, 1)];
  F = zeros(0,3);
  for i=1:(N-1)
    bases = i:N:N^2-N;
    F((end+1):(end+N-1),:) = [bases; bases+1; bases+1+N]';
    F((end+1):(end+N-1),:) = [bases; bases+N; bases+1+N]';
  endfor
endfunction

[test_v,test_f] = load_surface_test_flat(4);
assert(size(test_v)==[16,3]);
assert(size(test_f)==[18,3]);

% plot surface in 3D
function plot_surface(V,F)
  for i=1:size(F)(1)
    draw_verts = V([F(i,:) F(i,1)],:);
    plot3(draw_verts(:,1), draw_verts(:,2), draw_verts(:,3));
    hold on;
  endfor
endfunction
