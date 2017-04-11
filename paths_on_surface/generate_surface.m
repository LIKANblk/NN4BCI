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
    F((end+1):(end+N-1),:) = [bases; bases+N+1; bases+N]';
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

% reorder edges to make cycle or chain (such that ed(i,2)==ed(i+1,1))
function ord_ed = order_edges(ed)
  % check is there a cycle or not (if there are any vertices 
  % in column 1 but not 2 then it is not a cycle)
  is_cycle = 1;
  non_pair = -1;
  for i=1:size(ed)(1)
      if size(ed(ed(:,2)==ed(i,1),1))(1)==0
	 is_cycle = 0;
	 non_pair = ed(i,1);
      endif
  endfor
  start_v = -1;
  % if ed has cycle, start from the minimal ed; else --- from start of chain
  if is_cycle
     start_v = min(ed(:,1));
  else
    start_v = non_pair;
  endif
  % add points from start to results
  ord_ed = zeros(0,2);
  curr_v = start_v;
  has_next = 1;
  while has_next
	new_points = ed(ed(:,1)==curr_v,:);
	num_p = size(new_points)(1);
	if(num_p>1)
	  error("non-manifold vertice found!") % TODO: add description
	elseif(num_p==1)
	  ord_ed(end+1,:) = new_points;
	  ed(ed(:,1)==curr_v,:) = [];% remove row from eds
	  curr_v = new_points(2);
	else
	  has_next = 0;
	endif
  endwhile
  if size(ed)(1)>0
     error("non-manifold vertice found!");
  endif
endfunction

assert(order_edges([3,4; 1,3; 4,5])==[1,3;3,4;4,5]);
assert(order_edges([3,4; 1,3; 4,1])==[1,3;3,4;4,1]);

% get first set of triangles
function variants = initialize_variants(v,f, init_v, target_v)
  % get all triangles where init_v is first, second or third vertex,
  % and separate one vertex
  edges1 = f(f(:,1)==init_v,2:3);
  edges2 = f(f(:,2)==init_v,:);
  edges3 = f(f(:,3)==init_v,1:2);
  edges = [edges1; edges2(:,3) edges2(:,1); edges3];
  assert(size(edges)(1)>0); % can not process separate vertex
  % sort edges to make chain or cycle (if possible)
  sorted_ed = order_edges(edges);
  % set angles and possible positions starting from 
  % the first point in sorted list
  curr_angle = 0;
  % variants: v1 v2 v1_flat_(x,y) v2_flat_(x,y) angle1 angle2 dist
  variants = zeros(0,9);
  for c_e = 1:size(sorted_ed)(1)
      first_v = sorted_ed(c_e,:)(1);
      second_v = sorted_ed(c_e,:)(2);
      % calculate flat coordinates for two triangle vertices
      len_l1 = sum((v(first_v,:)-v(init_v,:)).^2)^0.5;
      len_l2 = sum((v(second_v,:)-v(init_v,:)).^2)^0.5;
      % calculate angle between 0-v1 and 0-v2
      angle_12 = acos(sum((v(first_v,:)-v(init_v,:)) .* (v(second_v,:)-v(init_v,:)))/len_l1/len_l2);
      % calculate flat coordinates of each vertex processed
      v1_flat = [len_l1*cos(curr_angle), len_l1*sin(curr_angle)];
      v2_flat = [len_l2*cos(curr_angle+angle_12), len_l2*sin(curr_angle+angle_12)]; 
      % calculate measure of distance from v1-v2 to target
      e12_med = (v(first_v,:)+v(second_v,:))/2;
      dist = sum((e12_med-v(target_v,:)).^2).^0.5;
      variants(end+1,:) = [first_v second_v v1_flat v2_flat curr_angle (curr_angle+angle_12) dist];
      curr_angle += angle_12;
  endfor
endfunction

%initialize_variants(test_v,test_f, 6, 15)

% one step in depth for dfs
% tr_variants: each row is [v1 v2 v1_flat_(x,y) v2_flat_(x,y) angle1 angle2 est_dis]
% stack: last expanded edges
% init_vertex,target_vertex: positions
function [Found, tr_variants, stack, init_vertex, target_vertex] = dfs(v, f, tr_variants, stack, init_vertex, target_vertex)
  Found = 0;
  % if there are no any more variants, return 0
  if size(tr_variants)(1)==0
     return
  endif
  % find the best of all current variants
  min_dist = min(tr_variants(:,9))
  possible_tr = tr_variants(tr_variants(:,9)==min_dist,:);
  best_v1 = possible_tr(1,1);
  best_v2 = possible_tr(1,2);
  % find triangle with this pair of vertices in reverse order
  tr1 = f(f(:,1)==best_v2 & f(:,2)==best_v1,:);
  tr2 = f(f(:,2)==best_v2 & f(:,3)==best_v1,:);
  tr3 = f(f(:,3)==best_v2 & f(:,1)==best_v1,:);
  tr = [tr1; tr2; tr3];
  
endfunction

init_v = 6;
finish_v = 16;
start_cond = initialize_variants(test_v,test_f, init_v, finish_v);
[Found, tr_variants, stack, init_vertex, target_vertex] = dfs(test_v,test_f, start_cond, zeros(0,2), init_v, finish_v);
