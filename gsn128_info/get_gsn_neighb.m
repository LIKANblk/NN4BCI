%cfg_neighb.method   = 'triangulation';
%sfp_path = '/home/jogick/brainstorm/fieldtrip/template/electrode/GSN-HydroCel-128.sfp';
%cfg_neighb.layout   = sfp_path; 
%neighbours_gsn128          = ft_prepare_neighbours(cfg_neighb);
%neighbours = neighbours_gsn128(4:end); %because first three channels purely auxiliar
%save('gsn128_neighb.mat','neighbours')


cfg_neighb.method   = 'triangulation';
bs_topography = load('bs_topography.mat');
bs_topography = bs_topography.topography_3D;
data.label = arrayfun(@(x) x.Name,bs_topography.Channel,'UniformOutput',false);
data.chanpos = zeros(length(data.label),3);
for ch_ind=1:length(data.label)
   data.chanpos(ch_ind,:) = bs_topography.Channel(ch_ind).Loc;
end
neighbours_gsn128          = ft_prepare_neighbours(cfg_neighb,data);
for ch_ind=1:length(data.label)
    neighbours_gsn128(ch_ind).neighblabel = neighbours_gsn128(ch_ind).neighblabel';
end

neighbours = neighbours_gsn128;
save('gsn128_neighb.mat','neighbours')

%Saving 2D topography projection in .lay format for MNE processing
topography3D_txt_file = 'GSN-128.lay';
fileID = fopen(topography3D_txt_file,'w');
[y,x] = bst_project_2d(data.chanpos(:,1), data.chanpos(:,2), data.chanpos(:,3), '2dcap'); %Weird thing with wapping x and y. I really don't know why it happends.
for ch_ind=1:length(data.label)
   fprintf(fileID,'%d %f %f %f %f %s\n',ch_ind,x(ch_ind),y(ch_ind),0,0,data.label{ch_ind}); %Two zeros - sensors width and heigth - unknown. For .lay format compatibility
end
fclose(fileID);
