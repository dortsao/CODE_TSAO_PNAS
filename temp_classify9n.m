%load FI_full_200.mat  %%two frames of alternating seg map   %%%FINAL data set with 25 frames, 200 points per condition
clear all
close all

%load FI_series_1.mat
%load FI_fig2_1.mat
%load new_large_rotate_random_synthetic_leaf_3colors_6/FI_fig2_1.mat

NumFrames = 5;  %%to 12

thresh = 10e3;
%thresh = 3e4;

load C:\Users\Doris\Dropbox\PAPERS_WORKING\space\code\codenew_103020\synthetic_bear\nbhds_HPC\nbhds\FrameInfo.mat;


nbhd_path = 'C:\\Users\\Doris\\Dropbox\\PAPERS_WORKING\\space\\code\\codenew_103020\\synthetic_bear\\nbhds_HPC\nbhds';

%%%compute Gaussian
Sigma = 30;
a = 201;  %%size of N1a
cx = (a-1)/2;
cy= cx;
for x = 1:a
    for y = 1:a
        G(x,y) = mygaussian_point(Sigma, cx, cy, x, y);
    end
end

for frame = 1:4
    %for frame = 11:12
    
    leafdir = 'synthetic_bear';
    fname = sprintf('C:\\Users\\Doris\\Dropbox\\PAPERS_WORKING\\space\\code\\codenew_103020\\%s\\leaf_%d.png', leafdir, frame);
    LM1 = imread(fname);  %%superseg map
    %     %ax1 = subplot(NumFrames-1,4,(frame-1)*4+1);
    %     %ax1 = subplot(1,4,1);
    %     imshow(LM1)
    %     colormap(ax1, gray(256));
    
    
    fname = sprintf('C:\\Users\\Doris\\Dropbox\\PAPERS_WORKING\\space\\code\\codenew_103020\\%s\\superseg_%d.png', leafdir, frame);
    SSM1 = imread(fname);  %%superseg map
    
    %     %ax2 = subplot(NumFrames-1,4,(frame-1)*4+2);
    %     %ax2 = subplot(1,4,2);
    %     imshow(SSM1)
    %     colormap(ax2, jet)
    twosided_edgeppoints = FrameInfo{frame}.tse;
    
%     cmd = sprintf('load C:\\Users\\Doris\\Dropbox\\PAPERS_WORKING\\space\\code\\codenew_103020\\%s\\FI_%d.mat', leafdir, frame);
%     eval([cmd]);
    %load synthetic_leaves\FI_1.mat
    
    labels_vec = double(unique(SSM1));
    
    
    for i = 1:length(twosided_edgeppoints)  %%%run through two sided neighborhoods
        bi_labels(i,:) = [twosided_edgeppoints{i}.label1 twosided_edgeppoints{i}.label2];
    end
    w = unique(bi_labels, 'rows');
    
    %table_bi_diffeo = cell(length(w));
    for i = 1:length(w)   %%%run through two sided neighborhoods
        table_bi_diffeo{i}.NumEls = 0;
        table_bi_diffeo{i}.twosided = 0;
        table_bi_diffeo{i}.twosided_count = 0;
        table_bi_diffeo{i}.onesided_count_side1 = 0;
        table_bi_diffeo{i}.onesided_count_side2 = 0;
        table_bi_diffeo{i}.onesided_side1_Lnext = -1;
        table_bi_diffeo{i}.onesided_side2_Lnext = -1;
        table_bi_diffeo{i}.Lnextlist1 = [];
        table_bi_diffeo{i}.Lnextlist2 = [];
    end
    
    for i = 1:length(twosided_edgeppoints)   %%%run through two sided neighborhoods and assemble table
        k = find((w(:,1) == twosided_edgeppoints{i}.label1) & (w(:,2) == twosided_edgeppoints{i}.label2));
        
        
        
        
        E_grid1 = twosided_edgeppoints{i}.E_grid1;
        rr_grid1 =twosided_edgeppoints{i}.rr_grid1;
        
        E_grid2 = twosided_edgeppoints{i}.E_grid2;
        rr_grid2 = twosided_edgeppoints{i}.rr_grid2;
        
        
        
        corrected_rr_mat_grid1 = twosided_edgeppoints{i}.corrected_rr_mat_grid1;
        corrected_rr_mat_grid2 = twosided_edgeppoints{i}.corrected_rr_mat_grid2;
        E_warp_diff_grid1 = twosided_edgeppoints{i}.E_warp_diff_grid1 ;
        E_warp_diff_grid2 = twosided_edgeppoints{i}.E_warp_diff_grid2 ;
        
        [i1 j1]= findminE(E_grid1);
        [i2 j2]= findminE(E_grid2);
        
        E_mat1 = E_grid1{i1,j1};
        E_mat2 = E_grid2{i2,j2};
        
        rr_mat1 = rr_grid1{i1,j1};
        rr_mat2 = rr_grid2{i2,j2};
        
        corrected_rr_mat1 = corrected_rr_mat_grid1{i1,j1};
        corrected_rr_mat2 = corrected_rr_mat_grid2{i2,j2};
        E_warp_diff1 = E_warp_diff_grid1{i1,j1};
        E_warp_diff2 = E_warp_diff_grid2{i2,j2};
        
        
        ind = table_bi_diffeo{k}.NumEls;
        table_bi_diffeo{k}.list1{ind +1} = rr_mat1;
        table_bi_diffeo{k}.list2{ind +1} = rr_mat2;
        table_bi_diffeo{k}.Elist1{ind +1} = E_mat1;
        table_bi_diffeo{k}.Elist2{ind +1} = E_mat2;
        table_bi_diffeo{k}.corrected_list1{ind +1} = corrected_rr_mat1;
        table_bi_diffeo{k}.corrected_list2{ind +1} = corrected_rr_mat2;
        table_bi_diffeo{k}.E_warp_diff_list1{ind +1} =E_warp_diff1;
        table_bi_diffeo{k}.E_warp_diff_list2{ind +1} =E_warp_diff2;
        table_bi_diffeo{k}.Lnextlist1{ind +1} = twosided_edgeppoints{i}.label1_next;
        table_bi_diffeo{k}.Lnextlist2{ind +1} = twosided_edgeppoints{i}.label2_next;
        table_bi_diffeo{k}.NumEls = table_bi_diffeo{k}.NumEls + 1;

        
        %%%%load acutal and warped images so you can compute difference
        fname = sprintf('%s\\N1a_%d_%d.png', nbhd_path, frame, i);
        cmd = sprintf('N1a = imread("%s");', fname);
        eval([cmd]);
        
        fname = sprintf('%s\\N1b_%d_%d.png', nbhd_path, frame, i);
        cmd = sprintf('N1b = imread("%s");', fname);
        eval([cmd]);
        
        fname = sprintf('%s\\N2a_%d_%d.png', nbhd_path, frame, i);
        cmd = sprintf('N2a =  imread("%s");', fname);
        eval([cmd]);
        
        fname = sprintf('%s\\N2b_%d_%d.png', nbhd_path, frame, i);
        cmd = sprintf('N2b = imread("%s");', fname);
        eval([cmd]);
                               
        N1a = double(N1a);
        N1b = double(N1b);
        N2a = double(N2a);
        N2b = double(N2b);

        
        draw_path = sprintf('%s\\Im_warp_%d_%d', nbhd_path, frame, i);
        
        fname= sprintf('%s\\N1a_warp_%d_%d_%d_%d.png', draw_path, frame, i, i1,j1);
        cmd = sprintf('N1a_warp = imread("%s");', fname);
        eval([cmd]);
        
        fname= sprintf('%s\\N2a_warp_%d_%d_%d_%d.png', draw_path, frame, i, i2,j2);
        cmd = sprintf('N2a_warp = imread("%s");', fname);
        eval([cmd]);
                
        D1 = double(N1b)-double(N1a_warp);
        D2 = double(N2b)-double(N2a_warp);
                
        E_warp_diff1 = sum(sum((G.*D1).^2));     %%%perhaps this should be weighted by Gaussian receptive field to not count edge points are strongly
        E_warp_diff2 = sum(sum((G.*D2).^2));
        
        table_bi_diffeo{k}.E_warp_diff_list1{ind +1} =E_warp_diff1;
        table_bi_diffeo{k}.E_warp_diff_list2{ind +1} =E_warp_diff2;
                        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
    end
    
    
    NbhdType = cell(length(labels_vec));
    for i = 1:length(labels_vec)
        NbhdType{i}.twosided = 0;
        NbhdType{i}.onesided = 0;
    end
    
    for i = 1:length(w)   %%%run through two sided neighborhoods
        %        for i = 8:8   %%%run through two sided neighborhoods
        w(i,:)
        %pause
        for j = 1: table_bi_diffeo{i}.NumEls
            rr_mat1 = table_bi_diffeo{i}.list1{j}
            rr_mat2 = table_bi_diffeo{i}.list2{j}
            
            E_mat1 = table_bi_diffeo{i}.Elist1{j};
            E_mat2 = table_bi_diffeo{i}.Elist2{j};
            
            corrected_rr_mat1 = table_bi_diffeo{i}.corrected_list1{j};
            corrected_rr_mat2 = table_bi_diffeo{i}.corrected_list2{j};
            
            E1_warp_diff = table_bi_diffeo{i}.E_warp_diff_list1{j};
            E2_warp_diff = table_bi_diffeo{i}.E_warp_diff_list2{j};

            dist1 = norm(corrected_rr_mat1(1:4)-corrected_rr_mat2(1:4));
            dist2 = norm(corrected_rr_mat1(5:6)-corrected_rr_mat2(5:6));
            
%             v1 = rr_mat1(length(rr_mat1),:);
%             v2 = rr_mat2(length(rr_mat2),:);
%             
%             dist1 = norm(v1(1:4)-v2(1:4));
%             dist2 = norm(v1(5:6)-v2(5:6));
            
            %         epsilon1 = norm(rr_mat1(length(rr_mat1),:) - rr_mat1(length(rr_mat1)-1,:))/norm(rr_mat1(length(rr_mat1),:));
            %         epsilon2 = norm(rr_mat2(length(rr_mat2),:) - rr_mat2(length(rr_mat2)-1,:))/norm(rr_mat2(length(rr_mat2),:));
            %
            %         energy_frac1 = E_mat1(length(E_mat1)) / E_mat1(1);
            %         energy_frac2 = E_mat2(length(E_mat2)) / E_mat2(1);
            
             E1 = E_mat1(length(E_mat1))
             E2 = E_mat2(length(E_mat2))
            
            

            
            
            if ((dist1 < 0.1) & (dist2 < 3))
                %type = 1; %%%two_sided
                table_bi_diffeo{i}.twosided_count = table_bi_diffeo{i}.twosided_count + 1;
            elseif (E1_warp_diff < E2_warp_diff)
                %type = 2; %%%one sided, side 1
                table_bi_diffeo{i}.onesided_count_side1 = table_bi_diffeo{i}.onesided_count_side1 + 1;
            else
                %type = 3; %%%one sided, side 2
                table_bi_diffeo{i}.onesided_count_side2 = table_bi_diffeo{i}.onesided_count_side2 + 1;                     
            end
%             if ((E1 < thresh) & (E2 < thresh))
%                 table_bi_diffeo{i}.twosided_count = table_bi_diffeo{i}.twosided_count + 1;
%             elseif ((E1 < thresh) & (E2 >= thresh))
%                 table_bi_diffeo{i}.onesided_count_side1 = table_bi_diffeo{i}.onesided_count_side1 + 1;
%             elseif ((E1 >= thresh) & (E2 < thresh))
%                 table_bi_diffeo{i}.onesided_count_side2 = table_bi_diffeo{i}.onesided_count_side2 + 1;             
%             end

        end
        percent_2sided = table_bi_diffeo{i}.twosided_count/table_bi_diffeo{i}.NumEls;
        percent_1sided_side1 = table_bi_diffeo{i}.onesided_count_side1 /table_bi_diffeo{i}.NumEls;
        percent_1sided_side2 = table_bi_diffeo{i}.onesided_count_side2 /table_bi_diffeo{i}.NumEls;
        label1 = w(i,1);
        label2 = w(i,2);
        
        
        if (percent_2sided > 0.1)
            table_bi_diffeo{i}.twosided = 1;
        end
        if (percent_1sided_side1 > 0.1)
            table_bi_diffeo{i}.onesided_side1 = 1;
            table_bi_diffeo{i}.onesided_side1_Lnext = mode(cell2mat(table_bi_diffeo{i}.Lnextlist1));
        end
        if (percent_1sided_side2 > 0.1)
            table_bi_diffeo{i}.onesided_side2 = 1;
            table_bi_diffeo{i}.onesided_side2_Lnext = mode(cell2mat(table_bi_diffeo{i}.Lnextlist2));
            %             w(i,:)
            %             cell2mat(table_bi_diffeo{i}.Lnextlist2)
            %             pause
        end
        
        %         NbhdType{find(labels_vec == label1)}.twosided =  NbhdType{find(labels_vec == label1)}.twosided + 1;
        %         NbhdType{find(labels_vec == label2)}.twosided =  NbhdType{find(labels_vec == label2)}.twosided + 1;
        %         NbhdType{find(labels_vec == label1)}.twosided_partner =  label2;
        %         NbhdType{find(labels_vec == label2)}.twosided_partner =  label1;
        %     else
        %         NbhdType{find(labels_vec == label1)}.onesided =  NbhdType{find(labels_vec == label1)}.onesided + 1;
        % %         if (percent_1sided_label1_converge > 0.25)
        % %             NbhdType{find(labels_vec == label1)}.;
        %         NbhdType{find(labels_vec == label2)}.onesided =  NbhdType{find(labels_vec == label2)}.onesided + 1;
        
    end
    
    www=zeros(length(w), 3);
    
    for i = 1:length(w)
        www(i,1:2) = double(w(i,:));
        www(i,3) =  double(table_bi_diffeo{i}.twosided_count/table_bi_diffeo{i}.NumEls);
        www(i,4) =  table_bi_diffeo{i}.onesided_count_side1 /table_bi_diffeo{i}.NumEls;
        www(i,5) =  table_bi_diffeo{i}.onesided_count_side2 /table_bi_diffeo{i}.NumEls;
        www(i,6) =  table_bi_diffeo{i}.onesided_side1_Lnext;
        www(i,7) =  table_bi_diffeo{i}.onesided_side2_Lnext;
        www(i,8) =  table_bi_diffeo{i}.NumEls;
    end
    
    www
    pause
       
        
    %%%%Implement winner take all on www1.3.2021
 for j = 1:length(w)
     v = www(j,3:5);
     if (v(1) == max(v))
         www(j,3:5) = [ 1 0 0];
     elseif (v(2) == max(v))
         www(j,3:5) = [ 0 1 0];
     elseif (v(3) == max(v))
       www(j,3:5) = [ 0 0 1];
     end
 end
%  www(4,4) = 0; www(4,5) = 1;
% www(8,4) = 1; www(8,5) = 0;    
%     
%     %%%%
    
    %pause
    
    for i = 1:length(labels_vec)
        nbhd_type{i}.onesided = 0;
        nbhd_type{i}.twosided = 0;
        nbhd_type{i}.onesided_Lnext =  -1;
        nbhd_type{i}.background = 0;
    end
    for i = 1:length(labels_vec)
        label = labels_vec(i);
        for j = 1:length(w)
            if (www(j,8) < 5)  %%%less then 5 neighborhoods of this type
                continue;                
            elseif ((www(j,1) == label) & (www(j,3) > 0.1))  %%%double sided
                nbhd_type{i}.twosided = 1;
                nbhd_type{i}.twosided_partner = www(j,2);
            elseif ((www(j,2) == label) & (www(j,3) > 0.10))  %%%double sided
                nbhd_type{i}.twosided = 1;
                nbhd_type{i}.twosided_partner = www(j,1);
            elseif ((www(j,1) == label) & (www(j,4) > 0.10))
                nbhd_type{i}.onesided = 1;   %%%one sided owner
                nbhd_type{i}.onesided_Lnext = www(j,6);   %%%one sided owner
            elseif ((www(j,2) == label) & (www(j,5) > 0.10))
                nbhd_type{i}.onesided = 1;   %%%one sided owner
                nbhd_type{i}.onesided_Lnext = www(j,7);   %%%one sided owner
            end
        end
        
        %nbhd_type{i}
    end
    
    

    
    
    
    %%%%Find the background label: any surface point that is part of an owner surface (i.e., either part of two sided diffeo, or is the owner in a one sided diffeo) is NOT background
    for i = 1:length(labels_vec)  %%%find background label
        nbhd_type{i}.background = 1;
        label = labels_vec(i);
        for j = 1:length(w)
            if ((www(j,1) == label) & ((www(j,3) > 0.1) ||  (www(j,4) > 0.1) ))
                nbhd_type{i}.background = 0;
            elseif ((www(j,2) == label) & ((www(j,3) > 0.1) ||  (www(j,5) > 0.1) ))
                nbhd_type{i}.background = 0;
            end
        end
    end
    
    
    
    if (frame > 1)
        prev_labels_vec = seg_labels_vec;
    end
    
    seg_labels_vec = labels_vec;
    for i = 1:length(labels_vec)
        if ((nbhd_type{i}.onesided == 0) & (nbhd_type{i}.twosided == 1))  %%part of 2-sided nbhd but never a one-sided owner
            seg_labels_vec(i) =  nbhd_type{i}.twosided_partner;
        end
    end
    
    
    
    SSM2 =  SSM1;
    
    %%%%delete pure texture segments
    initial_color_vec = [0 60 100 120 100 120 60];
    for i = 1:length(labels_vec)
        SSM2(find(SSM1 == labels_vec(i))) = seg_labels_vec(i);
        %SSM2(find(SSM1 == labels_vec(i))) = initial_color_vec(i);
    end
    
    
    
    fname = sprintf('C:\\Users\\Doris\\Dropbox\\PAPERS_WORKING\\space\\code\\codenew_103020\\%s\\SSM2_%d.png', leafdir, frame);
    cmap = jet(256);
    imwrite(uint8(SSM2), cmap, fname);
    
    
    %ax3 = subplot(NumFrames-1,4,(frame-1)*4+3);
    %     %ax3 = subplot(1,4,3);
    %     imshow(SSM2)
    %     colormap(ax3, jet)
    %     %pause
    
    %%%%connect to previous frame
    SSM3 =  SSM2;
    
    
    
    
    %%%%%%Heart of the tracking algorithm
    
    if (frame > 1)   %%%next_labels_vec has been created in previous frame
        
        for i = 1:length(seg_labels_vec)
            if (nbhd_type{i}.background == 1)
                SSM3(find(SSM2 == seg_labels_vec(i))) = 0;
                continue;
            end
            persistent_label = next_labels_vec(i);
            if (persistent_label ==  -1)
                continue;
            end
            origin_persistent_label = prev_labels_vec(i);
            x = find(fate(:,1) == origin_persistent_label);
            %%%Assign SSM3 points with the persistent label to the proper fates
            %%%For example, if SSM3 currently has persistent label 5, we
            %%%know this is diffeomorphic to surface patch with label 4 (=origin_persistent_label) in
            %%%previous frame, and in the previous frame, label 4 was
            %%%assigned 120 (fate(x(1),2)). So we want to assign all points in SSM3 with
            %%%persistent label 5 to 120.
            SSM3(find(SSM2 == persistent_label)) = fate(x(1),2);
        end
        
        [labels_vec seg_labels_vec prev_labels_vec next_labels_vec ];
        
        
    end
    
    
    
    
    
    fname = sprintf('C:\\Users\\Doris\\Dropbox\\PAPERS_WORKING\\space\\code\\codenew_103020\\%s\\SSM3_%d.png', leafdir, frame);
    cmap = jet(256);
    
    imwrite(uint8(SSM3), cmap, fname);
    
    
    %Fate is a table that keeps track of what each unique surface label in the current segmented image (after removal of texture boundaries) gets labeled as after *invariant tracking*.
    
    
    for i = 1:length(seg_labels_vec)
        fate(i,1) = seg_labels_vec(i);
        fate(i,2) = mode(SSM3(find(SSM2 == seg_labels_vec(i))));
    end
    
    
    
    
    %ax4 = subplot(NumFrames-1,4,(frame-1)*4+4);
    %ax4 = subplot(1,4,4);
    
    %imshow(SSM3)
    %colormap(ax4, jet)
    %cmd = sprintf('title("Frame %d")', frame);
    %eval([cmd]);
    %set(gcf, 'Position', [85         124        1266         736]);
    %set(gcf, 'Color', [1 1 1])
    %pause
    
    
    
    %     Finally, we update next_labels_vec.
    % We first set it to be equal to the current frame.
    % Then we go through and for each element for which the current surface segment is identified as having a diffeomorphic counterpart in next frame, we set next_label_vec to be the counterpart label.
    % If the current surface segment dies, then we set next_label_vec to be -1.
    
    
    next_labels_vec = seg_labels_vec;  %%%%the labels in next frame corresponding to those in current frame
    for i = 1:length(seg_labels_vec)
        if (nbhd_type{i}.onesided == 1)   %%one-sided owner
            next_labels_vec(i) = nbhd_type{i}.onesided_Lnext;
        else
            next_labels_vec(i) = -1;     %%%convention: -1 is an impossibe surface label
        end
    end
end


function [min_i min_j] = findminE(E_grid)

v = E_grid{1,1};
minE = v(10);
if isnan(minE)
    minE = 1e50;
end
min_i = 1;
min_j = 1;
for i = 1:9
    for j = 1:9
        v = E_grid{i,j};
        E = v(10);
        if (E < minE)
            minE = E
            min_i = i;
            min_j = j;
        end
        
    end
end
end



function F = mygaussian_point(Sigma, cx, cy, x, y)

    % Generate mesh
    %[x y] = meshgrid(1:width, 1:height);
    %[x y] = meshgrid(-(width-1)/2:(width-1)/2, -(height-1)/2:(height-1)/2);
    
  
    % Generate gabor
    F = exp(-.5*((x-cx).^2/Sigma^2+(y-cy).^2/Sigma^2));  

end