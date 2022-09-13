
clear all
close all


%%%%plan for debugging
%1. Make sure you can converge to the same value if you take 200x200 patches of
%large image subjected to same transform as used in the synthesize routine.
% Try over a range of starting values and increments.  Make a version of
% compute_diffeo where all the code is in one place and easy to test.

%2. If this works, then the texture detection on the ovals should work
%cause it's the exact same thing.

cd C:\Users\Doris\Dropbox\PAPERS_WORKING\space\code\codenew_103020

%%%%Solve Leaf: code to segment and track leaves
% 3. identify edges with two clear neigbors
% 4. perform two sided diffeo detection for each edge identified in 3
% 5. group super segmentation map: if an edge element of the surface contains two sided diffeomorphism, the component is a texture component,
% and the labels on both sides of edge should be same



NumFrames = 161;


%C:\Users\Doris\Dropbox\PAPERS_WORKING\space\code\codenew_103020
%%%load leaf sequence
%%%10.31.2020: causes out of memory error. Better to load one by one





%     %for i = 1:2
%     close all
%     leaf_dir = 'synthetic_leaves'; %%%synthetic leaves
%     fname = sprintf('C:\\Users\\Doris\\Dropbox\\PAPERS_WORKING\\space\\code\\codenew_103020\\%s\\leaf_%d.png', leaf_dir, i);
%     LM{i} = imread(fname);  %%leaf map
%     fname = sprintf('C:\\Users\\Doris\\Dropbox\\PAPERS_WORKING\\space\\code\\codenew_103020\\%s\\leaf_edge_%d.png', leaf_dir, i);
%     EM{i} = imread(fname);  %%edge map
%     fname = sprintf('C:\\Users\\Doris\\Dropbox\\PAPERS_WORKING\\space\\code\\codenew_103020\\%s\\superseg_%d.png', leaf_dir, i);
%     SSM{i} = imread(fname);  %%superseg map
% end

%
% %%%%For every point in edge map, compute whether nbhd centered on circle is
% %%%%a 2-sided nbhd, and create patches for the left and rigth nbhd
%
%[a b] = size(EM{1});
%nsize = 100;  %%%nbhd size
%


nsize = 100;
% for frame = 16:16
hold off
%for frame = 1:1
clear twosided_edgeppoints
%   imagesc(SSM{frame}); colormap(jet(256));



%for frame = 1:NumFrames
 for frame = 1:5
    
    leaf_dir = 'synthetic_bear'; %%%synthetic leaves
    fname = sprintf('C:\\Users\\Doris\\Dropbox\\PAPERS_WORKING\\space\\code\\codenew_103020\\%s\\leaf_%d.png', leaf_dir, frame);
    LM1 = imread(fname);  %%leaf map
    fname = sprintf('C:\\Users\\Doris\\Dropbox\\PAPERS_WORKING\\space\\code\\codenew_103020\\%s\\leaf_%d.png', leaf_dir, frame+1);
    LM1_next = imread(fname);  %%leaf map
    
    fname = sprintf('C:\\Users\\Doris\\Dropbox\\PAPERS_WORKING\\space\\code\\codenew_103020\\%s\\leaf_edge_%d.png', leaf_dir, frame);
    EM1 = imread(fname);  %%edge map
    fname = sprintf('C:\\Users\\Doris\\Dropbox\\PAPERS_WORKING\\space\\code\\codenew_103020\\%s\\leaf_edge_%d.png', leaf_dir, frame+1);
    EM1_next = imread(fname);  %%edge map
    
    fname = sprintf('C:\\Users\\Doris\\Dropbox\\PAPERS_WORKING\\space\\code\\codenew_103020\\%s\\superseg_%d.png', leaf_dir, frame);
    SSM1 = imread(fname);  %%superseg map
    fname = sprintf('C:\\Users\\Doris\\Dropbox\\PAPERS_WORKING\\space\\code\\codenew_103020\\%s\\superseg_%d.png', leaf_dir, frame+1);
    SSM1_next = imread(fname);  %%superseg map
    
    [a b] = size(EM1);
    
    
    %     LM1 = LM{frame};
    %     LM1_next = LM{frame+1};
    %     %     imagesc(LM1_next);
    %     %     pause
    %     EM1 = EM{frame};
    %     SSM1 = SSM{frame};
    %     SSM1_next = SSM{frame+1};
    
  

    
    nbhd_space = 20;
    %nbhd_space = 5;
    ind = 1;
    for  i = (nbhd_space+1):(a-nbhd_space)
        for j = (nbhd_space+1):(b-nbhd_space)
            if (EM1(i,j) > 0)
                edgepoints{ind}.y = i;
                edgepoints{ind}.x = j;
                edgepoints{ind}.n = SSM1(i-nbhd_space,j);
                edgepoints{ind}.s = SSM1(i+nbhd_space,j);
                edgepoints{ind}.e = SSM1(i,j-nbhd_space);
                edgepoints{ind}.w = SSM1(i,j+nbhd_space);
                ind = ind +1;
            end
        end
    end
    
    
    u = randperm(length(edgepoints));
    for i = 1:length(edgepoints)  %%%run through two sided neighborhoods
        new_edgepoints{i} = edgepoints{u(i)};
    end
    
    
    %%%just look at 10000 points
    edgepoints = edgepoints(1:10000);
    new_edgepoints = new_edgepoints(1:10000);
    
    
    
    for i = 1:length(edgepoints)  %%%run through two sided neighborhoods
        quad_labels(i,:) = [new_edgepoints{i}.n new_edgepoints{i}.s new_edgepoints{i}.e new_edgepoints{i}.w ];
    end
    w = unique(quad_labels, 'rows'); %%%these are the unique nbhds...cycle through nbhds trying to find these
    
    ind = 1;
    for i = 1:length(edgepoints)   %%%index into reordered list
        crossed_off(i) = 0;
    end
    
    % clear reordered_edgepoints
    NumNbhds = 100;    %%number of nbhds that we will go through to find unique two-sided nbhds
    %
    
    NumNbhdsPerLabel =     floor(NumNbhds /length(w));
    
    
    for i = 1:NumNbhds   %%%index into reordered list
        %for i = 80:80   %%%index into reordered list
        %i
        target_found = 0;
        while ~target_found
            target = w(ind,:)  %%uique nbhd
            pause
            for j = 1:length(edgepoints)
                [ i/NumNbhds j/length(edgepoints)]
                current =  [new_edgepoints{j}.n new_edgepoints{j}.s new_edgepoints{j}.e new_edgepoints{j}.w ];
                if ((sum(current == target) == length(target)) & (crossed_off(j) == 0))  %%%%found target and it hasn't yet been crossed off
                    cx = new_edgepoints{j}.x;
                    cy = new_edgepoints{j}.y;
                    nbhd_superseg = SSM1(cy-nsize:cy+nsize, cx-nsize:cx+nsize);
                    v = unique(nbhd_superseg);
                    if (length(v) == 3)
                        reordered_edgepoints{i} = new_edgepoints{j};
                        crossed_off(j) = 1;
                        target_found = true;
                        break;
                    end
                end
            end
            ind = mod(ind, length(w))+1; %%%%wraps around
        end
    end
    
    
    
    
    
    
    
    
    
    % for point_num = 1:200
    %      [reordered_edgepoints{point_num}.n reordered_edgepoints{point_num}.s reordered_edgepoints{point_num}.e reordered_edgepoints{point_num}.w ]
    % end
    %pause
    
    
    ind2 = 0;  %%counter for edge points with two sides
    %for point_num = 1:length(edgepoints)
    
    %    w = randperm(length(edgepoints));
    
    %for point_num = 1:length(edgepoints)
    AA = SSM1;
    
    %     %%%%%comment everything before this
    
    
    
    
    
    for point_num = 1:NumNbhds
        point_num
        %cx = twosided_edgeppoints{point_num}.x;
        %cy = twosided_edgeppoints{point_num}.y;
        
        
        cx = reordered_edgepoints{point_num}.x;
        cy = reordered_edgepoints{point_num}.y;
        current =  [reordered_edgepoints{point_num}.n reordered_edgepoints{point_num}.s reordered_edgepoints{point_num}.e reordered_edgepoints{point_num}.w ]
        
        nbhd_superseg = SSM1(cy-nsize:cy+nsize, cx-nsize:cx+nsize);
        nbhd_superseg_next = SSM1_next(cy-nsize:cy+nsize, cx-nsize:cx+nsize);
        v = unique(nbhd_superseg);
        vv = unique(nbhd_superseg_next);
        
        v = setdiff(v,0);
        vv = setdiff(vv,0);
        
        if (length(v) == 2)
            ind2 = ind2+1;
            nbhd_leaf = LM1(cy-nsize:cy+nsize, cx-nsize:cx+nsize);
            nbhd_leaf_next = LM1_next(cy-nsize:cy+nsize, cx-nsize:cx+nsize);
            twosided_edgeppoints{ind2}.x = cx;
            twosided_edgeppoints{ind2}.y = cy;
            
            AA(cy-nsize:cy+nsize, cx-nsize:cx+nsize) = 0;
%             figure(1)
%             imshow(AA);
%             axis off
%             hold on
%             plot(cx, cy,'r.');
%             plot(cx, cy,'r.');
            
            
            %figure(2)
            N1a = zeros(size(nbhd_superseg));
            N2a = zeros(size(nbhd_superseg));
            N1b = zeros(size(nbhd_superseg));
            N2b = zeros(size(nbhd_superseg));
            support_v1 = find(nbhd_superseg == v(1));
            %%%find the regions with maximum overlap in next frame
            
            clear overlap;
            for label = 1:length(vv)
                overlap(label) = length(find((nbhd_superseg == v(1)) & (nbhd_superseg_next == vv(label))));
            end
            [overlap_val match_ind] = max(overlap);
            target1 = vv(match_ind);
            
            
            support_v1_next = find(nbhd_superseg_next == target1);
            N1a(support_v1) = nbhd_leaf(support_v1);
            N1b(support_v1_next) = nbhd_leaf_next(support_v1_next);
            support_v2 = find(nbhd_superseg == v(2));
            
            clear overlap;
            for label = 1:length(vv)
                overlap(label) = length(find((nbhd_superseg == v(2)) & (nbhd_superseg_next == vv(label))));
            end
            [overlap_val match_ind] = max(overlap);
            
            target2 = vv(match_ind);
            support_v2_next = find(nbhd_superseg_next == target2);
            N2a(support_v2) = nbhd_leaf(support_v2);
            N2b(support_v2_next) = nbhd_leaf_next(support_v2_next);
            
            %             [X Y] = meshgrid(-25:1:25, -25:1:25);
            %             for i  = 1:length(X)
            %                 for j   = 1:length(Y)
            %                      N1b_t= imtranslate(N1b,[X(i,j) Y(i,j)]);
            %                      C = N1b_t - N1a;
            %                     %imagesc(AA-B);
            %                      DP1(i,j) = sum(sum(C.^2));
            %                 end
            %             end
            %             [M,I] = min(DP1(:));
            %             [x1 y1] = ind2sub(size(DP1),I);
            %
            %
            %
            %             for i  = 1:length(X)
            %                 for j   = 1:length(Y)
            %                      N2b_t= imtranslate(N2b,[X(i,j) Y(i,j)]);
            %                      C = N2b_t - N2a;
            %                     %imagesc(AA-B);
            %                      DP2(i,j) = sum(sum(C.^2));
            %                 end
            %             end
            %             [M,I] = min(DP2(:));
            %             [x2 y2] = ind2sub(size(DP2),I)
            
            %             if (DP1(x1, y1) < DP2(x2, y2))
            %                 N1b_t = imtranslate(N1b, [X(x1,y1), Y(x1,y1)]);
            %                 N2b_t = imtranslate(N2b, [X(x1,y1), Y(x1,y1)]);
            %                 x_shift = x1;
            %                 y_shift = y1;
            %             else
            %                 N1b_t = imtranslate(N1b, [X(x2,y2), Y(x2,y2)]);
            %                 N2b_t = imtranslate(N2b, [X(x2,y2), Y(x2,y2)]);
            %                 x_shift = x2;
            %                 y_shift = y2;
            %             end
            
            %                 N1b_t = imtranslate(N1b, [X(x1,y1), Y(x1,y1)]);
            %                 N2b_t = imtranslate(N2b, [X(x2,y2), Y(x2,y2)]);
            %                 %x1_shift = x1;
            %y1_shift = y1;
            
            
            
            
            %%%compute neighborhoods centered on centroids rather than on
            %%%edge point.
            
            N1a_centroid = zeros(size(nbhd_superseg));
            N2a_centroid = zeros(size(nbhd_superseg));
            N1b_centroid = zeros(size(nbhd_superseg));
            N2b_centroid = zeros(size(nbhd_superseg));
            
            
            [x1 y1] = FindCentroid(N1a);
            [x2 y2] = FindCentroid(N1b);
            x_centroid = round(0.5*(x1 + x2));
            y_centroid = round(0.5*(y1 + y2));
            
            
            cx_centroid = cx + (x_centroid - nsize);
            cy_centroid = cy + (y_centroid - nsize); 
            nbhd_leaf = LM1(cy_centroid-nsize:cy_centroid+nsize, cx_centroid-nsize:cx_centroid+nsize);
            nbhd_leaf_next = LM1_next(cy_centroid-nsize:cy_centroid+nsize, cx_centroid-nsize:cx_centroid+nsize);
            nbhd_superseg = SSM1(cy_centroid-nsize:cy_centroid+nsize, cx_centroid-nsize:cx_centroid+nsize);
            nbhd_superseg_next = SSM1_next(cy_centroid-nsize:cy_centroid+nsize, cx_centroid-nsize:cx_centroid+nsize);
            
            support_v1 = find(nbhd_superseg == v(1));
            support_v1_next = find(nbhd_superseg_next == target1);
            N1a_centroid(support_v1) = nbhd_leaf(support_v1);
            N1b_centroid(support_v1_next) = nbhd_leaf_next(support_v1_next);
            
            
            [x1 y1] = FindCentroid(N2a);
            [x2 y2] = FindCentroid(N2b);
            x_centroid = round(0.5*(x1 + x2));
            y_centroid = round(0.5*(y1 + y2));
            
            
            cx_centroid = cx + (x_centroid - nsize);
            cy_centroid = cy + (y_centroid - nsize); 
            nbhd_leaf = LM1(cy_centroid-nsize:cy_centroid+nsize, cx_centroid-nsize:cx_centroid+nsize);
            nbhd_leaf_next = LM1_next(cy_centroid-nsize:cy_centroid+nsize, cx_centroid-nsize:cx_centroid+nsize);
            nbhd_superseg = SSM1(cy_centroid-nsize:cy_centroid+nsize, cx_centroid-nsize:cx_centroid+nsize);
            nbhd_superseg_next = SSM1_next(cy_centroid-nsize:cy_centroid+nsize, cx_centroid-nsize:cx_centroid+nsize);
            
            
            
            
            support_v2 = find(nbhd_superseg == v(2));
            support_v2_next = find(nbhd_superseg_next == target2);
            N2a_centroid(support_v2) = nbhd_leaf(support_v2);
            N2b_centroid(support_v2_next) = nbhd_leaf_next(support_v2_next);
            
                        
            %%%%end compute nbhds centered on centroids
            
            
            
            
            
            
            
%             figure(2)
%             subplot(2,2,1)
%             imagesc(N1a);
%             subplot(2,2,2)
%             imagesc(N1b);
%             subplot(2,2,3)
%             imagesc(N2a);
%             subplot(2,2,4)
%             imagesc(N2b);
%             %pause
            
            res1 = 0;
            res2 = 0;
            rr1 = zeros(6,1);
            rr2 = zeros(6,1);
            
            rr = [1 0 0 1 0 0];
            NumIters = 10;
            %rr = rho;
            A = N1a;
            B = N1b;
            
            E_mat1 = zeros(NumIters,1);
            rr_mat1 = zeros(NumIters,6);
            E_mat2 = zeros(NumIters,1);
            rr_mat2 = zeros(NumIters,6);
            
            
            
            x_shift = -20:5:20;
            y_shift = -20:5:20;
            himdim = 0.5*(length(N1a) - 1);
            for i = 1:length(x_shift)
                for j = 1:length(y_shift)
                    rr = [1 0 0 1 x_shift(i) y_shift(j)];
                    %[ E_mat1 rr_mat1] = compute_diffeon_mex(N1a, N1b, NumIters, rr);
                    E_grid1{i, j} = E_mat1;
                    rr_grid1{i, j} = rr_mat1;
                    %%%%%[ E_mat1 rr_mat1] = compute_diffeon(RA, RB, NumIters, rr);
                    %RC = affine_warp(N1a, rr_mat1(NumIters,:), himdim);
                    %draw_path = 'C:\\Users\\Doris\\Dropbox\\PAPERS_WORKING\\space\\code\\codenew_103020\\synthetic_leaves_rotate\\nbhds';
                    %fname= sprintf('%s\\N1a_warp_%d_%d_%d_%d.png', draw_path, frame, point_num, i,j);
                    %cmd = sprintf('imwrite(RC/255, "%s");', fname);
                    %eval([cmd]);
                end
            end
            
            
            for i = 1:length(x_shift)
                for j = 1:length(y_shift)
                    rr = [1 0 0 1 x_shift(i) y_shift(j)];
                    %[ E_mat1 rr_mat1] = compute_diffeon_mex(N2a, N2b, NumIters, rr);
                    E_grid2{i, j} = E_mat1;
                    rr_grid2{i, j} = rr_mat1;
                    %%%%[ E_mat1 rr_mat1] = compute_diffeon(RA, RB, NumIters, rr);
                    %RC = affine_warp(N2a, rr_mat1(NumIters,:), himdim);
                    %draw_path = 'C:\\Users\\Doris\\Dropbox\\PAPERS_WORKING\\space\\code\\codenew_103020\\synthetic_leaves_rotate\\nbhds';                    
                    %fname= sprintf('%s\\N2a_warp_%d_%d_%d_%d.png', draw_path, frame, point_num, i,j);
                    %cmd = sprintf('imwrite(RC/255, "%s");', fname);
                    %eval([cmd]);
                end
            end
            
            
            
            
            %    [ E_mat1 rr_mat1] = compute_diffeon_mex(N1a, N1b, NumIters, rr);
            %    [ E_mat2 rr_mat2] = compute_diffeon_mex(N2a, N2b, NumIters, rr);
            %    rr_mat1
            %    rr_mat2
            %    E_mat1(length(E_mat1))
            %    E_mat2(length(E_mat2))
            
            
            draw_path = 'C:\\Users\\Doris\\Dropbox\\PAPERS_WORKING\\space\\code\\codenew_103020\\synthetic_bear\\nbhds2';
            fname = sprintf('%s/N1a_%d_%d.png', draw_path, frame, ind2);
            imwrite(uint8(N1a), fname);
            pause(2)
            fname = sprintf('%s/N1b_%d_%d.png', draw_path, frame, ind2);
            imwrite(uint8(N1b), fname);
            pause(2)
            
            fname = sprintf('%s/N2a_%d_%d.png', draw_path, frame, ind2);
            imwrite(uint8(N2a), fname);
            pause(2)
            
            fname = sprintf('%s/N2b_%d_%d.png', draw_path, frame, ind2);
            imwrite(uint8(N2b), fname);
            pause(2)
            
            fname = sprintf('%s/N1a_centroid_%d_%d.png', draw_path, frame, ind2);
            imwrite(uint8(N1a_centroid), fname);
            pause(2)
            
            fname = sprintf('%s/N1b_centroid_%d_%d.png', draw_path, frame, ind2);
            imwrite(uint8(N1b_centroid), fname);
            pause(2)
            
            fname = sprintf('%s/N2a_centroid_%d_%d.png', draw_path, frame, ind2);
            imwrite(uint8(N2a_centroid), fname);
            pause(2)
            
            fname = sprintf('%s/N2b_centroid_%d_%d.png', draw_path, frame, ind2);
            imwrite(uint8(N2b_centroid), fname);
            pause(2)
            
            
            %             rr_mat1sasa
            %             rr_mat2
            %             pause
            %             res1/res2
            %             norm(rr1-rr2)
            twosided_edgeppoints{ind2}.label1 = v(1);
            twosided_edgeppoints{ind2}.label2 = v(2);
            twosided_edgeppoints{ind2}.label1_next = target1;
            twosided_edgeppoints{ind2}.label2_next = target2;
            twosided_edgeppoints{ind2}.E_grid1 = E_grid1;
            twosided_edgeppoints{ind2}.E_grid2 = E_grid2;
            twosided_edgeppoints{ind2}.rr_grid1 = rr_grid1;
            twosided_edgeppoints{ind2}.rr_grid2 = rr_grid2;
            %             twosided_edgeppoints{ind2}.x_shift = x_shift;
            %             twosided_edgeppoints{ind2}.y_shift = y_shift;
            %
            %             figure(3)
            %             subplot(1,2,1)
            %             plot(E_mat1)
            %             subplot(1,2,2)
            %             plot(E_mat2)
            %             pause
        end        %%%end if at edge point with two sides
        
    end        %%%end for loop through all nbhd points
    FrameInfo{frame}.tse =  twosided_edgeppoints;
    cmd = sprintf('save C:\\Users\\Doris\\Dropbox\\PAPERS_WORKING\\space\\code\\codenew_103020\\%s\\FI_%d  twosided_edgeppoints', leaf_dir, frame);
    eval([cmd])
    %FrameInfo{frame}.SSM =  SSM1;
end  %%end loop through frames

draw_path = 'C:\\Users\\Doris\\Dropbox\\PAPERS_WORKING\\space\\code\\codenew_103020\\synthetic_bear\\nbhds2';
fname = sprintf('%s\\FrameInfo.mat', draw_path);
save(fname, 'FrameInfo'); 




function A_centroid = ReCenter(A, x1, y1)
x1 = round(x1);
y1 = round(y1);
[a b] = size(A);
imdim = a;
B = zeros(imdim);
himdim = (imdim-1)/2;
minx = max(1, x1-himdim);
maxx = min(imdim, x1+himdim);
miny = max(1, y1-himdim);
maxy = min(imdim, y1+himdim);
B((miny+himdim+1-y1):(maxy + himdim+1-y1), (minx+himdim+1-x1):(maxx + himdim+1-x1)) = A(miny:maxy, minx:maxx);
A_centroid = B;
% figure(1)
% subplot(1,2,1)
% imagesc(A);
% subplot(1,2,2)
% imagesc(A_centroid);

end

function [x y] = FindCentroid(A)
A_sil = A;
A_sil(find(A~=0)) = 1;
s  = regionprops(A_sil, 'centroid');
v = s(1).Centroid;
x = v(1);
y = v(2);
end


function B = affine_warp(A, rho, himdim)
aa=zeros(2,2);
aa(1,1) = rho(1); aa(1,2) = rho(2); aa(2,1) = rho(3); aa(2,2) = rho(4);
tx=  -rho(5); ty= -rho(6);
if (det(aa) > 1e-6)
    inva = inv(aa);
else
    inva = aa;
end
invrho = [inva(1,1) inva(1,2) inva(2,1) inva(2,2) tx ty];


[W H] = size(A);
%[Xq Yq] = meshgrid(1:0.2:a, 1:0.2:b);  %%%make five times finer
%Aq = interp2(double(A),Xq,Yq,'cubic');

B = zeros(W);

for i = 1:W
    for j = 1:H
        u = round(Aff_loc(i-(himdim+1),j-(himdim+1),invrho));
        a = u(1)+(himdim+1);
        b = u(2)+(himdim+1);
        x1 = floor(a);
        y1 = floor(b);
        
        %Bicubic interpolation (applies grayscale image)
        if ((x1 >= 2) && (y1 >= 2) && (x1 <= W-2) && (y1 <= H-2))
            %Load 4x4 pixels
            P = A(y1-1:y1+2, x1-1:x1+2);
            
            %Interpolation weights
            dx = a - x1;
            dy = b - y1;
            
            %Bi-bicubic interpolation
            B(j, i) = bicubicInterpolate(P, dx, dy);
        end
    end
end

end


function q = bicubicInterpolate(p, x, y)
q1 = cubicInterpolate(p(1,:), x);
q2 = cubicInterpolate(p(2,:), x);
q3 = cubicInterpolate(p(3,:), x);
q4 = cubicInterpolate(p(4,:), x);
q = cubicInterpolate([q1, q2, q3, q4], y);
end

function q = cubicInterpolate(p, x)
q = p(2) + 0.5 * x*(p(3) - p(1) + x*(2.0*p(1) - 5.0*p(2) + 4.0*p(3) - p(4) + x*(3.0*(p(2) - p(3)) + p(4) - p(1))));
end
%
% figure; hold on
% for i = 1:11
%     plot(E_vec(i,:));
% end

function u = Aff_loc(i,j,r)
u = zeros(2,1);
u(1) = r(1)*i + r(2)*j + r(5);
u(2) = r(3)*i + r(4)*j + r(6);
end