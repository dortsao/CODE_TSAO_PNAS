
clear all
close all

ew = 1; %%default thin line: 2

leafdir = 'C:\\Users\\Doris\\Dropbox\\PAPERS_WORKING\\space\\code\\codenew_103020\\synthetic_bear';

scale = 8;
l = 640*scale*1.5;
w = 480*scale*1.5;

start_offset = 0;

blurscale = 10;
cropsize = 240*1.5-1;

cx1 = l/2;
cy1 = w/2;
cr1 = 200*scale*1.5;


for leaf_num = 1:4   %%%leaves 1-3 are ovals with ovals inside, leaf 4 is bear with oval inside
    if (leaf_num < 4)        
        close all
        %%%%%naturalistic leaf with leaf shaped texture inside
        
        %%Generate Random dot background
        RA = spatialPattern([w,l],-2);
        
        
        
        RA = RA - min(min(RA));
        RA = RA / max(max(RA));
        
        
        %%create outer leaf
        
        if (leaf_num ~=3)
            leaf_im = double(imread('C:\Users\Doris\Dropbox\PAPERS_WORKING\space\code\codenew_103020\synthetic_bear\silhouette_fun\leaf_silhouette_straight_2x.png'));
        else
            leaf_im = double(imread('C:\Users\Doris\Dropbox\PAPERS_WORKING\space\code\codenew_103020\synthetic_bear\silhouette_fun\leaf_silhouette_straight_2x.png'));
        end
        
        
        leaf_scale = 1.2;
        leaf_im = HardZoom(leaf_im, leaf_scale);
        
        [a b] = size(leaf_im);
        
        
        mask1 = (255-leaf_im) > 128;
        
        
        croppedImage(:,:) = RA(:,:).*mask1;    %%%%leaf-shaped pink noise        
        croppedImage = croppedImage * 0.5 / mean(mean(croppedImage(mask1)));  %%set mean to be 0.5
        
        leaf{2*leaf_num-1} = croppedImage;
              
        
        %%create inner leaf
        imageSize = size(RA);
        mask2 = imresize(mask1, 0.3);
        [a b] = size(mask2);
        [a1 b1] = size(mask1);
        numzero_l = a1-a;
        numzero_w = b1-b;
        mask2 = padarray(mask2, [numzero_l/2 numzero_w/2]);
        
        croppedImage(:,:) = RA(:,:).*mask2;    %%%%leaf-shaped pink noise        
        croppedImage = croppedImage * 0.5 / mean(mean(croppedImage(mask2)));  %%set mean to be 0.5
        
        leaf{2*leaf_num} = croppedImage;
        

        SuperSeg{2*leaf_num-1} = mask1;
        SuperSeg{2*leaf_num} = mask2;
                
        
    else
        
        close all
        
        %%%%%bear shaped leaf
        
        %%Generate Random dot background
        RA = spatialPattern([w,l],-2);
        
        
        RA = RA - min(min(RA));
        RA = RA / max(max(RA));
        
        
        
        bear = double(imread('C:\Users\Doris\Dropbox\PAPERS_WORKING\space\code\codenew_103020\synthetic_bear\silhouette_fun\bear_silhouette_2x.png'));
        bear_scale = 1.5;
        bear = HardZoom(bear, bear_scale);
        mask1 = (255-bear) > 128;
        
                
        croppedImage(:,:) = RA(:,:).*mask1;        
        croppedImage = croppedImage * 0.5 / mean(mean(croppedImage(mask1)));                
        leaf{2*leaf_num-1} = croppedImage;
        
        %%Draw inner leaf
        
        leaf_im = double(imread('C:\Users\Doris\Dropbox\PAPERS_WORKING\space\code\codenew_103020\synthetic_bear\silhouette_fun\leaf_silhouette_straight_tattoo_2x.png'));
        
        leaf_scale = 1;
        leaf_im = HardZoom(leaf_im, leaf_scale);
        
        [a b] = size(leaf_im);
        leaf_mask1 = (255-leaf_im) > 128;
        
        imageSize = size(RA);
        mask2 = imresize(leaf_mask1, 0.5);
        [a b] = size(mask2);
        [a1 b1] = size(leaf_mask1);
        numzero_l = a1-a;
        numzero_w = b1-b;
        mask2 = padarray(mask2, [numzero_l/2 numzero_w/2]);
        mask2 = imtranslate(mask2,[650*2+2*100 300*2-2*150],'FillValues',0);  %%%horz vert
        
        croppedImage(:,:) = RA(:,:).*mask2;        
        croppedImage = croppedImage * 0.5 / mean(mean(croppedImage(mask2)));                
        leaf{2*leaf_num} = croppedImage;
             
        SuperSeg{2*leaf_num-1} = mask1;
        SuperSeg{2*leaf_num} = mask2;        
    end
end


%%%%%%%%%%
%%%%Blit to background

%Generate Random dot background and save to bg

RA = spatialPattern([w,l],-2);
RA = RA - min(min(RA));
RA = RA / max(max(RA));
RA = RA * 0.5 / mean(mean(RA));

bg = RA;

y_off = 60*scale*1.5;
buf = 10*scale;
ind = 0;
rotscale = 2;
NumSteps = 50;

for i = -60:1:100   
    ii = i;
    if (i<-60)
        ii = -60;
    end
    if (i> 60)
        ii = 60;
    end
    
    ind = ind+1;
    

%%%%%%%%%%%%%%compute four transformations    
    angx1 = i/rotscale;
    angy1 = i/rotscale;
    angz1 = i/rotscale;
    
    angx2 = -i/rotscale;
    angy2 = i/rotscale;
    angz2 = 0;
    
    angx3 = -i/rotscale;
    angy3 = -0;
    angz3 = 0;
    
    
    
    rho_i1_init = [0 1.05 -400*2 -250*2]; %%%%angle, scale, x, y    
    rho_i2_init = [-60 1.05 0 -250*2]; %%%%angle, scale, x, y    
    rho_i3_init = [0 1.05 400*2 -250*2]; %%%%angle, scale, x, y    
    rho_i4_init = [0 1 -100*2 400*2]; %%%%angle, scale, x, y
         
    rho_i1_final = [-30 1.5 -500*2 0];    
    rho_i2_final = [-5 1.2 0 0];    
    rho_i3_final = [20 1.5 500*2 0];    
    rho_i4_final = [0 1 -100*2 200*2];
    
    phase = 2*(triangle(ii*2*pi/50)-0.5);
    angx1 = interp_leaf(rho_i1_init(1),rho_i1_final(1), ii, NumSteps);
    
    scalex1 = interp_leaf(rho_i1_init(2),rho_i1_final(2), ii, NumSteps);
    tx1 = interp_leaf(rho_i1_init(3),rho_i1_final(3), ii,  NumSteps);
    ty1 = interp_leaf(rho_i1_init(4),rho_i1_final(4), ii,  NumSteps);
    
    angx2 = interp_leaf(rho_i2_init(1),rho_i2_final(1), ii,  NumSteps);
    scalex2 = interp_leaf(rho_i2_init(2),rho_i2_final(2), ii,  NumSteps);
    tx2 = interp_leaf(rho_i2_init(3),rho_i2_final(3), ii,  NumSteps);
    ty2 = interp_leaf(rho_i2_init(4),rho_i2_final(4), ii,  NumSteps);
    
    angx3 = interp_leaf(rho_i3_init(1),rho_i3_final(1), ii,  NumSteps);
    scalex3 = interp_leaf(rho_i3_init(2),rho_i3_final(2), ii,  NumSteps);
    tx3 = interp_leaf(rho_i3_init(3),rho_i3_final(3), ii,  NumSteps);
    ty3 = interp_leaf(rho_i3_init(4),rho_i3_final(4), ii,  NumSteps);
    
    angx4 = interp_leaf(rho_i4_init(1),rho_i4_final(1), ii,  NumSteps);
    scalex4 = interp_leaf(rho_i4_init(2),rho_i4_final(2), ii,  NumSteps);
    tx4 = interp_leaf(rho_i4_init(3),rho_i4_final(3), ii,  NumSteps);
    ty4 = interp_leaf(rho_i4_init(4),rho_i4_final(4), ii,  NumSteps);
    
    ty4
    
    
    
    a1 = 0.2;
    a2 = 0.5;
    a3 = 0.2;
    a4 = 0.3;
    
    
    rho_i4 = [scalex4*cos(angx4*pi/180) scalex4*sin(angx4*pi/180) -scalex4*sin(angx4*pi/180) scalex4*cos(angx4*pi/180) tx4 ty4] + [0 0 0 0  0 0];
    
    
    lscale = 1.2;    
    
    if (i<60)
     tx1_extra = 0;
     tx3_extra = 0;
     ty3_extra = 0;
    else
     tx1_extra = -(i-60)*5;
     tx3_extra = (i-60)*15;
     ty3_extra = -(i-60)*15;
    end
    
    rho_i1 = lscale*[1 ii/120 ii/120 1  tx1+tx1_extra ty1]
    rho_i2 = lscale*[1-ii/150 0 0 1+ii/150 tx2 ty2]
    rho_i3 = lscale*[1 -ii/120 -ii/120 1 tx3+tx3_extra ty3+ty3_extra]
        
    
    scaleup = 1.2;
    rho_i1(1:4) = scaleup*rho_i1(1:4);
    rho_i2(1:4) = scaleup*rho_i2(1:4);
    rho_i3(1:4) = scaleup*rho_i3(1:4);
    rho_i4(1:4) = scaleup*rho_i4(1:4);
%%%%%%%%%%%%%%%%%%%%%%%%    
            
    %%%%draw random dot surfaces with edges
    A = bg;
        
    [a b] = size(leaf{1});
    c = max([a b]);
    himdim = (c-1)/2;
    
    EM = zeros(a, b);
    
    
  
    
    leaf_1_piece = affine_warp_big_mex(leaf{1}, rho_i1, himdim);    
    leaf_2_piece = affine_warp_big_mex(leaf{2}, rho_i1, himdim);    
    leaf_3_piece = affine_warp_big_mex(leaf{3}, rho_i2, himdim);    
    leaf_4_piece = affine_warp_big_mex(leaf{4}, rho_i2, himdim);
    leaf_5_piece = affine_warp_big_mex(leaf{5}, rho_i3, himdim);    
    leaf_6_piece = affine_warp_big_mex(leaf{6}, rho_i3, himdim);    
    leaf_7_piece = affine_warp_big_mex(leaf{7}, rho_i4, himdim);    
    leaf_8_piece = affine_warp_big_mex(leaf{8}, rho_i4, himdim);
   
    
    %%%%draw edges
    
    
    seg_mask1 = affine_warp_big_mex(double(SuperSeg{1}), rho_i1, himdim);
    seg_mask2 = affine_warp_big_mex(double(SuperSeg{2}), rho_i1, himdim);
    seg_mask3 = affine_warp_big_mex(double(SuperSeg{3}), rho_i2, himdim);
    seg_mask4 = affine_warp_big_mex(double(SuperSeg{4}), rho_i2, himdim);
    seg_mask5 = affine_warp_big_mex(double(SuperSeg{5}), rho_i3, himdim);
    seg_mask6 = affine_warp_big_mex(double(SuperSeg{6}), rho_i3, himdim);
    seg_mask7 = affine_warp_big_mex(double(SuperSeg{7}), rho_i4, himdim);
    seg_mask8 = affine_warp_big_mex(double(SuperSeg{8}), rho_i4, himdim);
    
    
    
    leaf1_mask = double(seg_mask1 >= 1);
    leaf2_mask = double(seg_mask2 >= 1);
    leaf3_mask = double(seg_mask3 >= 1);
    leaf4_mask = double(seg_mask4 >= 1);
    leaf5_mask = double(seg_mask5 >= 1);
    leaf6_mask = double(seg_mask6 >= 1);
    leaf7_mask = double(seg_mask7 >= 1);
    leaf8_mask = double(seg_mask8 >= 1);

        
%%%draw surfaces and edges (need to do in proper order)  
    A = ...
        (1 - leaf7_mask) .* A + leaf7_mask .* leaf_7_piece;   
    
    image_edge_inner= bwperim(leaf7_mask,8);
    image_edge_inner_thick = imdilate(image_edge_inner, strel('disk',ew));
    A( image_edge_inner_thick > 0) =  1;
    
    A = ...
        (1 - leaf8_mask) .* A + leaf8_mask .* leaf_8_piece;
    image_edge_inner= bwperim(leaf8_mask,8);
    image_edge_inner_thick = imdilate(image_edge_inner, strel('disk',ew));
    A( image_edge_inner_thick > 0) =  1;    
 
    
    A = ...
        (1 - leaf5_mask) .* A + leaf5_mask .* leaf_5_piece;       
    image_edge_inner= bwperim(leaf5_mask,8);
    image_edge_inner_thick = imdilate(image_edge_inner, strel('disk',ew));
    A( image_edge_inner_thick > 0) =  1;    
    
    A = ...
        (1 - leaf6_mask) .* A + leaf6_mask .* leaf_6_piece;
    image_edge_inner= bwperim(leaf6_mask,8);
    image_edge_inner_thick = imdilate(image_edge_inner, strel('disk',ew));
    A( image_edge_inner_thick > 0) =  1;
    
    
     
    A = ...
        (1 - leaf3_mask) .* A + leaf3_mask .* leaf_3_piece; 
    image_edge_inner= bwperim(leaf3_mask,8);
    image_edge_inner_thick = imdilate(image_edge_inner, strel('disk',ew));
    A( image_edge_inner_thick > 0) =  1;
    
    A = ...
        (1 - leaf4_mask) .* A + leaf4_mask .* leaf_4_piece;    
    image_edge_inner= bwperim(leaf4_mask,8);
    image_edge_inner_thick = imdilate(image_edge_inner, strel('disk',ew));
    A( image_edge_inner_thick > 0) =  1;
    

    
    A = ...
        (1 - leaf1_mask) .* A + leaf1_mask .* leaf_1_piece;
    image_edge_inner= bwperim(leaf1_mask,8);
    image_edge_inner_thick = imdilate(image_edge_inner, strel('disk',ew));
    A( image_edge_inner_thick > 0) =  1;    
    
    
    A = ...
        (1 - leaf2_mask) .* A + leaf2_mask .* leaf_2_piece;        
    image_edge_inner= bwperim(leaf2_mask,8);
    image_edge_inner_thick = imdilate(image_edge_inner, strel('disk',ew));
    A( image_edge_inner_thick > 0) =  1;
    

    
    A_crop = A(cy1 - cropsize*scale:cy1 + cropsize*scale, cx1 - cropsize*scale:cx1 + cropsize*scale);
    
    fname = sprintf('%s\\leaf_%d.png', leafdir, ind);
    imwrite(uint8(A_crop*255), fname);
    pause(1)   
    
    
    figure(1)
    imagesc(A_crop);    
   
            
    %%%%draw edge image (exactly same as leaf image, except draw 8 black
    %%%%surfaces instead of textured surfaces, each followed by edges

    
    EM = ...
        (1 - leaf7_mask) .* EM + leaf7_mask .* 0;   
    
    image_edge_inner= bwperim(leaf7_mask,8);
    image_edge_inner_thick = imdilate(image_edge_inner, strel('disk',ew));
    EM( image_edge_inner_thick > 0) =  1;
    
    EM = ...
        (1 - leaf8_mask) .* EM + leaf8_mask .* 0;
    image_edge_inner= bwperim(leaf8_mask,8);
    image_edge_inner_thick = imdilate(image_edge_inner, strel('disk',ew));
    EM( image_edge_inner_thick > 0) =  1;    
 
    
    EM = ...
        (1 - leaf5_mask) .* EM + leaf5_mask .* 0;       
    image_edge_inner= bwperim(leaf5_mask,8);
    image_edge_inner_thick = imdilate(image_edge_inner, strel('disk',ew));
    EM( image_edge_inner_thick > 0) =  1;    
    
    EM = ...
        (1 - leaf6_mask) .* EM + leaf6_mask .* 0;
    image_edge_inner= bwperim(leaf6_mask,8);
    image_edge_inner_thick = imdilate(image_edge_inner, strel('disk',ew));
    EM( image_edge_inner_thick > 0) =  1;
    
    
     
    EM = ...
        (1 - leaf3_mask) .* EM + leaf3_mask .* 0; 
    image_edge_inner= bwperim(leaf3_mask,8);
    image_edge_inner_thick = imdilate(image_edge_inner, strel('disk',ew));
    EM( image_edge_inner_thick > 0) =  1;
    
    EM = ...
        (1 - leaf4_mask) .* EM + leaf4_mask .* 0;    
    image_edge_inner= bwperim(leaf4_mask,8);
    image_edge_inner_thick = imdilate(image_edge_inner, strel('disk',ew));
    EM( image_edge_inner_thick > 0) =  1;
    

    
    EM = ...
        (1 - leaf1_mask) .* EM + leaf1_mask .* 0;
    image_edge_inner= bwperim(leaf1_mask,8);
    image_edge_inner_thick = imdilate(image_edge_inner, strel('disk',ew));
    EM( image_edge_inner_thick > 0) =  1;    
    
    
    EM = ...
        (1 - leaf2_mask) .* EM + leaf2_mask .* 0;        
    image_edge_inner= bwperim(leaf2_mask,8);
    image_edge_inner_thick = imdilate(image_edge_inner, strel('disk',ew));
    EM( image_edge_inner_thick > 0) =  1;
        
    

    EM_crop = EM(cy1 - cropsize*scale:cy1 + cropsize*scale, cx1 - cropsize*scale:cx1 + cropsize*scale);
    
    
    fname = sprintf('%s\\leaf_edge_%d.png',leafdir, ind);
    imwrite(uint8(255*EM_crop), fname);
    pause(1)
        
    %%%%draw super segmentation image (srart woth edge map and find
    %%%%independent pieces
    
    SSM = zeros(size(A));  %%super seg map
 
    im = EM;
    im = 1-im;
    CC = bwconncomp(im);
    
    SSM = zeros(size(im));
    
    %L = labelmatrix(CC);
    
    w = 10:255;
    w_rand = randperm(length(w));
    v = w(w_rand);
    
    %v = randperm(255);
    for i = 1:CC.NumObjects
        if (length(CC.PixelIdxList{i}) > 100)
            SSM(CC.PixelIdxList{i}) = v(i);
        end
    end
    %SSM_norm = SSM * 255/max(max(SSM));
    SSM_norm_crop = SSM(cy1 - cropsize*scale:cy1 + cropsize*scale, cx1 - cropsize*scale:cx1 + cropsize*scale);
    fname = sprintf('%s\\superseg_%d.png', leafdir, ind);
    imwrite(SSM_norm_crop, jet(256), fname);
    pause(1)        
end  %%%%end for loop going through frames





function draw_circle(cx, cy, r, NumPoints)

ind = 1;
for ang = 1:360/NumPoints:360
    ang_rad = ang*pi/180;
    circle{ind}.x = cx + r*cos(ang_rad);
    circle{ind}.y = cy + r*sin(ang_rad);
    ind = ind+1;
end

[b] =length(circle);
hold on
for i = 1:b
    if (i<b)
        
        h=plot([circle{i}.x circle{i+1}.x], [circle{i}.y circle{i+1}.y], 'Color', [1 1 1]);
    else
        h=plot([circle{i}.x circle{1}.x], [circle{i}.y circle{1}.y], 'Color', [1 1 1]);
    end
end
end

function [A_circle] = draw_circle_mat(A, cx, cy, r)

%%%draws a line

% ind = 1;
% for ang = 1:360/NumPoints:360
%     ang_rad = ang*pi/180;
%     circle{ind}.x = cx + r*cos(ang_rad);
%     circle{ind}.y = cy + r*sin(ang_rad);
%     ind = ind+1;
% end

A_circle = A;
[a b] = size(A);
[xx,yy] = meshgrid(1:b, 1:a);
A_circle(  (sqrt((xx-cx).^2 + (yy-cy).^2) > (r-2) )  & (sqrt((xx-cx).^2 + (yy-cy).^2) < (r+2) )) = 1;
%imagesc(A_circle)
end

function [A_rotate] = apply_rotate(A, angx, angy, angz)

[a b] = size(A);
cyy = (a+1)/2;
cxx = (b+1)/2;

A_rotate = zeros(size(A));

for i = 1:a
    for j = 1:b
        X = (j-cxx);
        Y = (i-cyy);
        Z = 0;
        
        
        RotMat = zeros(3);
        RotMat = rotate(angx, angy, angz);
        
        V_new = RotMat * ([X Y Z])';
        
        %x = round(V_new(1)/(1+V_new(3)) + cx);
        %y = round(V_new(2)/(1+V_new(3)) + cy);
        x = round(V_new(1)) + cxx;
        y = round(V_new(2)) + cyy;
        
        
        if ((x<=b) & (x>=1) & (y<=a) & (y>=1))
            A_rotate(y,x) = A(i,j);
            %   imagesc(B)
        end
    end
end
end


function [RotMat] = rotate(a, b, c)

a1 = a*pi/180;
RotMatx(1,1) = 1; RotMatx(1,2) = 0; RotMatx(1,3) = 0;
RotMatx(2,1) = 0; RotMatx(2,2) = cos(a1); RotMatx(2,3) = sin(a1);
RotMatx(3,1) = 0; RotMatx(3,2) = -sin(a1); RotMatx(3,3) = cos(a1);


b1 = b*pi/180;
RotMaty(1,1) = cos(b1); RotMaty(1,2) = 0; RotMaty(1,3) = sin(b1);
RotMaty(2,1) = 0; RotMaty(2,2) = 1; RotMaty(2,3) = 0;
RotMaty(3,1) = -sin(b1); RotMaty(3,2) = 0; RotMaty(3,3) = cos(b1);


c1 = c*pi/180;
RotMatz(1,1) = cos(c1); RotMatz(1,2) = sin(c1); RotMatz(1,3) = 0;
RotMatz(2,1) = -sin(c1); RotMatz(2,2) = cos(c1); RotMatz(2,3) = 0;
RotMatz(3,1) = 0; RotMatz(3,2) = 0; RotMatz(3,3) = 1;

RotMat = RotMatx * RotMaty * RotMatz;
end


function [A_fixed] = fix_holes_func(A)

[a b] =size(A);
A_fixed = A;

for i = 2:a-1
    for j = 2:b-1
        n = A(i-1, j);
        s = A(i+1, j);
        w = A(i, j-1);
        e = A(i, j+1);
        v = [n s e w];
        w = find(v ~= 0);
        if ((A(i,j) == 0) & (length(w) > 2) )
            %A_fixed(i,j) = v(w(1));
            A_fixed(i,j) = mean(v);
        end
    end
end
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


[H W] = size(A);

%[Xq Yq] = meshgrid(1:0.2:a, 1:0.2:b);  %%%make five times finer
%Aq = interp2(double(A),Xq,Yq,'cubic');

B = zeros(H, W);




for i = 1:W   %%%hprizontal
    for j = 1:H   %%%vertical
        u = round(Aff_loc(i-(himdim+1),j-(himdim+1),invrho));
        a = u(1)+(himdim+1);
        b = u(2)+(himdim+1);
        x1 = floor(a);
        y1 = floor(b);
        
        %Bicubic interpolation (applies grayscale image)
        if ((y1 >= 2) && (x1 >= 2) && (y1 <= H-2) && (x1 <= W-2))
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

function OutPicture = HardZoom(InPicture, ZoomFactor)
if ZoomFactor == 0
    OutPicture = [];
    return;
end
if ZoomFactor < 0
    InPicture = rot90(InPicture, 2);
    ZoomFactor = 0 - ZoomFactor;
end
if ZoomFactor == 1
    OutPicture = InPicture;
    return;
end
xSize = size(InPicture, 1);
ySize = size(InPicture, 2);
xCrop = xSize / 2 * abs(ZoomFactor - 1);
yCrop = ySize / 2 * abs(ZoomFactor - 1);
zoomPicture = imresize(InPicture, ZoomFactor);
if ZoomFactor > 1
    OutPicture = zoomPicture( 1+xCrop:end-xCrop, 1+yCrop:end-yCrop );
else
    OutPicture = zeros(xSize, ySize);
    OutPicture( 1+xCrop:end-xCrop, 1+yCrop:end-yCrop ) = zoomPicture;
end
end

% The usage is the same as sin(2*pi*f*t)
function y = triangle(t)
y = abs(mod((t+pi)/pi, 2)-1);
end

function y = interp_leaf(a, b, i, NumSteps)
y =(1-i/NumSteps)*a + (i/NumSteps)*b;
end