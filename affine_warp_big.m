%codegen affine_warp_big -args {zeros(2880,3840,'double'), zeros(1,6,'double'), zeros(1,1,'double')}

%codegen affine_warp_big -args {zeros(5760,7680,'double'), zeros(1,6,'double'), zeros(1,1,'double')}

function B = affine_warp_big(A, rho, himdim)

aa=zeros(2,2);
aa(1,1) = rho(1); aa(1,2) = rho(2); aa(2,1) = rho(3); aa(2,2) = rho(4);
tx=  -rho(5); ty= -rho(6);
if (det(aa) > 1e-6)
    inva = inv(aa);
else
    inva = aa;
end

v = zeros(2,1);
v = inva*[tx ty]';

invrho = [inva(1,1) inva(1,2) inva(2,1) inva(2,2) v(1) v(2)];


[H W] = size(A);

%[Xq Yq] = meshgrid(1:0.2:a, 1:0.2:b);  %%%make five times finer
%Aq = interp2(double(A),Xq,Yq,'cubic');

B = zeros(H, W);

himdim1 = (H-1)/2;   %vertical
himdim2 = (W-1)/2;  %%horizontal

for i = 1:W   %%%horizontal
    for j = 1:H   %%%vertical        
        u = round(Aff_loc_big(i-(himdim2+1),j-(himdim1+1),invrho));
        
        
%         u = zeros(2,1);
%         u(1) = rho(1)*(i-(himdim2+1)) + rho(2)*(j-(himdim1+1)) + rho(5);
%         u(2) = rho(3)*(i-(himdim2+1)) + rho(4)*(j-(himdim1+1)) + rho(6);
        
        a = u(1)+(himdim2+1);
        b = u(2)+(himdim1+1);
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

function u = Aff_loc_big(i,j,r)
u = zeros(2,1);
u(1) = r(1)*i + r(2)*j + r(5);
u(2) = r(3)*i + r(4)*j + r(6);
end

