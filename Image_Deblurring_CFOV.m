close all
clear all
clc

nx  = 128 ; % Resize to reduce Problem

maxit =1000; 
beta =1;  alpha = 1e-6;
tol1=1e-7; 

%----------------Images-----------------------------------

u_exact = double(imread('moon.tif'));

%-----------------Kernel-----------------------------------

  N=size(u_exact,1); 
  kernel=ke_gen(N,100,4); 
%--------------------------------------------------------- 
 u_exact=imresize(u_exact,[nx nx]); %image resize

kernel=imresize(kernel,[nx nx]);     
ny = nx; hx = 1 / nx; hy = 1 / ny; N=nx; 
kernel=kernel/sum(kernel(:));
m2 = 2*nx; nd2 = nx / 2; kernele = zeros(m2, m2) ;
kernele(nd2+1:nx+nd2,nd2+1:nx+nd2) = kernel ; %extention kernel

k_hat = fft2(fftshift(kernele)) ; clear kernele
%------------------------------------------------------------------
figure;   imagesc(u_exact);colormap(gray);
s=sprintf('Exact image');s=title(s);  

z = integral_ke(u_exact,k_hat,nx,nx);%Blurry image
Blur_psnr = ppsnr(z,u_exact)

 figure;  imagesc(z);colormap(gray);
 s=sprintf('Blurry image');s=title(s);  

%-------------- Assemble the RHS --------------------------

b = integral_ke(z,conj(k_hat),nx,nx); %k*kz

n = nx^2;  m = 2*nx*(nx-1);  nm = n + m;

%---------------- Assembel the LHS-----------------------------------------
U = zeros(nx,nx); 

[B] = computeB(nx);
[D] = computeD(U,nx,m,beta);

L = B'*inv(D)*B ;

LLL=L;
%----------------------------CFOV (CG)----------------------------------------
 [u,flu,~,~,rvu] = pcg(@(x)KKL(nx,x,k_hat,L,alpha),b(:),tol1,maxit);
 
u = reshape(u,nx,nx); 
CFOV_psnr = ppsnr(u,u_exact)

%---------------------------- PCG -----------------------------------------
c_hat = fft2(kernel, nx, nx);

gam = 1e-4;

[t1,flt1,~,~,rvt1] = pcg(@(x)KKL(nx,x,k_hat,L,alpha),b(:),tol1,maxit,@(x)CCDL(nx,x,c_hat,L,alpha,gam));
 t1 = reshape(t1,nx,nx); 
PCG_ppsnr = ppsnr(u,u_exact)
 
figure;  imagesc(u);colormap(gray);
s=sprintf('Deblurred image by CG');s=title(s);  

figure;  imagesc(t1);colormap(gray);
s=sprintf('Deblurred image by PCG');s=title(s);  


%--------------Functions-----------------------------------
function K = ke_gen(n, tau, radi);
if nargin<1,help ke_gen;return; end
if nargin<2, tau=200; end
if nargin<3, radi=4; end
K=zeros(n);
R=n/2; h=1/n; h2=h^2;
%RR=n^2/radi+1; 
RR=radi^2;

if radi>0 %___________________________________________

for j=1:n
  for k=1:n
    v=(j-R)^2+(k-R)^2;
    if v <= RR,
      K(j,k)=exp(-v/4/tau^2);
    end;
  end;
end;
sw=sum(K(:));
K=K/sw; %*tau/pi;

else radi<0 %___________________________________________
 range=R-2:R+2;
 K(range,range)=1/25;
end
end

function Ku = integral_ke(u,k_hat,nux,nuy) % BCCB only
%
%  Ku = integral_ke(u,k_hat)
%
%  Use 2-D FFT's to evaluate discrete approximation to the 
%  2-D convolution integral
%
%    Ku(x,y) = \int \int k(x-x',y-y') u(x',y') dx' dy'.
%
%  k_hat is the shifted 2-D discrete Fourier transform of the 2_D 
%  kernel evaluated at node points (x_i,y_j), and then extended.
%  u is also assumed to be evaluated at node points (x_i,y_j).
%  The size of k_hat may be different that of u, due to extension.

  [nkx,nky] = size(k_hat);
  n=size(u,1);
  Ku = real(ifft2( ((fft2(u,nkx,nky)) .* k_hat)));
  if nargin == 4
    Ku = Ku(1:nux,1:nuy);
  end
  end
function [B] = computeB(nx)

e = ones(nx,1);
E = spdiags([0*e -1*e e], -1:1, nx, nx);
E1 =E(1:nx-1,:);
 
M1=eye(nx,nx);
B1=kron(E1,M1);
 
E2 = eye(nx);
M2 = spdiags([0*e -1*e e], -1:1, nx-1, nx);
B2 = kron(E2,M2);
 
B = [B1;B2];
% L = B'*D*B;
end
function [D] = computeD(U,nx,m,beta)
h0=1/nx;
[X,Y] = meshgrid(h0/2:h0:1-h0/2);
% U = [20 26 100;5 10 30;25 30 40]  % give me U from previouw computations

nn = size(U,1);
UU = sparse(nn+2,nn+2);

% we are using reflection bounday conditions 
% another word, we are using normal boundary condition to be zero
UU(2:nn+1,2:nn+1) = U;
UU(1,:) = UU(2,:);
UU(nn+2,:) = UU(nn+1,:);
UU(:,1) = UU(:,2);
UU(:,nn+2) = UU(:,nn+1);
%------------------ Matrix D ------------------
Uxr = diff(U,1,2)/h0; % x-deriv at red points
xb = h0/2:h0:1-h0/2;   yr=xb;
yb = h0:h0:1-h0;       xr=yb;
[Xb,Yb]=meshgrid(xb,yb);
[Xr,Yr]=meshgrid(xr,yr);
Uxb = interp2(Xr,Yr,Uxr,Xb,Yb,'spline');
 
 
 Uyb = diff(U,1,1)/h0; % y-deriv at blue points
 Uyr = interp2(Xb,Yb,Uyb,Xr,Yr,'spline');
  

 
 Dr = sqrt( Uxr.^2 + Uyr.^2 + beta^2 );
 Db = sqrt( Uxb.^2 + Uyb.^2 + beta^2 );
 mm1 = size(Dr,1);
 
 Dvr = Dr(:);  Dvb = Db(:); Dv=[Dvr;Dvb];
 
 siz_u1 =size(Dvr);
 siz_u2 =size(Dvb);
 ddd = [ sparse(m,1) , Dv , sparse(m,1) ];
 D = spdiags(ddd,[-1 0 1],m,m);
 
end
function [w] = KKL(nx,x,k_hat,L,alpha)

x_mat = reshape(x,nx,nx);
y_mat = integral_ke(x_mat,k_hat,nx,nx);
w1_mat = integral_ke(y_mat,conj(k_hat),nx,nx);
% L = speye(nx^2);
  w = w1_mat(:) + alpha*L*x ; % TV
%   w = w1_mat(:) + alpha*L*x +alpha*x ; % Tik
end

  function Ku = integral_ccc(u,c_hat,nux,nuy) % Jun 2008 BCCB only

  [nkx,nky] = size(c_hat);
  n=size(u,1);
  Ku = real(ifft2( ((fft2(u,nkx,nky)) .* c_hat)));
  if nargin == 4
    Ku = Ku(1:nux,1:nuy);
  end
  end
function [w] = CCDL(nx,x,c_hat,L,alpha,gam)

x_mat = reshape(x,nx,nx);
y_mat = integral_ccc(x_mat,c_hat,nx,nx);
w1_mat = integral_ccc(y_mat,conj(c_hat),nx,nx);

w = w1_mat(:) + gam*diag(diag(L))*x;
end
function p = ppsnr(x,y)

d = mean( mean( (x(:)-y(:)).^2 ) );
m1 = max( abs(x(:)) );
m2 = max( abs(y(:)) );
m = max(m1,m2);

p = 10*log10( m^2/d );
end





