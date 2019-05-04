global Rhom
Rhom = [cos(0) sin(0); cos(pi/3.)+cos(0) sin(pi/3.)+sin(0); cos(pi/3.) sin(pi/3.); 0 0]';


azim = 0;
elev = 45;


x = -1:0.04:1.;
y = x;


z = x;
[x,y,z] = meshgrid(x,y,z);


sz = size(x);
x_ = reshape(x,1,[]);
y_ = reshape(y,1,[]);
z_ = reshape(z,1,[]);
X = vertcat(x_,y_,z_);


n = 4;

A = zeros(n, 2, 3);
B = zeros(n, 3, 3);
F = zeros(n, size(X,2));
size(A);


for i = 1:n
% ---------------------
B(i,[1;2],[1;2]) = Rhom(:,[1; 3]);
end


% 
% i = 1;
% B(i,[1;2],3) = [.1; 0.3];
% B(i,3,3)=1;
% B = B.*.85;
% Ai = inv(reshape(B(i,1:3,1:3),3,3));
% A(i,:,:) = [cos(pi*0.3) sin(pi*0.3) ; -sin(pi*0.3) cos(pi*0.3)]*Ai(1:2,:);
% % A(i,:,:) = Ai(1:2,:);
% 
% i = 2;
% B(i,[1;2],3) = randn(2,1)*0.4;
% B(i,3,3)=1;
% B = B.*.85;
% Ai = inv(reshape(B(i,1:3,1:3),3,3));
% A(i,:,:) = Ai(1:2,:);
% % A(i,:,:) = Ai(1:2,:);



for i = 1:n
% ---------------------

% B(i,[1;2],[1;2]) = Rhom(:,[1; 3]);
B(i,:,3) = [0;0;1];

% B(i,[1;2],3) = randn(2,1);
% B(i,3,3)=1;



B = B.*(.9 + (rand-0.5)*2*0.1);
Ai = inv(reshape(B(i,1:3,1:3),3,3));
A(i,:,:) = Ai(1:2,:)*RandOrthMat(3);
% A(i,:,:) = Ai(1:2,:);
end

% A(1,:,:) = A1;
% A(2,:,:) = A6;
% % A(3,:,:) = A3;
% % A(4,:,:) = A4;
% A(3,:,:) = A5;
% A(4,:,:) = A3;

for i = 1:n
Ai  = reshape(A(i,:,:), 2,3);
fi  = can_distance(Ai*X, zeros(2,size(X,2)) );
fi   = exp(-fi.^2/1.1);
fi = fi./sum(fi, 'all');
F(i,:) = fi;

% ---------------------
end

r=0.95;
clf;
for i = 1:n
% ---------------------
    
colormap jet


f  = F(i,:);
f = reshape(f, sz);
% f = f - min(f, [],'all');

% f = f./max(f, [],'all');
% f = f./sum(f, 'all');


thresh = max(f, [],'all')*r;
subplot(3,n,i);
isosurface(x, y, z, f, thresh);
isocaps(x, y, z, f, thresh, 'above');
xlim([-1 1])
ylim([-1 1])
zlim([-1 1])

lightangle(azim,elev);
daspect([1,1,1]);
camlight
lighting gouraud
grid on
view(20,20)

set(gca, 'XTickLabel', [])
set(gca, 'YTickLabel', [])
set(gca, 'ZTickLabel', [])

colormap jet

f = F(1,:);
for k = 2:i
    f  = f.*F(k,:);
end




f = reshape(f, sz);
% f = f - min(f, [],'all');

% f = f./max(f, [],'all');
% f = f./sum(f, 'all');

% max(f, [],'all')
% min(f, [],'all')


if i==n  R = 0.92;, else R = r;, end;
thresh=max(f, [],'all')*R;
subplot(3,n,i+n);
isosurface(x, y, z, f, thresh);
isocaps(x, y, z, f, thresh, 'above');
xlim([-1 1])
ylim([-1 1])
zlim([-1 1])

lightangle(azim,elev);
daspect([1,1,1]);
camlight
lighting gouraud
grid on
view(20,20)

set(gca, 'XTickLabel', [])
set(gca, 'YTickLabel', [])
set(gca, 'ZTickLabel', [])

probpl = subplot(3,n,i+n+n);

% size(f)
g = max(f, [],3);
% g = max(g, [],2);
% g = reshape(g,1,[]);
% g = g(1:2:end,1:2:end);
% g = g(1:2:end);
% % size(g);
% bar(g, 1.);
% contourf(g);
% area(g);
surf(g);
% plot(g);
% hline = refline([0. thresh]);

% hline.Color = 'r';
% fmesh(g);
% bar(g);
% view(0,90)
view(20,0);
% view(2);

grid off
set(gca, 'XTickLabel', [])
set(gca, 'YTickLabel', [])
% set(gca, 'ZTickLabel', [])
% xlim([-1 1])
% zlim([0 0.00000000001])



end








function d = can_distance(X, Y)
    global Rhom
    Z = mod(X - Y,1); 
    Z = Rhom(:,[1; 3])*Z;
    n1 = vecnorm(Z - Rhom(:,1), 2, 1);
    n2 = vecnorm(Z - Rhom(:,2), 2, 1);
    n3 = vecnorm(Z - Rhom(:,3), 2, 1);
    n4 = vecnorm(Z - Rhom(:,4), 2, 1);
    D = vertcat(n1,n2,n3,n4);
    d = min(D,[],1);
end


function g = normalize(f)
    g_ = f - min(f,[],'all');
    g_ = g_./max(g_, [],'all');
    g  = 1 - g_;
end

function M=RandOrthMat(n, tol)
% M = RANDORTHMAT(n)
% generates a random n x n orthogonal real matrix.
%
% M = RANDORTHMAT(n,tol)
% explicitly specifies a thresh value that measures linear dependence
% of a newly formed column with the existing columns. Defaults to 1e-6.
%
% In this version the generated matrix distribution *is* uniform over the manifold
% O(n) w.r.t. the induced R^(n^2) Lebesgue measure, at a slight computational 
% overhead (randn + normalization, as opposed to rand ). 
% 
% (c) Ofek Shilon , 2006.
    if nargin==1
	  tol=1e-6;
    end
    
    M = zeros(n); % prealloc
    
    % gram-schmidt on random column vectors
    
    vi = randn(n,1);  

    % the n-dimensional normal distribution has spherical symmetry, which implies
    % that after normalization the drawn vectors would be uniformly distributed on the
    % n-dimensional unit sphere.
    M(:,1) = vi ./ norm(vi);
    
    for i=2:n
	  nrm = 0;
	  while nrm<tol
		vi = randn(n,1);
		vi = vi -  M(:,1:i-1)  * ( M(:,1:i-1).' * vi )  ;
		nrm = norm(vi);
	  end
	  M(:,i) = vi ./ nrm;
    end %i
        
end  % RandOrthMat


