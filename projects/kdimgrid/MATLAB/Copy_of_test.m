global Rhom
Rhom = [cos(0) sin(0); cos(pi/3.)+cos(0) sin(pi/3.)+sin(0); cos(pi/3.) sin(pi/3.); 0 0]';



x = -1:0.02:1.;
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
size(A);
for i = 1:n
% ---------------------

B(i,[1;2],[1;2]) = Rhom(:,[1; 3]);
% B(i,:,3) = [0;0;1];

B(i,[1;2],3) = randn(2,1);
B(i,3,3)=1;
B = B.*1.;
Ai = inv(reshape(B(i,1:3,1:3),3,3));
A(i,:,:) = Ai(1:2,:)*RandOrthMat(3);

% ---------------------
end


clf;
for i = 2:n
% ---------------------

Ai = reshape(A(i,:,:), 2,3);
f  = can_distance(Ai*X, zeros(2,size(X,2)) );
f  = exp(-f.^2/.1);
f  = reshape(f, sz);
colormap jet

thresh=0.8;


azim = 0;
elev = 45;

subplot(2,n,n+i);


a=.5; b=-.0;c=1;d=0;
[x_ y_] = meshgrid(-1:0.01:1); % Generate x and y data
z_ = -1/c*(a*x_ + b*y_ + d); % Solve for z data
h =slice(x,y,z,f,x_,y_,z_, 'linear');
set(h,'edgecolor','none');
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



subplot(2,n,i);
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
% ---------------------
end


h = subplot(2,n,1);


vert = [0 0 0;1 0 0;1 1 0;0 1 0;0 0 1;1 0 1;1 1 1;0 1 1];
fac = [1 2 6 5;2 3 7 6;3 4 8 7;4 1 5 8;1 2 3 4;5 6 7 8];
patch('Vertices',vert,'Faces',fac,...
      'FaceVertexCData',hsv(6), 'FaceAlpha', 0.2);


lightangle(azim,elev);
daspect([1,1,1]);
camlight
grid on
view(20,20)

set(gca, 'XTickLabel', [])
set(gca, 'YTickLabel', [])
set(gca, 'ZTickLabel', [])


h = subplot(2,n,n+1);
a=.5; b=-.0;c=1;d=0;
[x_ y_] = meshgrid(-1:0.01:1); % Generate x and y data
z_ = -1/c*(a*x_ + b*y_ + d); % Solve for z data
surf(x_,y_,z_, 'EdgeColor', 'none', 'FaceColor', [0,0,0],...
    'FaceLighting', 'flat',...
       'FaceAlpha', 0.3);

xlim([-1 1])
ylim([-1 1])
zlim([-1 1])
lightangle(azim,elev);
daspect([1,1,1]);
camlight
grid on
view(20,20)

set(gca, 'XTickLabel', [])
set(gca, 'YTickLabel', [])
set(gca, 'ZTickLabel', [])

print('tuning', '-dpng', '-r300');


save('mydata', "A","B")


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


