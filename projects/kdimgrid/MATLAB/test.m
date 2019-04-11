global Rhom
Rhom = [cos(0) sin(0); cos(pi/3.)+cos(0) sin(pi/3.)+sin(0); cos(pi/3.) sin(pi/3.); 0 0]';




x = -1:0.05:1.;
y = x;


z = x;
[x,y,z] = meshgrid(x,y,z);


sz = size(x);
x_ = reshape(x,1,[]);
y_ = reshape(y,1,[]);
z_ = reshape(z,1,[]);

X = vertcat(x_,y_,z_);


B = zeros(3,3);
B([1;2],[1;2]) = Rhom(:,[1; 3]);
B(:,3) = [0;0;1];
% B([1;2],3) = randn(2,1);
% B(3,3)=1;



A = inv(B);

A1 = 1.2*A(1:2,:)*RandOrthMat(3);

% B = zeros(3,3);
% B([1;2],[1;2]) = Rhom(:,[1; 3]);
% % B(:,3) = [0;0;1];
% B([1;2],3) = randn(2,1);
% B(3,3)=1;



A = inv(B);

A2 = 1.*A(1:2,:)*RandOrthMat(3);



r = pi/2.;
% A = [cos(r) sin(r); -sin(r) cos(r)]*A;




f1 = can_distance(A1*X, zeros(2,size(X,2)) );
f2 = can_distance(A2*X, zeros(2,size(X,2)) );

f1 = reshape(f1, sz);
f2 = reshape(f2, sz);

% f1 = normalize(f1);
% f2 = normalize(f2);

f1 = exp(-f1.^2/.03);
f2 = exp(-f2.^2);
clf;


g =  f1.*f2;

colormap jet


thresh=0.2;


subplot(1,3,1);


a=.5; b=-.0;c=1;d=0;

[x_ y_] = meshgrid(-1:0.1:1); % Generate x and y data
z_ = -1/c*(a*x_ + b*y_ + d); % Solve for z data
h =slice(x,y,z,f1,x_,y_,z_, 'linear')
set(h,'edgecolor','none')

isosurface(x, y, z, f1, thresh);
isocaps(x, y, z, f1, thresh, 'above');
xlim([-1 1])
ylim([-1 1])
zlim([-1 1])
azim = 0;
elev = 45;
lightangle(azim,elev);
daspect([1,1,1]);
camlight
lighting gouraud
grid on
view(20,20)
set(gca, 'XTickLabel', [])
set(gca, 'YTickLabel', [])
set(gca, 'ZTickLabel', [])









subplot(1,3,2);
isosurface(x, y, z, f2, thresh);
isocaps(x, y, z, f2, thresh, 'above');

xlim([-1 1])
ylim([-1 1])
zlim([-1 1])
azim = 0;
elev = 45;
lightangle(azim,elev);
daspect([1,1,1]);
camlight
lighting gouraud
grid on
view(20,20)
set(gca, 'XTickLabel', [])
set(gca, 'YTickLabel', [])
set(gca, 'ZTickLabel', [])



subplot(1,3,3);

gthresh=0.96;
% for n = 1:4
%     isosurface(x, y, z,g, gthresh + n*0.01);
%     
% end
isosurface(x, y, z,g, gthresh );

isocaps(x, y, z, g, gthresh, 'above');
alpha(1.); 

view(20,20)

xlim([-1 1])
ylim([-1 1])
zlim([-1 1])
azim = 0;
elev = 45;
lightangle(azim,elev);
daspect([1,1,1]);
camlight
lighting gouraud
grid on;
xslice= [];
yslice= [];
zslice= [];
yslice = [0.];   

% contourslice(x,y,z,g,xslice,yslice,zslice)


set(gca, 'XTickLabel', [])
set(gca, 'YTickLabel', [])
set(gca, 'ZTickLabel', [])
print('tuning', '-dpng', '-r300');

% figure(2)
% 
% 
% pr = reshape(exp(-f1).*exp(-f2)./sum(exp(-f1).*exp(-f2)),1, []);
% % pr = reshape( (f1+f2)./sum(f1+f2),1, []);
% plot(pr);
% xlim([0 size(pr,2)])
% box off;
% 
% fig = gcf;
% fig.PaperUnits = 'inches';
% fig.PaperPosition = [0 0 3 2];
% % set(gcf, 'PaperSize', [2 1]);
% set(gca,'FontSize',12)
% set(gca,'xtick',[])
% set(gca,'xticklabel',[])
% set(gca,'ytick',[])
% set(gca,'yticklabel',[])
% 
% 
% xlabel('Position')
% ylabel('Pr')

print('pr','-dpng','-r0')





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


