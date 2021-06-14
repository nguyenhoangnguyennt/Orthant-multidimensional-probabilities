function  est=mvncdf(l,u,Sig,n)
% computes an estimator of the probability Pr(l<X<u),
% where 'X' is a zero-mean multivariate normal vector
% with covariance matrix 'Sig', that is, X~N(0,Sig)
% infinite values for vectors 'u' and 'l' are accepted;
% Monte Carlo method uses sample size 'n'; the larger
% the 'n', the smaller the relative error of the estimator;
% output: structure 'est' with
%              1. estimated value of probability Pr(l<X<u)
%              2. estimated relative error of estimator
%              3. theoretical upper bound on true Pr(l<X<u)
%              Remark: If you want to estimate Pr(l<Y<u),
%                   where Y~N(m,Sig) has mean vector 'm',
%                     then use 'mvncdf(Sig,l-m,u-m,n)'.
% Example: (run from directory with saved mvncdf.m)
% clear all,clc,d=25;
% l=ones(d,1)/2;u=ones(d,1);
% Sig=inv(0.5*eye(d)+.5*ones(d,d));
% est=mvncdf(l,u,Sig,10^4) % output of our method
%
% % Executing Matlab's toolbox\stats\stats\mvncdf.m
% % with n=10^7 below is slow and inaccurate
% cd(matlabroot) % change to Matlab default path
% options=optimset('TolFun',0,'MaxFunEvals',10^7,'Display','iter');
% [prob,err]=mvncdf(l,u,zeros(d,1),Sig,options)

% Reference:
% Z. I. Botev (2015),
% "The Normal Law Under Linear Restrictions:
%  Simulation and Estimation via Minimax Tilting",
%  submitted to JRSS(B)
l=l(:); u=u(:); % set to column vectors
d=length(l); % basic input check
if  (length(u)~=d)|(d~=sqrt(prod(size(Sig)))|any(l>u))
    error('l, u, and Sig have to match in dimension with u>l')
end
% Cholesky decomposition of matrix
[L, l, u]=cholperm( Sig, l, u ); D=diag(L);
if any(D<eps)
    warning('Method may fail as covariance matrix is singular!')
end
L=L./repmat(D,1,d);u=u./D; l=l./D; % rescale
L=L-eye(d); % remove diagonal
% find optimal tilting parameter via non-linear equation solver
options=optimset('Diagnostics','off','Display','off',...
    'Algorithm','trust-region-dogleg','Jacobian','on');
[soln,fval,exitflag] = fsolve(@(x)gradpsi(x,L,l,u),zeros(2*(d-1),1),options);
if exitflag~=1
    warning('Method may fail as covariance matrix is close to singular!')
end
x=soln(1:(d-1));mu=soln(d:(2*d-2)); % assign saddlepoint x* and mu*
est=mvnpr(n,L,l,u,mu);
% compute psi star
est.upbnd=exp(psy(x,L,l,u,mu));
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function est=mvnpr(n,L,l,u,mu)
% computes P(l<X<u), where X is normal with
% 'Cov(X)=L*L' and zero mean vector;
% exponential tilting uses parameter 'mu';
% Monte Carlo uses 'n' samples;
d=length(l); % Initialization
mu(d)=0;
Z=zeros(d,n); % create array for variables
p=0;
for k=1:(d-1)
    % compute matrix multiplication L*Z
    col=L(k,1:k)*Z(1:k,:);
    % compute limits of truncation
    tl=l(k)-mu(k)-col;
    tu=u(k)-mu(k)-col;
    %simulate N(mu,1) conditional on [tl,tu]
    Z(k,:)=mu(k)+trandn(tl(:),tu(:));
    % update likelihood ratio
    p = p+lnNpr(tl,tu)+.5*mu(k)^2-mu(k)*Z(k,:);
end
% deal with final Z(d) which need not be simulated
col=L(d,:)*Z;tl=l(d)-col;tu=u(d)-col;
p=p+lnNpr(tl,tu); % update LR corresponding to Z(d)
p=exp(p); % now switch back from logarithmic scale
est.prob=mean(p);
est.relErr=std(p)/sqrt(n)/est.prob; % relative error
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function p=psy(x,L,l,u,mu)
% implements psi(x,mu); assumes scaled 'L' without diagonal;
d=length(u);x(d)=0;mu(d)=0;x=x(:);mu=mu(:);
% compute now ~l and ~u
c=L*x;l=l-mu-c;u=u-mu-c;
p=sum(lnNpr(l,u)+.5*mu.^2-x.*mu);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [grad,J]=gradpsi(y,L,l,u)
% implements gradient of psi(x) to find optimal exponential twisting;
% assumes scaled 'L' with zero diagonal;
d=length(u);c=zeros(d,1);x=c;mu=c;
x(1:(d-1))=y(1:(d-1));mu(1:(d-1))=y(d:end);
% compute now ~l and ~u
c(2:d)=L(2:d,:)*x;lt=l-mu-c;ut=u-mu-c;
% compute gradients avoiding catastrophic cancellation
w=lnNpr(lt,ut);
pl=exp(-0.5*lt.^2-w)/sqrt(2*pi);
pu=exp(-0.5*ut.^2-w)/sqrt(2*pi);
P=pl-pu;
% output the gradient
dfdx=-mu(1:(d-1))+(P'*L(:,1:(d-1)))';
dfdm= mu-x+P;
grad=[dfdx;dfdm(1:(d-1))];
if nargout>1 % here compute Jacobian matrix
    lt(isinf(lt))=0; ut(isinf(ut))=0;
    dP=-P.^2+lt.*pl-ut.*pu; % dPdm
    DL=repmat(dP,1,d).*L;
    mx=-eye(d)+DL;
    xx=L'*DL;
    mx=mx(1:d-1,1:d-1);
    xx=xx(1:d-1,1:d-1);
    J=[xx,mx';
        mx,diag(1+dP(1:d-1))];
end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function x=trandn(l,u)
% samples a vector of length=length(l)=length(u)
% from the standard multivariate normal distribution,
% truncated over the region [l,u];
% input 'l' and 'u' are column vectors;
% inifinite values for 'u' and 'l' are accepted
x=nan(size(l));
a=.66; % treshold for switching between methods
% threshold can be tuned for maximum speed for each platform
% three cases to consider:
% case 1: a<l<u
I=l>a;
if any(I)
    tl=l(I); tu=u(I); x(I)=ntail(tl,tu);
end
% case 2: l<u<-a
J=u<-a;
if any(J)
    tl=-u(J); tu=-l(J); x(J)=-ntail(tl,tu);
end
% case 3: otherwise use inverse transform or accept-reject
I=~(I|J);
if  any(I)
    tl=l(I); tu=u(I); x(I)=tn(tl,tu);
end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function x=ntail(l,u)
% samples a column vector of length=length(l)=length(u)
% from the standard multivariate normal distribution,
% truncated over the region [l,u], where l>0 and
% l and u are column vectors; modification of Marsaglia (1964)
% who uses acceptance-rejection from Rayleigh distr.
c=l.^2/2; n=length(l); f=expm1(c-u.^2/2);
x=c-reallog(1+rand(n,1).*f); % sample using Rayleigh
% keep list of rejected
I=find(rand(n,1).^2.*x>c); d=length(I);
while d>0 % while there are rejections
    cy=c(I); % find the thresholds of rejected
    y=cy-reallog(1+rand(d,1).*f(I));
    idx=rand(d,1).^2.*y<cy; % accepted
    x(I(idx))=y(idx); % store the accepted
    I=I(~idx); % remove accepted from list
    d=length(I); % number of rejected
end
x=sqrt(2*x); % this Rayleigh transform can be delayed till the end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function x=tn(l,u)
% samples a column vector of length=length(l)=length(u)
% from the standard multivariate normal distribution,
% truncated over the region [l,u], where -a<l<u<a for some
% 'a' and l and u are column vectors;
% uses acceptance rejection and inverse-transform method;
tol=2; % controls switch between methods
% threshold can be tuned for maximum speed for each platform
% case: abs(u-l)>tol, uses accept-reject from randn
I=abs(u-l)>tol; x=l;
if any(I)
    tl=l(I); tu=u(I); x(I)=trnd(tl,tu);
end
% case: abs(u-l)<tol, uses inverse-transform
I=~I;
if any(I)
    tl=l(I); tu=u(I); pl=erfc(tl/sqrt(2))/2; pu=erfc(tu/sqrt(2))/2;
    x(I)=sqrt(2)*erfcinv(2*(pl-(pl-pu).*rand(size(tl))));
end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function  x=trnd(l,u)
% uses acceptance rejection to simulate from truncated normal
x=randn(size(l)); % sample normal
% keep list of rejected
I=find(x<l|x>u); d=length(I);
while d>0 % while there are rejections
    ly=l(I); % find the thresholds of rejected
    uy=u(I);
    y=randn(size(ly));
    idx=y>ly&y<uy; % accepted
    x(I(idx))=y(idx); % store the accepted
    I=I(~idx); % remove accepted from list
    d=length(I); % number of rejected
end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function p=lnNpr(a,b)
% computes ln(P(a<Z<b))
% where Z~N(0,1) very accurately for any 'a', 'b'
p=zeros(size(a));
% case b>a>0
I=a>0;
if any(I)
    pa=lnPhi(a(I)); % log of upper tail
    pb=lnPhi(b(I));
    p(I)=pa+log1p(-exp(pb-pa));
end
% case a<b<0
idx=b<0;
if any(idx)
    pa=lnPhi(-a(idx)); % log of lower tail
    pb=lnPhi(-b(idx));
    p(idx)=pb+log1p(-exp(pa-pb));
end
% case a<0<b
I=(~I)&(~idx);
if any(I)
    pa=erfc(-a(I)/sqrt(2))/2; % lower tail
    pb=erfc(b(I)/sqrt(2))/2;  % upper tail
    p(I)=log1p(-pa-pb);
end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function p=lnPhi(x)
% computes logarithm of  tail of Z~N(0,1) mitigating
% numerical roundoff errors;
p=-0.5*x.^2-log(2)+reallog(erfcx(x/sqrt(2)));
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [ L, l, u, perm ] = cholperm( Sig, l, u )
%  Computes permuted lower Cholesky factor L for Sig
%  by permuting integration limit vectors l and u.
%  Outputs perm, such that Sig(perm,perm)=L*L'.
%
% Reference: 
%  Gibson G. J., Glasbey C. A., Elston D. A. (1994),
%  "Monte Carlo evaluation of multivariate normal integrals and
%  sensitivity to variate ordering", 
%  In: Advances in Numerical Methods and Applications, pages 120--126

d=length(l);perm=1:d; % keep track of permutation
L=zeros(d,d);z=zeros(d,1);
for j=1:d
    pr=Inf(d,1); % compute marginal prob.
    I=j:d; % search remaining dimensions
    D=diag(Sig);
    s=D(I)-sum(L(I,1:j-1).^2,2);
    s(s<0)=eps;s=sqrt(s);
    tl=(l(I)-L(I,1:j-1)*z(1:j-1))./s;
    tu=(u(I)-L(I,1:j-1)*z(1:j-1))./s;
    pr(I)=lnNpr(tl,tu);
    % find smallest marginal dimension
    [dummy,k]=min(pr);
    % flip dimensions k-->j
    jk=[j,k];kj=[k,j];
    Sig(jk,:)=Sig(kj,:);Sig(:,jk)=Sig(:,kj); % update rows and cols of Sig
    L(jk,:)=L(kj,:); % update only rows of L
    l(jk)=l(kj);u(jk)=u(kj); % update integration limits
    perm(jk)=perm(kj); % keep track of permutation
    % construct L sequentially via Cholesky computation
    s=Sig(j,j)-sum(L(j,1:j-1).^2);s(s<0)=eps;L(j,j)=sqrt(s);
    L(j+1:d,j)=(Sig(j+1:d,j)-L(j+1:d,1:j-1)*(L(j,1:j-1))')/L(j,j);
    % find mean value, z(j), of truncated normal:
    tl=(l(j)-L(j,1:j-1)*z(1:j-1))/L(j,j);
    tu=(u(j)-L(j,1:j-1)*z(1:j-1))/L(j,j);
    w=lnNpr(tl,tu); % aids in computing expected value of trunc. normal
    z(j)=(exp(-.5*tl.^2-w)-exp(-.5*tu.^2-w))/sqrt(2*pi);
end
end


