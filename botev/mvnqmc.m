function  est=mvnqmc(l,u,Sig,n)
% computes an estimator of the probability Pr(l<X<u),
% where 'X' is a zero-mean multivariate normal vector
% with covariance matrix 'Sig', that is, X~N(0,Sig)
% infinite values for vectors 'u' and 'l' are accepted;
%
% This version uses a Quasi Monte Carlo (QMC) pointset
% of size ceil(n/12) and estimates the relative error
% using 12 independent randomized QMC estimators; QMC
% is slower than ordinary Monte Carlo (see my mvncdf.m), 
% but is also likely to be more accurate when d<50.
%
% output:      structure 'est' with
%              1. estimated value of probability Pr(l<X<u)
%              2. estimated relative error of estimator
%              3. theoretical upper bound on true Pr(l<X<u)
%              Remark: If you want to estimate Pr(l<Y<u),
%                   where Y~N(m,Sig) has mean vector 'm',
%                     then use 'mvnqmc(Sig,l-m,u-m,n)'.
%
% Example: (run from directory with saved mvnqmc.m)
% clear all,clc,d=25;
% l=ones(d,1)*5;u=Inf(d,1);
% Sig=0.5*eye(d)+.5*ones(d,d);
% est=mvnqmc(l,u,Sig,10^4) % output of our method
%
% % Executing Matlab's toolbox\stats\stats\mvncdf.m
% % with n=10^7 below is slow and inaccurate
% options=optimset('TolFun',0,'MaxFunEvals',10^7,'Display','iter');
% [prob,err]=mvncdf(l,u,zeros(d,1),Sig,options)
%
% Reference: Z. I. Botev (2015),
% "The Normal Law Under Linear Restrictions:
%  Simulation and Estimation via Minimax Tilting", submitted to JRSS(B)
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
for i=1:12 % repeat randomized QMC
    p(i)=mvnpr(ceil(n/12),L,l,u,mu);
end
est.prob=mean(p); % average of QMC estimates
est.relErr=std(p)/sqrt(12)/est.prob; % relative error
est.upbnd=exp(psy(x,L,l,u,mu)); % compute psi star
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function p=mvnpr(n,L,l,u,mu)
% computes P(l<X<u), where X is normal with
% 'Cov(X)=L*L' and zero mean vector;
% exponential tilting uses parameter 'mu';
% Quasi Monte Carlo uses 'n' samples;
d=length(l); % Initialization
mu(d)=0;
Z=zeros(d,n); % create array for variables
% QMC pointset
x = scramble(sobolset(d-1),'MatousekAffineOwen');
p=0;
for k=1:(d-1)
    % compute matrix multiplication L*Z
    col=L(k,1:k)*Z(1:k,:);
    % compute limits of truncation
    tl=l(k)-mu(k)-col;
    tu=u(k)-mu(k)-col;
    %simulate N(mu,1) conditional on [tl,tu] via QMC
    Z(k,:)=mu(k)+norminvp(x(1:n,k),tl,tu);
    % update likelihood ratio
    p = p+lnNpr(tl,tu)+.5*mu(k)^2-mu(k)*Z(k,:);
end
% deal with final Z(d) which need not be simulated
col=L(d,:)*Z;tl=l(d)-col;tu=u(d)-col;
p=p+lnNpr(tl,tu); % update LR corresponding to Z(d)
p=mean(exp(p)); % now switch back from logarithmic scale
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
function x=norminvp(p,l,u)
% computes with tail-precision the quantile function
% of the standard normal distribution at 0<=p<=1,
% and truncated to the interval [l,u];
% Inf values for vectors 'l' and 'u' accepted;

% %Example 1:
% %Suppose you wish to simulate a random variable
% %'Z' from the non-standard Gaussian N(m,s^2)
% %conditional on l<Z<u. First compute
% X=norminvp(rand,(l-m)/s,(u-m)/s); % and then set
% Z=m+s*X

% %Example 2:
% %Suppose you desire the median of Z~N(0,1), truncated to Z>9;
% norminvp(0.5,9,Inf) %our method
% % Matlab's toolbox\stats\stats\norminv.m fails
%  pl=normcdf(9);
%  norminv(0.5*(1-pl)+pl)

% Reference:
% Z. I. Botev (2015),
% "The Normal Law Under Linear Restrictions:
%  Simulation and Estimation via Minimax Tilting", submitted to JRSS(B)

l=l(:);u=u(:);p=p(:); % set to column vectors
if (length(l)~=length(u))|any(l>u)|any(p>1)|any(p<0)
    error('l, u, and p must be the same length with u>l and 0<=p<=1')
end
x=nan(size(l)); % allocate memory
I=(p==1);x(I)=u(I); % extreme values of quantile
J=(p==0);x(J)=l(J);
I=~(I|J); % cases for which 0<x<1
if  any(I)
    x(I)=cases(p(I),l(I),u(I));
end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function x=cases(p,l,u)
a=35; % treshold for switching between erfcinv and newton method
% case 1: a<l<u
I=l>a;
if any(I)
    tl=l(I); tu=u(I); tp=p(I); x(I)=normq(tp,tl,tu);
end
% case 2: l<u<-a
J=u<-a;
if any(J)
    tl=-u(J); tu=-l(J); tp=p(J); x(J)=-normq(1-tp,tl,tu);
end
% case 3: otherwise use erfcinv
I=~(I|J);
if  any(I)
    tl=l(I); tu=u(I); tp=p(I); x(I)=Phinv(tp,tl,tu);
end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function x=normq(p,l,u)
% computes with precision the quantile function
% of the standard normal distribution,
% truncated to the interval [l,u];
% normq assumes 0<l<u and 0<p<1
q=@(x)(erfcx(x/sqrt(2))/2); % define Q function
ql=q(l);qu=q(u);
l=l.^2;u=u.^2;
% set initial value for Newton iteration
x=sqrt(l-2*reallog(1+p.*expm1(l/2-u/2)));
% initialize Newton method
err=Inf;
while err>10^-10
    del=-q(x)+(1-p).*exp(.5*(x.^2-l)).*ql+p.*exp(.5*(x.^2-u)).*qu;
    x=x-del; % Newton's step
    err=max(abs(del)); % find the maximum error
end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function x=Phinv(p,l,u)
% computes with precision the quantile function
% of the standard normal distribution,
% truncated to the interval [l,u], using erfcinv.
I=u<0;l(I)=-l(I);u(I)=-u(I); % use symmetry of normal
pu=erfc(u/sqrt(2))/2;pl=erfc(l/sqrt(2))/2;
x=sqrt(2)*erfcinv(2*(pl+(pu-pl).*p));
x(I)=-x(I); % adjust sign due to symmetry
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
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [ L, l, u, perm ] = cholperm( Sig, l, u )
%  Computes permuted lower Cholesky factor L for Sig
%  by permuting integration limit vectors l and u.
%  Outputs perm, such that Sig(perm,perm)=L*L'.
%
% Reference: Gibson G. J., Glasbey C. A., Elston D. A. (1994),
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
    % find median value, z(j), of truncated normal:
    tl=(l(j)-L(j,1:j-1)*z(1:j-1))/L(j,j);
    tu=(u(j)-L(j,1:j-1)*z(1:j-1))/L(j,j);
    z(j)=norminvp(.5,tl,tu); % median value
end
end