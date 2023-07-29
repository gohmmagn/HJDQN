Lu = 25;
Ll = 5;
delta = 0.3;
maxiter = 20000;

eta = ((-maxiter*log(1+delta))/(log(Ll)-log(Lu)));

t = 0:1:maxiter;
Ls = Lu*(1/(1+delta)).^(t/eta);

plot(t,Ls);