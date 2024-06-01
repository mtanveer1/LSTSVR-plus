function [train_Time,RMSE]=LSTSVR_PI_func(FunPara,train_data,test_data)


[~,no_col]=size(train_data);
%Y=train_data(:,no_col);
c1=FunPara.c1;
c2=FunPara.c2;
c3=FunPara.c3;
eps1=FunPara.eps1;
sigma=FunPara.sigma;
feature=round((no_col-1)/2);  %%%since the last feature is the label.
A=train_data(:,1:feature);
Y=train_data(:,end);
%train_data_NPI=[train_data_NPI, train_data(:,end)];
A_PI=train_data(:,feature+1:end-1);
test_data_NPI_no_Y=test_data(:,1:feature);
Y_test=test_data(:,end);
%test_data_NPI=[test_data_NPI_no_Y, test_data(:,end)];


[m1,~]=size(A);
e=ones(m1,1);
I=eye(m1+1);

%%%calculating the kernel matrix of A and A^t.
XXh = sum(A.^2,2)*ones(1,m1);
K = XXh+XXh'-2*(A*A');
K = exp(-K./(2*sigma^2));
G=[K e];

XXh = sum(A_PI.^2,2)*ones(1,m1);
K_PI = XXh+XXh'-2*(A_PI*A_PI');
K_PI = exp(-K_PI./(2*sigma^2));
G_PI=[K_PI e];
tic;
GGt=G*G';
GGT_PI= G_PI*G_PI';
deno=(GGt+(c1/c2)*GGT_PI+(1/c2)*GGt*GGT_PI);
deno=deno+10^-4*eye(size(deno,1));
temp=(c1*eps1*e-(c1*c3/c2)*GGT_PI*e+GGt*eps1*e-(c3/c2)*GGt*GGT_PI*e);
numer1=c1*Y+temp;
alpha=deno\numer1;
temp2=(c1*I+G'*G);
u1=temp2\(G'*(Y+alpha));
numer2=-c1*Y+temp;
beta=deno\numer2;
u2=temp2\(G'*(Y-beta));


train_Time=toc;

T=test_data_NPI_no_Y;
nt=size(T,1);


K_Test = - 2*T*A';
T = sum(T.^2,2)*ones(1,size(A,1));
A = sum(A.^2,2)*ones(1,nt);
K_Test = K_Test + T + A';
K_Test = exp(-K_Test./(2*sigma^2));


f1_func=K_Test*u1(1:end-1)+u1(end);
f2_func=K_Test*u2(1:end-1)+u2(end);

predicted_Y=(f1_func+f2_func)*0.5;


%% 
RMSE=rmse(predicted_Y,Y_test);     
end

