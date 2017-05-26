function setpath
% Add the current folder & subfolders to the top of the search path of MATLAB
% by faruto @ faruto's Studio~
% Email:patrick.lee@foxmai.com
% QQ : 516667408
% last modified 2010.09.21
pwdstr = pwd;
p = genpath(pwdstr);
addpath(p, '-begin');
savepath;
end