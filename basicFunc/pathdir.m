

function [dirnames1,num1]=pathdir(path1)
dirs=dir( path1);
dircell = struct2cell(dirs);
dirnames1 = dircell(1,:);
num1 = length(dirnames1);
end