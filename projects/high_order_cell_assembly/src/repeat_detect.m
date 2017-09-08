function [Nreps,locs] = repeat_detect( spkraster )
%
% Usage: [Nreps,locs] = repeat_detect( spkraster )
%
% locs is a Nreps x 2 matrix with the beginning and end
% of each repeat (automatically excluding -1s)

if spkraster(end) >= 0
  spkraster(end+1) = -1;
end

% Check for empty repeats
negs = find(spkraster < 0);
if length(negs) == 1
  if length(find(diff(spkraster) < 0)) > 1
    % Then not using -1s
    negs = find(diff(spkraster) < 0);
  end
end

Nreps = length(negs);

locs = ones(Nreps,2);
startspk = 1;
for i = 1:Nreps
  locs(i,1) = startspk;
  if spkraster(negs(i)) < 0
    locs(i,2) = negs(i)-1;
  else
    locs(i,2) = negs(i);
  end
  startspk = negs(i)+1;
end
