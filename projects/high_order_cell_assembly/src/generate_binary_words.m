%!/usr/bin/env python
% ----------------------------------------------------------------------
% Numenta Platform for Intelligent Computing (NuPIC)
% Copyright (C) 2016, Numenta, Inc.  Unless you have an agreement
% with Numenta, Inc., for a separate license for this software code, the
% following terms and conditions apply:
%
% This program is free software: you can redistribute it and/or modify
% it under the terms of the GNU Affero Public License version 3 as
% published by the Free Software Foundation.
%
% This program is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
% See the GNU Affero Public License for more details.
%
% You should have received a copy of the GNU Affero Public License
% along with this program.  If not, see http://www.gnu.org/licenses.
%
% http://numenta.org/licenses/
% ----------------------------------------------------------------------

function binaryWords = generate_binary_words(wordLength, maxActiveBits)
binaryWords = zeros(2^wordLength, wordLength);
for i = 1:2^10
    binstr = dec2bin(i-1);
    for j = 1:length(binstr)
        binaryWords(i,end-j+1) = str2num(binstr(end-j+1));
    end
end
binaryWords = binaryWords(sum(binaryWords, 2)<maxActiveBits, :);
numSpikesInWord = sum(binaryWords, 2);