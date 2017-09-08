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