% calculate the euclidean distance between each pair of neurons
neuronDistance = zeros(numNeuron, numNeuron);
for i = 1: numNeuron
    for j=i: numNeuron
        neuronDistance(i,j) = norm(centerLocation(i)-centerLocation(j));
    end
end 