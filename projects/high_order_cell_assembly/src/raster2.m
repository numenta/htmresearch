function h=raster2(times1, numFramesPerStim, dt, color)

timeInMovie = mod(times1, numFramesPerStim);
trial = floor(times1/numFramesPerStim)+1;

for i = 1:length(times1)
    set(line,'XData',[1 1]*timeInMovie(i)*dt,...
        'YData',[trial(i) trial(i)+0.78],...
        'Color',color, 'LineWidth', .8)
end
