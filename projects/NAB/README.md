# Description

Experimental [NAB](https://github.com/numenta/NAB) detectors.

Requires NAB to be installed.

~~~
cd ./NAB
python run.py -d null distalTimestamps1CellPerColumn --dataDir ~/nta/NAB/data --windowsFile ~/nta/NAB/labels/combined_windows.json --profilesFile ~/nta/NAB/config/profiles.json --detect --score --normalize
~~~

See latest results in [results/final_results.json](results/final_results.json)
