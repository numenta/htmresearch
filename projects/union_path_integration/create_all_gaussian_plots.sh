python plot_capacity_heatmap.py --inFile results/capacityHeatmap_gaussian.json --outFile capacityHeatmap_gaussian.pdf

python plot_capacity_trends.py --inFile results/capacityTrends_gaussian.json --outFile capacityTrends_gaussian.pdf

python plot_comparison_to_ideal.py --inFile results/comparisonToIdeal_gaussian.json --outFile comparisonToIdeal_gaussian.pdf --locationModuleWidth 26 30 40

python plot_narrowing.py --inFile results/narrowing_gaussian.json --outFile1 narrowing_singleTrials_gaussian.pdf --outFile2 narrowing_aggregated_gaussian.pdf --exampleObjectCount 50 --aggregateObjectCounts 40 50 60 70 --exampleObjectNumbers 28 27 30 --scrambleCells

python plot_summary.py --inFile results/convergenceSummary_gaussian.json --outFile summary_gaussian.pdf

python plot_feature_distributions.py --inFile results/featureDistributions_gaussian.json --outFile1 featureDistributions_gaussian_1.pdf --outFile2 featureDistributions_gaussian_2.pdf --xlim2 30
