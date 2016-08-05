@0xa3d7c406756614db;

# Next ID: 17
struct SparseNetProto {
  filterDim @0 :UInt32;
  outputDim @1 :UInt32;
  batchSize @2 :UInt32;
  losses @3 :List(IterationLossHistory);
  iteration @4 :UInt32;
  basis @5 :List(Float64);
  learningRate @6 :Float64;
  decayCycle @7: UInt32;
  learningRateDecay @8 :Float64;
  numLcaIterations @9 :UInt32;
  lcaLearningRate @10 :Float64;
  thresholdDecay @11 :Float64;
  minThreshold @12 :Float64;
  thresholdType @13 :Text;
  verbosity @14 :UInt8;
  showEvery @15 :UInt32;
  seed @16 :UInt32;

  # Next ID: 2
  struct IterationLossHistory {
    iteration @0 :UInt32;
    loss @1 :Float64;
  }
}