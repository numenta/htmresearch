@0xdaafba3afb142e21;

struct ClassifierResult {
  label @0 :UInt16;
  distance @1 :Float64;
}

struct RelationalMemoryLog {
  # Unix timestamp for measuring relative time per record and overall time
  ts @0 :Float64;
  # Active indices of sensory input
  sensation @1 :List(UInt16);
  # Active indices of predicted cells in L4
  predictedL4 @2 :List(UInt16);
  # Active indices of active cells in L4
  activeL4 @3 :List(UInt16);
  # List of active L6 cells by module. Each module is a list containing
  # the active cells per time step, with most recent first
  activeL6History @4 :List(List(List(UInt16)));
  # List of active L5 cells per module
  activeL5 @5 :List(List(UInt16));
  classifierResults @6 :List(ClassifierResult);
  # List of offsets for each module dimension
  motorDelta @7 :List(Int16);
  # List of predicted cells per L6 module
  predictedL6 @8 :List(List(UInt16));
}
