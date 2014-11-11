
# Assumes NUPIC_CORE is set to point to the root of the nupic.core repo.
NUPIC_CORE_EXTERNAL = $(NUPIC_CORE)/external

##########################################
# Include paths
#
# Three include paths: nupic core, platform specific external libraries, 
# platform independent external libraries
#
NUPIC_CORE_INC = $(NUPIC_CORE)/build/release/include
NUPIC_EXTERNAL_INC = $(NUPIC_CORE_EXTERNAL)/common/include
NUPIC_EXTERNAL_PLATFORM_INC = $(NUPIC_CORE_EXTERNAL)/darwin64/include

##########################################
#
# Library paths
#
NUPIC_CORE_LIB = $(NUPIC_CORE)/external/darwin64/lib

##########################################
#
# All compiler flags
#
#
# Note: compilation flags for your system may be different. To get the
# compilation flags for your system, look in:
# $nupic.core/build/scripts/CMakeFiles/testcpphtm.dir/flags.make
# and:
# $nupic.core/build/scripts/CMakeFiles/testcpphtm.dir/link.txt
#


CXX_FLAGS = -mmacosx-version-min=10.7  \
-I$(NUPIC_CORE_INC) -I$(NUPIC_EXTERNAL_INC) \
-I$(NUPIC_EXTERNAL_PLATFORM_INC) -isystem \
-std=c++11 -stdlib=libc++ -fPIC -DPIC \
-m64 -DHAVE_CONFIG_H -DNTA_INTERNAL -DNTA_PLATFORM_darwin64 \
-DNTA_COMPILER_CLANG -DBOOST_NO_WREGEX -DNUPIC2 -gfull -DNTA_ASSERTIONS_ON \
-fvisibility=hidden -Wall -Wno-c++11-extensions\
-Wreturn-type -Wunused -Wno-unused-parameter -O3 -pipe -DNTA_ASM




sdr: sdr_calculations.cpp
	clang++ -c sdr_calculations.cpp $(CXX_FLAGS)
	clang++    -mmacosx-version-min=10.7 -Wl,-search_paths_first \
    -Wl,-headerpad_max_install_names     -Wl,-u,_munmap -m64 \
    -stdlib=libc++ \
    -lc++abi -liconv -lsqlite3 -framework CoreServices -framework Accelerate \
    -L$(NUPIC_CORE_EXTERNAL)/darwin64/lib sdr_calculations.o -o sdr_calculations \
    $(NUPIC_CORE)/build/temp/lib/libnupic_core.a \
    $(NUPIC_CORE_EXTERNAL)/darwin64/lib/libyaml.a \
    $(NUPIC_CORE_EXTERNAL)/darwin64/lib/libapr-1.a \
    $(NUPIC_CORE_EXTERNAL)/darwin64/lib/libaprutil-1.a \
    $(NUPIC_CORE_EXTERNAL)/darwin64/lib/libz.a 
