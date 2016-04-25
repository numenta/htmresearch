
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
NUPIC_CORE_LIB = $(NUPIC_CORE)/build/release/lib

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

CXX_FLAGS = -DNTA_ARCH_64 -DNTA_OS_DARWIN -DBOOST_NO_WREGEX -DHAVE_CONFIG_H -DNTA_ASM -DNTA_ASSERTIONS_ON \
	-DNTA_INTERNAL -DNTA_PLATFORM_darwin64 -DNUPIC2 \
	-isystem $(NUPIC_EXTERNAL_PLATFORM_INC) -isystem $(NUPIC_EXTERNAL_INC) \
	-isystem $(NUPIC_CORE)/src -I$(NUPIC_CORE_INC) -fPIC -std=c++11 -m64 \
	-stdlib=libc++ -fvisibility=hidden -Wall -Wreturn-type -Wunused \
	-Wno-unused-parameter -O3 -mpopcnt


LINK_FLAGS = $(NUPIC_CORE_LIB)/libnupic_core.a $(NUPIC_CORE_LIB)/libyaml.a \
		$(NUPIC_CORE_LIB)/libapr-1.a $(NUPIC_CORE_LIB)/libaprutil-1.a $(NUPIC_CORE_LIB)/libz.a \
		$(NUPIC_CORE_LIB)/libcapnpc.a $(NUPIC_CORE_LIB)/libcapnp.a $(NUPIC_CORE_LIB)/libkj.a $(NUPIC_CORE_LIB)/libnpymath.a


##########################################
#
# Our applications
#

sdr: sdr_calculations2.cpp
	clang++ -O4 -c sdr_calculations.cpp $(CXX_FLAGS)
	clang++ -O4 -c sdr_calculations2.cpp $(CXX_FLAGS)
	clang++ -O4 -c sdr_utilities.cpp $(CXX_FLAGS)
	clang++ -O4 -c binary_algorithms.cpp $(CXX_FLAGS)
	clang++ -O4 -c test_binary_matrix.cpp $(CXX_FLAGS)

	clang++ -Wl,-search_paths_first \
		-Wl,-headerpad_max_install_names -Wl,-u,_munmap -m64 \
		-stdlib=libc++ \
		-lc++abi -liconv -lsqlite3 -framework CoreServices -framework Accelerate \
		-L$(NUPIC_CORE_LIB) sdr_calculations.o sdr_utilities.o binary_algorithms.o \
		-o sdr_calculations $(LINK_FLAGS)


	clang++ -Wl,-search_paths_first \
		-Wl,-headerpad_max_install_names -Wl,-u,_munmap -m64 \
		-stdlib=libc++ \
		-lc++abi -liconv -lsqlite3 -framework CoreServices -framework Accelerate \
		-L$(NUPIC_CORE_LIB) sdr_calculations2.o sdr_utilities.o binary_algorithms.o \
		-o sdr_calculations2  $(LINK_FLAGS)

	clang++ -Wl,-search_paths_first \
		-Wl,-headerpad_max_install_names -Wl,-u,_munmap -m64 -stdlib=libc++ \
		-lc++abi -liconv -lsqlite3 -framework CoreServices -framework Accelerate \
		-L$(NUPIC_CORE_LIB) test_binary_matrix.o binary_algorithms.o \
		-o test_binary_matrix $(LINK_FLAGS)


clean: sdr_calculations2.cpp
	rm -f *.o
	rm -f sdr_calculations
	rm -f test_binary_matrix
