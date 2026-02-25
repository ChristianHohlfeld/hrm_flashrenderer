.PHONY: all build build-hrm build-flash test test-hrm test-flash

all: build

build: build-hrm build-flash

build-hrm:
	cmake -S hrm_core -B hrm_core/build -DCMAKE_BUILD_TYPE=Release
	cmake --build hrm_core/build -j

build-flash:
	TORCH_CUDA_ARCH_LIST=7.5 python3 setup.py install --user --break-system-packages

test: test-hrm test-flash

test-hrm: build-hrm
	ctest --test-dir hrm_core/build

test-flash: build-flash
	flash-kernel-test
	flash-append-test

