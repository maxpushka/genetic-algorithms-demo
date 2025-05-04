FROM ubuntu:22.04 AS builder

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive

# Install base dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    ninja-build \
    git \
    wget \
    python3 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Clang
RUN apt-get update && apt-get install -y \
    clang \
    libc++-dev \
    libc++abi-dev \
    libstdc++-12-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Boost and other libraries
RUN apt-get update && apt-get install -y \
    libboost-all-dev \
    libtbb-dev \
    libeigen3-dev \
    libnlopt-dev \
    libpthread-stubs0-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Build and install PaGMO (based on CI script)
RUN git clone https://github.com/esa/pagmo2.git /tmp/pagmo2 && \
    cd /tmp/pagmo2 && \
    mkdir build && \
    cd build && \
    # Configure with explicit C++ standard and Clang
    cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        # -DCMAKE_CXX_STANDARD=17 \
        -DCMAKE_CXX_COMPILER=clang++ \
        -DCMAKE_C_COMPILER=clang \
        -DPAGMO_BUILD_TESTS=OFF \
        -DPAGMO_WITH_EIGEN3=ON \
        -DPAGMO_WITH_NLOPT=ON && \
    make -j$(nproc) && \
    make install && \
    ldconfig

# Copy source code
COPY . .

# Update CMakeLists.txt to use C++17 instead of C++23 and fix MacOS-specific flags
# Also update to use find_package for PaGMO since we installed it system-wide
#RUN sed -i 's/set(CMAKE_CXX_STANDARD 23)/set(CMAKE_CXX_STANDARD 17)/' CMakeLists.txt && \
Run sed -i 's/-isysroot \/Applications\/Xcode.app\/Contents\/Developer\/Platforms\/MacOSX.platform\/Developer\/SDKs\/MacOSX.sdk//' CMakeLists.txt && \
    sed -i 's/set(CMAKE_CXX_SCAN_FOR_MODULES 1)/# set(CMAKE_CXX_SCAN_FOR_MODULES 1)/' CMakeLists.txt && \
    sed -i 's/FetchContent_Declare(pagmo/#FetchContent_Declare(pagmo/' CMakeLists.txt && \
    sed -i 's/FetchContent_MakeAvailable(pagmo)/#FetchContent_MakeAvailable(pagmo)/' CMakeLists.txt && \
    sed -i 's/#find_package(Pagmo REQUIRED)/find_package(pagmo REQUIRED)/' CMakeLists.txt && \
    sed -i 's/#    Pagmo::pagmo/    pagmo/' CMakeLists.txt && \
    sed -i 's/     pagmo/#     pagmo/' CMakeLists.txt

# Configure and build
RUN mkdir -p build && \
    cd build && \
    cmake -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CXX_STANDARD=17 \
        -DCMAKE_CXX_COMPILER=clang++ \
        -DCMAKE_C_COMPILER=clang \
        .. && \
    cmake --build . -j $(nproc)

# Runtime stage
FROM ubuntu:22.04

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libboost-serialization-dev \
    libtbb-dev \
    libeigen3-dev \
    libnlopt-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create directories
WORKDIR /app

# Copy only the built executable and any necessary runtime files
COPY --from=builder /app/build/bin/ga /app/
COPY --from=builder /app/build/_deps/spdlog-build/libspdlog.a /usr/local/lib/
COPY --from=builder /usr/local/lib/libpagmo.so* /usr/local/lib/

# Set environment variables
ENV LD_LIBRARY_PATH=/usr/local/lib

# Create results and logs directories
RUN mkdir -p results logs

# Command to run the application
ENTRYPOINT ["/app/ga"]
