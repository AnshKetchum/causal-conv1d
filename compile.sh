nvcc -o benchmark benchmark.cpp \
  -I./csrc -I./causal_conv1d \
  csrc/*.cu causal_conv1d/*.cu \
  -std=c++17 \
  -O3 \
  --use_fast_math \
  -arch=sm_80  # adjust for your GPU architecture