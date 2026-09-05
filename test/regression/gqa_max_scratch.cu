#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "cutlass/util/device_memory.h"
#include "tgv_gqa.cuh"

namespace {

void check(cudaError_t status) {
  if (status != cudaSuccess) {
    std::fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(status));
    std::exit(2);
  }
}

template<int Splits, int HeadDim = 64, int Stages = 3>
bool run_case(char const* name, int length, int repetitions, bool equal_logits = false,
              bool constant_values = false) {
  using Element = cutlass::bfloat16_t;
  constexpr int kv_heads = 8;
  constexpr int local_heads = 8;
  constexpr int q_heads = kv_heads * local_heads;
  std::vector<Element> q(q_heads * HeadDim, Element(0.0f));
  std::vector<Element> k(size_t(kv_heads) * length * HeadDim, Element(0.0f));
  std::vector<Element> v(k.size(), Element(1.0f));
  std::vector<Element> out(q.size());
  for (int head = 0; head < q_heads; ++head) q[head * HeadDim] = Element(1.0f);
  for (int head = 0; head < kv_heads; ++head) {
    for (int token = 0; token < length; ++token) {
      k[(size_t(head) * length + token) * HeadDim] =
          Element(equal_logits ? 0.0f : float((token / 128) % 2));
      if (!constant_values) {
        for (int feature = 0; feature < HeadDim; ++feature) {
          v[(size_t(head) * length + token) * HeadDim + feature] =
              Element(0.25f * float((token / 128 + head + feature) % 7 - 3));
        }
      }
    }
  }
  std::vector<float> expected(q.size());
  for (int head = 0; head < q_heads; ++head) {
    for (int feature = 0; feature < HeadDim; ++feature) {
      double numerator = 0.0;
      double denominator = 0.0;
      for (int token = 0; token < length; ++token) {
        // Independent scalar softmax oracle; Q/K produce base-2 logits 0 or 1.
        double weight = std::exp2(equal_logits ? 0.0 : double((token / 128) % 2));
        double value = constant_values ? 1.0 :
            0.25 * double((token / 128 + head / local_heads + feature) % 7 - 3);
        numerator += weight * value;
        denominator += weight;
      }
      expected[head * HeadDim + feature] = float(Element(float(numerator / denominator)));
    }
  }
  cutlass::device_memory::allocation<Element> dq(q.size()), dk(k.size()), dv(v.size()), dout(out.size());
  cutlass::device_memory::allocation<int> dlength(1);
  dq.copy_from_host(q.data()); dk.copy_from_host(k.data()); dv.copy_from_host(v.data());
  dlength.copy_from_host(&length);
  cudaStream_t stream{};
  check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  for (int iteration = 0; iteration < repetitions; ++iteration) {
    // Poison output each time so an unwritten element cannot pass using a prior launch.
    check(cudaMemsetAsync(dout.get(), 0xff, out.size() * sizeof(Element), stream));
    TGV::gqa::gqa_host<Element, Element, float,
        local_heads, 1, 128, HeadDim, Stages, Stages, Splits, Splits>(
        dk.get(), dq.get(), dv.get(), dout.get(), dlength.get(), nullptr,
        kv_heads, local_heads, 1, length, HeadDim, 1,
        length * HeadDim, HeadDim, 1, kv_heads * length * HeadDim,
        local_heads * HeadDim, HeadDim, q_heads * HeadDim, 1, q_heads * HeadDim,
        length * HeadDim, HeadDim, 1, kv_heads * length * HeadDim,
        local_heads * HeadDim, HeadDim, q_heads * HeadDim, 1, q_heads * HeadDim,
        1.0f / TGV::gqa::Log2_E, 0, false, -1, stream);
    check(cudaGetLastError());
    check(cudaStreamSynchronize(stream));
    dout.copy_to_host(out.data());
    int mismatches = 0;
    for (size_t i = 0; i < out.size(); ++i) {
      float actual = float(out[i]);
      // Nonconstant values expose wrong relative attention weights. The absolute
      // tolerance is one BF16 step at 0.5; the constant-value control stays exact.
      float tolerance = constant_values ? 0.0f : 1.0f / 256.0f;
      if (!std::isfinite(actual) || std::fabs(actual - expected[i]) > tolerance) {
        if (mismatches < 8)
          std::printf("%s iteration=%d head=%zu feature=%zu actual=%g expected=%g tolerance=%g\n",
                      name, iteration, i / HeadDim, i % HeadDim, actual, expected[i], tolerance);
        ++mismatches;
      }
    }
    if (mismatches) {
      std::printf("%s splits=%d KV=%d head_dim=%d stages=%d mismatches=%d FAIL\n",
                  name, Splits, length, HeadDim, Stages, mismatches);
      check(cudaStreamDestroy(stream));
      return false;
    }
  }
  check(cudaStreamDestroy(stream));
  std::printf("%s splits=%d KV=%d head_dim=%d stages=%d checked_launches=%d PASS\n",
              name, Splits, length, HeadDim, Stages, repetitions);
  return true;
}

} // namespace

int main(int argc, char** argv) {
  bool all = false;
  int repetitions = 100;
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--all") == 0) all = true;
    else if (std::strcmp(argv[i], "--repetitions") == 0 && i + 1 < argc) {
      char* end{};
      long count = std::strtol(argv[++i], &end, 10);
      if (*end || count < 1 || count > 10000) {
        std::fprintf(stderr, "repetitions must be in [1,10000]\n");
        return 2;
      }
      repetitions = int(count);
    }
    else {
      std::fprintf(stderr, "usage: %s [--all] [--repetitions 1..10000]\n", argv[0]);
      return 2;
    }
  }
  cudaDeviceProp prop{};
  check(cudaGetDeviceProperties(&prop, 0));
  std::printf("device=%s compute=%d.%d\n", prop.name, prop.major, prop.minor);
  if (prop.major != 10 || prop.minor != 0) {
    std::fprintf(stderr, "This sm_100a executable requires an SM100 GPU; no test executed.\n");
    return 2;
  }
  bool okay = run_case<1>("one-CTA-two-tiles", 256, repetitions);
  if (all) {
    okay &= run_case<1>("constant-value-control", 256, repetitions, false, true);
    okay &= run_case<1>("one-CTA-one-tile", 128, repetitions);
    okay &= run_case<1>("equal-logits", 256, repetitions, true);
    okay &= run_case<8>("shipped-default", 2048, repetitions);
    okay &= run_case<8>("one-tile-per-split", 1024, repetitions);
    okay &= run_case<1>("stage-wrap", 640, repetitions);
    okay &= run_case<1>("partial-tail", 656, repetitions);
    okay &= run_case<1,64,2>("two-stages", 640, repetitions);
    okay &= run_case<1,128>("head-dimension-128", 256, repetitions);
  }
  return okay ? 0 : 1;
}
