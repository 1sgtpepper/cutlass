#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "cute/tensor.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/util/device_memory.h"

using namespace cute;

namespace {

void check(cudaError_t status) {
  if (status != cudaSuccess) {
    std::fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(status));
    std::exit(2);
  }
}

void check(cutlass::Status status) {
  if (status != cutlass::Status::kSuccess) {
    std::fprintf(stderr, "CUTLASS error: %s\n", cutlassGetStatusString(status));
    std::exit(2);
  }
}

template<int TileM, int GranM, int GranN, int ExtraCarveout>
struct Configuration {
  using FP8 = cutlass::float_e4m3_t;
  using Tile = Shape<Int<TileM>, _128, _128>;
  using Cluster = Shape<_1, _1, _1>;
  using Scale = cutlass::detail::Sm90BlockwiseScaleConfig<
      GranM, GranN, 128, GMMA::Major::MN, GMMA::Major::MN>;
  using Epilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      Tile, Cluster, cutlass::epilogue::collective::EpilogueTileAuto,
      float, float, float, cutlass::layout::RowMajor, 4,
      float, cutlass::layout::RowMajor, 4,
      cutlass::epilogue::TmaWarpSpecializedCooperative>::CollectiveOp;
  using Mainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      FP8, tuple<cutlass::layout::RowMajor, typename Scale::LayoutSFA>, 16,
      FP8, tuple<cutlass::layout::ColumnMajor, typename Scale::LayoutSFB>, 16,
      float, Tile, Cluster,
      cutlass::gemm::collective::StageCountAutoCarveout<sizeof(typename Epilogue::SharedStorage) + ExtraCarveout>,
      cutlass::gemm::KernelTmaWarpSpecializedCooperativeFP8Blockwise>::CollectiveOp;
  using Kernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int, int, int, int>, Mainloop, Epilogue>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;
};

// Observe the actual mainloop types without executing a GPU kernel.
// This diagnostic is separate from numerical confirmation.
int inspect_layouts() {
  using Config = Configuration<256, 128, 64, 0>;
  using Mainloop = Config::Mainloop;
  using SmemB = Mainloop::SmemLayoutSFB;
  std::vector<float> storage(cosize(SmemB{}), 1.0f);
  auto sSFB = make_tensor(make_smem_ptr(storage.data()), make_layout(
      make_shape(_256{}, get<0>(shape(SmemB{})),
                 make_shape(get<1>(shape(SmemB{})), get<2>(shape(SmemB{})))),
      make_stride(_0{}, get<0>(stride(SmemB{})),
                  make_stride(get<1>(stride(SmemB{})), get<2>(stride(SmemB{}))))));
  typename Mainloop::TiledMma mma;
  auto coordinates = make_identity_tensor(make_shape(_256{}, _128{}));
  int aliased_threads = 0;
  for (int tid = 0; tid < size(mma); ++tid) {
    auto shared = mma.get_slice(tid).partition_C(sSFB);
    auto reg = make_tensor_like<float>(shared(_, _, _, _0{}));
    auto split = tiled_divide(reg, tuple<_1, _2>{});
    auto first = split(0, _, _, _);
    auto second = split(1, _, _, _);
    bool aliases = true;
    for (int i = 0; i < size(first); ++i) {
      aliases &= (&first(i) == &second(i));
    }
    aliased_threads += aliases;
    fill(reg, 1.0f);
    auto unique = filter_zeros(first);
    for (int i = 0; i < size(unique); ++i) unique(i) *= 2.0f;
    if (tid == 0 || tid == 128) {
      auto coord = mma.get_slice(tid).partition_C(coordinates);
      auto coord_split = tiled_divide(coord, tuple<_1, _2>{});
      std::printf("thread=%d reg_layout=", tid);
      print(reg.layout());
      std::printf(" wave0_coord="); print(coord_split(0, 0, 0, 0));
      std::printf(" wave1_coord="); print(coord_split(1, 0, 0, 0));
      std::printf(" wave1_B_after_wave0_write=%g aliases=%d\n", second(0), aliases);
    }
  }
  std::printf("HOST_LAYOUT_OBSERVATION aliased_threads=%d/%d; GPU numerical execution not performed\n",
              aliased_threads, int(size(mma)));
  return aliased_threads == int(size(mma)) ? 0 : 1;
}

template<int TileM = 256, int GranM = 128, int GranN = 64, int ExtraCarveout = 0>
bool run_case(char const* name, int M, int N, int K, int L = 1,
              bool unit_a = false, bool padded = false,
              float alpha = 1.0f, float beta = 0.0f, bool benchmark = false) {
  using Config = Configuration<TileM, GranM, GranN, ExtraCarveout>;
  using Gemm = typename Config::Gemm;
  using Kernel = typename Config::Kernel;
  using FP8 = typename Config::FP8;
  using Scale = typename Config::Scale;
  int lda = K + (padded ? 32 : 0);
  int ldb = K + (padded ? 16 : 0);
  int ldd = N + (padded ? 8 : 0);
  typename Kernel::StrideA stride_a = make_stride(int64_t(lda), _1{}, int64_t(M) * lda);
  typename Kernel::StrideB stride_b = make_stride(int64_t(ldb), _1{}, int64_t(N) * ldb);
  typename Kernel::StrideC stride_c = make_stride(int64_t(ldd), _1{}, int64_t(M) * ldd);
  typename Kernel::StrideD stride_d = stride_c;
  auto problem = make_shape(M, N, K, L);
  auto layout_a = Scale::tile_atom_to_shape_SFA(problem);
  auto layout_b = Scale::tile_atom_to_shape_SFB(problem);
  int sm = (M + GranM - 1) / GranM;
  int sn = (N + GranN - 1) / GranN;
  int sk = (K + 127) / 128;
  std::vector<FP8> a(size_t(M) * lda * L, FP8(1.0f));
  std::vector<FP8> b(size_t(N) * ldb * L, FP8(1.0f));
  std::vector<float> c(size_t(M) * ldd * L, 3.0f);
  std::vector<float> d(c.size(), -12345.0f);
  std::vector<float> sa(size_t(sm) * sk * L);
  std::vector<float> sb(size_t(sn) * sk * L);
  for (int l = 0; l < L; ++l) {
    for (int k = 0; k < sk; ++k) {
      for (int m = 0; m < sm; ++m)
        sa[m + sm * (k + sk * l)] = unit_a ? 1.0f : float(2 << ((m + k + l) % 3));
      for (int n = 0; n < sn; ++n)
        sb[n + sn * (k + sk * l)] = float(1 << ((n + k + l) % 2));
    }
  }
  cutlass::device_memory::allocation<FP8> da(a.size()), db(b.size());
  cutlass::device_memory::allocation<float> dc(c.size()), dd(d.size()), dsa(sa.size()), dsb(sb.size());
  da.copy_from_host(a.data()); db.copy_from_host(b.data());
  dc.copy_from_host(c.data()); dd.copy_from_host(d.data());
  dsa.copy_from_host(sa.data()); dsb.copy_from_host(sb.data());
  typename Gemm::Arguments args{
      cutlass::gemm::GemmUniversalMode::kGemm, problem,
      {da.get(), stride_a, db.get(), stride_b, dsa.get(), layout_a, dsb.get(), layout_b},
      {{alpha, beta}, dc.get(), stride_c, dd.get(), stride_d}};
  Gemm gemm;
  check(gemm.can_implement(args));
  cutlass::device_memory::allocation<uint8_t> workspace(Gemm::get_workspace_size(args));
  check(gemm.initialize(args, workspace.get()));
  check(gemm.run());
  check(cudaDeviceSynchronize());
  dd.copy_to_host(d.data());
  int mismatches = 0;
  for (int l = 0; l < L; ++l) {
    for (int m = 0; m < M; ++m) {
      for (int n = 0; n < ldd; ++n) {
        // Independent block-product oracle: no CuTe scale layout or tensor-core reference.
        float expected = -12345.0f;
        if (n < N) {
          int sum = 0;
          for (int k = 0; k < sk; ++k) {
            int count = (K - 128 * k < 128) ? K - 128 * k : 128;
            int scale_a = unit_a ? 1 : 2 << (((m / GranM) + k + l) % 3);
            int scale_b = 1 << (((n / GranN) + k + l) % 2);
            sum += count * scale_a * scale_b;
          }
          expected = alpha * float(sum) + beta * 3.0f;
        }
        float actual = d[(size_t(l) * M + m) * ldd + n];
        if (actual != expected) {
          if (mismatches < 4)
            std::printf("%s l=%d m=%d n=%d actual=%g expected=%g\n", name, l, m, n, actual, expected);
          ++mismatches;
        }
      }
    }
  }
  std::printf("%s tile=%dx128x128 scale=%dx%dx128 stages=%d problem=%dx%dx%dx%d mismatches=%d %s\n",
      name, TileM, GranM, GranN, Config::Mainloop::DispatchPolicy::Stages,
      M, N, K, L, mismatches, mismatches ? "FAIL" : "PASS");
  if (benchmark) {
    // Capture only device work, keeping allocation and the host oracle outside timing.
    cudaStream_t stream;
    cudaGraph_t graph;
    cudaGraphExec_t executable;
    cudaEvent_t begin, end;
    check(cudaStreamCreate(&stream));
    check(cudaEventCreate(&begin));
    check(cudaEventCreate(&end));
    check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    for (int i = 0; i < 100; ++i) check(gemm.run(stream));
    check(cudaStreamEndCapture(stream, &graph));
    check(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
    check(cudaGraphLaunch(executable, stream));
    check(cudaStreamSynchronize(stream));
    for (int sample = 0; sample < 5; ++sample) {
      check(cudaEventRecord(begin, stream));
      check(cudaGraphLaunch(executable, stream));
      check(cudaEventRecord(end, stream));
      check(cudaEventSynchronize(end));
      float milliseconds;
      check(cudaEventElapsedTime(&milliseconds, begin, end));
      std::printf("TIMING %s sample=%d us=%g\n", name, sample, milliseconds * 10.0f);
    }
    check(cudaGraphExecDestroy(executable));
    check(cudaGraphDestroy(graph));
    check(cudaEventDestroy(begin));
    check(cudaEventDestroy(end));
    check(cudaStreamDestroy(stream));
  }
  return mismatches == 0;
}

} // namespace

int main(int argc, char** argv) {
  if (argc == 2 && std::strcmp(argv[1], "--layouts") == 0) return inspect_layouts();
  bool benchmark = argc == 2 && std::strcmp(argv[1], "--benchmark") == 0;
  bool all = argc == 2 && std::strcmp(argv[1], "--all") == 0;
  if (argc > 1 && !all && !benchmark) {
    std::fprintf(stderr, "usage: %s [--layouts|--all|--benchmark]\n", argv[0]);
    return 2;
  }
  cudaDeviceProp prop{};
  check(cudaGetDeviceProperties(&prop, 0));
  std::printf("device=%s compute=%d.%d\n", prop.name, prop.major, prop.minor);
  if (prop.major != 9 || prop.minor != 0) {
    std::fprintf(stderr, "This executable requires an SM90 Hopper GPU; no test executed.\n");
    return 2;
  }
  if (benchmark) {
    bool okay = run_case("drain", 1024, 1024, 128, 1, false, false, 1, 0, true);
    okay &= run_case("multi-K", 1024, 1024, 512, 1, false, false, 1, 0, true);
    okay &= run_case<128,128,64>("one-wave", 1024, 1024, 512, 1, false, false, 1, 0, true);
    okay &= run_case<256,128,128>("one-B-scale", 1024, 1024, 512, 1, false, false, 1, 0, true);
    return okay ? 0 : 1;
  }
  bool okay = run_case("minimal", 256, 128, 128);
  if (all) {
    okay &= run_case<128,128,64>("one-wave", 256, 128, 128);
    okay &= run_case<256,128,128>("one-B-scale", 256, 128, 128);
    okay &= run_case<256,64,64>("two-A-scales-per-wave", 256, 128, 128);
    okay &= run_case("unit-A-scales", 256, 128, 128, 1, true);
    okay &= run_case("steady-state", 256, 128, 256);
    okay &= run_case("stage-wrap", 256, 128, 512);
    okay &= run_case("partial-K", 256, 128, 656);
    okay &= run_case<256,128,64,65536>("extra-carveout", 256, 128, 656);
    okay &= run_case("partial-MN-strided-epilogue-batched", 192, 80, 144, 2, false, true, 0.5f, 2.0f);
  }
  return okay ? 0 : 1;
}
