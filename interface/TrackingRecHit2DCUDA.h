#ifndef CUDADataFormats_TrackingRecHit_interface_TrackingRecHit2DCUDA_h
#define CUDADataFormats_TrackingRecHit_interface_TrackingRecHit2DCUDA_h


//ask if this include is required
////

#include "CUDADataFormats/Common/interface/HeterogeneousSoA.h"
#include "RecoLocalTracker/SiPixelRecHits/interface/pixelCPEforGPU.h"


#include <cuda_runtime.h>

#include "CUDADataFormats/SiPixelCluster/interface/gpuClusteringConstants.h"
#include "HeterogeneousCore/CUDAUtilities/interface/HistoContainer.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCompat.h"
#include "Geometry/CommonTopologies/interface/SimplePixelTopology.h"
#include "CUDADataFormats/TrackingRecHit/interface/SiPixelHitStatus.h"

#include "DataFormats/SoATemplate/interface/SoALayout.h"
#include "CUDADataFormats/Common/interface/PortableDeviceCollection.h"


struct TrackingChargeStatus{
  uint32_t charge : 5;
  uint32_t status : 5;
  };

namespace pixelCPEforGPU {
  struct ParamsOnGPU;
}


GENERATE_SOA_LAYOUT(TrackingRecHit2DCUDATemplate,
                    SOA_COLUMN(uint32_t, nHits),
                    SOA_COLUMN(uint32_t, nMaxModules),
                    SOA_COLUMN(float, xLocal),
                    SOA_COLUMN(float, yLocal),
                    SOA_COLUMN(float, xerrLocal),
                    SOA_COLUMN(float, yerrLocal),
                    SOA_COLUMN(float, xGlobal),
                    SOA_COLUMN(float, yGlobal),
                    SOA_COLUMN(float, zGlobal),
                    SOA_COLUMN(float, rGlobal),
                    SOA_COLUMN(uint16_t, iphi),
                    SOA_COLUMN(int16_t, clusterSizeX),
                    SOA_COLUMN(int16_t, clusterSizeY),
                    SOA_COLUMN(uint16_t, detectorIndex),
                    SOA_COLUMN(uint32_t, hitsModuleStart),
                    SOA_COLUMN(uint32_t, hitsLayerStart),
                    SOA_COLUMN(TrackingChargeStatus, CandS))


  using TrackingRecHit2DCUDALayout = TrackingRecHit2DCUDATemplate<>;
  using TrackingRecHit2DCUDAView = TrackingRecHit2DCUDALayout::View;
  using TrackingRecHit2DCUDAConstView = TrackingRecHit2DCUDALayout::ConstView;

// While porting from previous code, we decorate the base PortableCollection. XXX/TODO: improve if possible...
class TrackingRecHit2DCUDA : public cms::cuda::PortableDeviceCollection<TrackingRecHit2DCUDALayout> {
public:
  using Status = SiPixelHitStatus;
  static_assert(sizeof(Status) == sizeof(uint8_t));
  using hindex_type = uint32_t;  // if above is <=2^3
  using PhiBinner = cms::cuda::
      HistoContainer<int16_t, 256, -1, 8 * sizeof(int16_t), hindex_type, pixelTopology::maxLayers>;  //28 for phase2 geometr
  using AverageGeometry = pixelTopology::AverageGeometry;

  template <typename>
  friend class TrackingRecHit2DHeterogeneous;
  friend class TrackingRecHit2DReduced;      //TrackingChargeStatus CandS;
  __device__ __forceinline__ void setChargeAndStatus(int i, uint32_t ich, Status is) {
     view()[i].CandS().charge = ich;
     view()[i].CandS().status = *reinterpret_cast<uint8_t*>(&is);
     }

   __device__ __forceinline__ PhiBinner& phiBinner() { return *m_phiBinner; }
   __device__ __forceinline__ PhiBinner const& phiBinner() const { return *m_phiBinner; }
   __device__ __forceinline__ pixelCPEforGPU::ParamsOnGPU const& cpeParams() const { return *m_cpeParams; }
   __device__ __forceinline__ AverageGeometry& averageGeometry() { return *m_averageGeometry; }
   __device__ __forceinline__ AverageGeometry const& averageGeometry() const { return *m_averageGeometry; }

  //auto nHits() const { return m_nHits; }
  //auto nMaxModules() const { return m_nMaxModules; }
  auto offsetBPIX2() const { return m_offsetBPIX2; }

  auto hitsModuleStart() const { return m_hitsModuleStart; }
  auto hitsLayerStart() { return m_hitsLayerStart; }
  //auto phiBinner() { return m_phiBinner; }
  auto phiBinnerStorage() { return m_phiBinnerStorage; }
  //auto iphi() { return m_iphi; }


  

  enum class Storage32 {
    kXLocal = 0,
    kYLocal = 1,
    kXerror = 2,
    kYerror = 3,
    kCharge = 4,
    kXGlobal = 5,
    kYGlobal = 6,
    kZGlobal = 7,
    kRGlobal = 8,
    kPhiStorage = 9,
    kLayers = 10
  };

  enum class Storage16 {
    kDetId = 0,
    kPhi = 1,
    kXSize = 2,
    kYSize = 3,
  };

  // using PhiBinner = TrackingRecHit2DSOAView::PhiBinner;

  TrackingRecHit2DCUDA() = default;

  explicit TrackingRecHit2DCUDA(
      uint32_t nHits,
      bool isPhase2,
      int32_t offsetBPIX2,
      pixelCPEforGPU::ParamsOnGPU const* cpeParams,
      uint32_t const* hitsModuleStart,
      cudaStream_t stream);
      //TrackingRecHit2DCUDA<cms::cudacompat::GPUTraits> const* input = nullptr);

  explicit TrackingRecHit2DCUDA(
      float* store32, uint16_t* store16, uint32_t* modules, int nHits, cudaStream_t stream = nullptr);
  ~TrackingRecHit2DCUDA() = default;

  TrackingRecHit2DCUDA(const TrackingRecHit2DCUDA&) = delete;
  TrackingRecHit2DCUDA& operator=(const TrackingRecHit2DCUDA&) = delete;
  TrackingRecHit2DCUDA(TrackingRecHit2DCUDA&&) = default;
  TrackingRecHit2DCUDA& operator=(TrackingRecHit2DCUDA&&) = default;


  // auto nHits() const { return m_nHits; }
  // auto nMaxModules() const { return m_nMaxModules; }
  // auto offsetBPIX2() const { return m_offsetBPIX2; }
  //
  // auto hitsModuleStart() const { return m_hitsModuleStart; }
  // auto hitsLayerStart() { return m_hitsLayerStart; }
  // auto phiBinner() { return m_phiBinner; }
  // auto phiBinnerStorage() { return m_phiBinnerStorage; }
  // auto iphi() { return m_iphi; }

  cms::cuda::host::unique_ptr<float[]> localCoordToHostAsync(cudaStream_t stream) const;

  cms::cuda::host::unique_ptr<uint32_t[]> hitsModuleStartToHostAsync(cudaStream_t stream) const;

  cms::cuda::host::unique_ptr<uint16_t[]> store16ToHostAsync(cudaStream_t stream) const;
  cms::cuda::host::unique_ptr<float[]> store32ToHostAsync(cudaStream_t stream) const;

  // needs specialization for Host
  void copyFromGPU(TrackingRecHit2DCUDA const* input, cudaStream_t stream);

private:
  uint32_t const* m_hitsModuleStart;
  uint32_t* m_hitsLayerStart;

   AverageGeometry* m_averageGeometry;              // owned by TrackingRecHit2DHeterogeneous
   pixelCPEforGPU::ParamsOnGPU const* m_cpeParams;
   PhiBinner* m_phiBinner;
   PhiBinner::index_type* m_phiBinnerStorage;
   int32_t m_offsetBPIX2;

};




//template <typename Traits>
TrackingRecHit2DCUDA::TrackingRecHit2DCUDA( 
    uint32_t nHits,
    bool isPhase2,
    int32_t offsetBPIX2,
    pixelCPEforGPU::ParamsOnGPU const* cpeParams,
    uint32_t const* hitsModuleStart,
    cudaStream_t stream)
   // TrackingRecHit2DHeterogeneous<cms::cudacompat::GPUTraits> const* input)
    : m_offsetBPIX2(offsetBPIX2){
  //auto view = Traits::template make_host_unique<TrackingRecHit2DCUDA>(stream);

  m_nMaxModules = isPhase2 ? phase2PixelTopology::numberOfModules : phase1PixelTopology::numberOfModules;

  view()[i].nHits() = nHits;
  view()[i].nMaxModules() = m_nMaxModules;
  //m_view = Traits::template make_unique<TrackingRecHit2DSOAView>(stream);  // leave it on host and pass it by value?
  m_AverageGeometryStore = nullptr;//Traits::template make_unique<TrackingRecHit2DSOAView::AverageGeometry>(stream);
  view->m_averageGeometry = m_AverageGeometryStore.get();
  view->m_cpeParams = cpeParams;
  view()[i].hitsModuleStart() = hitsModuleStart;

  // if empy do not bother
//  if (0 == nHits) {
//    if constexpr (std::is_same_v<Traits, cms::cudacompat::GPUTraits>) {
//      cms::cuda::copyAsync(m_view, view, stream);
//    } else {
//      m_view.reset(view.release());  // NOLINT: std::move() breaks CUDA version
//    }
//    return;
//  }

  // the single arrays are not 128 bit alligned...
  // the hits are actually accessed in order only in building
  // if ordering is relevant they may have to be stored phi-ordered by layer or so
  // this will break 1to1 correspondence with cluster and module locality
  // so unless proven VERY inefficient we keep it ordered as generated

.//   host copy is "reduced"  (to be reviewed at some point)
/*  if constexpr (std::is_same_v<Traits, cms::cudacompat::HostTraits>) {
    // it has to compile for ALL cases
    copyFromGPU(input, stream);
  } else {
    assert(input == nullptr);

    auto nL = isPhase2 ? phase2PixelTopology::numberOfLayers : phase1PixelTopology::numberOfLayers;

    m_store16 = Traits::template make_unique<uint16_t[]>(nHits * n16, stream);
    m_store32 = Traits::template make_unique<float[]>(nHits * n32 + nL + 1, stream);
    m_PhiBinnerStore = Traits::template make_unique<TrackingRecHit2DCUDA::PhiBinner>(stream);
  }*/

  static_assert(sizeof(TrackingRecHit2DCUDA::hindex_type) == sizeof(float));
  static_assert(sizeof(TrackingRecHit2DCUDA::hindex_type) == sizeof(TrackingRecHit2DCUDA::PhiBinner::index_type));

  auto get32 = [&](Storage32 i) { return m_store32.get() + static_cast<int>(i) * nHits; };

  // copy all the pointers
  m_phiBinner = view->m_phiBinner = m_PhiBinnerStore.get();
  m_phiBinnerStorage = view->m_phiBinnerStorage =
      reinterpret_cast<TrackingRecHit2DCUDA::PhiBinner::index_type*>(get32(Storage32::kPhiStorage));

  view()[i].xLocal() = get32(Storage32::kXLocal);
  view()[i].yLocal() = get32(Storage32::kYLocal);
  view()[i].xerrLocal() = get32(Storage32::kXerror);
  view()[i].yerrLocal() = get32(Storage32::kYerror);
  view()[i].CandS.charge = reinterpret_cast<uint32_t*>(get32(Storage32::kCharge));

  //if constexpr (!std::is_same_v<Traits, cms::cudacompat::HostTraits>) {
    assert(input == nullptr);
    view()[i].xGlobal = get32(Storage32::kXGlobal);
    view()[i].yGlobal = get32(Storage32::kYGlobal);
    view()[i].zGlobal = get32(Storage32::kZGlobal);
    view()[i].rGlobal = get32(Storage32::kRGlobal);

    auto get16 = [&](Storage16 i) { return m_store16.get() + static_cast<int>(i) * nHits; };
    m_iphi = view()[i].iphi = reinterpret_cast<int16_t*>(get16(Storage16::kPhi));

    view()[i].clusterSizeX = reinterpret_cast<int16_t*>(get16(Storage16::kXSize));
    view()[i].clusterSizeY = reinterpret_cast<int16_t*>(get16(Storage16::kYSize));
    view()[i].detectorIndex = get16(Storage16::kDetId);

    m_phiBinner = view->m_phiBinner = m_PhiBinnerStore.get();
    m_hitsLayerStart = view()[i].hitsLayerStart = reinterpret_cast<uint32_t*>(get32(Storage32::kLayers));
  //  }

  // transfer view
/*  if constexpr (std::is_same_v<Traits, cms::cudacompat::GPUTraits>) {
    cms::cuda::copyAsync(m_view, view, stream);
  } else {
    m_view.reset(view.release());  // NOLINT: std::move() breaks CUDA version
  }*/
}





//The Second Function

//this is intended to be used only for CPU SoA but doesn't hurt to have it for all cases
//template <typename Traits>
TrackingRecHit2DCUDA::TrackingRecHit2DCUDA(
    float* store32, uint16_t* store16, uint32_t* modules, int nHits, cudaStream_t stream)
    : m_nHits(nHits), m_hitsModuleStart(modules) {
  //auto view = Traits::template make_host_unique<TrackingRecHit2DCUDA>(stream);

  //m_view = Traits::template make_unique<TrackingRecHit2DSOAView>(stream);

  TrackingRecHit2DCUDA = nHits;

  //if (0 == nHits) {
  //  if constexpr (std::is_same_v<Traits, cms::cudacompat::GPUTraits>) {
  //    cms::cuda::copyAsync(m_view, view, stream);
  //  } else {
  //    m_view = std::move(view);
  //  }
  //  return;
  //}

  //m_store16 = Traits::template make_unique<uint16_t[]>(nHits * n16, stream);
  //m_store32 = Traits::template make_unique<float[]>(nHits * n32, stream);
  //m_PhiBinnerStore = Traits::template make_unique<TrackingRecHit2DCUDA::PhiBinner>(stream);
  //m_AverageGeometryStore = Traits::template make_unique<TrackingRecHit2DCUDA::AverageGeometry>(stream);
  m_AverageGeometryStore = nullptr;
  view->m_averageGeometry = m_AverageGeometryStore.get();
  view()[i].hitsModuleStart() = m_hitsModuleStart;

  //store transfer
  //if constexpr (std::is_same_v<Traits, cms::cudacompat::GPUTraits>) {
  //  cms::cuda::copyAsync(m_store16, store16, stream);
  //  cms::cuda::copyAsync(m_store32, store32, stream);
  //} else {
  //  std::copy(store32, store32 + nHits * n32, m_store32.get());  // want to copy it
  //  std::copy(store16, store16 + nHits * n16, m_store16.get());
  //}

  //getters
  auto get32 = [&](Storage32 i) { return m_store32.get() + static_cast<int>(i) * nHits; };
  auto get16 = [&](Storage16 i) { return m_store16.get() + static_cast<int>(i) * nHits; };

  //Store 32
  view()[i].xLocal() = get32(Storage32::kXLocal);
  view()[i].yLocal() = get32(Storage32::kYLocal);
  view()[i].xerrLocal() = get32(Storage32::kXerror);
  view()[i].yerrLocal() = get32(Storage32::kYerror);
  view()[i].CandS.charge = reinterpret_cast<uint32_t*>(get32(Storage32::kCharge));
  view()[i].xGlobal() = get32(Storage32::kXGlobal);
  view()[i].yGlobal() = get32(Storage32::kYGlobal);
  view()[i].zGlobal() = get32(Storage32::kZGlobal);
  view()[i].rGlobal() = get32(Storage32::kRGlobal);

  m_phiBinner = view->m_phiBinner = m_PhiBinnerStore.get();
  m_phiBinnerStorage = view->m_phiBinnerStorage =
      reinterpret_cast<TrackingRecHit2DCUDA::PhiBinner::index_type*>(get32(Storage32::kPhiStorage));

  //Store 16
  view()[i].detectorIndex() = get16(Storage16::kDetId);
  m_iphi = view()[i].iphi() = reinterpret_cast<int16_t*>(get16(Storage16::kPhi));
  view()[i].clusterSizeX() = reinterpret_cast<int16_t*>(get16(Storage16::kXSize));
  view()[i].clusterSizeY() = reinterpret_cast<int16_t*>(get16(Storage16::kYSize));

  // transfer view
  //if constexpr (std::is_same_v<Traits, cms::cudacompat::GPUTraits>) {
 //   cms::cuda::copyAsync(m_view, view, stream);
  //} else {
//m_view = std::move(view);
  //}
}

#endif  // CUDADataFormats_TrackingRecHit_interface_TrackingRecHit2DCUDA_h
