#ifndef CUDADataFormats_TrackingRecHit_interface_TrackingRecHit2DSOAView_h
#define CUDADataFormats_TrackingRecHit_interface_TrackingRecHit2DSOAView_h

#include <cuda_runtime.h>

#include "CUDADataFormats/SiPixelCluster/interface/gpuClusteringConstants.h"
#include "HeterogeneousCore/CUDAUtilities/interface/HistoContainer.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCompat.h"
#include "Geometry/CommonTopologies/interface/SimplePixelTopology.h"
#include "CUDADataFormats/TrackingRecHit/interface/SiPixelHitStatus.h"
#include "DataFormats/SoATemplate/interface/SoALayout.h"
//#include "DataFormats/Portable/interface/PortableCUDADeviceCollection.h"
#include "CUDADataFormats/Common/interface/PortableDeviceCollection.h"


 struct TrackingChargeStatus{
  uint32_t charge : 5; 
  uint32_t status : 5;
  };

GENERATE_SOA_LAYOUT(TrackingRecHit2DSOAViewTemplate,
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

namespace pixelCPEforGPU {
  struct ParamsOnGPU;
}
  
 
  class TrackingRecHit2DSOAView : public PortableDeviceCollection<TrackingRecHit2DSOAViewTemplate<>> {
  public:
      using Status = SiPixelHitStatus;
      static_assert(sizeof(Status) == sizeof(uint8_t));

      using hindex_type = uint32_t;  // if above is <=2^32

      using PhiBinner = cms::cuda::
          HistoContainer<int16_t, 256, -1, 8 * sizeof(int16_t), hindex_type, pixelTopology::maxLayers>;  //28 for phase2 geometry

      using AverageGeometry = pixelTopology::AverageGeometry;

      template <typename>
      friend class TrackingRecHit2DHeterogeneous;
      friend class TrackingRecHit2DReduced;

      TrackingRecHit2DSOAView() = default;
      explicit TrackingRecHit2DSOAView(size_t maxModules, cudaStream_t stream)
      : PortableDeviceCollection<TrackingRecHit2DSOAViewTemplate<>>(maxModules + 1, stream) {}

      //TrackingChargeStatus CandS;
   __device__ __forceinline__ void setChargeAndStatus(int i, uint32_t ich, Status is) {
      view()[i].CandS.charge = ich
      view()[i].CandS.status = *reinterpret_cast<uint8_t*>(&is);
      }

    __device__ __forceinline__ PhiBinner& phiBinner() { return *m_phiBinner; }
    __device__ __forceinline__ PhiBinner const& phiBinner() const { return *m_phiBinner; }
    __device__ __forceinline__ pixelCPEforGPU::ParamsOnGPU const& cpeParams() const { return *m_cpeParams; }
    __device__ __forceinline__ AverageGeometry& averageGeometry() { return *m_averageGeometry; }
    __device__ __forceinline__ AverageGeometry const& averageGeometry() const { return *m_averageGeometry; }

  private:
   

    // supporting objects
    // m_averageGeometry is corrected for beam spot, not sure where to host it otherwise
    AverageGeometry* m_averageGeometry;              // owned by TrackingRecHit2DHeterogeneous
    pixelCPEforGPU::ParamsOnGPU const* m_cpeParams;  // forwarded from setup, NOT owned
    uint32_t const* m_hitsModuleStart;               // forwarded from clusters

    uint32_t* m_hitsLayerStart;

    PhiBinner* m_phiBinner;
    PhiBinner::index_type* m_phiBinnerStorage;

};

#endif