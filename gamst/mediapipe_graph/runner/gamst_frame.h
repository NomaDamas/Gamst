#ifndef MEDIAPIPE_GAMST_GAMST_FRAME_H_
#define MEDIAPIPE_GAMST_GAMST_FRAME_H_

#include <string>
#include <utility>
#include <vector>

#include "mediapipe/framework/packet.h"

namespace mediapipe {
namespace gamst {

// Everything a graph produced at one input timestamp (see GamstFrameCalculator).
struct GamstFrame {
  // False when the FlowLimiter dropped the frame before any model ran.
  bool processed = false;
  // (graph stream name, packet) for each output present at this timestamp.
  std::vector<std::pair<std::string, Packet>> packets;
};

}  // namespace gamst
}  // namespace mediapipe

#endif  // MEDIAPIPE_GAMST_GAMST_FRAME_H_
