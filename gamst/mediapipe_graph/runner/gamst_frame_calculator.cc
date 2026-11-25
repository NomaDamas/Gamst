// GamstFrameCalculator: one bundle per input timestamp, emitted once every
// declared output stream has settled at that timestamp.
//
// Why: graph output observers run on separate threads, so per-stream callbacks
// can reach stdout out of order. GAMST needs "all outputs of frame T are known"
// (including "no face at T"). This calculator waits (default input stream
// handler) until every input is settled at T, then emits one GamstFrame with the
// packets present at T. The runner observes only this stream, so stdout order is
// timestamp order and a frame's packets always precede its done line.
//
//   node {
//     calculator: "GamstFrameCalculator"
//     input_stream: "FRAME:throttled_input_video"   # present = the frame passed the FlowLimiter
//     input_stream: "STREAM:0:pose_landmarks"
//     input_stream: "STREAM:1:face_landmarks"
//     output_stream: "BUNDLE:gamst_frame"
//   }
//
// The output also feeds FlowLimiterCalculator's FINISHED back edge: a bundle at
// T releases every in-flight frame at or before T.

#include <string>
#include <utility>
#include <vector>

#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/gamst/gamst_frame.h"

namespace mediapipe {

namespace {
constexpr char kFrameTag[] = "FRAME";
constexpr char kStreamTag[] = "STREAM";
constexpr char kBundleTag[] = "BUNDLE";
}  // namespace

class GamstFrameCalculator : public CalculatorBase {
 public:
  static absl::Status GetContract(CalculatorContract* cc) {
    RET_CHECK(cc->Inputs().HasTag(kFrameTag)) << "FRAME input required";
    cc->Inputs().Tag(kFrameTag).SetAny();
    for (CollectionItemId id = cc->Inputs().BeginId(kStreamTag);
         id < cc->Inputs().EndId(kStreamTag); ++id) {
      cc->Inputs().Get(id).SetAny();
    }
    cc->Outputs().Tag(kBundleTag).Set<gamst::GamstFrame>();
    // Called for settled timestamps even when every input is empty, so a
    // dropped frame (no FRAME packet) still produces a bundle.
    cc->SetProcessTimestampBounds(true);
    return absl::OkStatus();
  }

  absl::Status Open(CalculatorContext* cc) override {
    cc->SetOffset(TimestampDiff(0));
    const auto& names = cc->Inputs().TagMap()->Names();
    for (CollectionItemId id = cc->Inputs().BeginId(kStreamTag);
         id < cc->Inputs().EndId(kStreamTag); ++id) {
      ids_.push_back(id);
      names_.push_back(names[id.value()]);
    }
    return absl::OkStatus();
  }

  absl::Status Process(CalculatorContext* cc) override {
    const Timestamp ts = cc->InputTimestamp();
    if (!ts.IsRangeValue() || (has_last_ && ts <= last_)) {
      return absl::OkStatus();
    }
    has_last_ = true;
    last_ = ts;
    auto bundle = absl::make_unique<gamst::GamstFrame>();
    bundle->processed = !cc->Inputs().Tag(kFrameTag).IsEmpty();
    for (size_t i = 0; i < ids_.size(); ++i) {
      const Packet& packet = cc->Inputs().Get(ids_[i]).Value();
      if (!packet.IsEmpty() && packet.Timestamp() == ts) {
        bundle->packets.emplace_back(names_[i], packet);
      }
    }
    cc->Outputs().Tag(kBundleTag).Add(bundle.release(), ts);
    return absl::OkStatus();
  }

 private:
  std::vector<CollectionItemId> ids_;
  std::vector<std::string> names_;
  bool has_last_ = false;
  Timestamp last_;
};

REGISTER_CALCULATOR(GamstFrameCalculator);

}  // namespace mediapipe
