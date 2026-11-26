// gamst_graph_runner: runs one MediaPipe CalculatorGraph as a GAMST Pipeline
// module, in its own process. GAMST owns the graph: config, side packets,
// branch switches, run start/reset/stop; MediaPipe owns scheduling inside it.
//
// Protocol v1 (docs/mediapipe-graph.md is the reference):
//
// stdin, one text line per command; FRAME is followed by its raw bytes:
//   SIDE <name> <bool|int|float|double|string> <value>   side packet for the next run
//   BRANCH <name> <0|1>                                  feed enable_<name> = 0/1 from the next frame
//   START                                                start a run with the current side packets
//   FRAME <capture_us> <width> <height>\n<width*height*3 BGR bytes>
//   RESET                                                close inputs, wait, start a fresh run
//   CLOSE                                                stop and exit 0
//
// stdout, one JSON object per line (native logs are moved to stderr first):
//   {"type":"hello","protocol":1,"mediapipe":"<rev>","gpu":false,"input":..,"branches":[..],"side_packets":[..]}
//   {"type":"ready","run":1,"branches":{"pose":true,...}}
//   {"type":"packet","run":1,"stream":"pose_landmarks","ts_us":..,"kind":"landmarks","value":[[x,y,z,vis,pres],..]}
//   {"type":"done","run":1,"ts_us":..,"processed":true}     every frame <= ts_us is finished
//   {"type":"ack","op":"branch","name":"face","enabled":false}
//   {"type":"error","stage":"frame","code":"INVALID_ARGUMENT","fatal":false}
//   {"type":"closed","run":1}
//
// Packet timestamps are the GAMST capture time in microseconds, unchanged: the
// graph never sees a wall clock of its own. Error text goes to stderr only
// (MediaPipe messages may quote paths); stdout carries status codes.

#include <unistd.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/status/status.h"
#include "absl/strings/match.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/classification.pb.h"
#include "mediapipe/framework/formats/detection.pb.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/formats/rect.pb.h"
#include "mediapipe/framework/port/file_helpers.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/gamst/gamst_frame.h"
#ifdef GAMST_WITH_GPU
#include "mediapipe/gpu/gpu_shared_data_internal.h"
#endif

#ifndef GAMST_MEDIAPIPE_REVISION
#define GAMST_MEDIAPIPE_REVISION "unknown"
#endif

ABSL_FLAG(std::string, graph, "", "Text CalculatorGraphConfig (.pbtxt) to run.");
ABSL_FLAG(std::string, input_stream, "input_video",
          "Graph input stream that receives SRGB ImageFrame packets.");
ABSL_FLAG(std::string, frame_stream, "gamst_frame",
          "Graph stream produced by GamstFrameCalculator (one bundle per frame).");
ABSL_FLAG(std::string, branch_prefix, "enable_",
          "Graph input streams with this prefix get one bool per frame.");
ABSL_FLAG(int, max_side, 8192, "Largest accepted frame width or height.");

namespace {

using mediapipe::CalculatorGraph;
using mediapipe::CalculatorGraphConfig;
using mediapipe::Packet;
using mediapipe::Timestamp;

constexpr int kProtocol = 1;
constexpr size_t kMaxString = 4096;

class Out {
 public:
  explicit Out(FILE* file) : file_(file) {}
  void Line(const std::string& text) {
    std::lock_guard<std::mutex> lock(mutex_);
    std::fwrite(text.data(), 1, text.size(), file_);
    std::fputc('\n', file_);
    std::fflush(file_);
  }

 private:
  FILE* file_;
  std::mutex mutex_;
};

std::string Quote(absl::string_view text) {
  std::string out = "\"";
  for (unsigned char c : text.substr(0, kMaxString)) {
    switch (c) {
      case '"': out += "\\\""; break;
      case '\\': out += "\\\\"; break;
      case '\n': out += "\\n"; break;
      case '\r': out += "\\r"; break;
      case '\t': out += "\\t"; break;
      default:
        if (c < 0x20 || c >= 0x7f) {
          char buffer[8];
          std::snprintf(buffer, sizeof(buffer), "\\u%04x", c);
          out += buffer;
        } else {
          out += static_cast<char>(c);
        }
    }
  }
  return out + "\"";
}

std::string Num(double value) {
  if (!std::isfinite(value)) return "null";
  char buffer[32];
  std::snprintf(buffer, sizeof(buffer), "%.7g", value);
  return buffer;
}

std::string Bool(bool value) { return value ? "true" : "false"; }

template <typename List>
std::string Points(const List& list) {
  std::string out = "[";
  for (int i = 0; i < list.landmark_size(); ++i) {
    const auto& p = list.landmark(i);
    if (i) out += ",";
    out += "[" + Num(p.x()) + "," + Num(p.y()) + "," + Num(p.z()) + "," +
           (p.has_visibility() ? Num(p.visibility()) : "null") + "," +
           (p.has_presence() ? Num(p.presence()) : "null") + "]";
  }
  return out + "]";
}

std::string DetectionJson(const mediapipe::Detection& d) {
  const auto& box = d.location_data().relative_bounding_box();
  return "{\"score\":" + (d.score_size() ? Num(d.score(0)) : "null") +
         ",\"label_id\":" + (d.label_id_size() ? std::to_string(d.label_id(0)) : "null") +
         ",\"label\":" + (d.label_size() ? Quote(d.label(0)) : "null") +
         ",\"box\":[" + Num(box.xmin()) + "," + Num(box.ymin()) + "," + Num(box.width()) + "," +
         Num(box.height()) + "]}";
}

std::string RectJson(const mediapipe::NormalizedRect& r) {
  return "{\"x_center\":" + Num(r.x_center()) + ",\"y_center\":" + Num(r.y_center()) +
         ",\"width\":" + Num(r.width()) + ",\"height\":" + Num(r.height()) +
         ",\"rotation\":" + Num(r.rotation()) + "}";
}

std::string ClassificationsJson(const mediapipe::ClassificationList& list) {
  std::string out = "[";
  for (int i = 0; i < list.classification_size(); ++i) {
    const auto& c = list.classification(i);
    if (i) out += ",";
    out += "{\"index\":" + std::to_string(c.index()) + ",\"score\":" + Num(c.score()) +
           ",\"label\":" + Quote(c.label()) + "}";
  }
  return out + "]";
}

template <typename T, typename F>
std::string Join(const std::vector<T>& items, F each) {
  std::string out = "[";
  for (size_t i = 0; i < items.size(); ++i) {
    if (i) out += ",";
    out += each(items[i]);
  }
  return out + "]";
}

// Known packet types become bounded JSON; images report only their size.
void Serialize(const Packet& p, std::string* kind, std::string* value) {
  using namespace mediapipe;  // NOLINT
  if (p.ValidateAsType<NormalizedLandmarkList>().ok()) {
    *kind = "landmarks";
    *value = Points(p.Get<NormalizedLandmarkList>());
  } else if (p.ValidateAsType<LandmarkList>().ok()) {
    *kind = "world_landmarks";
    *value = Points(p.Get<LandmarkList>());
  } else if (p.ValidateAsType<std::vector<NormalizedLandmarkList>>().ok()) {
    *kind = "landmarks_list";
    *value = Join(p.Get<std::vector<NormalizedLandmarkList>>(),
                  [](const NormalizedLandmarkList& l) { return Points(l); });
  } else if (p.ValidateAsType<std::vector<Detection>>().ok()) {
    *kind = "detections";
    *value = Join(p.Get<std::vector<Detection>>(), DetectionJson);
  } else if (p.ValidateAsType<Detection>().ok()) {
    *kind = "detections";
    *value = "[" + DetectionJson(p.Get<Detection>()) + "]";
  } else if (p.ValidateAsType<NormalizedRect>().ok()) {
    *kind = "rect";
    *value = RectJson(p.Get<NormalizedRect>());
  } else if (p.ValidateAsType<std::vector<NormalizedRect>>().ok()) {
    *kind = "rects";
    *value = Join(p.Get<std::vector<NormalizedRect>>(), RectJson);
  } else if (p.ValidateAsType<ClassificationList>().ok()) {
    *kind = "classifications";
    *value = ClassificationsJson(p.Get<ClassificationList>());
  } else if (p.ValidateAsType<std::vector<ClassificationList>>().ok()) {
    *kind = "classifications_list";
    *value = Join(p.Get<std::vector<ClassificationList>>(), ClassificationsJson);
  } else if (p.ValidateAsType<bool>().ok()) {
    *kind = "bool";
    *value = Bool(p.Get<bool>());
  } else if (p.ValidateAsType<int>().ok()) {
    *kind = "number";
    *value = std::to_string(p.Get<int>());
  } else if (p.ValidateAsType<float>().ok()) {
    *kind = "number";
    *value = Num(p.Get<float>());
  } else if (p.ValidateAsType<double>().ok()) {
    *kind = "number";
    *value = Num(p.Get<double>());
  } else if (p.ValidateAsType<std::string>().ok()) {
    *kind = "string";
    *value = Quote(p.Get<std::string>());
  } else if (p.ValidateAsType<ImageFrame>().ok()) {
    const auto& image = p.Get<ImageFrame>();  // never the pixels
    *kind = "image";
    *value = "{\"width\":" + std::to_string(image.Width()) +
             ",\"height\":" + std::to_string(image.Height()) + "}";
  } else {
    *kind = "opaque";
    *value = Quote(p.DebugTypeName());
  }
}

class Session {
 public:
  explicit Session(Out* out) : out_(out) {}

  absl::Status Load(const std::string& path) {
    std::string text;
    MP_RETURN_IF_ERROR(mediapipe::file::GetContents(path, &text));
    if (!mediapipe::ParseTextProto<CalculatorGraphConfig>(text, &config_)) {
      return absl::InvalidArgumentError("graph is not a text CalculatorGraphConfig");
    }
    input_ = absl::GetFlag(FLAGS_input_stream);
    frame_stream_ = absl::GetFlag(FLAGS_frame_stream);
    const std::string prefix = absl::GetFlag(FLAGS_branch_prefix);
    bool has_input = false;
    for (const std::string& name : config_.input_stream()) {
      if (name == input_) {
        has_input = true;
      } else if (absl::StartsWith(name, prefix)) {
        branches_[name.substr(prefix.size())] = true;
      }
    }
    if (!has_input) return absl::InvalidArgumentError("graph lacks the image input stream");
    prefix_ = prefix;
    return absl::OkStatus();
  }

  std::string Hello() const {
    std::string branches = "[", sides = "[";
    for (const auto& [name, on] : branches_) branches += (branches.size() > 1 ? "," : "") + Quote(name);
    for (const std::string& name : config_.input_side_packet())
      sides += (sides.size() > 1 ? "," : "") + Quote(name);
#ifdef GAMST_WITH_GPU
    const bool gpu = true;
#else
    const bool gpu = false;
#endif
    return "{\"type\":\"hello\",\"protocol\":" + std::to_string(kProtocol) +
           ",\"mediapipe\":" + Quote(GAMST_MEDIAPIPE_REVISION) + ",\"gpu\":" + Bool(gpu) +
           ",\"input\":" + Quote(input_) + ",\"frame_stream\":" + Quote(frame_stream_) +
           ",\"branches\":" + branches + "],\"side_packets\":" + sides + "]}";
  }

  bool running() const { return graph_ != nullptr; }

  absl::Status Start() {
    if (graph_) return absl::FailedPreconditionError("already running");
    auto graph = std::make_unique<CalculatorGraph>();
#ifdef GAMST_WITH_GPU
    MP_ASSIGN_OR_RETURN(auto gpu, mediapipe::GpuResources::Create());
    MP_RETURN_IF_ERROR(graph->SetGpuResources(std::move(gpu)));
#endif
    MP_RETURN_IF_ERROR(graph->Initialize(config_));
    const int run = run_ + 1;
    MP_RETURN_IF_ERROR(graph->ObserveOutputStream(
        frame_stream_, [this, run](const Packet& p) { return OnFrame(run, p); }));
    MP_RETURN_IF_ERROR(graph->StartRun(side_packets_));
    graph_ = std::move(graph);
    run_ = run;
    has_last_ = false;
    std::string branches = "{";
    for (const auto& [name, on] : branches_)
      branches += (branches.size() > 1 ? "," : "") + Quote(name) + ":" + Bool(on);
    out_->Line("{\"type\":\"ready\",\"run\":" + std::to_string(run_) + ",\"branches\":" + branches + "}}");
    return absl::OkStatus();
  }

  absl::Status Stop() {
    if (!graph_) return absl::OkStatus();
    std::unique_ptr<CalculatorGraph> graph = std::move(graph_);
    MP_RETURN_IF_ERROR(graph->CloseAllInputStreams());
    return graph->WaitUntilDone();  // the observer has flushed every bundle
  }

  // Returns a non-ok status only for graph failures (fatal); protocol misuse
  // that leaves the stream in sync is reported and skipped.
  absl::Status Frame(int64_t us, int width, int height, const std::string& bgr) {
    if (!graph_) {
      Error("frame", absl::FailedPreconditionError("START first"), false);
      return absl::OkStatus();
    }
    if (has_last_ && us <= last_us_) {
      Error("frame", absl::InvalidArgumentError("timestamps must increase"), false);
      return absl::OkStatus();
    }
    auto frame = std::make_unique<mediapipe::ImageFrame>(
        mediapipe::ImageFormat::SRGB, width, height,
        mediapipe::ImageFrame::kDefaultAlignmentBoundary);
    uint8_t* pixels = frame->MutablePixelData();
    const int step = frame->WidthStep();
    const auto* source = reinterpret_cast<const uint8_t*>(bgr.data());
    for (int y = 0; y < height; ++y) {
      uint8_t* row = pixels + static_cast<size_t>(y) * step;
      const uint8_t* in = source + static_cast<size_t>(y) * width * 3;
      for (int x = 0; x < width; ++x) {  // BGR -> RGB
        row[3 * x] = in[3 * x + 2];
        row[3 * x + 1] = in[3 * x + 1];
        row[3 * x + 2] = in[3 * x];
      }
    }
    const Timestamp ts(us);
    MP_RETURN_IF_ERROR(graph_->AddPacketToInputStream(
        input_, mediapipe::Adopt(frame.release()).At(ts)));
    for (const auto& [name, on] : branches_) {
      MP_RETURN_IF_ERROR(graph_->AddPacketToInputStream(
          prefix_ + name, mediapipe::MakePacket<bool>(on).At(ts)));
    }
    has_last_ = true;
    last_us_ = us;
    return absl::OkStatus();
  }

  void Side(const std::string& name, const std::string& type, const std::string& value) {
    Packet packet;
    bool ok = true;
    if (type == "bool") {
      ok = value == "0" || value == "1" || value == "true" || value == "false";
      packet = mediapipe::MakePacket<bool>(value == "1" || value == "true");
    } else if (type == "int") {
      int v = 0;
      ok = absl::SimpleAtoi(value, &v);
      packet = mediapipe::MakePacket<int>(v);
    } else if (type == "float") {
      float v = 0;
      ok = absl::SimpleAtof(value, &v);
      packet = mediapipe::MakePacket<float>(v);
    } else if (type == "double") {
      double v = 0;
      ok = absl::SimpleAtod(value, &v);
      packet = mediapipe::MakePacket<double>(v);
    } else if (type == "string") {
      packet = mediapipe::MakePacket<std::string>(value);
    } else {
      ok = false;
    }
    if (!ok) {
      Error("side", absl::InvalidArgumentError("bad side packet"), false);
      return;
    }
    side_packets_[name] = packet;
    out_->Line("{\"type\":\"ack\",\"op\":\"side\",\"name\":" + Quote(name) +
               ",\"applies\":\"" + (graph_ ? "next_run" : "this_run") + "\"}");
  }

  void Branch(const std::string& name, const std::string& value) {
    auto it = branches_.find(name);
    if (it == branches_.end() || (value != "0" && value != "1")) {
      Error("branch", absl::NotFoundError("unknown branch"), false);
      return;
    }
    it->second = value == "1";
    out_->Line("{\"type\":\"ack\",\"op\":\"branch\",\"name\":" + Quote(name) +
               ",\"enabled\":" + Bool(it->second) + "}");
  }

  void Error(const char* stage, const absl::Status& status, bool fatal) {
    std::cerr << "gamst_graph_runner " << stage << ": " << status << std::endl;
    out_->Line(std::string("{\"type\":\"error\",\"stage\":\"") + stage + "\",\"code\":" +
               Quote(absl::StatusCodeToString(status.code())) + ",\"fatal\":" + Bool(fatal) + "}");
  }

  int run() const { return run_; }

 private:
  absl::Status OnFrame(int run, const Packet& packet) {
    const auto& frame = packet.Get<mediapipe::gamst::GamstFrame>();
    const std::string ts = std::to_string(packet.Timestamp().Value());
    const std::string head = "{\"type\":\"packet\",\"run\":" + std::to_string(run) + ",\"ts_us\":" + ts;
    for (const auto& [stream, item] : frame.packets) {
      std::string kind, value;
      Serialize(item, &kind, &value);
      out_->Line(head + ",\"stream\":" + Quote(stream) + ",\"kind\":\"" + kind + "\",\"value\":" + value + "}");
    }
    out_->Line("{\"type\":\"done\",\"run\":" + std::to_string(run) + ",\"ts_us\":" + ts +
               ",\"processed\":" + Bool(frame.processed) + "}");
    return absl::OkStatus();
  }

  Out* out_;
  CalculatorGraphConfig config_;
  std::string input_, frame_stream_, prefix_;
  std::map<std::string, bool> branches_;
  std::map<std::string, Packet> side_packets_;
  std::unique_ptr<CalculatorGraph> graph_;
  int run_ = 0;
  bool has_last_ = false;
  int64_t last_us_ = 0;
};

}  // namespace

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);
  // Keep a private protocol descriptor, then send native stdout to stderr so
  // library prints can never corrupt protocol lines.
  const int protocol_fd = dup(STDOUT_FILENO);
  dup2(STDERR_FILENO, STDOUT_FILENO);
  FILE* protocol = fdopen(protocol_fd, "w");
  if (protocol == nullptr) return 3;
  Out out(protocol);
  Session session(&out);
  std::ios::sync_with_stdio(false);

  absl::Status loaded = session.Load(absl::GetFlag(FLAGS_graph));
  if (!loaded.ok()) {
    session.Error("load", loaded, true);
    return 2;
  }
  out.Line(session.Hello());

  const int max_side = absl::GetFlag(FLAGS_max_side);
  std::string line;
  while (std::getline(std::cin, line)) {
    if (line.empty()) continue;
    std::vector<std::string> parts = absl::StrSplit(line, absl::MaxSplits(' ', 3));
    const std::string& op = parts[0];
    absl::Status status;
    if (op == "FRAME" && parts.size() == 4) {
      int64_t us = 0;
      int width = 0, height = 0;
      if (!absl::SimpleAtoi(parts[1], &us) || !absl::SimpleAtoi(parts[2], &width) ||
          !absl::SimpleAtoi(parts[3], &height) || us < 0 || width < 1 || height < 1 ||
          width > max_side || height > max_side) {
        // The byte count is unknown, so the stream cannot be resynchronized.
        session.Error("frame", absl::InvalidArgumentError("bad frame header"), true);
        return 4;
      }
      std::string bgr(static_cast<size_t>(width) * height * 3, '\0');
      std::cin.read(&bgr[0], static_cast<std::streamsize>(bgr.size()));
      if (static_cast<size_t>(std::cin.gcount()) != bgr.size()) break;  // EOF mid-frame
      status = session.Frame(us, width, height, bgr);
      if (!status.ok()) {
        session.Error("frame", status, true);
        return 1;
      }
    } else if (op == "SIDE" && parts.size() == 4) {
      session.Side(parts[1], parts[2], parts[3]);
    } else if (op == "BRANCH" && parts.size() == 3) {
      session.Branch(parts[1], parts[2]);
    } else if (op == "START" && parts.size() == 1) {
      status = session.Start();
      if (!status.ok()) {
        session.Error("start", status, !session.running());
        if (!session.running()) return 1;
      }
    } else if (op == "RESET" && parts.size() == 1) {
      status = session.Stop();
      if (status.ok()) status = session.Start();
      if (!status.ok()) {
        session.Error("reset", status, true);
        return 1;
      }
    } else if (op == "CLOSE" && parts.size() == 1) {
      status = session.Stop();
      if (!status.ok()) session.Error("close", status, true);
      out.Line("{\"type\":\"closed\",\"run\":" + std::to_string(session.run()) + "}");
      return status.ok() ? 0 : 1;
    } else {
      session.Error("command", absl::InvalidArgumentError("unknown command"), true);
      return 4;
    }
  }
  absl::Status status = session.Stop();  // stdin closed: finish what is in flight
  out.Line("{\"type\":\"closed\",\"run\":" + std::to_string(session.run()) + "}");
  return status.ok() ? 0 : 1;
}
