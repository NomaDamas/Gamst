// Optional ORB-SLAM3 native bridge. Link only in an explicitly owned worker.
// Upstream ORB-SLAM3 and its linked artifacts retain their own GPL-3.0 license.
#include "System.h"
#include <atomic>
#include <cmath>
#include <cstdint>
#include <memory>
#include <mutex>
#include <stdexcept>

namespace {
std::atomic<bool> claimed{false};
struct Session {
    std::unique_ptr<ORB_SLAM3::System> system;
    std::mutex mutex;
    unsigned long calls=0;
    unsigned long limit;
    double last=-1;
    bool closed=false;
    Session(const char* vocabulary,const char* settings,unsigned long max_calls):limit(max_calls) {
        system.reset(new ORB_SLAM3::System(vocabulary,settings,ORB_SLAM3::System::MONOCULAR,false));
    }
};
}
extern "C" {
int gamst_orb_abi() { return 1; }
const char* gamst_orb_source() { return "4452a3c4ab75b1cde34e5505a36ec3f9edcdc4c4"; }
// One System construction per process, including failures. Upstream owns native
// threads and raw internal allocations: restart the worker to create another.
void* gamst_orb_open(const char* vocabulary,const char* settings,unsigned long max_calls) {
    if(!vocabulary || !settings || !max_calls || max_calls>3000 || claimed.exchange(true)) return nullptr;
    try { return new Session(vocabulary,settings,max_calls); } catch(...) { return nullptr; }
}
// Returns 0 on success, -1 on invalid/error, -2 at the lifetime frame limit.
// Output Tcw is populated ONLY for tracking state2 (OK); otherwise all zeros.
int gamst_orb_track(void* handle,const uint8_t* gray,int height,int width,double seconds,
                    float* tcw,int* state,int* changed) {
    if(!handle || !gray || !tcw || !state || !changed || height<32 || width<32 ||
       height>1080 || width>1920 || !std::isfinite(seconds) || seconds<0) return -1;
    auto* s=static_cast<Session*>(handle);
    std::lock_guard<std::mutex> guard(s->mutex);
    for(int i=0;i<16;++i) tcw[i]=0;
    *state=-1;*changed=0;
    if(s->closed || seconds<=s->last) return -1;
    if(s->calls>=s->limit) return -2;
    ++s->calls;s->last=seconds;
    try {
        cv::Mat image(height,width,CV_8UC1,const_cast<uint8_t*>(gray));
        auto pose=s->system->TrackMonocular(image,seconds);
        *state=s->system->GetTrackingState();
        *changed=s->system->MapChanged()?1:0;
        if(*state==2) {
            auto matrix=pose.matrix();
            for(int row=0;row<4;++row) for(int col=0;col<4;++col) tcw[row*4+col]=matrix(row,col);
        }
        return 0;
    } catch(...) { s->closed=true; return -1; }
}
int gamst_orb_reset(void* handle) {
    if(!handle) return -1;
    auto* s=static_cast<Session*>(handle);std::lock_guard<std::mutex> guard(s->mutex);
    if(s->closed) return -1;
    try { s->system->Reset();s->last=-1;return 0; } catch(...) { s->closed=true;return -1; }
}
// Call only once, after other ABI calls have completed. Python wrapper enforces
// this ownership. Upstream may block in Shutdown; parent worker deadline owns it.
int gamst_orb_close(void* handle) {
    if(!handle) return 0;
    auto* s=static_cast<Session*>(handle);
    try { s->system->Shutdown();delete s;return 0; } catch(...) { return -1; }
}
}
