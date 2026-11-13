// gamst GBSC001 callback encoder. Adapter code Apache-2.0.
// Integrate with the explicitly licensed Bosch BSEC2 firmware; no BSEC code or
// configuration is included. See docs/bsec2-context.md for the callback wiring.
#ifndef GAMST_BSEC2_PACKET_H
#define GAMST_BSEC2_PACKET_H
#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <limits>

namespace gamst {
static const size_t bsec2_packet_capacity = 398;
inline void bsec2_le(uint8_t *out, uint64_t value, size_t bytes) {
    for (size_t i=0;i<bytes;++i) out[i]=static_cast<uint8_t>(value>>(8*i));
}
inline uint32_t bsec2_crc(const uint8_t *data,size_t size) {
    uint32_t value=0xffffffffU;
    for (size_t i=0;i<size;++i) {
        value^=data[i];
        for (unsigned bit=0;bit<8;++bit) value=(value>>1)^((value&1)?0xedb88320U:0U);
    }
    return value^0xffffffffU;
}
// Outputs is Bosch bsecOutputs; Version is bsec_version_t. Preserve native
// nanosecond timestamps, float32 values, IDs, accuracy and dimensions exactly.
template<class Outputs,class Version>
size_t encode_bsec2(uint8_t *out,size_t capacity,const Outputs &outputs,
                   const Version &version,uint32_t sequence,const uint8_t *config_sha256) {
    static_assert(sizeof(float)==4 && std::numeric_limits<float>::is_iec559,"IEEE754 binary32 required");
    const size_t count=outputs.nOutputs;
    const size_t size=49+15*count+4;
    if (!out || !config_sha256 || count<1 || count>23 ||
        count>sizeof(outputs.output)/sizeof(outputs.output[0]) || capacity<size || version.major!=2) return 0;
    for (size_t i=0;i<count;++i) {
        const auto &row=outputs.output[i];
        if (row.time_stamp<0 || !isfinite(row.signal) || row.accuracy>3 || row.signal_dimensions!=1) return 0;
    }
    const uint8_t magic[8]={'G','B','S','C','0','0','1',0};
    memcpy(out,magic,8);
    out[8]=version.major;out[9]=version.minor;out[10]=version.major_bugfix;out[11]=version.minor_bugfix;
    out[12]=static_cast<uint8_t>(count);bsec2_le(out+13,sequence,4);memcpy(out+17,config_sha256,32);
    for (size_t i=0;i<count;++i) {
        const auto &row=outputs.output[i];uint8_t *target=out+49+15*i;
        bsec2_le(target,static_cast<uint64_t>(row.time_stamp),8);
        uint32_t bits;memcpy(&bits,&row.signal,4);bsec2_le(target+8,bits,4);
        target[12]=row.sensor_id;target[13]=row.accuracy;target[14]=row.signal_dimensions;
    }
    bsec2_le(out+size-4,bsec2_crc(out,size-4),4);return size;
}
}
#endif
