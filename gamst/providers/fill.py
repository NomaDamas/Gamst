"""Fixed-view colored-liquid level estimates with an explicit volume calibration."""
import hashlib
import json
import math
import threading

from ..capabilities import CapabilitySpec, OutputField, SensorMetadata, identifier
from ..plugins import InputFrame


def _configuration(*, width, height, roi, hsv_lower, hsv_upper, volume_curve_ml,
                   calibration_id, container_id, coordinate_frame):
    for value in (calibration_id,container_id,coordinate_frame):
        identifier(value)
    if any(type(v) is not int or not 32 <= v <= 2048 for v in (width,height)) or width*height > 2097152:
        raise ValueError('image dimensions must be32..2048 and at most2Mi pixels')
    if type(roi) not in (list,tuple) or len(roi)!=4 or any(type(v) is not int for v in roi):
        raise ValueError('ROI must be integer left,top,right,bottom with exclusive upper bounds')
    left,top,right,bottom=roi
    if not 0<=left<right<=width or not 0<=top<bottom<=height or right-left<8 or bottom-top<16:
        raise ValueError('ROI must be inside the image and at least8x16')
    for hsv in (hsv_lower,hsv_upper):
        if type(hsv) not in (list,tuple) or len(hsv)!=3 or any(type(v) is not int or not 0<=v<=bound for v,bound in zip(hsv,(179,255,255))):
            raise ValueError('HSV requires integer hue0..179 and saturation/value0..255')
    if any(a>b for a,b in zip(hsv_lower,hsv_upper)) or hsv_lower[1]<32 or hsv_lower[2]<16:
        raise ValueError('ordered HSV bounds with minimum saturation32 and value16 required')
    if type(volume_curve_ml) not in (list,tuple) or not 2<=len(volume_curve_ml)<=32:
        raise ValueError('volume curve requires2..32 height-fraction/mL pairs')
    curve=[]
    for pair in volume_curve_ml:
        if type(pair) not in (list,tuple) or len(pair)!=2 or any(type(v) not in (int,float) or not math.isfinite(v) for v in pair):
            raise ValueError('volume curve must be finite number pairs')
        f,v=map(float,pair)
        if not 0<=f<=1 or not 0<=v<=1e6 or curve and (f<=curve[-1][0] or v<=curve[-1][1]):
            raise ValueError('height fractions and volumes must strictly increase within bounds')
        curve.append((f,v))
    if curve[0]!=(0.,0.) or curve[-1][0]!=1 or curve[-1][1]<=0:
        raise ValueError('curve must span empty(0,0) to full(1,positive mL)')
    return dict(width=width,height=height,roi=tuple(roi),hsv_lower=tuple(hsv_lower),hsv_upper=tuple(hsv_upper),
        volume_curve_ml=tuple(curve),calibration_id=calibration_id,container_id=container_id,coordinate_frame=coordinate_frame)


class CalibratedColorFill:
    def __init__(self, **config):
        self.config=_configuration(**config)
        self.spec=self.capability(**self.config)
        try:
            import cv2
            import numpy as np
        except ImportError as exc:
            raise RuntimeError('Install optional humansense-sdk[vision] in the selected runtime for calibrated fill') from exc
        self.cv,self.np=cv2,np
        self._lock=threading.RLock()
        self.closed=False

    @staticmethod
    def capability(**config):
        checked=_configuration(**config)
        version='1.'+hashlib.sha256(json.dumps(checked,sort_keys=True).encode()).hexdigest()
        return CapabilitySpec('vision.calibrated_fill','vision.container_fill','video','gamst.opencv_color_fill',version,
            (OutputField('fill','object'),),'frame',interpretation='estimate',freshness_ms=1000,
            devices=('cpu',),sensor_requirements=('fixed-view-colored-liquid','declared-height-volume-curve'),license='Apache-2.0')

    def reset(self):
        # Frame-independent: no retained images or temporal smoothing.
        pass

    def close(self):
        with self._lock:
            self.closed=True

    def _volume(self,fraction):
        curve=self.config['volume_curve_ml']
        for (f0,v0),(f1,v1) in zip(curve,curve[1:]):
            if fraction<=f1:
                return v0+(v1-v0)*(fraction-f0)/(f1-f0)
        return curve[-1][1]

    def process(self,frame):
        with self._lock:
            if self.closed:
                raise RuntimeError('fill provider is closed')
            np,cv,c=self.np,self.cv,self.config
            if not isinstance(frame,InputFrame) or frame.modality!='video' or not isinstance(frame.sensor,SensorMetadata):
                raise ValueError('video InputFrame with sensor metadata required')
            identifier(frame.subject_id)
            if (frame.sensor.calibration_id,frame.sensor.coordinate_frame)!=(c['calibration_id'],c['coordinate_frame']):
                raise ValueError('fill calibration and coordinate frame must match')
            if type(frame.timestamp_ms) is not int or not 0<=frame.timestamp_ms<=2**53-1:
                raise ValueError('capture timestamp must be nonnegative integer milliseconds')
            data=frame.data
            if type(data) is not np.ndarray or data.dtype!=np.uint8 or data.shape!=(c['height'],c['width'],3):
                raise ValueError('plain uint8 BGR image must match configured dimensions')
            left,top,right,bottom=c['roi'];height=bottom-top
            crop=data[top:bottom,left:right].copy()
            hsv=cv.cvtColor(crop,cv.COLOR_BGR2HSV)
            mask=cv.inRange(hsv,c['hsv_lower'],c['hsv_upper'])
            coverage=np.mean(mask!=0,axis=1)
            dry=coverage<=.2;wet=coverage>=.8
            dry_prefix=0
            while dry_prefix<height and dry[dry_prefix]:dry_prefix+=1
            wet_suffix=0
            while wet_suffix<height and wet[height-1-wet_suffix]:wet_suffix+=1
            transition=height-dry_prefix-wet_suffix
            result=dict(status='unknown',reason=None,height_fraction=None,volume_ml=None,
                meniscus_image_y_px=None,height_resolution_interval=None,volume_resolution_interval_ml=None,
                container_id=c['container_id'],calibration_id=c['calibration_id'],coordinate_frame=c['coordinate_frame'],
                roi_px=list(c['roi']),image_size_px=[c['width'],c['height']],
                hsv_lower=list(c['hsv_lower']),hsv_upper=list(c['hsv_upper']),
                volume_curve_ml=[list(p) for p in c['volume_curve_ml']],
                detected_color_fraction=float(np.mean(mask!=0)),dry_prefix_rows=dry_prefix,
                wet_suffix_rows=wet_suffix,transition_rows=transition,
                calibration_verified=False,liquid_identity_verified=False,
                empty_or_full_verified=False,opencv_version=cv.__version__,numpy_version=np.__version__)
            if not np.any(mask):result['reason']='no_color_evidence'
            elif dry_prefix<2:result['reason']='headspace_not_visible'
            elif wet_suffix<2:result['reason']='bottom_color_support_missing'
            elif transition>2:result['reason']='fragmented_or_sloped_profile'
            else:
                boundary=(dry_prefix+height-wet_suffix)/2
                fraction=(height-boundary)/height
                half_width=(transition/2+.5)/height
                lo,hi=max(0.,fraction-half_width),min(1.,fraction+half_width)
                result.update(status='estimated',reason='single_horizontal_color_boundary',
                    height_fraction=fraction,volume_ml=self._volume(fraction),
                    meniscus_image_y_px=top+boundary,height_resolution_interval=[lo,hi],
                    volume_resolution_interval_ml=[self._volume(lo),self._volume(hi)])
            return self.spec.validate({'fill':result})
