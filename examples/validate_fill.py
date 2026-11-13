"""Actual OpenCV synthetic fill smoke; not physical container acceptance."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import time

from humansense import SensorMetadata
from humansense.plugins import InputFrame
from humansense.providers.fill import CalibratedColorFill


def run(args):
    import numpy as np
    config=dict(width=128,height=128,roi=[16,16,112,112],hsv_lower=[100,100,50],hsv_upper=[130,255,255],
        volume_curve_ml=[[0,0],[.5,100],[1,400]],calibration_id='synthetic',container_id='fixture',coordinate_frame='image')
    p=CalibratedColorFill(**config);cases=[]
    try:
        for seq,(fraction,expected) in enumerate(((.25,50),(.5,100),(.75,250),(0,None),(1,None))):
            image=np.full((128,128,3),255,np.uint8)
            image[112-round(96*fraction):112,16:112]=(255,0,0)
            f=InputFrame(image,'video',seq*100,sensor=SensorMetadata('fixture','camera',seq,'image',calibration_id='synthetic'))
            start=time.perf_counter();result=p.process(f);elapsed=(time.perf_counter()-start)*1000
            passed=result['fill']['volume_ml']==expected and result['fill']['status']==('unknown' if expected is None else 'estimated')
            cases.append(dict(height_fraction=fraction,expected_volume_ml=expected,
                input_bgr_sha256=hashlib.sha256(image.tobytes()).hexdigest(),passed=passed,processing_ms=elapsed,context=result))
    finally:p.close()
    report=dict(kind='synthetic_opencv_color_geometry_not_actual_liquid',config=config,
        capability=p.spec.describe(),python=platform.python_version(),platform=platform.platform(),
        cases=cases,passed=all(c['passed'] for c in cases),physical_calibration_verified=False,
        actual_liquid_validated=False,end_to_end_latency_validated=False)
    with Path(args.output).open('x') as target:json.dump(report,target,indent=2,allow_nan=False);target.write('\n')
    if not report['passed']:raise RuntimeError('synthetic fill oracle failed; inspect report')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True)
    run(parser.parse_args())
