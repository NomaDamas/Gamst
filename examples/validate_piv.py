"""Deterministic translated particle smoke; source mode is not installed-package acceptance."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import time
import types

from humansense import SensorMetadata
from humansense.plugins import InputFrame
from humansense.providers.piv import OpenPIVFlow

SOURCE_SHA256 = '09019b6486fbd28628d977eb86112bde97310c2381289ab145340e0811030601'
SOURCE_COMMIT = '048474c4db04c66156bca2b89632a76d778cddcb'


def run(args):
    import numpy as np
    import scipy
    from scipy.ndimage import gaussian_filter
    config=dict(meters_per_pixel=.001,calibration_id='synthetic',coordinate_frame='image')
    if args.reference_source:
        with Path(args.reference_source).open('rb') as source:payload=source.read(256*1024+1)
        if hashlib.sha256(payload).hexdigest()!=SOURCE_SHA256:
            raise ValueError('reference source must match the exact documented OpenPIV commit/hash')
        # Execute only verified bytes, avoiding a second mutable-path read.
        module=types.ModuleType('verified_openpiv_reference')
        exec(compile(payload,'verified_openpiv_pyprocess.py','exec'),module.__dict__)
        provider=OpenPIVFlow(OpenPIVFlow._scipy_backend(module),**config)
        mode='pinned_core_source_not_installed_package'
    else:
        provider=OpenPIVFlow.from_installed(**config)
        mode='installed_openpiv_0.26.1_synthetic_smoke'
    rng=np.random.default_rng(104);image=np.zeros((128,128),dtype=float)
    for _ in range(500):
        x,y=rng.integers(4,120,size=2);image[y,x]=rng.uniform(100,255)
    base=np.clip(gaussian_filter(image,.8)*4,0,255).astype(np.uint8)
    cases=[]
    try:
        for dx,dy,dt_ms in ((3,2,100),(-3,-2,100),(0,0,100),(3,2,200)):
            provider.reset()
            for seq,gray in enumerate((base,np.roll(base,(dy,dx),(0,1)))):
                f=InputFrame(np.repeat(gray[:,:,None],3,axis=2),'video',seq*dt_ms,
                    sensor=SensorMetadata('fixture','camera',seq,'image',calibration_id='synthetic'))
                start=time.perf_counter();result=provider.process(f);elapsed=(time.perf_counter()-start)*1000
            vectors=[v['velocity_m_s'] for v in result['flow']['vectors'] if v['valid']]
            expected=np.array([dx,dy])*.001/(dt_ms/1000)
            error=None if not vectors else float(np.max(np.abs(np.asarray(vectors)-expected)))
            passed=len(vectors)==9 and error is not None and error<.002
            cases.append(dict(shift_px=[dx,dy],dt_ms=dt_ms,expected_m_s=expected.tolist(),
                valid_vectors=len(vectors),maximum_component_error_m_s=error,passed=passed,
                processing_ms=elapsed,context=result))
    finally:provider.close()
    report=dict(kind=mode,source_commit=SOURCE_COMMIT,reference_source_sha256=SOURCE_SHA256 if args.reference_source else None,
        fixture='seed104_500_blurred_particles_128_square_wrapped_translation',fixture_sha256=hashlib.sha256(base.tobytes()).hexdigest(),
        runtime=dict(python=platform.python_version(),platform=platform.platform(),numpy=np.__version__,scipy=scipy.__version__),
        cases=cases,passed=all(case['passed'] for case in cases),physical_calibration_verified=False,
        actual_fluid_validated=False,end_to_end_latency_validated=False)
    with Path(args.output).open('x') as target:json.dump(report,target,indent=2,allow_nan=False);target.write('\n')
    if not report['passed']:raise RuntimeError('synthetic PIV oracle failed; inspect report')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-source',help='optional exact pinned pyprocess.py already present locally')
    parser.add_argument('--output',required=True)
    run(parser.parse_args())
