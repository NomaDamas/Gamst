"""Bounded conditional background similarity from actual sparse pixel tracks.

This is an internal algorithm, not a camera-pose or physical-truth estimator.
The owned trajectory provider supplies source/lifecycle/evidence binding.
"""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class SimilarityResult:
    previous_to_current: tuple | None
    diagnostics: dict


def background_roi(value):
    if (type(value) not in (list,tuple) or len(value)!=4 or any(
            type(v) not in (int,float) or not math.isfinite(v) or not 0<=v<=1 for v in value)):
        raise ValueError('explicit normalized static-background rectangle required')
    a,b,c,d=value
    if c-a<.5 or d-b<.5 or (c-a)*(d-b)<.3:
        raise ValueError('background rectangle must cover sufficient image support')
    return tuple(float(v) for v in value)


class BackgroundSimilarity:
    """Fixed policy; admitted results remain conditional, uncalibrated estimates."""
    VERSION='opencv-lk-similarity-1'

    def __init__(self,roi):
        self.roi=background_roi(roi)
        import cv2
        import numpy as np
        self.cv,self.np=cv2,np

    def prepare(self,image):
        np,cv=self.np,self.cv
        if (type(image) is not np.ndarray or image.dtype!=np.uint8 or image.ndim!=3
                or image.shape[2]!=3 or min(image.shape[:2])<16
                or max(image.shape[:2])>4096 or image.nbytes>16*1024*1024):
            raise ValueError('bounded plain BGR uint8 image required')
        h,w=image.shape[:2];ratio=min(1.,640/max(h,w))
        size=(max(1,round(w*ratio)),max(1,round(h*ratio)))
        gray=cv.cvtColor(image,cv.COLOR_BGR2GRAY)
        if size!=(w,h):gray=cv.resize(gray,size,interpolation=cv.INTER_AREA)
        return gray.copy(),(w,h)

    def _mask(self,shape,boxes):
        np=self.np;h,w=shape
        if type(boxes) not in (list,tuple) or len(boxes)>8:
            raise ValueError('bounded complete foreground inventory required')
        from .tracking import normalize_bbox
        mask=np.zeros((h,w),np.uint8)
        a,b,c,d=self.roi
        mask[math.ceil(b*h):math.floor(d*h),math.ceil(a*w):math.floor(c*w)]=255
        for value in boxes:
            box=normalize_bbox(value)
            x,y,bw,bh=box['x']*w,box['y']*h,box['width']*w,box['height']*h
            px,py=.05*bw+3,.05*bh+3
            mask[max(0,math.floor(y-py)):min(h,math.ceil(y+bh+py)),
                 max(0,math.floor(x-px)):min(w,math.ceil(x+bw+px))]=0
        return mask

    def _support(self,points,shape):
        np,cv=self.np,self.cv;h,w=shape
        cells=np.floor(points/np.array([w,h])*4).astype(int)
        occupied=len({tuple(v) for v in cells})
        area=float(cv.contourArea(cv.convexHull(points.astype(np.float32))))/(w*h)
        return occupied,area

    def geometry(self,matrix,size,*,cumulative=False):
        """Check an original-pixel transform, with no determinant/finite guessing."""
        np,cv=self.np,self.cv;w,h=size
        m=np.asarray(matrix,dtype=np.float64)
        if m.shape!=(3,3) or not np.isfinite(m).all() or not np.allclose(m[2],[0,0,1],atol=1e-12,rtol=0):
            return None
        linear=m[:2,:2];det=float(np.linalg.det(linear))
        if det<=0:return None
        singular=np.linalg.svd(linear,compute_uv=False)
        if singular[-1]<=0 or singular[0]/singular[-1]>1.02:return None
        scale=math.sqrt(det);angle=math.degrees(math.atan2(linear[1,0],linear[0,0]))
        low,high,rotation,overlap=(.8,1.25,15.,.5) if cumulative else (.95,1.05,5.,.7)
        if not low<=scale<=high or abs(angle)>rotation:return None
        if not cumulative and float(np.linalg.norm(m[:2,2]))>.1*math.hypot(w,h):return None
        corners=np.array([[0,0],[w,0],[w,h],[0,h]],np.float64)
        warped=corners@linear.T+m[:2,2]
        if not np.isfinite(warped).all() or np.max(np.abs(warped))>10*max(w,h):return None
        area,_=cv.intersectConvexConvex(corners.astype(np.float32),warped.astype(np.float32))
        ratio=float(area)/(w*h)
        if ratio<overlap:return None
        inverse=np.linalg.inv(m)
        if not np.isfinite(inverse).all():return None
        return dict(scale=scale,rotation_degrees=angle,overlap_fraction=min(1.,ratio))

    def estimate(self,previous,current,original_size,previous_boxes,current_boxes):
        np,cv=self.np,self.cv
        if (type(previous) is not np.ndarray or type(current) is not np.ndarray
                or previous.dtype!=np.uint8 or current.dtype!=np.uint8 or previous.ndim!=2
                or current.shape!=previous.shape or min(previous.shape)<1 or max(previous.shape)>640):
            raise ValueError('matching bounded prepared grayscale images required')
        if (type(original_size) not in (tuple,list) or len(original_size)!=2 or any(
                type(v) is not int or not 16<=v<=4096 for v in original_size)):
            raise ValueError('explicit original image dimensions required')
        h,w=previous.shape;ow,oh=original_size
        ratio=min(1.,640/max(oh,ow))
        if (w,h)!=(max(1,round(ow*ratio)),max(1,round(oh*ratio))):
            raise ValueError('prepared dimensions do not match original resize policy')
        info=dict(method=self.VERSION,opencv_version=cv.__version__,status='unavailable',
            reason='insufficient_background',conditional_estimate=True,
            camera_motion_verified=False,analysis_size=[w,h],original_size=[ow,oh])
        def fail(reason):
            return SimilarityResult(None,dict(info,reason=reason))
        if min(previous.shape)<16:return fail('insufficient_analysis_resolution')
        pm=self._mask(previous.shape,previous_boxes);cm=self._mask(current.shape,current_boxes)
        if min(float(np.mean(pm>0)),float(np.mean(cm>0)))<.3:return fail('insufficient_background')
        points=cv.goodFeaturesToTrack(previous,maxCorners=256,qualityLevel=.01,minDistance=5,
                                      mask=pm,blockSize=7)
        if points is None or len(points)<24:return fail('insufficient_texture')
        options=dict(winSize=(21,21),maxLevel=3,
                     criteria=(cv.TERM_CRITERIA_EPS|cv.TERM_CRITERIA_COUNT,30,.01))
        ends,status,_=cv.calcOpticalFlowPyrLK(previous,current,points,None,**options)
        if ends is None or status is None:return fail('forward_tracking_failed')
        starts=points.reshape(-1,2);ends=ends.reshape(-1,2)
        valid=status.reshape(-1).astype(bool)&np.isfinite(ends).all(axis=1)
        valid &= (ends[:,0]>=0)&(ends[:,0]<w)&(ends[:,1]>=0)&(ends[:,1]<h)
        starts,ends=starts[valid],ends[valid]
        if len(ends)<24:return fail('insufficient_correspondences')
        xy=np.floor(ends).astype(int);keep=cm[xy[:,1],xy[:,0]]!=0
        starts,ends=starts[keep],ends[keep]
        if len(ends)<24:return fail('foreground_or_support_loss')
        back,status,_=cv.calcOpticalFlowPyrLK(current,previous,ends.reshape(-1,1,2),None,**options)
        if back is None or status is None:return fail('backward_tracking_failed')
        back=back.reshape(-1,2)
        valid=status.reshape(-1).astype(bool)&np.isfinite(back).all(axis=1)
        valid &= np.linalg.norm(back-starts,axis=1)<=1.
        starts,ends=starts[valid],ends[valid]
        info['correspondence_count']=len(starts)
        if len(starts)<24:return fail('forward_backward_disagreement')
        affine,inliers=cv.estimateAffinePartial2D(starts,ends,method=cv.RANSAC,
            ransacReprojThreshold=2.,maxIters=1000,confidence=.99,refineIters=10)
        if affine is None or inliers is None or not np.isfinite(affine).all():return fail('fit_failed')
        keep=inliers.reshape(-1).astype(bool);count=int(keep.sum())
        info.update(inlier_count=count,inlier_fraction=count/len(starts))
        if count<24 or count/len(starts)<.75:return fail('insufficient_inliers')
        residual=np.linalg.norm(starts[keep]@affine[:,:2].T+affine[:,2]-ends[keep],axis=1)
        median,p95=float(np.median(residual)),float(np.percentile(residual,95))
        support=[self._support(p,previous.shape) for p in (starts[keep],ends[keep])]
        info.update(residual_median_analysis_px=median,residual_p95_analysis_px=p95,
            occupied_tiles=min(v[0] for v in support),hull_area_fraction=min(v[1] for v in support))
        if median>1 or p95>2:return fail('excessive_residual')
        if info['occupied_tiles']<6 or info['hull_area_fraction']<.25:return fail('poor_spatial_support')
        analysis=np.eye(3);analysis[:2]=affine
        resize=np.diag([w/ow,h/oh,1.])
        original=np.linalg.inv(resize)@analysis@resize
        geometry=self.geometry(original,original_size)
        if geometry is None:return fail('unsupported_transform')
        info.update(geometry,status='estimated',reason=None,transform_direction='previous_to_current_pixels')
        return SimilarityResult(tuple(tuple(float(v) for v in row) for row in original),info)
