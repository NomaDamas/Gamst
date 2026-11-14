"""Actual OpenCV rendered-image mechanics, not real camera/sports accuracy."""
import unittest
try:
    import cv2 as cv
    import numpy as np
except ImportError:
    cv=np=None
from humansense.camera_similarity import BackgroundSimilarity


@unittest.skipIf(cv is None,'optional OpenCV/NumPy missing')
class Similarity(unittest.TestCase):
    def setUp(self):
        self.engine=BackgroundSimilarity([0,0,1,1])

    def texture(self,w=480,h=320):
        rng=np.random.default_rng(9)
        image=rng.integers(0,256,(h,w),dtype=np.uint8)
        image=cv.GaussianBlur(image,(5,5),.8)
        return cv.cvtColor(image,cv.COLOR_GRAY2BGR)

    def estimate(self,image,transform,boxes=(),other_boxes=()):
        h,w=image.shape[:2]
        second=cv.warpAffine(image,np.array(transform,dtype=float)[:2],(w,h))
        a,size=self.engine.prepare(image);b,_=self.engine.prepare(second)
        return self.engine.estimate(a,b,size,boxes,other_boxes)

    def test_actual_translation_and_independent_moving_foreground(self):
        image=self.texture();h,w=image.shape[:2]
        image[130:190,200:260]=255
        transform=np.array([[1,0,5],[0,1,-3],[0,0,1]],float)
        second=cv.warpAffine(image,transform[:2],(w,h))
        second[150:210,280:340]=255
        a,size=self.engine.prepare(image);b,_=self.engine.prepare(second)
        box=dict(x=190/w,y=120/h,width=90/w,height=90/h)
        other=dict(x=190/w,y=120/h,width=160/w,height=100/h)
        result=self.engine.estimate(a,b,size,[box],[other])
        self.assertEqual(result.diagnostics['status'],'estimated',result.diagnostics)
        np.testing.assert_allclose(result.previous_to_current,transform,atol=.12)
        self.assertFalse(result.diagnostics['camera_motion_verified'])

    def test_rotation_scale_and_original_resize_conjugation(self):
        for w,h in ((480,320),(1001,663)):
            image=self.texture(w,h)
            transform=np.eye(3);transform[:2]=cv.getRotationMatrix2D((w/2,h/2),2.,1.02)
            result=self.estimate(image,transform)
            self.assertIsNotNone(result.previous_to_current,result.diagnostics)
            actual=np.asarray(result.previous_to_current)
            points=np.array([[w*.2,h*.2,1],[w*.7,h*.6,1]])
            np.testing.assert_allclose(points@actual.T,points@transform.T,atol=.6)

    def test_blank_cut_foreground_dominance_and_large_motion_abstain(self):
        image=self.texture()
        for other in (np.zeros_like(image),255-image):
            a,size=self.engine.prepare(image);b,_=self.engine.prepare(other)
            result=self.engine.estimate(a,b,size,[],[])
            self.assertIsNone(result.previous_to_current,result.diagnostics)
        gray,size=self.engine.prepare(image)
        result=self.engine.estimate(gray,gray,size,[dict(x=0,y=0,width=1,height=1)],[])
        self.assertIsNone(result.previous_to_current)
        result=self.estimate(image,[[1,0,100],[0,1,0],[0,0,1]])
        self.assertIsNone(result.previous_to_current,result.diagnostics)
        blank=np.zeros_like(gray)
        self.assertIsNone(self.engine.estimate(blank,blank,size,[],[]).previous_to_current)

    def test_bounds_and_reflections_fail_without_clipping(self):
        for roi in ([0,0,True,1],[0,0,.1,.1],[0,0,float('nan'),1]):
            with self.assertRaises(ValueError):BackgroundSimilarity(roi)
        with self.assertRaises(ValueError):self.engine.prepare(np.zeros((640,640,3),np.float32))
        image=self.texture();gray,size=self.engine.prepare(image)
        with self.assertRaises(ValueError):self.engine.estimate(gray,gray,(999,999),[],[])
        self.assertIsNone(self.engine.geometry([[-1,0,480],[0,1,0],[0,0,1]],size))
        self.assertIsNone(self.engine.geometry([[1,1,0],[0,1,0],[0,0,1]],size))
        self.assertIsNone(self.engine.geometry([[1,0,0],[0,1,0],[0,1,1]],size))

    def test_source_pixels_remain_owned_by_caller(self):
        image=self.texture();gray,size=self.engine.prepare(image)
        original=gray.copy();image[:]=0
        np.testing.assert_array_equal(gray,original)
        before=gray.copy();result=self.engine.estimate(gray,gray,size,[],[])
        self.assertIsNotNone(result.previous_to_current)
        np.testing.assert_array_equal(gray,before)

    def test_concentrated_texture_and_extreme_aspect_ratio_abstain(self):
        image=self.texture();image[100:]=0;image[:,150:]=0
        result=self.estimate(image,[[1,0,2],[0,1,1],[0,0,1]])
        self.assertIsNone(result.previous_to_current,result.diagnostics)
        image=self.texture(4096,16);gray,size=self.engine.prepare(image)
        result=self.engine.estimate(gray,gray,size,[],[])
        self.assertIsNone(result.previous_to_current)
        self.assertEqual(result.diagnostics['reason'],'insufficient_analysis_resolution')
