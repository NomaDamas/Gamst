import asyncio
import unittest
from humansense import HumanSense
from humansense.dashboard import Dashboard


class VisualizationTests(unittest.IsolatedAsyncioTestCase):
    async def test_dashboard_geometry_survives_context_compaction(self):
        async with HumanSense(consent=True) as human:
            points=[{'x':.5,'y':.4,'z':0,'visibility':.9,'presence':.9} for _ in range(33)]
            await human.push_signal('scene.poses',[{'frame_pose_index':0,'landmarks':points,'geometry':{'left_wrist_above_shoulder':True},'track_id':'body-1'}],'video')
            await human.push_signal('scene.faces',[{'frame_face_index':0,'landmarks':points,'blendshapes':{'jawOpen':.5}}],'video')
            data=await Dashboard(human,asyncio.get_running_loop()).dispatch({'op':'state'})
            for name in ('scene.faces','scene.poses'):
                self.assertEqual(len(data['guides'][name]['value'][0]['landmarks']),33)
                self.assertNotIn('landmarks',data['context']['observations'][name]['value'][0])
                self.assertTrue(data['guides'][name]['fresh'])
            self.assertEqual(data['context']['observations']['scene.poses']['value'][0]['landmark_count'],33)
