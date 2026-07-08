"""
Unit tests for bbox_utils.py.

Formalizes the IoU checks we already verified by hand: partial
overlap, identical boxes, and no overlap at all.
"""
import unittest 

from src.bbox_utils import BoxMath

class TestBoxMath(unittest.TestCase):
    """Tests for the BoxMath class -- area, intersection, and IoU."""
    
    def test_area_of_simple_box(self):
        """A 200x200 box should have area 40000."""
        box = [100,100,300,300]
        result = BoxMath.area(box)
        self.assertEqual(result,40000)
    
    def test_iou_partial_overlap(self):
        """Two boxes overlapping partially should give IoU ~0.143."""
        box_a= [100,100,300,300]
        box_b = [200,200,400,400]
        
        result = BoxMath.iou(box_a, box_b)
        self.assertAlmostEqual(result, 0.142857, places =4)
    
    def test_iou_identical_boxes(self):
        #Two identical boxes should give IoU exactly 1.0.
        box_a = [100,100,300,300]
        box_b = [100,100,300,300]
        result = BoxMath.iou(box_a,box_b)
        self.assertEqual(result,1.0)
        
    def test_iou_no_overlap(self):
        """Two far-apart boxes should give IoU exactly 0.0."""
        box_a = [0,0,50,50]
        box_b = [500, 500, 600, 600]
        result = BoxMath.iou(box_a,box_b)
        self.assertEqual(result, 0.0)

if __name__ =='__main__':
    unittest.main()