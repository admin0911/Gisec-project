import unittest
import numpy as np
import torch
from torch.utils.data import TensorDataset
from poison_features.attacks import poison_dataset
class MixedTests(unittest.TestCase):
 def test_counts_pixels_labels_and_repeatability(self):
  for channels in [1,3]:
   x=torch.zeros(1000,channels,28,28);y=torch.arange(1000)%10;data=TensorDataset(x,y)
   for kind,expected in [('mixed_noise',{'label_flip':25,'blended_injection':25}),('mixed_all',{'label_flip':17,'backdoor':17,'blended_injection':16})]:
    a=poison_dataset(data,kind,poison_rate=.05,seed=42)
    b=poison_dataset(data,kind,poison_rate=.05,seed=42)
    np.testing.assert_array_equal(a.metadata.poison_type,b.metadata.poison_type)
    self.assertEqual(int(a.metadata.is_poisoned.sum()),50)
    for t,n in expected.items():self.assertEqual(int(np.sum(a.metadata.poison_type==t)),n)
    for i in range(1000):
     pix,label=a[i];t=a.metadata.poison_type[i]
     if t in ['clean','label_flip']:self.assertTrue(torch.equal(pix,x[i]))
     if t=='label_flip':self.assertEqual(label,(int(y[i])+1)%10)
     if t in ['backdoor','blended_injection']:self.assertEqual(label,0);self.assertNotEqual(int(y[i]),0);self.assertGreater(pix.sum(),0)
     if t=='backdoor':self.assertEqual(int(torch.count_nonzero(pix)),channels*9)
    self.assertEqual(int(torch.count_nonzero(x)),0)
 def test_cap(self):
  data=TensorDataset(torch.zeros(101,1,28,28),torch.arange(101)%10)
  self.assertEqual(int(poison_dataset(data,'mixed_all',poison_rate=.1).metadata.is_poisoned.sum()),10)
  for kw in [dict(poison_rate=.11),dict(poison_count=11),dict(poison_count=1)]:
   with self.assertRaises(ValueError):poison_dataset(data,'mixed_all',**kw)
if __name__=='__main__':unittest.main()
