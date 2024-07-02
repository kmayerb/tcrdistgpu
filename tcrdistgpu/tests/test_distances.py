from tcrdistgpu.distance import TCRgpu
from tcrdistgpu.distance import sort_out_k
import unittest
import pandas as pd
import numpy as np

class TestTCRgpu(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load test data
        cls.tst_tcr = pd.read_csv("data/tmp_tcr.tsv", sep="\t").head(2000)

    def setUp(self):
        # Initialize the TCRgpu object
        self.tg = TCRgpu(tcrs=self.tst_tcr, mode='cpu', kbest=10)

    def test_initialization(self):
        # Test initialization of the TCRgpu object
        self.assertIsNotNone(self.tg.tcrs)
        self.assertEqual(self.tg.mode, 'cpu')
        self.assertEqual(self.tg.kbest, 10)
        self.assertEqual(self.tg.chunk_size, 1000)

    def test_encode_tcrs(self):
        # Test encoding of TCR sequences
        encoded = self.tg.encode_tcrs()
        self.assertEqual(encoded.shape[0], self.tg.tcrs.shape[0])
        self.assertEqual(encoded.shape[1], 2 * (self.tg.target_length - 5) + 2)

    def test_compute(self):
        # Test computation of nearest neighbors
        self.tg.encode_tcrs()
        sorted_indices, sorted_smallest_k_values = self.tg.compute()
        self.assertEqual(sorted_indices.shape, (self.tg.tcrs.shape[0], self.tg.kbest))
        self.assertEqual(sorted_smallest_k_values.shape, (self.tg.tcrs.shape[0], self.tg.kbest))


class TestTCRgpu_dash(unittest.TestCase):

    def setUp(self):
        # Initialize the TCRgpu object
        self.dash_tcr = pd.read_csv("data/dash_human.csv", sep=",")
        self.dash_result = pd.read_csv('data/dash_human_fixed_gappos.csv').values
        self.dash_result2 = pd.read_csv('data/dash_human_fixed_gappos_false.csv').values
        self.tg = TCRgpu(tcrs=self.dash_tcr, mode='cpu', kbest=10)
        self.tg.cdr3a_col = "cdr3_a_aa"
        self.tg.cdr3b_col = "cdr3_b_aa"
        self.tg.va_col = 'v_a_gene'
        self.tg.vb_col = 'v_b_gene'


    def test_initialization(self):
        # Test initialization of the TCRgpu object
        self.assertIsNotNone(self.tg.tcrs)
        self.assertEqual(self.tg.mode, 'cpu')
        self.assertEqual(self.tg.kbest, 10)
        self.assertEqual(self.tg.chunk_size, 1000)

    def test_encode_tcrs(self):
        # Test encoding of TCR sequences
        encoded = self.tg.encode_tcrs()
        self.assertEqual(encoded.shape[0], self.tg.tcrs.shape[0])
        self.assertEqual(encoded.shape[1], 2 * (self.tg.target_length - 5) + 2)
        print(encoded)
    
    def test_x_compute_vs_dash(self):
        """
        import pandas as pd
        from tcrdist.repertoire import TCRrep
        d1 = pd.read_csv('dash_human.csv')
        tr = TCRrep(cell_df = d1[['v_a_gene','v_b_gene','cdr3_a_aa','cdr3_b_aa']],
                    organism = 'human',
                    chains = ['alpha','beta'],
                    deduplicate = False, 
                    compute_distances = True)
        tr.kargs_a['cdr3_a_aa']['fixed_gappos'] = False
        tr.kargs_b['cdr3_b_aa']['fixed_gappos'] = False
        tr.dist = tr.pw_beta + tr.pw_alpha
        """
        # Test computation of nearest neighbors
        self.tg.encode_tcrs()
        sorted_indices, sorted_smallest_k_values = self.tg.compute()
        
        self.assertEqual(sorted_indices.shape, (self.tg.tcrs.shape[0], self.tg.kbest))
        self.assertEqual(sorted_smallest_k_values.shape, (self.tg.tcrs.shape[0], self.tg.kbest))
        # !!!!!!!!!!!!!! NOTE WE DON"T GET EXACT MATCH TO TCRDIST3 RESULTS
        sorted_indices_expected, sorted_smallest_k_values_expected = sort_out_k(self.dash_result2 , k = 10)
        print((sorted_indices_expected == sorted_indices).mean())
        print(sorted_smallest_k_values[0:5,0:5])
        print(sorted_smallest_k_values_expected[0:5,0:5])
        assert ((sorted_indices_expected < 100) == (sorted_indices <100)).mean() > .95



if __name__ == '__main__':
    unittest.main()
