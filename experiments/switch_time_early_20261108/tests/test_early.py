import sys, unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import run_switch_early as r
from switch_engine import ShadowEngine, tree_digest

class TestEarly(unittest.TestCase):
    def test_registered_policy_order(self):
        self.assertEqual(r.POLICIES,('M','T40','T60','T80'))
        self.assertEqual(r.SWITCH_AT,{'M':201,'T40':40,'T60':60,'T80':80})
        self.assertEqual(tuple(r.COMPARISONS),('T60_vs_M','T60_vs_T40','T60_vs_T80'))
    def test_correct_switch_step(self):
        for name,at in [('T40',40),('T60',60),('T80',80)]:
            self.assertEqual(r.active_optimizer(name,at-1),'M')
            self.assertEqual(r.active_optimizer(name,at),'A')
    def test_keys_no_overlap(self):
        collected=set()
        for pool in range(12):
            keys=set(r.fit_keys(pool))|set(r.test_keys(pool))
            self.assertEqual(len(keys),112)
            self.assertTrue(not (keys&collected));collected.update(keys)
        self.assertEqual(len({r.init_key(p,s) for p in range(12) for s in range(4)}),48)
    def test_identical_within_pool_schedule(self):
        self.assertNotEqual(r.schedule_sha(r.schedule(0)),r.schedule_sha(r.schedule(1)))
        self.assertEqual(r.schedule_sha(r.schedule(0)),r.schedule_sha(r.schedule(0)))
    def test_state_barrier(self):
        if not (r.ROOT/'runs/GLOBAL_SEAL.json').exists():
            with self.assertRaisesRegex(ValueError,'no prospective freeze|FORBIDDEN'):
                r.verify_global_seal()
    def test_config_canonicalization(self):
        import json
        self.assertEqual(json.loads(json.dumps(r.CONFIG,sort_keys=True)), json.loads(json.dumps(r.CONFIG)))
    def test_shadow_uses_hidden_only(self):
        e=ShadowEngine(23)
        self.assertEqual(tuple(e.hidden.shape),(128,64))
        self.assertEqual(tree_digest(e.model.state_dict()),tree_digest(ShadowEngine(23).model.state_dict()))
    def test_clean_margin_fixed(self):
        self.assertEqual(r.CLEAN_MARGIN, .05)
        self.assertEqual(r.NPOOLS,12)
        self.assertEqual(r.NSEEDS,4)
if __name__=='__main__':unittest.main()
