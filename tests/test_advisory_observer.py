"""Observer persistence must not alter numerical checkpoints or control flow."""
from copy import deepcopy
import hashlib
import json

import numpy as np
import pytest
import torch

from block_solver import solve_blocks
from field_store import BlockStore


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Journal:
    """Test observer; real quota/atomic journal is CLI-owned and tested there."""
    def __init__(self,path):
        self.path=path;self.closures=[];self.prepared=[]

    def restore(self,snapshot):
        states=json.loads(self.path.read_bytes()) if self.path.exists() else {}
        return states.get(snapshot['checkpoint_sha256'])

    def prepare(self,snapshot,metrics):
        self.prepared.append(deepcopy(snapshot))
        states=json.loads(self.path.read_bytes()) if self.path.exists() else {}
        states[snapshot['checkpoint_sha256']]=deepcopy(metrics)
        self.path.write_text(json.dumps(states,allow_nan=False))

    def complete(self,snapshot):
        self.closures.append(deepcopy(snapshot))


def options():
    shape=(3,3,6)
    def load(lo,hi):
        local=tuple(hi-lo);observed=np.broadcast_to(np.arange(lo[2],hi[2])%2==0,local).copy()
        target=np.zeros((*local,3),'f4');target[...,0]=np.where(observed,2.,10.)
        return target,np.ones(local,'f4'),observed,{}
    return dict(shape=shape,chunk=3,identity={'synthetic':'observed update counters'},load_observations=load,
                origin_xyz=[0.]*3,spacing_xyz=[1.]*3,order=[(0,0,0),(0,0,1)],
                halo=1,sweeps=2,reserve_bytes=0,rtol=1e-6)


def known_updates(target,weights,spacing,*,initial,fixed,**kwargs):
    result=torch.as_tensor(initial+target[...,0],dtype=torch.float64)
    fixed=torch.as_tensor(fixed);valid=torch.isfinite(fixed);result[valid]=fixed[valid]
    return result,{'converged':True}


def test_prepare_hook_binds_exact_future_manifest_and_preserves_scientific_schema(tmp_path):
    records=[]
    root=tmp_path/'field'
    with BlockStore(root,(2,2,2),2,{'test':'observer'},reserve_bytes=0) as store:
        before=sha(root/'CHECKPOINT.json')
        store.commit((0,0,0),np.ones((2,2,2)),{'cursor':1},np.ones((2,2,2),bool),
                     prepare=lambda snapshot:records.append(snapshot))
    saved=json.loads((root/'CHECKPOINT.json').read_bytes())
    assert records[0]['previous_checkpoint_sha256']==before
    assert records[0]['checkpoint_sha256']==sha(root/'CHECKPOINT.json')
    assert records[0]['manifest']==saved
    assert saved['progress']=={'cursor':1} and 'advice' not in saved


def test_prepare_sink_failure_does_not_poison_store_or_lose_scientific_work(tmp_path):
    def fail(snapshot):raise OSError('diagnostic disk unavailable')
    with BlockStore(tmp_path/'field',(2,2,2),2,{'test':'observer'},reserve_bytes=0) as store:
        receipt=store.commit((0,0,0),np.ones((2,2,2)),{'cursor':1},prepare=fail)
        assert receipt['prepare_error']
        assert store.manifest['progress']=={'cursor':1}
        store.commit((0,0,0),np.full((2,2,2),2.),{'cursor':2})
        np.testing.assert_array_equal(store.read([0]*3,[2]*3),2.)


def test_observed_counters_exclude_unobserved_and_halos_and_include_final_sweep(tmp_path,monkeypatch):
    monkeypatch.setattr('block_solver.solve_field',known_updates)
    journal=Journal(tmp_path/'journal.json')
    result=solve_blocks(tmp_path/'field',**options(),observer=journal)
    assert result['status']=='SWEEP_BUDGET_REACHED'
    assert len(journal.closures)==2
    for closure in journal.closures:
        metrics=closure['metrics']
        assert metrics['measurement_scope']=='pre_encoding_owned_observed'
        assert metrics['observed_count']==27
        assert metrics['observed_max_update']==2.
        assert metrics['observed_sum_sq']==pytest.approx(104.,abs=1e-10)
        assert metrics['observed_rms_update']==pytest.approx(np.sqrt(104/27))
        assert metrics['complete'] and metrics['work_s']>=0 and metrics['control_s']>=0
        assert closure['manifest']['progress']['completed_sweeps'][-1]['max_core_update']==10.
    assert journal.closures[-1]['checkpoint_sha256']==sha(tmp_path/'field/CHECKPOINT.json')


def test_resuming_a_partial_sweep_restores_counters_without_replaying_blocks(tmp_path,monkeypatch):
    monkeypatch.setattr('block_solver.solve_field',known_updates)
    root=tmp_path/'field';journal=Journal(tmp_path/'journal.json')
    def stop():
        path=root/'CHECKPOINT.json'
        return path.exists() and json.loads(path.read_bytes())['progress'].get('next_block')==1
    stopped=solve_blocks(root,**options(),observer=journal,stop=stop)
    assert stopped['status']=='CHECKPOINTED_STOP'
    solve_blocks(root,**options(),observer=journal,resume=True)
    assert [c['metrics']['sweep'] for c in journal.closures]==[1,2]
    assert all(c['metrics']['observed_count']==27 and c['metrics']['observed_sum_sq']==pytest.approx(104.,abs=1e-10)
               for c in journal.closures)
    assert len(journal.prepared)==4


def test_observer_does_not_change_real_science_or_stop_policy(tmp_path):
    from test_block_solver import case,state
    kwargs=case(tmp_path);journal=Journal(tmp_path/'journal.json')
    a=solve_blocks(tmp_path/'plain',**kwargs)
    b=solve_blocks(tmp_path/'advice',**kwargs,observer=journal)
    ma,va,oa=state(tmp_path/'plain');mb,vb,ob=state(tmp_path/'advice')
    assert a['status']==b['status']=='SWEEP_BUDGET_REACHED'
    assert a['completed_sweeps']==b['completed_sweeps']
    ma['progress']['solver'].pop('elapsed_s');mb['progress']['solver'].pop('elapsed_s')
    assert ma['identity']==mb['identity'] and ma['progress']==mb['progress']
    np.testing.assert_array_equal(va,vb);np.testing.assert_array_equal(oa,ob)
    assert {k:v['sha256'] for k,v in ma['chunks'].items()}=={k:v['sha256'] for k,v in mb['chunks'].items()}


def test_complete_boundary_resume_is_not_a_new_sweep(tmp_path,monkeypatch):
    monkeypatch.setattr('block_solver.solve_field',known_updates)
    journal=Journal(tmp_path/'journal.json');root=tmp_path/'field'
    solve_blocks(root,**options(),observer=journal)
    before=(root/'CHECKPOINT.json').read_bytes()
    resumed=Journal(tmp_path/'journal.json')
    solve_blocks(root,**options(),observer=resumed,resume=True)
    assert len(resumed.closures)==1 and resumed.closures[0]['metrics']['sweep']==2
    assert (root/'CHECKPOINT.json').read_bytes()==before


def test_optional_snapshot_allocation_failure_does_not_poison_science(tmp_path,monkeypatch):
    import field_store
    original=field_store.deepcopy;raised=False
    def fail_snapshot(value):
        nonlocal raised
        if not raised and isinstance(value,dict) and value.get('progress')=={'cursor':1}:
            raised=True
            raise MemoryError('diagnostic snapshot allocation')
        return original(value)
    with BlockStore(tmp_path/'field',(2,2,2),2,{'test':'observer'},reserve_bytes=0) as store:
        monkeypatch.setattr(field_store,'deepcopy',fail_snapshot)
        receipt=store.commit((0,0,0),np.ones((2,2,2)),{'cursor':1},prepare=lambda event:None)
        assert raised and receipt['prepare_error']
        np.testing.assert_array_equal(store.read([0]*3,[2]*3),1.)


def test_advisory_accumulator_overflow_cannot_fail_a_valid_scientific_commit(tmp_path,monkeypatch):
    monkeypatch.setattr('block_solver.solve_field',known_updates)
    shape=(2,2,8);constant=float(2**511)
    def load(lo,hi):
        z,y,x=np.meshgrid(*(np.arange(a,b) for a,b in zip(lo,hi)),indexing='ij')
        observed=(z==1)&(y==1)&(x%2==0)
        return np.zeros((*observed.shape,3),'f4'),np.ones(observed.shape,'f4'),observed,{}
    kwargs=dict(shape=shape,chunk=2,identity={'test':'finite constant scientific field'},
                load_observations=load,origin_xyz=[0.]*3,spacing_xyz=[1.]*3,
                order=[(0,0,i) for i in range(4)],sweeps=1,reserve_bytes=0,
                fixed=lambda lo,hi:np.full(tuple(hi-lo),constant))
    with pytest.warns(RuntimeWarning,match='advice'):
        result=solve_blocks(tmp_path/'field',**kwargs,observer=Journal(tmp_path/'journal.json'))
    assert result['status']=='SWEEP_BUDGET_REACHED' and result['advisory'] is None
    saved=json.loads((tmp_path/'field/CHECKPOINT.json').read_bytes())
    with BlockStore(tmp_path/'field',shape,2,saved['identity'],resume=True,reserve_bytes=0) as store:
        np.testing.assert_array_equal(store.read([0]*3,shape),constant)


def test_real_scientific_integrity_error_from_closure_is_not_swallowed(tmp_path,monkeypatch):
    from block_solver import AdvisoryIntegrityError
    monkeypatch.setattr('block_solver.solve_field',known_updates)
    class Corrupt(Journal):
        def complete(self,event):raise AdvisoryIntegrityError('scientific payload hash differs')
    root=tmp_path/'field'
    with pytest.raises(AdvisoryIntegrityError):
        solve_blocks(root,**options(),observer=Corrupt(tmp_path/'journal.json'))
    assert len(json.loads((root/'CHECKPOINT.json').read_bytes())['progress']['completed_sweeps'])==1


def test_diagnostics_do_not_consume_scientific_status_storage(tmp_path,monkeypatch):
    monkeypatch.setattr('block_solver.solve_field',known_updates)
    solve_blocks(tmp_path/'field',**options(),observer=Journal(tmp_path/'journal.json'))
    status=json.loads((tmp_path/'field/STATUS.json').read_bytes())
    assert 'advisory' not in status and 'advisory_errors' not in status


def test_closure_diagnostic_copy_failure_is_isolated(tmp_path,monkeypatch):
    import block_solver
    monkeypatch.setattr(block_solver,'solve_field',known_updates)
    original=block_solver.deepcopy
    def fail_completed(value):
        if isinstance(value,dict) and value.get('schema')=='field-observer-v1' and value.get('next_block')==2:
            raise MemoryError('diagnostic closure copy')
        return original(value)
    monkeypatch.setattr(block_solver,'deepcopy',fail_completed)
    with pytest.warns(RuntimeWarning,match='advice'):
        result=solve_blocks(tmp_path/'field',**options(),observer=Journal(tmp_path/'journal.json'))
    assert result['status']=='SWEEP_BUDGET_REACHED'
