from pathlib import Path
import pytest

from visual_prompt_guidance.config import resolve_settings
from visual_prompt_guidance.runtime import build_engine, build_scoring, build_observers


def test_shared_config_maps_to_engine_guidance_scoring_propagation_and_updater(tmp_path):
    settings=resolve_settings('generate',{
        'normalize_gradient':False,
        'selected_losses':['coverage'],
        'scoring_losses':['moment'],
        'guide_steps':3,'strength':0.02,
        'propagation_layers':['D1'],
        'use_propagation':True,
        'updater':'rms',
        'save_maps':False,
        'observer_interval':7,
    })
    obs=build_observers(settings,tmp_path/'diagnostics')
    engine=build_engine(settings,backend=object(),observers=obs)
    assert engine.guidance.normalize_gradient is False
    assert engine.guidance.max_steps==3
    assert engine.guidance.affinity_layers==('D1',)
    assert engine.transform is not None
    assert engine.scoring is not engine.objective
    assert [term.name for term in engine.scoring.terms]==['moment']
    assert engine.observers[0].interval==7
    settings['use_propagation']=False
    engine=build_engine(settings,backend=object())
    assert engine.guidance.affinity_layers is None
    assert engine.transform is None
    settings['updater']='invalid'
    with pytest.raises(ValueError,match='Unknown latent updater'):
        build_engine(settings,backend=object())
