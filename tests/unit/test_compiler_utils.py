import pytest
from dsperse.src.compile.utils.compiler_utils import CompilerUtils

def test_parse_complex_layer_backends():
    # 1. Simple mapping
    spec = "0,2:jstprove;3-4:ezkl"
    backends, indices = CompilerUtils.parse_complex_layer_backends(spec)
    assert backends == {0: 'jstprove', 2: 'jstprove', 3: 'ezkl', 4: 'ezkl'}
    assert indices == set()

    # 2. Mapping with default indices
    spec = "0; 2:jstprove; 3-4:ezkl"
    backends, indices = CompilerUtils.parse_complex_layer_backends(spec)
    assert backends == {2: 'jstprove', 3: 'ezkl', 4: 'ezkl'}
    assert indices == {0}

    # 3. Robust parsing: spaces and multiple separators
    spec = "0, 2:jstprove  3-4:ezkl"
    backends, indices = CompilerUtils.parse_complex_layer_backends(spec)
    assert backends == {0: 'jstprove', 2: 'jstprove', 3: 'ezkl', 4: 'ezkl'}
    assert indices == set()

    # 4. Mixed format
    spec = "0-1; 5:jstprove; 6:ezkl"
    backends, indices = CompilerUtils.parse_complex_layer_backends(spec)
    assert backends == {5: 'jstprove', 6: 'ezkl'}
    assert indices == {0, 1}

def test_parse_backend_and_layers():
    # 1. Backend name only
    be, fallback, idxs = CompilerUtils.parse_backend_and_layers("ezkl")
    assert be == "ezkl"
    assert fallback is False
    assert idxs is None

    # 2. Layer indices only
    be, fallback, idxs = CompilerUtils.parse_backend_and_layers("0,2-3")
    assert be is None
    assert fallback is True
    assert idxs == [0, 2, 3]

    # 3. Complex mapping
    be, fallback, idxs = CompilerUtils.parse_backend_and_layers("0:jstprove;1:ezkl")
    assert be is None
    assert fallback is True
    assert idxs == "PARSE_COMPLEX"

    # 4. None/Empty
    be, fallback, idxs = CompilerUtils.parse_backend_and_layers(None)
    assert be is None
    assert fallback is True
    assert idxs is None

def test_parse_layers():
    # 1. Simple indices
    assert CompilerUtils.parse_layers("0,2,5") == [0, 2, 5]

    # 2. Ranges
    assert CompilerUtils.parse_layers("0-2,5") == [0, 1, 2, 5]

    # 3. Spaces and trailing commas (Fixed)
    assert CompilerUtils.parse_layers(" 0, 2, ") == [0, 2]

    # 4. None/Empty
    assert CompilerUtils.parse_layers(None) is None
    assert CompilerUtils.parse_layers("") is None

def test_get_backends_to_build():
    layer_backends = {2: 'jstprove', 3: 'ezkl'}
    default_indices = {0}
    
    # Slice 0: In default_indices -> both backends
    assert set(CompilerUtils.get_backends_to_build(0, layer_backends, default_indices, None, True)) == {'jstprove', 'ezkl'}
    
    # Slice 2: In layer_backends -> jstprove only
    assert CompilerUtils.get_backends_to_build(2, layer_backends, default_indices, None, True) == ['jstprove']
    
    # Slice 3: In layer_backends -> ezkl only
    assert CompilerUtils.get_backends_to_build(3, layer_backends, default_indices, None, True) == ['ezkl']
    
    # Slice 1: Not in any -> default both (fallback mode)
    assert set(CompilerUtils.get_backends_to_build(1, layer_backends, default_indices, None, True)) == {'jstprove', 'ezkl'}
    
    # Slice 1: Forced backend mode
    assert CompilerUtils.get_backends_to_build(1, layer_backends, default_indices, 'ezkl', False) == ['ezkl']
