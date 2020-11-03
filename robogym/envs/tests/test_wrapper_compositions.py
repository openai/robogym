from copy import deepcopy

from robogym.wrappers.named_wrappers import edit_wrappers


def test_edit_wrappers():
    test_wrappers = [
        ["StopOnFallWrapper"],
        ["RandomizedActionLatency"],
        ["RandomizedCubeSizeWrapper"],
        ["RandomizedBodyInertiaWrapper"],
        ["RandomizedTimestepWrapper"],
        ["RandomizedFrictionWrapper"],
        ["RandomizedGravityWrapper"],
        ["RandomizedWindWrapper"],
    ]

    test_wrappers_insert_above_query = [
        ["RandomizedCubeSizeWrapper", ["TESTWRAPPER1", {"arg1": "yayyy"}]],
        ["RandomizedFrictionWrapper", ["TESTWRAPPER2", {"arg2": "yayyy"}]],
    ]
    test_wrappers_insert_above_answer = [
        ["StopOnFallWrapper"],
        ["RandomizedActionLatency"],
        ["TESTWRAPPER1", {"arg1": "yayyy"}],
        ["RandomizedCubeSizeWrapper"],
        ["RandomizedBodyInertiaWrapper"],
        ["RandomizedTimestepWrapper"],
        ["TESTWRAPPER2", {"arg2": "yayyy"}],
        ["RandomizedFrictionWrapper"],
        ["RandomizedGravityWrapper"],
        ["RandomizedWindWrapper"],
    ]

    test_wrappers_insert_below_query = [
        ["RandomizedCubeSizeWrapper", ["TESTWRAPPER1", {"arg1": "yayyy"}]],
        ["RandomizedFrictionWrapper", ["TESTWRAPPER2", {"arg2": "yayyy"}]],
    ]
    test_wrappers_insert_below_answer = [
        ["StopOnFallWrapper"],
        ["RandomizedActionLatency"],
        ["RandomizedCubeSizeWrapper"],
        ["TESTWRAPPER1", {"arg1": "yayyy"}],
        ["RandomizedBodyInertiaWrapper"],
        ["RandomizedTimestepWrapper"],
        ["RandomizedFrictionWrapper"],
        ["TESTWRAPPER2", {"arg2": "yayyy"}],
        ["RandomizedGravityWrapper"],
        ["RandomizedWindWrapper"],
    ]

    test_wrappers_delete_query = [
        "RandomizedBodyInertiaWrapper",
        "RandomizedFrictionWrapper",
    ]
    test_wrappers_delete_answer = [
        ["StopOnFallWrapper"],
        ["RandomizedActionLatency"],
        ["RandomizedCubeSizeWrapper"],
        ["RandomizedTimestepWrapper"],
        ["RandomizedGravityWrapper"],
        ["RandomizedWindWrapper"],
    ]

    test_wrappers_replace_query = [
        ["RandomizedBodyInertiaWrapper", ["TESTWRAPPER1", {"arg1": "yayyy"}]],
        ["RandomizedFrictionWrapper", ["TESTWRAPPER2", {"arg2": "yayyy"}]],
    ]
    test_wrappers_replace_answer = [
        ["StopOnFallWrapper"],
        ["RandomizedActionLatency"],
        ["RandomizedCubeSizeWrapper"],
        ["TESTWRAPPER1", {"arg1": "yayyy"}],
        ["RandomizedTimestepWrapper"],
        ["TESTWRAPPER2", {"arg2": "yayyy"}],
        ["RandomizedGravityWrapper"],
        ["RandomizedWindWrapper"],
    ]

    assert (
        edit_wrappers(
            wrappers=deepcopy(test_wrappers),
            insert_above=test_wrappers_insert_above_query,
        )
        == test_wrappers_insert_above_answer
    )
    assert (
        edit_wrappers(
            wrappers=deepcopy(test_wrappers),
            insert_below=test_wrappers_insert_below_query,
        )
        == test_wrappers_insert_below_answer
    )
    assert (
        edit_wrappers(
            wrappers=deepcopy(test_wrappers), delete=test_wrappers_delete_query
        )
        == test_wrappers_delete_answer
    )
    assert (
        edit_wrappers(
            wrappers=deepcopy(test_wrappers), replace=test_wrappers_replace_query
        )
        == test_wrappers_replace_answer
    )
