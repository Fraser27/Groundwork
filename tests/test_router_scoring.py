"""Turning similarity scores into a decision about which tiers to run.

Pure functions, so these need no cluster, no fakes and no credentials. That is the reason the
scoring lives in its own module: the rule is the part most likely to be retuned, and it should be
possible to change it and know immediately whether the behaviour still holds.

The two properties worth the most here: **a raw OpenSearch score is not a cosine** (so a threshold
set on the raw value silently means something else), and **selection is relative** (because no
absolute cosine means "relevant" across questions of different lengths).
"""

from __future__ import annotations

import pytest

from src.query.router_scoring import DEFAULT_TOP_N, LayerScore, cosine_of, score_layers


def _by_kind(layers: list[LayerScore]) -> dict[str, LayerScore]:
    return {layer.kind: layer for layer in layers}


def _selected(layers: list[LayerScore]) -> set[str]:
    return {layer.kind for layer in layers if layer.selected}


class TestScoreToCosine:
    """`cosinesimil` reports `1 / (2 - cos)`, not the cosine.

    An administrator setting 0.8 believing it is a cosine would really be asking for 0.75. Every
    number is converted before it is scored, compared or displayed.
    """

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [(1.0, 1.0), (0.9091, 0.9), (0.6667, 0.5), (0.5, 0.0), (0.3333, -1.0)],
    )
    def test_the_conversion_matches_the_engine(self, raw, expected):
        assert cosine_of(raw) == pytest.approx(expected, abs=1e-3)

    def test_a_raw_threshold_would_have_meant_something_else(self):
        """The concrete cost of not converting: 0.8 is really 0.75."""
        assert cosine_of(0.8) == pytest.approx(0.75)

    def test_a_zero_score_does_not_divide_by_zero(self):
        """The engine does not emit one, but a fake client in a test might, and a crash in the
        router would take down the whole answer rather than one layer."""
        assert cosine_of(0.0) == -1.0
        assert cosine_of(-5.0) == -1.0


class TestTheFloor:
    def test_items_below_the_floor_do_not_count(self):
        layers = _by_kind(
            score_layers({"metric": [0.9, 0.1, 0.05]}, min_similarity=0.5, margin=0.3)
        )
        assert layers["metric"].hit_count == 1

    def test_a_layer_with_nothing_above_the_floor_is_dropped_and_says_why(self):
        layers = _by_kind(score_layers({"entity": [0.2, 0.1]}, min_similarity=0.5, margin=0.3))
        assert layers["entity"].selected is False
        assert "floor" in layers["entity"].reason

    def test_nothing_above_the_floor_anywhere_selects_nothing(self):
        """The router reads this as "nothing looked relevant" and degrades to trying every tier,
        so it has to return cleanly rather than raising."""
        layers = score_layers({"metric": [0.1], "entity": [0.2]}, min_similarity=0.9, margin=1.0)
        assert _selected(layers) == set()


class TestTheLayerScore:
    def test_the_top_two_are_averaged(self):
        layers = _by_kind(score_layers({"metric": [0.8, 0.6, 0.1]}, min_similarity=0.0, margin=0.0))
        assert layers["metric"].raw_score == pytest.approx(0.7)

    def test_one_lucky_item_does_not_beat_two_good_ones(self):
        """Max would promote the lucky layer. That is the failure this averaging exists to avoid:
        a single coincidental match should not decide which tiers run."""
        layers = _by_kind(
            score_layers(
                {"lucky": [0.95, 0.10], "solid": [0.80, 0.78]},
                min_similarity=0.0,
                margin=0.0,
            )
        )
        assert layers["solid"].score > layers["lucky"].score

    def test_a_long_mediocre_tail_does_not_drag_an_excellent_match_down(self):
        """A full mean would punish the layer that holds the one thing the question is about."""
        layers = _by_kind(
            score_layers(
                {"deep": [0.9, 0.85] + [0.5] * 20, "shallow": [0.6, 0.6]},
                min_similarity=0.0,
                margin=0.0,
            )
        )
        assert layers["deep"].score > layers["shallow"].score

    def test_a_layer_with_a_single_hit_is_scored_on_it(self):
        """A tenant with one governed metric is a normal state, not a degenerate one."""
        layers = _by_kind(score_layers({"metric": [0.7]}, min_similarity=0.0, margin=0.0))
        assert layers["metric"].raw_score == pytest.approx(0.7)
        assert layers["metric"].hit_count == 1

    def test_hit_count_is_reported_but_not_scored(self):
        """Forty mediocre hits must not outrank two good ones."""
        layers = _by_kind(
            score_layers(
                {"many": [0.55] * 40, "few": [0.9, 0.88]},
                min_similarity=0.0,
                margin=0.0,
            )
        )
        assert layers["many"].hit_count == 40
        assert layers["few"].score > layers["many"].score

    def test_the_caller_need_not_sort(self):
        """Otherwise forgetting to sort silently changes the decision."""
        ascending = score_layers({"metric": [0.1, 0.6, 0.8]}, min_similarity=0.0, margin=0.0)
        descending = score_layers({"metric": [0.8, 0.6, 0.1]}, min_similarity=0.0, margin=0.0)
        assert ascending[0].score == descending[0].score


class TestTheMargin:
    """How much worse than the best layer a layer may be and still be searched.

    Relative rather than absolute because similarity is not calibrated: there is no cosine that
    means "relevant" across questions of different lengths and phrasings.
    """

    def test_a_margin_of_zero_selects_only_the_winner(self):
        layers = score_layers(
            {"metric": [0.9, 0.9], "entity": [0.6, 0.6]}, min_similarity=0.0, margin=0.0
        )
        assert _selected(layers) == {"metric"}

    def test_a_margin_of_one_selects_everything_above_the_floor(self):
        layers = score_layers(
            {"metric": [0.9, 0.9], "entity": [0.3, 0.3]}, min_similarity=0.0, margin=1.0
        )
        assert _selected(layers) == {"metric", "entity"}

    def test_a_close_second_is_kept(self):
        layers = score_layers(
            {"metric": [0.80, 0.80], "entity": [0.72, 0.72]}, min_similarity=0.0, margin=0.2
        )
        assert _selected(layers) == {"metric", "entity"}

    def test_a_distant_second_is_dropped_and_names_the_cutoff(self):
        layers = _by_kind(
            score_layers(
                {"metric": [0.90, 0.90], "entity": [0.40, 0.40]},
                min_similarity=0.0,
                margin=0.2,
            )
        )
        assert layers["entity"].selected is False
        assert "margin" in layers["entity"].reason

    def test_ties_are_all_selected(self):
        """Declining on a tie is right for choosing one metric to compile; here the question is
        which layers to search, and searching both is the safe answer rather than neither."""
        layers = score_layers(
            {"metric": [0.7, 0.7], "entity": [0.7, 0.7]}, min_similarity=0.0, margin=0.0
        )
        assert _selected(layers) == {"metric", "entity"}


class TestTheBoost:
    """Governed metrics may outrank an equally-scoring layer, by a configurable amount.

    This is what makes "the router may skip tier 1" safe: without it, a paraphrase scoring
    slightly low routes away from deterministic SQL to tier 4, where a model writes the query --
    which is the exact failure the router was built to remove.
    """

    def test_an_equal_metric_layer_wins(self):
        layers = score_layers(
            {"metric": [0.70, 0.70], "entity": [0.70, 0.70]},
            min_similarity=0.0,
            margin=0.0,
            boosts={"metric": 0.05},
        )
        assert _selected(layers) == {"metric"}

    def test_the_boost_can_rescue_a_near_miss(self):
        layers = score_layers(
            {"metric": [0.66, 0.66], "entity": [0.70, 0.70]},
            min_similarity=0.0,
            margin=0.0,
            boosts={"metric": 0.05},
        )
        assert "metric" in _selected(layers)

    def test_the_boost_cannot_rescue_a_layer_below_the_floor(self):
        """The floor is a data question -- did anything match at all -- and a preference must not
        answer it. Otherwise a boost invents relevance where the index held none."""
        layers = _by_kind(
            score_layers(
                {"metric": [0.10]},
                min_similarity=0.5,
                margin=1.0,
                boosts={"metric": 0.9},
            )
        )
        assert layers["metric"].selected is False

    def test_the_unboosted_score_is_kept_for_the_trace(self):
        """So the diagram can show what the boost did, not just its effect."""
        layers = _by_kind(
            score_layers(
                {"metric": [0.60, 0.60]}, min_similarity=0.0, margin=0.0, boosts={"metric": 0.05}
            )
        )
        assert layers["metric"].raw_score == pytest.approx(0.60)
        assert layers["metric"].score == pytest.approx(0.65)
        assert "+0.05" in layers["metric"].reason

    def test_a_boost_cannot_push_a_score_past_one(self):
        """`relative` is a ratio against the best score, and a score above 1.0 makes the margin
        arithmetic meaningless."""
        layers = _by_kind(
            score_layers(
                {"metric": [0.99, 0.99]}, min_similarity=0.0, margin=0.0, boosts={"metric": 0.5}
            )
        )
        assert layers["metric"].score <= 1.0


class TestTheTrace:
    def test_every_layer_asked_about_comes_back(self):
        """Including the losers. A trace of only the winners records the outcome, not the
        reasoning, and the reasoning is what an auditor came for."""
        layers = score_layers(
            {"metric": [0.9, 0.9], "entity": [0.1], "table": []},
            min_similarity=0.5,
            margin=0.1,
        )
        assert {layer.kind for layer in layers} == {"metric", "entity", "table"}

    def test_every_layer_carries_a_reason(self):
        layers = score_layers(
            {"metric": [0.9, 0.9], "entity": [0.1], "table": []},
            min_similarity=0.5,
            margin=0.1,
        )
        assert all(layer.reason for layer in layers)

    def test_the_reason_names_the_number_that_decided_it(self):
        layers = _by_kind(
            score_layers(
                {"metric": [0.9, 0.9], "entity": [0.4, 0.4]}, min_similarity=0.0, margin=0.1
            )
        )
        assert "0.81" in layers["entity"].reason  # the cutoff, 0.9 * (1 - 0.1)

    def test_layers_come_back_best_first(self):
        layers = score_layers(
            {"weak": [0.3, 0.3], "strong": [0.9, 0.9]}, min_similarity=0.0, margin=1.0
        )
        assert [layer.kind for layer in layers] == ["strong", "weak"]

    def test_no_layers_at_all_is_an_empty_list(self):
        """A tenant with nothing indexed, which is a normal state on a fresh deployment."""
        assert score_layers({}, min_similarity=0.5, margin=0.3) == []


class TestDefaults:
    def test_top_n_is_two(self):
        """Documented as a constant so the tradeoff is visible where it is set rather than buried
        in a signature default."""
        assert DEFAULT_TOP_N == 2
