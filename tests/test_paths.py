"""Tests for chain assembly.

The load-bearing claim is that a chain is a *reading* of edges the reader already has, never a new
fact. So the tests that matter are the ones pinning what may not appear in a chain: an edge the walk
did not return, a hop through a document, or a link the graph does not hold.

The worked example throughout is the one that motivated the module. "Who helped Sam Parker commit
fraud" is answered by three edges that share two intermediate entities, and measured against the
live retail demo all three came back at `hops=1`, none adjacent in the list, among distractors
differing by four characters.
"""

from __future__ import annotations

from typing import Any

from src.query.paths import MAX_CHAIN_LENGTH, MAX_FANOUT, chains

FLOOR = 0.8


def edge(
    assertion_id: str,
    subject: str,
    predicate: str,
    obj: str,
    *,
    confidence: float = 0.95,
    epistemic_class: str = "EXTRACTED_MODEL",
) -> dict[str, Any]:
    return {
        "assertion_id": assertion_id,
        "subject_id": subject,
        "predicate": predicate,
        "object_id": obj,
        "epistemic_class": epistemic_class,
        "confidence": confidence,
    }


#: The chain the retail demo needed and could not show: a customer, an order, its return, and the
#: associate who approved it.
FRAUD_CHAIN = [
    edge("a1", "customer:sam-parker", "PLACED_ORDER", "order:ord-2026-04417"),
    edge("a2", "return:rtn-2026-00912", "RETURN_OF", "order:ord-2026-04417"),
    edge("a3", "associate:curtis-lindgren", "APPROVED_RETURN", "return:rtn-2026-00912"),
]


def only(paths: list[dict[str, Any]]) -> dict[str, Any]:
    assert len(paths) == 1, [p["nodes"] for p in paths]
    return paths[0]


class TestTheConnectionIsFound:
    def test_three_edges_sharing_two_entities_become_one_chain(self):
        chain = only(chains(FRAUD_CHAIN))
        assert set(chain["nodes"]) == {
            "customer:sam-parker",
            "order:ord-2026-04417",
            "return:rtn-2026-00912",
            "associate:curtis-lindgren",
        }
        assert len(chain["steps"]) == 3

    def test_a_distractor_sharing_no_entity_is_not_joined_in(self):
        """The failure this replaces: `ord-2024-00008` and `ord-2026-04417` differ by four
        characters, and a model asked to join a flat list has nothing but those characters to go on.
        The join is on entity identity here, so a near-miss id is simply a different node."""
        noise = edge("z1", "customer:other", "PLACED_ORDER", "order:ord-2024-00008")
        chain = only(chains([*FRAUD_CHAIN, noise]))
        assert "z1" not in [s["assertion_id"] for s in chain["steps"]]

    def test_the_order_the_edges_arrived_in_does_not_change_the_chain(self):
        """A walk ranks by predicate and confidence, so the three edges of one connection arrive
        scattered. If arrival order changed the result, a chain would be an artefact of ranking."""
        forward = chains(FRAUD_CHAIN)
        backward = chains(list(reversed(FRAUD_CHAIN)))
        assert [p["nodes"] for p in forward] == [p["nodes"] for p in backward]

    def test_a_disconnected_pair_of_edges_yields_nothing(self):
        separate = [
            edge("b1", "party:acme", "REPRESENTS", "matter:m-1"),
            edge("b2", "party:beta", "REPRESENTS", "matter:m-2"),
        ]
        assert chains(separate) == []


class TestWhatMayNotAppearInAChain:
    def test_a_single_edge_is_not_reported_as_a_connection(self):
        """One hop is the row the flat list already shows. Presenting it as a connection would claim
        the graph found something it did not."""
        assert chains([FRAUD_CHAIN[0]]) == []

    def test_no_chain_passes_through_a_document(self):
        """The same call `expand` makes for its frontier. A document node touches every fact
        extracted from it, so a hop through one would report "these appeared in the same file" as a
        relationship -- and would multiply one connection into a copy per citing document."""
        via_document = [
            edge("d1", "document:doc-a59", "MENTIONS", "party:acme", confidence=FLOOR),
            edge("d2", "document:doc-a59", "MENTIONS", "party:beta", confidence=FLOOR),
        ]
        assert chains(via_document) == []

    def test_no_chain_passes_through_a_catalogued_table(self):
        """`HAS_COLUMN` is schema, not evidence. A forty-column table would otherwise produce a
        chain for every pair of its columns, which is the noise `expand`'s ranking already fights.
        """
        schema = [
            edge("c1", "table:iceberg_db.orders", "HAS_COLUMN", "column:order_id", confidence=1.0),
            edge("c2", "table:iceberg_db.orders", "HAS_COLUMN", "column:total", confidence=1.0),
        ]
        assert chains(schema) == []

    def test_a_self_loop_is_not_a_hop(self):
        loop = [
            edge("s1", "party:acme", "SAME_AS", "party:acme"),
            edge("s2", "party:acme", "REPRESENTS", "matter:m-1"),
        ]
        assert chains(loop) == []

    def test_a_row_with_no_assertion_id_cannot_be_a_step(self):
        """Every step has to be citable. A hop nobody can take to `get_provenance` is exactly the
        unauditable join this module exists to remove."""
        anonymous = {k: v for k, v in FRAUD_CHAIN[1].items() if k != "assertion_id"}
        assert chains([FRAUD_CHAIN[0], anonymous, FRAUD_CHAIN[2]]) == []

    def test_a_malformed_row_does_not_take_the_rest_down(self):
        assert only(chains([*FRAUD_CHAIN, None, "not an edge", {}]))["steps"]


class TestDirection:
    def test_a_step_walked_against_its_own_direction_is_marked(self):
        """`subject_id` and `object_id` stay as written, so `reversed` is the only thing that says
        the chain read the edge backwards. A renderer that ignored it would print `A PLACED_ORDER B`
        for a chain arriving at A from B, which is a false statement rather than a layout bug."""
        chain = only(chains(FRAUD_CHAIN))
        by_id = {s["assertion_id"]: s for s in chain["steps"]}
        # The walk starts at the associate, so it meets every edge at its object.
        assert by_id["a3"]["reversed"] is False
        assert by_id["a2"]["reversed"] is False
        assert by_id["a1"]["reversed"] is True

    def test_the_flags_are_consistent_with_the_node_order(self):
        for chain in chains(FRAUD_CHAIN):
            for i, step in enumerate(chain["steps"]):
                entered, left = chain["nodes"][i], chain["nodes"][i + 1]
                assert {entered, left} == {step["subject_id"], step["object_id"]}
                assert step["reversed"] == (entered == step["object_id"])

    def test_a_chain_and_the_same_chain_backwards_are_one_result(self):
        assert len(chains(FRAUD_CHAIN)) == 1


class TestSubChainsAreSuppressed:
    def test_a_three_hop_chain_does_not_also_report_its_two_hop_stretches(self):
        """A four-node chain contains two two-hop stretches. Listing them would fill the panel with
        restatements of one connection, which is the flat-list problem again in a new shape."""
        assert len(chains(FRAUD_CHAIN)) == 1

    def test_two_routes_through_a_shared_entity_are_both_kept(self):
        """A shared middle is not a shared connection. Two associates who each approved a return on
        Sam's order are two findings, and reporting one would name one of them and not the other."""
        branching = [
            *FRAUD_CHAIN,
            edge("a4", "associate:naomi-ferreira", "APPROVED_RETURN", "return:rtn-2026-00700"),
            edge("a5", "return:rtn-2026-00700", "RETURN_OF", "order:ord-2026-04417"),
        ]
        found = chains(branching)
        from_sam = {
            c["nodes"][-1] for c in found if c["nodes"][0] == "customer:sam-parker"
        } | {c["nodes"][0] for c in found if c["nodes"][-1] == "customer:sam-parker"}
        assert from_sam == {"associate:curtis-lindgren", "associate:naomi-ferreira"}


class TestConfidenceIsTheWeakestHop:
    def test_the_weakest_hop_decides(self):
        weakened = [
            FRAUD_CHAIN[0],
            edge(
                "a2",
                "return:rtn-2026-00912",
                "RETURN_OF",
                "order:ord-2026-04417",
                confidence=FLOOR,
            ),
            FRAUD_CHAIN[2],
        ]
        assert only(chains(weakened))["confidence"] == FLOOR

    def test_a_chain_resting_on_a_floor_hop_sorts_below_one_that_does_not(self):
        """No second rule needed for presence claims: `MENTIONS` sits exactly on the trust floor by
        design, so weakest-hop ordering demotes any chain that rests on one."""
        strong = [
            edge("s1", "party:acme", "REPRESENTS", "matter:m-1"),
            edge("s2", "party:acme", "ADVERSE_TO", "party:beta"),
        ]
        weak = [
            edge("w1", "party:gamma", "MENTIONS", "matter:m-9", confidence=FLOOR),
            edge("w2", "party:gamma", "MENTIONS", "party:delta", confidence=FLOOR),
        ]
        found = chains([*weak, *strong])
        assert [c["confidence"] for c in found] == [0.95, FLOOR]

    def test_a_missing_confidence_is_zero_rather_than_assumed(self):
        unstated = {k: v for k, v in FRAUD_CHAIN[1].items() if k != "confidence"}
        assert only(chains([FRAUD_CHAIN[0], unstated, FRAUD_CHAIN[2]]))["confidence"] == 0.0


class TestBounds:
    def _line(self, length: int) -> list[dict[str, Any]]:
        return [
            edge(f"L{i}", f"party:p{i}", "REPRESENTS", f"party:p{i + 1}") for i in range(length)
        ]

    def test_a_chain_is_capped_at_the_module_length(self):
        found = chains(self._line(MAX_CHAIN_LENGTH + 3))
        assert found
        assert max(len(c["steps"]) for c in found) == MAX_CHAIN_LENGTH

    def test_the_longest_chain_is_reported_first(self):
        """Longest first: a long connection is precisely what the flat list could not show."""
        mixed = [*self._line(MAX_CHAIN_LENGTH), edge("x1", "party:q1", "REPRESENTS", "party:q2")]
        mixed.append(edge("x2", "party:q2", "REPRESENTS", "party:q3"))
        found = chains(mixed)
        assert len(found[0]["steps"]) == MAX_CHAIN_LENGTH

    def test_the_result_count_is_capped(self):
        many: list[dict[str, Any]] = []
        for group in range(30):
            many.append(edge(f"g{group}a", f"party:a{group}", "REPRESENTS", f"matter:m{group}"))
            many.append(edge(f"g{group}b", f"party:b{group}", "REPRESENTS", f"matter:m{group}"))
        assert len(chains(many, limit=5)) == 5

    def test_a_hub_is_only_crossed_to_its_strongest_neighbours(self):
        """A well-connected node -- a matter with two hundred documents -- would otherwise
        generate a chain for every pair of its edges. The cut is by confidence, so the neighbours
        reachable across a hub are the strongest rather than whichever the walk listed first."""
        spokes = [
            edge(f"h{i}", f"party:p{i}", "REPRESENTS", "matter:hub", confidence=0.9 - i / 1000)
            for i in range(60)
        ]
        crossed = {c["nodes"][2] for c in chains(spokes, limit=1000)}
        assert crossed == {f"party:p{i}" for i in range(MAX_FANOUT)}

    def test_an_empty_hit_list_is_an_empty_result(self):
        assert chains([]) == []


class TestDeterminism:
    def test_the_same_edges_produce_the_same_chains_twice(self):
        """Two runs of one question returning different connections is indistinguishable from the
        graph having changed, which is the whole reason `expand` sorts on `assertion_id` last."""
        edges = [
            *FRAUD_CHAIN,
            edge("a4", "associate:naomi-ferreira", "APPROVED_RETURN", "return:rtn-2026-00700"),
            edge("a5", "return:rtn-2026-00700", "RETURN_OF", "order:ord-2026-04417"),
        ]
        assert chains(edges) == chains(list(reversed(edges)))
