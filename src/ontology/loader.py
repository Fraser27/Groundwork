"""Load a domain ontology pack and expose the closed predicate vocabulary.

Domain-agnostic by construction: the platform reads whichever pack it is pointed
at, and `ontologies/legal.yaml` is simply the default. `ontologies/healthcare.yaml`
exists to keep that honest — if a second domain ever stops loading cleanly, the
abstraction has quietly rotted.

The pack's real job is answering one question at write time: *is this predicate
allowed?* Governing predicates are closed and validated; descriptive ones are open.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

ONTOLOGY_DIR = Path(__file__).resolve().parents[2] / "ontologies"


@dataclass(frozen=True)
class PredicateDef:
    id: str
    label: str
    description: str
    governing: bool
    domain: tuple[str, ...] = ()
    range: tuple[str, ...] = ()
    help: str | None = None
    symmetric: bool = False


@dataclass(frozen=True)
class EntityDef:
    id: str
    label: str
    description: str
    help: str | None = None

    layer: str = "domain"
    """Which half of the graph this belongs to: `domain` for facts read out of documents,
    `catalog` for schema declared by a system of record.

    One graph on purpose -- a metric over `matters` should reconcile with a fact from a page -- but
    a reader auditing a conflict check does not want a firm's Glue columns on screen. Declared here
    rather than inferred from an id prefix, so a domain pack can add an entity kind without a UI
    change.
    """

    @property
    def slug(self) -> str:
        """The id prefix form: `Party` -> `party`.

        Entity ids are written `kind:slug` in lowercase while this vocabulary is declared
        capitalised. Comparing one against the other without normalising is how `court:` came to
        exist in the graph while `Court` sat declared and apparently unused.
        """
        return self.id.lower()


@dataclass(frozen=True)
class RuleDef:
    id: str
    version: str
    description: str
    when: tuple[str, ...]
    then: str
    min_premise_class: str
    help: str | None = None

    @property
    def method(self) -> str:
        """The `method` string INFERRED assertions from this rule will carry."""
        return f"rule:{self.id}@{self.version}"


@dataclass(frozen=True)
class Ontology:
    domain: str
    version: int
    entities: dict[str, EntityDef]
    predicates: dict[str, PredicateDef]
    rules: tuple[RuleDef, ...]

    @functools.cached_property
    def governing_predicates(self) -> frozenset[str]:
        """The closed set. `build_assertion` rejects anything outside it."""
        return frozenset(p.id for p in self.predicates.values() if p.governing)

    @functools.cached_property
    def descriptive_predicates(self) -> frozenset[str]:
        return frozenset(p.id for p in self.predicates.values() if not p.governing)

    @functools.cached_property
    def rule_conclusions(self) -> frozenset[str]:
        """Predicates only a rule may write.

        Declaring the conclusions as governing predicates -- which they have to be, so
        `build_assertion` accepts them and a typo in a rule fails at load -- also put them in the
        list handed to the extractor's prompt. A model then proposed `POTENTIAL_CONFLICT`
        directly, which is exactly the collapse the epistemic axis exists to prevent: a conflict
        a model guessed at and one a rule derived from two signed-off facts became the same kind
        of object, and only the second carries a proof tree.
        """
        from src.ontology.patterns import PatternError, parse_rule

        out: set[str] = set()
        for rule in self.rules:
            try:
                out.add(parse_rule(rule).conclusion.predicate)
            except PatternError:
                # An unparseable rule is reported by the reasoner. Here it simply contributes no
                # conclusion, which errs toward letting an extractor propose it rather than
                # silently forbidding a predicate for an unrelated reason.
                continue
        return frozenset(out)

    @functools.cached_property
    def extractable_predicates(self) -> frozenset[str]:
        """What an extractor may propose: everything except what a rule concludes."""
        return (self.governing_predicates | self.descriptive_predicates) - self.rule_conclusions

    def is_governing(self, predicate: str) -> bool:
        return predicate in self.governing_predicates

    @property
    def entity_kinds(self) -> frozenset[str]:
        """Every declared entity kind, as it appears in an id prefix.

        Closed, like the governing predicates and for the same reason. An extractor free to invent
        a kind produces a graph nobody can query: `vessel:mv-aurelia` and `ship:mv-aurelia` are two
        nodes, and a traversal finds one of them. Unlike a descriptive predicate, an entity kind is
        structural -- it decides which node a fact hangs off -- so a new one is a deliberate pack
        change rather than something a model may improvise.
        """
        return frozenset(e.slug for e in self.entities.values())

    def entity_kind_of(self, entity_id: str) -> str | None:
        """The declared kind an id claims, or None if it claims one this pack does not have.

        Returns None for a bare id with no prefix too: an unprefixed id cannot be placed in the
        vocabulary, and guessing would defeat the point of having one.
        """
        kind, sep, rest = entity_id.partition(":")
        if not sep or not rest:
            return None
        kind = kind.lower()
        return kind if kind in self.entity_kinds else None

    def layer_of(self, entity_id: str) -> str:
        """`domain`, `catalog`, or `unknown` for an id outside the vocabulary.

        `unknown` rather than a default, because a filter that quietly files an unrecognised node
        under `domain` hides exactly the drift this vocabulary exists to surface.
        """
        kind = self.entity_kind_of(entity_id)
        if kind is None:
            return "unknown"
        for e in self.entities.values():
            if e.slug == kind:
                return e.layer
        return "unknown"

    def allowed_for(self, predicate: str) -> frozenset[str] | None:
        """The vocabulary to validate `predicate` against.

        Returns the closed set for governing predicates (so an unknown governing
        predicate fails) and None for descriptive ones (open, no validation).

        A predicate the pack has never seen is treated as descriptive: new
        subject-matter tags should not require an ontology release, whereas a new
        *governing* predicate is a deliberate schema decision.
        """
        if predicate in self.governing_predicates:
            return self.governing_predicates
        if predicate in self.descriptive_predicates:
            return None
        return None

    def canonical_pair(self, predicate: str, subject_id: str, object_id: str) -> tuple[str, str]:
        """Order the endpoints of a symmetric predicate deterministically.

        `ADVERSE_TO` declares `symmetric: true` in the pack, and until now that flag was
        only ever *reported* through the API — never enforced. So "A adverse to B" and
        "B adverse to A" produced two different content hashes and two edges for one
        fact, which fragments exactly the signal a conflict check reads.

        Sorting the endpoints collapses them: the same relationship yields the same
        `assertion_id` whichever way round a model happened to state it.
        """
        pdef = self.predicates.get(predicate)
        if pdef is None or not pdef.symmetric:
            return subject_id, object_id
        return tuple(sorted((subject_id, object_id)))  # type: ignore[return-value]

    def rule_premise_floor(self, rule_id: str) -> str | None:
        """The weakest epistemic class a rule will accept as a premise.

        Declared per rule in the pack (`min_premise_class`) and, like `symmetric`,
        previously parsed and exposed but enforced nowhere. `conflict_check` sets
        EXTRACTED_DET on purpose: a conflict flag resting on an unreviewed model guess
        would be worse than no flag, because someone would rely on it.
        """
        for rule in self.rules:
            if rule.id == rule_id:
                return rule.min_premise_class
        return None

    def ground(self, proposed: str) -> str | None:
        """STANDARD grounding: exact, case-insensitive match. No LLM.

        Deliberately conservative. An LLM deciding that `acts_on_behalf_of` means
        `REPRESENTS` is a schema decision made by a model, which is not something
        to accept silently in a system that has to be defensible.
        """
        needle = proposed.strip().upper().replace(" ", "_").replace("-", "_")
        for pid in self.predicates:
            if pid.upper() == needle:
                return pid
        for pid, pdef in self.predicates.items():
            if pdef.label.upper().replace(" ", "_") == needle:
                return pid
        return None


def _parse(raw: dict[str, Any]) -> Ontology:
    entities = {
        e["id"]: EntityDef(
            id=e["id"],
            label=e.get("label", e["id"]),
            description=e.get("description", ""),
            help=e.get("help"),
            layer=e.get("layer", "domain"),
        )
        for e in raw.get("entity_types", [])
    }

    predicates: dict[str, PredicateDef] = {}
    for group, governing in (("governing_predicates", True), ("descriptive_predicates", False)):
        for p in raw.get(group, []):
            predicates[p["id"]] = PredicateDef(
                id=p["id"],
                label=p.get("label", p["id"]),
                description=p.get("description", ""),
                governing=governing,
                domain=tuple(p.get("domain", ())),
                range=tuple(p.get("range", ())),
                help=p.get("help"),
                symmetric=bool(p.get("symmetric", False)),
            )

    rules = tuple(
        RuleDef(
            id=r["id"],
            version=r.get("version", "v1"),
            description=r.get("description", ""),
            when=tuple(r.get("when", ())),
            then=r["then"],
            min_premise_class=r.get("min_premise_class", "EXTRACTED_DET"),
            help=r.get("help"),
        )
        for r in raw.get("rules", [])
    )

    ontology = Ontology(
        domain=raw["domain"],
        version=int(raw.get("version", 1)),
        entities=entities,
        predicates=predicates,
        rules=rules,
    )
    _validate_rules(ontology)
    return ontology


def _validate_rules(ontology: Ontology) -> None:
    """Refuse a pack whose rules cannot fire, at load rather than at inference time.

    Both failures are silent otherwise, and silence is the problem: a rule that never fires and
    a rule that finds nothing look identical from the outside, and "no conflicts found" is
    exactly the answer nobody double-checks.

    Conclusions must be declared predicates because `build_assertion` validates against the
    closed vocabulary at write time. Catching it here means a typo in a rule's `then` fails the
    pack instead of throwing once, months later, the first time the rule matches something.
    """
    from src.ontology.patterns import PatternError, parse_rule

    known = ontology.governing_predicates | ontology.descriptive_predicates
    for rule in ontology.rules:
        parsed = parse_rule(rule)
        if parsed.conclusion.predicate not in known:
            raise PatternError(
                f"rule {rule.id!r} concludes {parsed.conclusion.predicate!r}, which the "
                f"{ontology.domain} pack does not declare; add it to governing_predicates or "
                "the write will be rejected at inference time"
            )
        for premise in parsed.premises:
            if premise.predicate not in known:
                raise PatternError(
                    f"rule {rule.id!r} matches on {premise.predicate!r}, which the "
                    f"{ontology.domain} pack does not declare, so it can never match"
                )


@functools.lru_cache(maxsize=8)
def load_ontology(domain: str = "legal") -> Ontology:
    path = ONTOLOGY_DIR / f"{domain}.yaml"
    if not path.exists():
        available = sorted(p.stem for p in ONTOLOGY_DIR.glob("*.yaml"))
        raise FileNotFoundError(f"no ontology pack {domain!r}; available: {available}")
    return _parse(yaml.safe_load(path.read_text()))
