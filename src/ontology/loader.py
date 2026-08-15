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

    def is_governing(self, predicate: str) -> bool:
        return predicate in self.governing_predicates

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
