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
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

ONTOLOGY_DIR = Path(__file__).resolve().parents[2] / "ontologies"

#: What each `blocks:` value means, as the endpoints of the edge it taints.
#:
#: A predicate has to say which end, because the answer differs per predicate and only the pack
#: author knows: a conflict taints both parties, stale authority taints the citing document but
#: not the authority itself, and a contraindication taints the drug but not the patient. Guessing
#: in Python would withhold "Brown was overruled" — the one fact a reader needs.
BLOCK_ENDPOINTS: dict[str, tuple[str, ...]] = {
    "subject": ("subject",),
    "object": ("object",),
    "both": ("subject", "object"),
}

#: Runs of anything that is not a letter or digit, in the part of an entity id after the colon.
#: Collapsed to one hyphen, so `Calder Shipping AG`, `calder_shipping_ag` and `calder--shipping`
#: are one id. Unicode-aware (`\w` minus `_`) rather than ASCII-only: a German or French party
#: name is an ordinary thing in this domain, and transliterating it would be a guess.
_ENTITY_LOCAL_NOISE = re.compile(r"[\W_]+", re.UNICODE)


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

    blocks: str = ""
    """Which endpoint a fact with this predicate forbids an answer about, or empty for one that
    only informs. See `BLOCK_ENDPOINTS`.

    Declared rather than derived from `rule_conclusions`, which it currently coincides with in
    both packs. The coincidence is not a guarantee: an enrichment rule concluding something benign
    would silently become a veto, and a veto nobody declared is as bad as a veto that never ran."""

    transitive: bool = False
    """Whether a rule may walk a chain of these edges as one premise.

    Declared per predicate because transitivity is a claim about the world that only the pack
    author can make: ownership carries through a chain of holding companies, and `ADVERSE_TO`
    emphatically does not — opposing A, who opposes B, does not put you against B."""


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

    external_id: bool = False
    """Whether the part after the colon is an identifier some other system owns.

    `matter:NTL-2026-0114` is a case-management id and `table:src-1:legal.matters` is a Glue
    name; normalising either breaks the join back to the system that issued it. A party's id, by
    contrast, is minted here from words on a page, so it is ours to canonicalise.

    Declared per pack because only the author knows which of their kinds are foreign keys.
    Defaults to False -- normalise -- deliberately: forgetting to mark a kind external mangles an
    id and breaks matter-scoped reads loudly, while forgetting the other way leaves the silent
    conflict miss this whole mechanism exists to prevent."""

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

    entity_suffixes: frozenset[str] = frozenset()
    """Trailing words that are legal-form noise rather than part of a name, for *detection only*.

    Declared per pack because which words those are is domain knowledge: `gmbh` means nothing in
    the healthcare pack, and hardcoding a legal list in Python would make the second pack a lie.
    Used by `entity_blocking_keys` to ask a human whether two ids are one company, and never by
    `canonical_entity_id` — see that method for why dropping a word must not reach a stored id."""

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

    @functools.cached_property
    def blocking_predicates(self) -> frozenset[str]:
        """Predicates whose facts forbid an answer rather than informing one.

        Closed, and closed for a stronger reason than the rest of the vocabulary: a veto nobody
        can query is a governance control that reports "nothing refused" because it is incapable
        of refusing, and that reads exactly like a clean conflict check.
        """
        return frozenset(p.id for p in self.predicates.values() if p.blocks)

    @functools.cached_property
    def transitive_predicates(self) -> frozenset[str]:
        """Predicates a rule may follow as a chain rather than a single edge.

        Closed, like the blocks. A path premise over a predicate nobody declared transitive
        would conclude something the pack never claimed followed from a chain.
        """
        return frozenset(p.id for p in self.predicates.values() if p.transitive)

    def blocked_endpoints(self, predicate: str) -> tuple[str, ...]:
        """Which ends of an edge with this predicate are tainted. Empty for a non-blocking one."""
        pdef = self.predicates.get(predicate)
        if pdef is None or not pdef.blocks:
            return ()
        return BLOCK_ENDPOINTS.get(pdef.blocks, ())

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

    def _entity_def_for(self, kind: str) -> EntityDef | None:
        for e in self.entities.values():
            if e.slug == kind:
                return e
        return None

    def canonical_entity_id(self, entity_id: str) -> str | None:
        """One spelling of an entity, so two documents naming it land on one node.

        The gap this closes: `MERGE (s:Entity {tenant_id, entity_id})` makes any variation a new
        node, and `entity_kind_of` lowercases the prefix before checking it, so `Party:Calder`,
        `party:calder` and `party: Calder ` all pass the closed-kind guard and fork. A conflict
        check joins on the shared node, so the fork returns nothing and reads as clean -- the
        failure the closed *predicate* vocabulary exists to prevent, unguarded for entity names.

        Deliberately conservative, in the shape of `ground()`: exact rules, no LLM, and None
        rather than a guess. Only presentation is normalised -- case, whitespace, punctuation
        runs. Nothing is *dropped*, so no information is lost: this runs before `_compute_id`, so
        a wrong collapse would leave no record anywhere that two names were merged. Stripping a
        corporate suffix would be such a collapse (`Calder Shipping AG` and `Calder Shipping Ltd`
        are commonly a parent and its subsidiary, which is what `AFFILIATE_OF` records), so it
        belongs in duplicate *detection*, where a human adjudicates, and never here.

        Not `_slugify` from `discovery.enrichment`, which is right for a term node and wrong
        here: it strips everything outside `[a-z0-9]`, turning `table:src-1:legal.matters` into
        one flat run. These two look like duplication and are not.
        """
        kind = self.entity_kind_of(entity_id)
        if kind is None:
            # The kind guard stays the thing that rejects. Normalising an id whose kind this pack
            # does not declare would launder an invented kind into a well-formed-looking id.
            return None
        _, _, local = entity_id.partition(":")
        definition = self._entity_def_for(kind)
        if definition is not None and definition.external_id:
            # Another system's identifier. Case and punctuation carry meaning there, and
            # `GraphExplorer` rebuilds `matter:${matter_id}` from the uncased property.
            return f"{kind}:{local}"
        local = _ENTITY_LOCAL_NOISE.sub("-", local.strip().lower()).strip("-")
        if not local:
            # Punctuation only. `kind:` alone is not an entity, and `entity_kind_of` rejects it.
            return None
        return f"{kind}:{local}"

    def entity_blocking_keys(self, entity_id: str) -> frozenset[str]:
        """Keys two ids share when they may name one thing, for putting a question to a human.

        Blocking keys, deliberately not similarity. No threshold, no edit distance, no fuzzy
        matching, because no threshold is correct at any setting: 0.7 merges "Acme Corp" with
        "Acme Holdings". Two ids collide here or they do not, which is reproducible forever and
        explainable to a lawyer in one sentence.

        Two keys per id. The first is the normalised local part with separators removed, so
        `calder-shipping-ag` and `Calder Shipping AG` collide. The second additionally drops
        trailing corporate-form words, so `calder-shipping` collides with `calder-shipping-ag`.

        That second key is exactly what `canonical_entity_id` refuses to do, and the asymmetry is
        the design: a *stored* id with its suffix dropped would silently merge a parent with its
        subsidiary, which `AFFILIATE_OF` exists to distinguish. A *key* is never stored, drives no
        query, and its only consequence is a reviewer being asked. Over-generating candidates
        costs three seconds; under-generating costs a missed conflict.

        Empty for an external id: the issuing system already guarantees uniqueness there, so two
        Glue tables with similar names are not a duplicate anybody should resolve.
        """
        canonical = self.canonical_entity_id(entity_id)
        if canonical is None:
            return frozenset()
        kind, _, local = canonical.partition(":")
        definition = self._entity_def_for(kind)
        if definition is not None and definition.external_id:
            return frozenset()
        squashed = local.replace("-", "")
        if not squashed:
            return frozenset()
        keys = {f"{kind}:{squashed}"}
        words = [w for w in local.split("-") if w]
        # One suffix at a time, and never the last word standing: `party:ag` is not a company.
        while len(words) > 1 and words[-1] in self.entity_suffixes:
            words = words[:-1]
        stripped = "".join(words)
        if stripped:
            keys.add(f"{kind}:{stripped}")
        return frozenset(keys)

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
            external_id=bool(e.get("external_id", False)),
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
                blocks=str(p.get("blocks", "") or ""),
                transitive=bool(p.get("transitive", False)),
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
        entity_suffixes=frozenset(str(s).lower() for s in raw.get("entity_suffixes", ())),
    )
    _validate_blocks(predicates, ontology.domain)
    _validate_transitive(predicates, ontology.domain)
    _validate_entity_ids(entities, ontology.domain)
    _validate_rules(ontology)
    return ontology


def _validate_entity_ids(entities: dict[str, EntityDef], domain: str) -> None:
    """Refuse a pack that would let a catalogued id be rewritten.

    A `catalog` entity's id is always some other system's name -- a Glue database, table or
    column -- so normalising it breaks the join back to that system. `layer: catalog` and
    `external_id: true` therefore have to agree, and the pack is the only place that can say so.

    Only this direction is checked. A *domain* kind may legitimately be either: `Party` is minted
    from words on a page, while `Matter` carries a case-management reference.
    """
    for e in entities.values():
        if e.layer == "catalog" and not e.external_id:
            raise ValueError(
                f"entity {e.id!r} in the {domain} pack is layer: catalog but not "
                "external_id: true; a catalogued id belongs to the system that issued it and "
                "normalising it would break the join back to that system"
            )


def _validate_blocks(predicates: dict[str, PredicateDef], domain: str) -> None:
    """Refuse a pack whose `blocks:` value is not one this code understands.

    Skipping an unreadable value would leave a predicate the author wrote as a veto informing
    answers instead of forbidding them, and nothing downstream can tell that apart from a
    predicate nobody meant to block. Loud at load, not silent per query.
    """
    for pdef in predicates.values():
        if pdef.blocks and pdef.blocks not in BLOCK_ENDPOINTS:
            raise ValueError(
                f"predicate {pdef.id!r} in the {domain} pack declares blocks: {pdef.blocks!r}, "
                f"which is not one of {sorted(BLOCK_ENDPOINTS)}; a veto that cannot be read is "
                "a veto that never fires"
            )
        if pdef.blocks and not pdef.governing:
            # A descriptive predicate is open and unvalidated, so a veto resting on one could be
            # minted by any extractor inventing a tag. Blocks have to come off the closed set.
            raise ValueError(
                f"predicate {pdef.id!r} in the {domain} pack declares blocks: {pdef.blocks!r} but "
                "is descriptive; only a governing predicate may veto an answer"
            )


def _validate_transitive(predicates: dict[str, PredicateDef], domain: str) -> None:
    """Refuse a `transitive:` declaration a chain could not soundly follow.

    Both refusals are about a walk that would produce conclusions the pack never claimed:

    - **Descriptive.** The descriptive half is open and unvalidated, so an extractor inventing a
      tag could mint the links of the chain. A path has to run over the closed set.
    - **Nothing to continue into.** A chain needs the object of one edge to be a legal subject of
      the next, so `domain` and `range` must overlap. `REPRESENTS` is Counsel->Party and stops
      after one hop; declaring it transitive would silently be a no-op rather than an error.

    Symmetric is deliberately *not* refused here. It is a real hazard — a walk oscillates between
    two nodes — but `_expand_path` breaks that with visited-node tracking, and a genuinely
    symmetric transitive predicate (shares an ingredient with) is a coherent thing for a pack to
    declare. Refusing it here would be refusing a sound pack on the strength of an engine detail.
    """
    for pdef in predicates.values():
        if not pdef.transitive:
            continue
        if not pdef.governing:
            raise ValueError(
                f"predicate {pdef.id!r} in the {domain} pack declares transitive: true but is "
                "descriptive; a chain a rule walks has to run over the closed vocabulary, or an "
                "extractor could invent its links"
            )
        if not (set(pdef.domain) & set(pdef.range)):
            raise ValueError(
                f"predicate {pdef.id!r} in the {domain} pack declares transitive: true but its "
                f"domain {list(pdef.domain)} and range {list(pdef.range)} do not overlap, so a "
                "chain has nothing to continue into"
            )


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
            if premise.is_path and premise.predicate not in ontology.transitive_predicates:
                raise PatternError(
                    f"rule {rule.id!r} walks a chain of {premise.predicate!r}, which the "
                    f"{ontology.domain} pack does not declare transitive; a conclusion drawn "
                    "along that chain would claim something the pack never said follows from it"
                )


@functools.lru_cache(maxsize=8)
def load_ontology(domain: str = "legal") -> Ontology:
    path = ONTOLOGY_DIR / f"{domain}.yaml"
    if not path.exists():
        available = sorted(p.stem for p in ONTOLOGY_DIR.glob("*.yaml"))
        raise FileNotFoundError(f"no ontology pack {domain!r}; available: {available}")
    return _parse(yaml.safe_load(path.read_text()))
