import re
import json
import uuid
import asyncio
from abc import ABC
from dataclasses import dataclass, field
from typing import Dict, Optional, Any, List, Tuple, DefaultDict
from collections import defaultdict, Counter

import networkx as nx
from langchain_chroma import Chroma

from agents.memo_structure import Sub_memo_layer, MemoStructure
from eval_envs.base_envs import Basic_Recorder
from utils.hire_agent import Agent, Embedding


# ---------------------------- Utility functions ---------------------------- #

_RELATION_KEYWORDS = [
    "under", "on", "in", "inside", "into", "behind", "near", "beneath", "above"
]


def _normalize_obj_name(token: str) -> str:
    """
    Convert 'alarmclock 1' -> 'alarmclock'; 'desk 2' -> 'desk'
    Lowercases and trims digits at the end.
    """
    token = token.strip().lower()
    parts = token.split()
    if parts and parts[-1].isdigit():
        parts = parts[:-1]
    return " ".join(parts)


def _extract_go_to_targets(actions_list: List[str]) -> List[str]:
    targets = []
    for a in actions_list or []:
        if a.lower().startswith("go to "):
            tgt = a[6:].strip()
            targets.append(tgt)
    return targets


def _extract_location_from_obs(obs: str) -> Optional[str]:
    # Example: "You arrive at desk 1."
    m = re.search(r"You arrive at (?:the )?([a-zA-Z0-9 _\-]+?)\.", obs, re.IGNORECASE)
    if m:
        return m.group(1).strip().lower()
    return None


def _extract_visible_from_obs(obs: str) -> List[str]:
    # Try patterns like:
    # "you see a bed 1, a desk 1, a drawer 17, ..."
    # "On the desk 1, you see a alarmclock 1, a bowl 1, ..."
    # "Inside the fridge 1, you see a apple 1, ..."
    text = obs.replace("\n", " ").lower()
    # Find the segment "... you see ... ."
    m = re.search(r"you see (.+?)(?:\.|$)", text, re.IGNORECASE)
    if not m:
        return []
    segment = m.group(1)

    # Unify separators: ", and a " -> ", a ", " and a " -> ", a ", same for "an"
    segment = re.sub(r"\band an\b", ", an", segment)
    segment = re.sub(r"\band a\b", ", a", segment)
    segment = re.sub(r"\b, and an\b", ", an", segment)
    segment = re.sub(r"\b, and a\b", ", a", segment)

    # Split by ", a " or ", an " or leading "a " / "an "
    items = []
    # Ensure first starts with a/an if not already; handle cases like "a bed 1, a desk 1..."
    # Split naive by ", "
    raw_items = [x.strip() for x in segment.split(",") if x.strip()]
    for it in raw_items:
        # Remove leading 'a ' or 'an '
        it = re.sub(r"^(a|an)\s+", "", it).strip()
        if it:
            items.append(it)
    # Normalize names (drop trailing numbers)
    return [_normalize_obj_name(i) for i in items]


def _parse_task_from_init_obs(obs: str) -> Dict[str, Any]:
    """
    Parse goal from the init observation:
    Returns {
        "text": str,
        "targets": [str],            # target object types detected
        "relation": {"type": str, "landmark": str} or None
    }
    """
    result = {"text": "", "targets": [], "relation": None}
    text = obs.strip()
    m = re.search(r"Your task is to:\s*(.+)$", text, flags=re.IGNORECASE | re.DOTALL)
    if m:
        goal = m.group(1).strip()
        result["text"] = goal
        # Extract relation and landmark if possible
        rel_found = None
        for rel in _RELATION_KEYWORDS:
            if f" {rel} " in (" " + goal.lower() + " "):
                rel_found = rel
                break
        if rel_found:
            # naive split: "look at alarmclock under the desklamp"
            parts = goal.lower().split(rel_found)
            left = parts[0].strip()
            right = parts[1].strip() if len(parts) > 1 else ""
            # Extract last noun in left and first noun in right
            left_tokens = re.findall(r"[a-zA-Z]+", left)
            right_tokens = re.findall(r"[a-zA-Z]+", right)
            if left_tokens:
                result["targets"].append(left_tokens[-1])
            if right_tokens:
                # landmark (e.g., desklamp)
                result["relation"] = {"type": rel_found, "landmark": right_tokens[0]}
        else:
            # No relation keyword; try to extract probable target nouns.
            # Heuristic: take nouns-like tokens excluding verbs like "look", "examine"
            tokens = re.findall(r"[a-zA-Z]+", goal.lower())
            stop = {"the", "a", "an", "to", "at", "and", "or", "of", "on", "in", "under",
                    "with", "from", "into", "inside", "behind", "near", "beneath", "above",
                    "look", "examine", "open", "close", "put", "take", "go"}
            candidates = [t for t in tokens if t not in stop]
            if candidates:
                # Deduplicate preserving order
                seen = set()
                dedup = []
                for c in candidates:
                    if c not in seen:
                        seen.add(c)
                        dedup.append(c)
                # Keep top 1-2 as targets
                result["targets"] = dedup[:2]
    return result


def _tokenize_action(action: str) -> Tuple[str, List[str]]:
    """
    Split action text into a verb and important object tokens (normalized).
    Returns (verb, [objects list])
    Examples:
        "take alarmclock 1 from desk 1" -> ("take", ["alarmclock", "desk"])
        "open drawer 1" -> ("open", ["drawer"])
        "examine desk 1" -> ("examine", ["desk"])
        "go to shelf 2" -> ("go", ["shelf"])  # we likely ignore "go"
    """
    a = action.strip().lower()
    verb = a.split()[0] if a else ""
    # Extract object phrases after verb, remove prepositions
    after = a[len(verb):].strip() if verb else ""
    # Split by prepositions to keep object mentions
    # Keep 'from', 'in', 'on', 'under', 'to' as separators.
    objs = re.split(r"\b(from|in|on|under|to|into|inside|behind|near|beneath|above)\b", after)
    # objs is alternating, keep non-prepositions
    obj_tokens = []
    for idx, seg in enumerate(objs):
        if idx % 2 == 0:  # non-preposition segment
            toks = seg.strip()
            if not toks:
                continue
            # split by spaces, but we want type words with possible trailing number
            name = _normalize_obj_name(toks)
            if name:
                # Sometimes have multiple words merged, try to keep last two if multiword
                # For ALFWorld usually single word.
                parts = [p for p in name.split() if p]
                if len(parts) >= 1:
                    # Keep the last one as primary type
                    obj_tokens.append(parts[-1])
    # Deduplicate
    seen = set()
    objects = []
    for o in obj_tokens:
        if o and o not in seen:
            seen.add(o)
            objects.append(o)
    return verb, objects


# ---------------------------- Memory Layers ---------------------------- #

@dataclass
class AffordanceMemory(Sub_memo_layer):
    """
    Object affordance memory:
    - Database: Dict[str, Counter] mapping object_type -> counts of verbs taken on that object.
    - Update: parse 'action_took' across steps to increment verb counts for mentioned objects.
    - Retrieve: given target object types, return ranked list of likely actions for each.
    """
    layer_intro: str = (
        "AffordanceMemory: A mapping of object types to common action verbs observed in past trajectories. "
        "database = Dict[object_type:str, Counter(verb->count)]. "
        "Update by parsing actions taken; Retrieve returns top verbs per object type."
    )
    database: Dict[str, Counter] = field(default_factory=lambda: defaultdict(Counter))

    async def retrieve(self, **kwargs) -> Dict[str, List[str]]:
        targets: List[str] = kwargs.get("targets", []) or []
        top_k: int = kwargs.get("top_k", 5)
        res: Dict[str, List[str]] = {}
        for t in targets:
            t_norm = _normalize_obj_name(t)
            counter = self.database.get(t_norm, Counter())
            if not counter:
                # Provide conservative defaults when unknown
                defaults = ["examine", "open", "close", "take", "put"]
                res[t_norm] = defaults[:top_k]
            else:
                res[t_norm] = [v for v, _ in counter.most_common(top_k)]
        return res

    async def update(self, **kwargs) -> None:
        steps: List[Dict[str, Any]] = kwargs.get("steps", []) or []
        if not isinstance(self.database, dict):
            self.database = defaultdict(Counter)
        for st in steps:
            actions = st.get("action_took", [])
            if not actions:
                continue
            action = actions[0] if isinstance(actions, list) else str(actions)
            verb, objs = _tokenize_action(action)
            if not verb or not objs:
                continue
            if verb == "go":
                continue  # navigation not an affordance of objects
            for obj in objs:
                self.database[obj][verb] += 1


@dataclass
class SpatialGraphMemory(Sub_memo_layer):
    """
    Spatial and state co-occurrence memory:
    - Database: networkx.Graph
        Nodes:
            - object:<type>    kind='object'
            - location:<type>  kind='location'
        Edges location<->object with attributes:
            - count: int       times observed
            - success: int     successes associated with this co-occurrence
    - Update: parse each observation to extract current location and visible objects.
    - Retrieve: given targets and current reachable locations, suggest best candidate locations by frequency.
    """
    layer_intro: str = (
        "SpatialGraphMemory: Maintains a bipartite graph of location types and object types with counts and success. "
        "Nodes: object:<type> (kind='object'), location:<type> (kind='location'). "
        "Edges track count and success-weighted observations."
    )
    database: nx.Graph = field(default_factory=nx.Graph)

    def _ensure_location_node(self, loc_type: str) -> str:
        node = f"location:{loc_type}"
        if node not in self.database:
            self.database.add_node(node, kind="location", name=loc_type)
        return node

    def _ensure_object_node(self, obj_type: str) -> str:
        node = f"object:{obj_type}"
        if node not in self.database:
            self.database.add_node(node, kind="object", name=obj_type)
        return node

    async def retrieve(self, **kwargs) -> Dict[str, List[Dict[str, Any]]]:
        targets: List[str] = kwargs.get("targets", []) or []
        reachable: List[str] = kwargs.get("reachable", []) or []
        # Map reachable instance names -> types
        reachable_types = [_normalize_obj_name(r) for r in reachable]
        reachable_base_types = [t for t in set(reachable_types)]

        suggestions: Dict[str, List[Dict[str, Any]]] = {}
        for t in targets:
            t_norm = _normalize_obj_name(t)
            obj_node = f"object:{t_norm}"
            if obj_node not in self.database:
                # No prior: return generic containers likely in ALF domains
                generic = ["desk", "table", "shelf", "drawer", "dresser", "cabinet", "counter", "bed", "sofa"]
                ranked = [g for g in generic if g in reachable_base_types] or generic
                suggestions[t_norm] = [{"location_type": r, "score": 0.0, "reason": "no prior memory"} for r in ranked[:5]]
                continue

            # Aggregate counts per location type from edges connected to object node
            loc_scores: Counter = Counter()
            succ_scores: Counter = Counter()
            for neighbor in self.database.neighbors(obj_node):
                if not str(neighbor).startswith("location:"):
                    continue
                edge = self.database.get_edge_data(obj_node, neighbor) or {}
                count = edge.get("count", 0)
                succ = edge.get("success", 0)
                loc_type = self.database.nodes[neighbor].get("name", neighbor.replace("location:", ""))
                loc_scores[loc_type] += count
                succ_scores[loc_type] += succ

            # Combine success signal first then count
            combined = []
            for lt, c in loc_scores.items():
                s = succ_scores.get(lt, 0)
                score = 2.0 * s + 1.0 * c
                combined.append((lt, score, s, c))
            combined.sort(key=lambda x: x[1], reverse=True)

            # Filter to reachable if available
            filtered = [x for x in combined if x[0] in reachable_base_types] or combined
            out_list = []
            for lt, sc, s, c in filtered[:8]:
                reason = f"freq={c}, success={s}"
                out_list.append({"location_type": lt, "score": float(sc), "reason": reason})
            suggestions[t_norm] = out_list
        return suggestions

    async def update(self, **kwargs) -> None:
        steps: List[Dict[str, Any]] = kwargs.get("steps", []) or []
        final_reward: float = float(kwargs.get("reward", 0.0) or 0.0)
        success_flag = 1 if final_reward and final_reward > 0 else 0

        for st in steps:
            obs = st.get("obs", "") or ""
            location_label = _extract_location_from_obs(obs)
            if not location_label:
                # could still contain list "you see ..." in the initial screen without arrival
                # but we need a container/location context; skip if unknown
                continue
            loc_type = _normalize_obj_name(location_label)

            visible_objs = _extract_visible_from_obs(obs)
            if not visible_objs:
                continue

            loc_node = self._ensure_location_node(loc_type)
            for obj_name in visible_objs:
                obj_type = _normalize_obj_name(obj_name)
                obj_node = self._ensure_object_node(obj_type)
                if not self.database.has_edge(loc_node, obj_node):
                    self.database.add_edge(loc_node, obj_node, count=0, success=0)
                self.database.edges[loc_node, obj_node]["count"] += 1
                if success_flag:
                    self.database.edges[loc_node, obj_node]["success"] += 1


@dataclass
class TrajectoryIndexMemory(Sub_memo_layer):
    """
    Trajectory-level memory using vector search (Chroma).
    - Stores completed episodes: task text, brief summary, outcomes.
    - Retrieve similar tasks for transfer hints.
    """
    layer_intro: str = (
        "TrajectoryIndexMemory: Vector index over past episode summaries using Chroma. "
        "page_content stores a concise summary. Metadata is flat: "
        "{type:'trajectory', reward:float, success:bool, task: <task text>}"
    )
    database: Optional[Chroma] = None
    embedder: Optional[Embedding] = None

    def __post_init__(self):
        if self.database is None:
            self.database = Chroma(embedding_function=self.embedder or Embedding())

    async def retrieve(self, **kwargs) -> List[Dict[str, Any]]:
        query: str = kwargs.get("query", "") or ""
        k: int = int(kwargs.get("k", 4) or 4)
        if not query.strip():
            return []
        try:
            docs = self.database.similarity_search(query=query, k=k)
        except Exception:
            return []
        results: List[Dict[str, Any]] = []
        for d in docs:
            results.append({
                "summary": d.page_content,
                "metadata": d.metadata or {}
            })
        return results

    async def update(self, **kwargs) -> None:
        init_obs: Dict[str, Any] = kwargs.get("init", {}) or {}
        steps: List[Dict[str, Any]] = kwargs.get("steps", []) or []
        reward: float = float(kwargs.get("reward", 0.0) or 0.0)

        init_text = str(init_obs.get("obs", "") or "")
        task_info = _parse_task_from_init_obs(init_text)
        task_text = task_info.get("text", "")

        # Build a compact trajectory summary
        actions_taken: List[str] = []
        visited_locations: List[str] = []
        interacted_objects: Counter = Counter()

        for st in steps:
            obs = st.get("obs", "") or ""
            act_list = st.get("action_took", [])
            if act_list:
                actions_taken.append(act_list[0])
                verb, objs = _tokenize_action(act_list[0])
                for o in objs:
                    interacted_objects[o] += 1
            loc = _extract_location_from_obs(obs)
            if loc:
                visited_locations.append(_normalize_obj_name(loc))

        visited_locations = list(dict.fromkeys(visited_locations))  # dedup preserve order
        obj_list = [o for o, _c in interacted_objects.most_common(6)]
        success = bool(reward and reward > 0)

        # Heuristic summary
        summary_lines = [
            f"Task: {task_text}",
            f"Outcome: {'SUCCESS' if success else 'FAIL'} (reward={reward})",
            f"Visited locations: {', '.join(visited_locations) if visited_locations else 'N/A'}",
            f"Key objects: {', '.join(obj_list) if obj_list else 'N/A'}",
        ]
        # Include final 3 actions as key steps
        tail_steps = actions_taken[-3:] if actions_taken else []
        if tail_steps:
            summary_lines.append("Final steps: " + " | ".join(tail_steps))

        text_to_store = "\n".join(summary_lines)
        meta = {
            "type": "trajectory",
            "reward": float(reward),
            "success": bool(success),
            "task": task_text[:512],
            "visited": json.dumps(visited_locations),
            "objects": json.dumps(obj_list)
        }
        try:
            self.database.add_texts(
                texts=[text_to_store],
                metadatas=[meta],
                ids=[str(uuid.uuid4())]
            )
        except Exception:
            # Avoid crashing updates if vector store is not available
            return


# ---------------------------- Orchestrator ---------------------------- #

class LayeredMemo(MemoStructure, ABC):
    """
    Orchestrates retrieval and updates across:
    - AffordanceMemory
    - SpatialGraphMemory
    - TrajectoryIndexMemory
    """
    def __init__(self):
        super().__init__()
        self.embedder = Embedding()
        self.affordance = AffordanceMemory()
        self.spatial = SpatialGraphMemory()
        self.trajectory = TrajectoryIndexMemory(embedder=self.embedder)

    async def general_retrieve(self, recorder: Basic_Recorder) -> Dict:
        if recorder is None or getattr(recorder, "init", None) is None:
            raise ValueError("recorder.init is required for retrieval")

        init_obs: Dict[str, Any] = recorder.init or {}
        obs_text: str = str(init_obs.get("obs", "") or "")
        actions_init: List[str] = init_obs.get("actions_list", []) or []

        # 1) Parse task and scene
        task_info = _parse_task_from_init_obs(obs_text)
        visible_now = _extract_visible_from_obs(obs_text)
        reachable_targets = _extract_go_to_targets(actions_init)
        # Only keep target names, not 'go to ' prefix
        reachable_instances = [r for r in reachable_targets]
        reachable_types = [_normalize_obj_name(r) for r in reachable_instances]

        # 2) Retrieve similar episodes from Trajectory index
        query = task_info.get("text", "")
        similar_cases = await self.trajectory.retrieve(query=query, k=4)

        # 3) Predict promising locations from Spatial graph memory
        spatial_suggestions = await self.spatial.retrieve(
            targets=task_info.get("targets", []),
            reachable=reachable_instances
        )

        # 4) Affordance suggestions for key target objects and landmarks
        affordance_targets = list(set(task_info.get("targets", []) +
                                     ([task_info["relation"]["landmark"]]
                                      if task_info.get("relation") else [])))
        affordances = await self.affordance.retrieve(
            targets=affordance_targets,
            top_k=5
        )

        # 5) Synthesize a ranked visitation plan per target:
        visitation_plan: Dict[str, List[str]] = {}
        for t, locs in spatial_suggestions.items():
            # Rank by score and keep only those that are actually reachable
            ranked_types = [x["location_type"] for x in locs]
            # Map ranked types to current instances of same type
            # Prefer those that are currently reachable
            instance_scores = []
            for inst in reachable_instances:
                inst_type = _normalize_obj_name(inst)
                if inst_type in ranked_types:
                    score = next((x["score"] for x in locs if x["location_type"] == inst_type), 0.0)
                    instance_scores.append((inst, score))
            instance_scores.sort(key=lambda x: x[1], reverse=True)
            visitation_plan[t] = [i for i, _s in instance_scores][:5] or reachable_instances[:5]

        # 6) Prepare clean output for agent
        out = {
            "task": {
                "text": task_info.get("text", ""),
                "targets": task_info.get("targets", []),
                "relation": task_info.get("relation", None)
            },
            "scene": {
                "visible_objects": visible_now,
                "reachable_locations": reachable_instances,
            },
            "knowledge": {
                "likely_locations": spatial_suggestions,
                "affordances": affordances,
                "visitation_plan": visitation_plan,
                "similar_cases": similar_cases
            },
            "meta": {
                "coverage": {
                    "affordance_objects": len(self.affordance.database or {}),
                    "spatial_nodes": int(self.spatial.database.number_of_nodes()
                                         if isinstance(self.spatial.database, nx.Graph) else 0),
                }
            }
        }
        return out

    async def general_update(self, recorder: Basic_Recorder) -> None:
        if recorder is None:
            raise ValueError("recorder is required for update")

        steps: List[Dict[str, Any]] = getattr(recorder, "steps", []) or []
        reward: float = float(getattr(recorder, "reward", 0.0) or 0.0)
        init_obs: Dict[str, Any] = getattr(recorder, "init", {}) or {}

        # Order: Affordances (local action stats) -> Spatial graph (co-occurrence) -> Trajectory index (episodic)
        await self.affordance.update(steps=steps, reward=reward)
        await self.spatial.update(steps=steps, reward=reward)
        await self.trajectory.update(init=init_obs, steps=steps, reward=reward)