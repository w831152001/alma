import re
import json
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field

import networkx as nx
from agents.memo_structure import Sub_memo_layer, MemoStructure
from eval_envs.base_envs import Basic_Recorder
from utils.hire_agent import Agent, Embedding
from langchain_chroma import Chroma


# ---------------------------- Utilities ----------------------------

def _extract_room_name(text: str) -> Optional[str]:
    if not text:
        return None
    m = re.search(r"-=\s*(.*?)\s*=-", text)
    if m:
        name = m.group(1).strip()
        if name:
            return name
    # Fallback: first non-empty line up to punctuation
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if len(line) < 80:
            return re.sub(r"[^A-Za-z0-9 _-]", "", line)
    return None


def _extract_directions(text: str) -> List[str]:
    if not text:
        return []
    dirs = set(re.findall(r"\bleading\s+([a-z]+)\b", text.lower()))
    # Also consider "door north", "passageway east" patterns
    dirs.update(re.findall(r"\b(?:door|passageway|portal|hatch)\s+leading\s+([a-z]+)\b", text.lower()))
    canonical = []
    for d in sorted(dirs):
        if d in ["north", "south", "east", "west", "up", "down", "northeast", "northwest",
                 "southeast", "southwest", "in", "out"]:
            canonical.append(d)
    return canonical


def _action_direction(action: str) -> Optional[str]:
    if not action:
        return None
    a = action.strip().lower()
    # Normalize common forms
    m = re.match(r"^(?:go|move|walk|run)\s+([a-z]+)$", a)
    if m:
        return m.group(1)
    # Single-token directions
    if a in ["north", "south", "east", "west", "up", "down", "n", "s", "e", "w", "u", "d"]:
        mapping = {"n": "north", "s": "south", "e": "east", "w": "west", "u": "up", "d": "down"}
        return mapping.get(a, a)
    return None


def _extract_objects_simple(text: str) -> List[str]:
    if not text:
        return []
    # Heuristics: capture nouns after "You see" or items split by commas
    objs: Set[str] = set()
    for line in text.splitlines():
        l = line.strip()
        if not l:
            continue
        m = re.search(r"You see (.+?)[\.\n]", l, flags=re.IGNORECASE)
        if m:
            raw = m.group(1)
            # Remove articles and descriptors
            parts = re.split(r",| and ", raw)
            for p in parts:
                noun = re.sub(r"\b(a|an|the)\b", "", p, flags=re.IGNORECASE).strip()
                noun = re.sub(r"[^A-Za-z0-9 _-]", "", noun)
                noun = re.sub(r"\s+", " ", noun).strip()
                if noun:
                    objs.add(noun)
        # Capture object-like phrases: "There is a X", "There is an X"
        m2 = re.findall(r"There is (?:a|an|the)\s+([A-Za-z0-9 _-]+?)[\.\n]", l, flags=re.IGNORECASE)
        for n in m2:
            noun = re.sub(r"[^A-Za-z0-9 _-]", "", n).strip()
            if noun:
                objs.add(noun)
    return sorted(objs)


def _canonical_task_id(goal: str, start_room: Optional[str]) -> str:
    base = f"{(goal or '').strip()}\n{(start_room or '').strip()}"
    return str(uuid.uuid5(uuid.NAMESPACE_URL, base))


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _reward_to_confidence(r: float) -> float:
    # Map reward [0,1] to confidence [0.2, 0.95]
    r = max(0.0, min(1.0, r))
    return round(0.2 + 0.75 * r, 3)


def _detect_failure_text(obs: str) -> bool:
    if not obs:
        return False
    fail_markers = [
        "you can't", "you cannot", "that's not possible", "you fail", "unable to",
        "doesn't work", "won't work", "is locked", "no such thing", "don't have that",
    ]
    o = obs.lower()
    return any(m in o for m in fail_markers)


def _estimate_action_success(prev_score: Optional[float], next_score: Optional[float], obs: str) -> bool:
    # Simple heuristic: improved score and no failure text
    if _detect_failure_text(obs):
        return False
    if (prev_score is not None) and (next_score is not None):
        return next_score >= prev_score
    return True


# ---------------------------- Layers ----------------------------

class TaskParsingLayer(Sub_memo_layer):
    def __init__(self) -> None:
        super().__init__(
            layer_intro="Parse initial observation/goal into structured fields.",
            database=None,
        )
        self.parser_agent = Agent(
            model="gpt-4o-mini",
            system_prompt=(
                "Extract structured fields from a TextWorld initial observation and goal. "
                "Be concise and avoid hallucinations. Use only what is explicitly present."
            ),
            output_schema={
                "start_room": {"type": ["string", "null"], "description": "Room name from header like '-= Room =-'"},
                "exits": {"type": "string", "description": "Comma-separated canonical directions"},
                "objects": {"type": "string", "description": "Comma-separated object names present in the start obs"},
                "goal_key_phrases": {"type": "string", "description": "Comma-separated essential goal keywords"},
            },
        )

    async def retrieve(self, **kwargs) -> Dict[str, Any]:
        init: Dict[str, Any] = kwargs.get("init", {})
        goal: str = (init.get("goal") or "").strip()
        obs: str = (init.get("obs") or "").strip()

        # Local parse
        start_room = _extract_room_name(obs)
        exits = _extract_directions(obs)
        objects = _extract_objects_simple(obs)

        # Refine via agent to fill missing details and keyword extraction
        try:
            parsed = await self.parser_agent.ask(
                f"Observation:\n{obs}\n\nGoal:\n{goal}\n\n"
                f"Return fields with directions lowercased canonical names."
            )
            agent_room = (parsed.get("start_room") or "").strip() or None
            agent_exits = [e.strip().lower() for e in (parsed.get("exits") or "").split(",") if e.strip()]
            agent_objects = [o.strip() for o in (parsed.get("objects") or "").split(",") if o.strip()]
            goal_kps = [k.strip() for k in (parsed.get("goal_key_phrases") or "").split(",") if k.strip()]
        except Exception:
            agent_room = None
            agent_exits = []
            agent_objects = []
            goal_kps = []

        final_room = agent_room or start_room
        final_exits = sorted(set(agent_exits or exits))
        final_objects = sorted(set(agent_objects or objects))
        parsed_out = {
            "goal": goal,
            "start_room": final_room,
            "exits": final_exits,
            "objects": final_objects,
            "goal_key_phrases": goal_kps,
        }
        return parsed_out

    async def update(self, **kwargs) -> None:
        # No persistent state
        return None


class SemanticTaskIndexLayer(Sub_memo_layer):
    def __init__(self) -> None:
        super().__init__(
            layer_intro="Chroma index of task signatures: goal + start room + start objects + reward.",
            database=Chroma(embedding_function=Embedding()),
        )
        self.db: Chroma = self.database  # type: ignore

    async def retrieve(self, **kwargs) -> Dict[str, Any]:
        goal: str = kwargs.get("goal", "")
        start_room: Optional[str] = kwargs.get("start_room")
        objects: List[str] = kwargs.get("objects", [])
        query = f"Goal: {goal}\nStart:{start_room or ''}\nObjects:{', '.join(objects)}"
        try:
            docs = self.db.similarity_search(query=query, k=4)
        except Exception:
            docs = []
        sims: List[Dict[str, Any]] = []
        for d in docs:
            meta = d.metadata or {}
            sims.append({
                "task_id": meta.get("task_id"),
                "reward": _safe_float(meta.get("reward"), 0.0),
                "start_room": meta.get("start_room"),
                "note": (d.page_content or "")[:280],
            })
        return {"similar_tasks": sims}

    async def update(self, **kwargs) -> None:
        recorder: Basic_Recorder = kwargs["recorder"]
        parsed: Dict[str, Any] = kwargs.get("parsed", {})
        strategy_ids: List[str] = kwargs.get("strategy_ids", [])

        start_room = parsed.get("start_room")
        goal = parsed.get("goal") or ""
        objects = parsed.get("objects", [])
        exits = parsed.get("exits", [])
        rew = _safe_float(getattr(recorder, "reward", 0.0), 0.0)
        task_id = _canonical_task_id(goal, start_room)
        content = (
            f"TASK {task_id}\n"
            f"Goal: {goal}\n"
            f"Start: {start_room}\n"
            f"Objects: {', '.join(objects)}\n"
            f"Exits: {', '.join(exits)}\n"
            f"Reward: {rew}\n"
            f"StrategyRefs: {', '.join(strategy_ids)}"
        )
        try:
            self.db.add_texts(
                texts=[content],
                metadatas=[{
                    "type": "task",
                    "task_id": task_id,
                    "reward": float(rew),
                    "start_room": (start_room or ""),
                    "timestamp": float(time.time()),
                }],
                ids=[f"task::{task_id}::{int(time.time())}"],
            )
        except Exception:
            # If add fails, ignore to avoid cascading failure
            return None


class StrategyLibraryLayer(Sub_memo_layer):
    def __init__(self) -> None:
        super().__init__(
            layer_intro="Chroma index of strategy summaries with outcomes; keyed by task semantics.",
            database=Chroma(embedding_function=Embedding()),
        )
        self.db: Chroma = self.database  # type: ignore
        self.summarizer = Agent(
            model="gpt-4o-mini",
            system_prompt=(
                "Summarize a TextWorld trajectory into a compact, transferable strategy.\n"
                "Extract key actions, preconditions, success and failure signals."
            ),
            output_schema={
                "strategy": {"type": "string", "description": "One-paragraph high-level plan"},
                "key_actions": {"type": "string", "description": "Comma-separated canonical actions"},
                "preconditions": {"type": "string", "description": "Comma-separated needed conditions/items"},
                "success_signals": {"type": "string", "description": "Comma-separated cues of progress"},
                "failure_signals": {"type": "string", "description": "Comma-separated cues of mistakes"},
            },
        )

    async def retrieve(self, **kwargs) -> Dict[str, Any]:
        goal: str = kwargs.get("goal", "")
        start_room: Optional[str] = kwargs.get("start_room")
        objects: List[str] = kwargs.get("objects", [])
        query = f"Goal: {goal}\nStart:{start_room or ''}\nObjects:{', '.join(objects)}"
        try:
            docs = self.db.similarity_search(query=query, k=5)
        except Exception:
            docs = []
        out: List[Dict[str, Any]] = []
        for d in docs:
            meta = d.metadata or {}
            r = _safe_float(meta.get("reward"), 0.0)
            conf = _reward_to_confidence(r if meta.get("success", False) else r * 0.6)
            out.append({
                "summary": (d.page_content or "")[:550],
                "confidence": conf,
                "success": bool(meta.get("success", False)),
            })
        # Deduplicate by summary start
        seen = set()
        deduped = []
        for s in out:
            key = s["summary"][:120]
            if key in seen:
                continue
            seen.add(key)
            deduped.append(s)
        return {"strategies": deduped[:5]}

    async def update(self, **kwargs) -> Dict[str, Any]:
        recorder: Basic_Recorder = kwargs["recorder"]
        parsed: Dict[str, Any] = kwargs.get("parsed", {})
        goal = parsed.get("goal") or ""
        start_room = parsed.get("start_room") or ""
        rew = _safe_float(getattr(recorder, "reward", 0.0), 0.0)
        success = rew >= 0.5  # heuristic threshold
        steps = getattr(recorder, "steps", []) or []
        # Build plain-text trajectory for the agent
        traj_lines: List[str] = []
        for st in steps:
            act = " / ".join(st.get("action_took", [])) if isinstance(st.get("action_took"), list) else (st.get("action_took") or "")
            score = st.get("scores")
            ob = st.get("obs") or ""
            ob = re.sub(r"\s+", " ", ob).strip()
            traj_lines.append(f"ACTION: {act} | SCORE: {score} | OBS: {ob[:240]}")
        traj_text = "\n".join(traj_lines[:60])

        try:
            parsed_sum = await self.summarizer.ask(
                f"Goal:\n{goal}\nStart room: {start_room}\nTrajectory:\n{traj_text}\n\n"
                "Answer with compact fields that generalize beyond specific room names."
            )
        except Exception:
            parsed_sum = {
                "strategy": "Explore systematically, collect required items, manipulate containers/doors, and verify progress signals.",
                "key_actions": "explore,inspect,take,open,unlock,use,move",
                "preconditions": "have-required-item,reachable-location,unlocked-path",
                "success_signals": "reward-increase,new-rooms,goal-item-acquired",
                "failure_signals": "repeated-failure,locked-without-key,stuck-loop",
            }

        page_content = (
            f"STRATEGY\n{parsed_sum.get('strategy', '')}\n"
            f"KEY_ACTIONS: {parsed_sum.get('key_actions', '')}\n"
            f"PRECONDITIONS: {parsed_sum.get('preconditions', '')}\n"
            f"SUCCESS_SIGNS: {parsed_sum.get('success_signals', '')}\n"
            f"FAILURE_SIGNS: {parsed_sum.get('failure_signals', '')}"
        )
        task_id = _canonical_task_id(goal, start_room)
        sid = f"strategy::{task_id}::{int(time.time())}"
        try:
            self.db.add_texts(
                texts=[page_content],
                metadatas=[{
                    "type": "strategy",
                    "task_id": task_id,
                    "reward": float(rew),
                    "success": bool(success),
                    "start_room": start_room,
                    "timestamp": float(time.time()),
                }],
                ids=[sid],
            )
        except Exception:
            return {"strategy_ids": []}
        return {"strategy_ids": [sid]}


class ObjectAffordanceLayer(Sub_memo_layer):
    def __init__(self) -> None:
        super().__init__(
            layer_intro="Chroma of object affordances discovered from action-outcome pairs.",
            database=Chroma(embedding_function=Embedding()),
        )
        self.db: Chroma = self.database  # type: ignore

    @staticmethod
    def _parse_affordances(steps: List[Dict[str, Any]]) -> List[Tuple[str, str, Optional[str], bool]]:
        # Returns list of (verb, obj, tool, success)
        affordances: List[Tuple[str, str, Optional[str], bool]] = []
        for i, st in enumerate(steps):
            act_raw = st.get("action_took")
            if isinstance(act_raw, list):
                act = " ".join(act_raw).strip().lower()
            else:
                act = (act_raw or "").strip().lower()
            if not act:
                continue
            prev_score = steps[i - 1].get("scores") if i > 0 else None
            next_score = st.get("scores")
            obs = st.get("obs") or ""
            ok = _estimate_action_success(prev_score, next_score, obs)

            def add(verb: str, obj: str, tool: Optional[str] = None) -> None:
                v = verb.strip()
                o = re.sub(r"[^A-Za-z0-9 _-]", "", obj).strip()
                t = re.sub(r"[^A-Za-z0-9 _-]", "", tool).strip() if tool else None
                if o:
                    affordances.append((v, o, t or None, ok))

            m = re.match(r"^(?:take|get|pick up)\s+(.+)$", act)
            if m:
                add("take", m.group(1))
                continue
            m = re.match(r"^(?:drop|leave)\s+(.+)$", act)
            if m:
                add("drop", m.group(1))
                continue
            m = re.match(r"^(?:open)\s+(.+)$", act)
            if m:
                add("open", m.group(1))
                continue
            m = re.match(r"^(?:close)\s+(.+)$", act)
            if m:
                add("close", m.group(1))
                continue
            m = re.match(r"^(?:unlock)\s+(.+)\s+with\s+(.+)$", act)
            if m:
                add("unlock", m.group(1), m.group(2))
                continue
            m = re.match(r"^(?:put|insert|place)\s+(.+)\s+(?:in|into|inside)\s+(.+)$", act)
            if m:
                add("put-into", m.group(2), m.group(1))
                continue
            m = re.match(r"^(?:use)\s+(.+)\s+(?:on|with)\s+(.+)$", act)
            if m:
                add("use-on", m.group(2), m.group(1))
                continue
            m = re.match(r"^(?:eat|consume)\s+(.+)$", act)
            if m:
                add("eat", m.group(1))
                continue
            m = re.match(r"^(?:drink)\s+(.+)$", act)
            if m:
                add("drink", m.group(1))
                continue
            m = re.match(r"^(?:wear|put on)\s+(.+)$", act)
            if m:
                add("wear", m.group(1))
                continue
            m = re.match(r"^(?:examine|look at)\s+(.+)$", act)
            if m:
                add("examine", m.group(1))
                continue
        return affordances

    async def retrieve(self, **kwargs) -> Dict[str, Any]:
        objects: List[str] = kwargs.get("objects", [])
        goal: str = kwargs.get("goal", "")
        results: Dict[str, Dict[str, Any]] = {}
        for obj in objects[:6]:  # limit fan-out
            query = f"{obj} related to {goal}"
            try:
                docs = self.db.similarity_search(query=query, k=3)
            except Exception:
                docs = []
            verb_stats: Dict[str, Dict[str, Any]] = {}
            for d in docs:
                meta = d.metadata or {}
                verb = str(meta.get("verb") or "").strip()
                success = bool(meta.get("success", False))
                if not verb:
                    continue
                if verb not in verb_stats:
                    verb_stats[verb] = {"count": 0, "success": 0}
                verb_stats[verb]["count"] += 1
                verb_stats[verb]["success"] += 1 if success else 0
            # Rank by success rate then count
            ranked = sorted(
                verb_stats.items(),
                key=lambda kv: (kv[1]["success"] / max(1, kv[1]["count"]), kv[1]["count"]),
                reverse=True,
            )
            if ranked:
                results[obj] = {
                    "affordances": [v for v, _ in ranked[:5]],
                    "evidence": [f"{v} ({s['success']}/{s['count']})" for v, s in ranked[:5]],
                }
        return {"object_affordances": results}

    async def update(self, **kwargs) -> None:
        recorder: Basic_Recorder = kwargs["recorder"]
        parsed: Dict[str, Any] = kwargs.get("parsed", {})
        goal = parsed.get("goal") or ""
        start_room = parsed.get("start_room") or ""
        steps = getattr(recorder, "steps", []) or []
        affs = self._parse_affordances(steps)
        if not affs:
            return None
        texts: List[str] = []
        metas: List[Dict[str, Any]] = []
        ids: List[str] = []
        task_id = _canonical_task_id(goal, start_room)
        ts = int(time.time())
        for i, (verb, obj, tool, success) in enumerate(affs[:200]):
            text = f"AFFORDANCE {verb} OBJ {obj} TOOL {tool or ''} SUCCESS {success}"
            texts.append(text)
            metas.append({
                "type": "affordance",
                "verb": verb,
                "object": obj,
                "tool": tool or "",
                "success": bool(success),
                "task_id": task_id,
                "timestamp": float(ts),
            })
            ids.append(f"aff::{task_id}::{verb}::{obj}::{i}::{ts}")
        try:
            self.db.add_texts(texts=texts, metadatas=metas, ids=ids)
        except Exception:
            return None


class FailureMemoryLayer(Sub_memo_layer):
    def __init__(self) -> None:
        super().__init__(
            layer_intro="Chroma of failure modes and recovery tips mined from low-reward trajectories.",
            database=Chroma(embedding_function=Embedding()),
        )
        self.db: Chroma = self.database  # type: ignore
        self.analyzer = Agent(
            model="gpt-4o-mini",
            system_prompt=(
                "Analyze failures in a TextWorld trajectory. Identify pitfalls and suggest corrective actions."
            ),
            output_schema={
                "pitfalls": {"type": "string", "description": "Comma-separated pitfalls"},
                "fixes": {"type": "string", "description": "Comma-separated recommended fixes"},
            },
        )

    async def retrieve(self, **kwargs) -> Dict[str, Any]:
        goal: str = kwargs.get("goal", "")
        try:
            docs = self.db.similarity_search(query=f"pitfalls for {goal}", k=3)
        except Exception:
            docs = []
        out: List[str] = []
        for d in docs:
            content = (d.page_content or "").strip()
            if content:
                out.append(content[:400])
        # Deduplicate
        seen = set()
        uniq = []
        for p in out:
            key = p[:120]
            if key in seen:
                continue
            seen.add(key)
            uniq.append(p)
        return {"pitfalls": uniq[:3]}

    async def update(self, **kwargs) -> None:
        recorder: Basic_Recorder = kwargs["recorder"]
        parsed: Dict[str, Any] = kwargs.get("parsed", {})
        goal = parsed.get("goal") or ""
        start_room = parsed.get("start_room") or ""
        rew = _safe_float(getattr(recorder, "reward", 0.0), 0.0)
        steps = getattr(recorder, "steps", []) or []

        # Detect repetition/loops
        actions = []
        for st in steps:
            act = " / ".join(st.get("action_took", [])) if isinstance(st.get("action_took"), list) else (st.get("action_took") or "")
            actions.append(act.strip().lower())
        loop_flag = len(actions) != len(set(actions)) and len(actions) > 6

        if rew >= 0.35 and not loop_flag:
            return None

        traj = []
        for st in steps[:80]:
            act = " / ".join(st.get("action_took", [])) if isinstance(st.get("action_took"), list) else (st.get("action_took") or "")
            obs = re.sub(r"\s+", " ", (st.get("obs") or "")).strip()
            traj.append(f"ACTION: {act} | OBS: {obs[:200]}")
        traj_text = "\n".join(traj)

        try:
            analysis = await self.analyzer.ask(
                f"Goal:\n{goal}\nTrajectory:\n{traj_text}\n\nIdentify pitfalls and fixes."
            )
            pitfalls = [p.strip() for p in (analysis.get("pitfalls") or "").split(",") if p.strip()]
            fixes = [f.strip() for f in (analysis.get("fixes") or "").split(",") if f.strip()]
            content = f"PITFALLS: {', '.join(pitfalls)}\nFIXES: {', '.join(fixes)}"
        except Exception:
            content = "PITFALLS: repeated actions, missing key, wrong container\nFIXES: search new rooms, acquire key, try different containers"

        task_id = _canonical_task_id(goal, start_room)
        try:
            self.db.add_texts(
                texts=[content],
                metadatas=[{
                    "type": "pitfall",
                    "task_id": task_id,
                    "reward": float(rew),
                    "loop": bool(loop_flag),
                    "timestamp": float(time.time()),
                }],
                ids=[f"pit::{task_id}::{int(time.time())}"],
            )
        except Exception:
            return None


class ProceduralPathMemoryLayer(Sub_memo_layer):
    def __init__(self) -> None:
        super().__init__(
            layer_intro="NetworkX graph of observed room transitions, with direction statistics.",
            database=nx.Graph(),
        )
        self.graph: nx.Graph = self.database  # type: ignore

    async def retrieve(self, **kwargs) -> Dict[str, Any]:
        start_room: Optional[str] = kwargs.get("start_room")
        exits: List[str] = kwargs.get("exits", [])
        if not exits:
            return {"exploration_prior": {"by_direction": [], "notes": "No exits parsed."}}

        dir_scores: Dict[str, float] = {d: 0.0 for d in exits}
        # Prefer room-specific data
        if start_room and (start_room in self.graph):
            for nbr in self.graph.neighbors(start_room):
                data = self.graph.get_edge_data(start_room, nbr) or {}
                dir_counts = data.get("dir_counts", {}) or {}
                for d in exits:
                    dir_scores[d] += float(dir_counts.get(d, 0))
        # Fallback to global
        if all(v == 0.0 for v in dir_scores.values()):
            for u, v, data in self.graph.edges(data=True):
                dir_counts = data.get("dir_counts", {}) or {}
                for d in exits:
                    dir_scores[d] += float(dir_counts.get(d, 0)) * 0.5  # down-weight global

        ranked = sorted(dir_scores.items(), key=lambda kv: kv[1], reverse=True)
        notes = "Prioritized by observed transition frequencies."
        return {"exploration_prior": {"by_direction": [d for d, _ in ranked], "notes": notes}}

    async def update(self, **kwargs) -> None:
        recorder: Basic_Recorder = kwargs["recorder"]
        steps = getattr(recorder, "steps", []) or []
        if not steps:
            return None

        # Build sequence of (room, after_action_direction)
        sequence: List[Tuple[Optional[str], Optional[str]]] = []
        for st in steps:
            obs = st.get("obs") or ""
            room = _extract_room_name(obs)
            act_raw = st.get("action_took")
            if isinstance(act_raw, list):
                act_text = " ".join(act_raw)
            else:
                act_text = act_raw or ""
            direction = _action_direction(act_text)
            sequence.append((room, direction))

        # Update edges using consecutive room pairs where a move occurred
        prev_room: Optional[str] = None
        prev_dir: Optional[str] = None
        for room, direction in sequence:
            if prev_room and room and prev_dir:
                if not self.graph.has_edge(prev_room, room):
                    self.graph.add_edge(prev_room, room, dir_counts={})
                data = self.graph.get_edge_data(prev_room, room)
                dir_counts = data.get("dir_counts", {})
                dir_counts[prev_dir] = dir_counts.get(prev_dir, 0) + 1
                data["dir_counts"] = dir_counts
                nx.set_edge_attributes(self.graph, {(prev_room, room): data})
            prev_room = room
            prev_dir = direction


# ---------------------------- Orchestrator ----------------------------

class TextWorldMemory(MemoStructure):
    def __init__(self) -> None:
        super().__init__()
        # Initialize layers
        self.parser = TaskParsingLayer()
        self.task_index = SemanticTaskIndexLayer()
        self.strategy_lib = StrategyLibraryLayer()
        self.affordance = ObjectAffordanceLayer()
        self.failures = FailureMemoryLayer()
        self.paths = ProceduralPathMemoryLayer()

        # Attach shared database handle if needed
        self.database = {
            "task_index": self.task_index.database,
            "strategy_lib": self.strategy_lib.database,
            "affordance": self.affordance.database,
            "failures": self.failures.database,
            "paths": self.paths.database,
        }

    async def general_retrieve(self, recorder: Basic_Recorder) -> Dict:
        init = getattr(recorder, "init", {}) or {}
        if not isinstance(init, dict):
            raise ValueError("recorder.init must be a dict")

        # 1) Parse task
        parsed = await self.parser.retrieve(init=init)
        goal = parsed.get("goal") or ""
        start_room = parsed.get("start_room")
        exits = parsed.get("exits", [])
        objects = parsed.get("objects", [])

        # 2) Semantic prior queries (chain: parsed -> task_index/strategy/affordance/failures/paths)
        similar = await self.task_index.retrieve(goal=goal, start_room=start_room, objects=objects)
        strategies = await self.strategy_lib.retrieve(goal=goal, start_room=start_room, objects=objects)
        afford = await self.affordance.retrieve(objects=objects, goal=goal)
        pitfalls = await self.failures.retrieve(goal=goal)
        path_prior = await self.paths.retrieve(start_room=start_room, exits=exits)

        # 3) Compose clean, compact output
        output = {
            "task_brief": {
                "goal": goal,
                "start_room": start_room or "",
                "exits": exits,
                "objects_seen": objects,
            },
            "exploration_prior": path_prior.get("exploration_prior", {"by_direction": [], "notes": ""}),
            "retrieved_strategies": strategies.get("strategies", []),
            "object_affordances": afford.get("object_affordances", {}),
            "pitfalls": pitfalls.get("pitfalls", []),
            "similar_past_tasks": similar.get("similar_tasks", []),
        }
        # Store into recorder for downstream visibility if needed
        recorder.memory_retrieved = output
        return output

    async def general_update(self, recorder: Basic_Recorder) -> None:
        init = getattr(recorder, "init", {}) or {}
        if not isinstance(init, dict):
            raise ValueError("recorder.init must be a dict in update")
        # 1) Re-parse to get structured signals
        parsed = await self.parser.retrieve(init=init)

        # 2) Update graph paths first (purely structural)
        await self.paths.update(recorder=recorder)

        # 3) Update affordances from action-outcome pairs
        await self.affordance.update(recorder=recorder, parsed=parsed)

        # 4) Summarize and store strategy, get ids to link
        strat_info = await self.strategy_lib.update(recorder=recorder, parsed=parsed)
        strategy_ids = strat_info.get("strategy_ids", []) if isinstance(strat_info, dict) else []

        # 5) Update task semantic index linking to strategies
        await self.task_index.update(recorder=recorder, parsed=parsed, strategy_ids=strategy_ids)

        # 6) Update failures if low reward or loops
        await self.failures.update(recorder=recorder, parsed=parsed)