"""Генератор многооперационных AST-деревьев CadQuery-программ.

Стратегия:
  * сложность модели определяется целевым числом операций (budget), которое
    выбирается из диапазона конфигурации для заданного уровня сложности;
  * дерево строится рекурсивно: на каждом шаге выбирается примитив / эскиз-операция,
    модификатор (fillet, chamfer, shell, hole, transform, pattern) либо булева
    операция (union, cut, intersect) с двумя поддеревьями;
  * каждая промежуточная операция сразу исполняется и проверяется валидатором;
    при неудаче состояние генератора (RNG, счётчики) откатывается, и выбирается
    другая операция;
  * соблюдаются лимиты на число операций каждого типа (отверстия, скругления,
    булевы операции, массивы и т.п.);
  * параметры модификаторов подбираются безопасно (радиус скругления <= 10% от
    минимального габарита, толщина оболочки <= 2% и т.д.) и фиксируются в узле;
  * генерация полностью детерминирована seed-ом.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import math
import time

import numpy as np

from ..syntax_tree.nodes import (
    ASTNode, PrimitiveNode, SketchNode, PathNode,
    ExtrudeNode, RevolveNode, TwistExtrudeNode, LoftNode, SweepNode,
    FilletNode, ChamferNode, ShellNode, HoleNode, SplitNode,
    TransformNode, PatternNode, BooleanNode,
)
from ..executor.executor import Executor
from ..validators.geometry_validator import GeometryValidator, ValidationError
from ..utils.geometry_utils import fast_bbox, unwrap_solid
from ..config import GeneratorConfig
from ..exceptions import GenerationError
from ..operations.solid_ops import select_edges, resolve_split


@dataclass
class BuildState:
    """Полное состояние генератора для отката при неудачной операции."""
    rng_state: Dict[str, Any]
    total_ops: int
    operation_counts: Dict[str, int]


class ASTBuilder:
    PRIMITIVE_OPS = ["box", "cylinder", "sphere", "cone", "wedge", "torus"]
    SKETCH_OPS = ["rect", "circle", "ellipse", "polygon", "slot"]
    EXTRA_SKETCH_OPS = ["polyline", "spline", "roundrect", "frame",
                        "sector", "arc_profile", "ellipse_arc", "bent", "mirrored"]
    # семейства, рисующие несколько проводов (непригодны для revolve/loft)
    MULTI_WIRE_OPS = ["frame"]
    # семейства, привязанные к началу координат (center должен быть (0,0))
    ORIGIN_LOCKED_OPS = ["mirrored", "ellipse_arc"]
    EXTRUDE_REVOLVE_OPS = ["extrude", "revolve", "twist_extrude"]
    LOFT_SWEEP_OPS = ["loft", "sweep"]
    UNARY_OPS = ["fillet", "chamfer", "shell", "hole", "split",
                 "translate", "rotate", "mirror",
                 "rarray", "polarArray", "scatter"]
    BINARY_OPS = ["union", "cut", "intersect"]

    def __init__(self, config: GeneratorConfig, seed: int = None, complexity: str = "medium"):
        self.config = config
        self.complexity = complexity
        self.base_seed = seed if seed is not None else config.global_seed
        self.rng: np.random.Generator = np.random.default_rng(self.base_seed)
        # масштаб модели: задаётся в `_reset` (первый же вызов сборки) из
        # лог-униформного диапазона; 1.0 — безопасный дефолт до сборки
        self.scale: float = 1.0
        self.executor = Executor()
        self.validator = GeometryValidator(config)

        ops_map = {
            "easy": config.easy_ops_range,
            "medium": config.medium_ops_range,
            "hard": config.hard_ops_range,
            "expert": config.expert_ops_range,
        }
        self.target_min, self.target_max = ops_map.get(complexity, config.medium_ops_range)
        self.target_min = max(1, self.target_min)
        self.target_max = max(self.target_min, min(self.target_max, config.max_operations))
        self.max_ops = self.target_max + 2

        # глубина ограничивает и «читаемость» программы, и время построения;
        # для больших бюджетов поднимаем лимит, чтобы цепочка всегда могла
        # «поглотить» весь бюджет даже без ветвления
        self.depth_limit = max(config.max_depth, self.target_max)
        self.attempt_timeout = {
            "easy": 8.0, "medium": 15.0, "hard": 20.0, "expert": 30.0,
        }.get(complexity, 15.0)
        self._deadline = 0.0
        self.backtrack_attempts = max(config.backtrack_attempts, 10)
        self.total_ops = 0
        self.operation_counts: Dict[str, int] = {
            "hole": 0, "fillet": 0, "chamfer": 0, "shell": 0,
            "split": 0,
            "union": 0, "cut": 0, "intersect": 0,
            "rarray": 0, "polarArray": 0, "scatter": 0,
            "extrude": 0, "revolve": 0, "twist_extrude": 0, "loft": 0, "sweep": 0,
        }
        self.target_ops = self.target_min
        self._validated: set = set()

    # ------------------------------------------------------------------ #
    # Точка входа                                                         #
    # ------------------------------------------------------------------ #
    def build(self) -> ASTNode:
        # абсолютный wall-clock лимит на всю сборку: бэктрекинг не должен
        # длиться минуты (попытка ограничена attempt_timeout, но попыток много),
        # а зависшая OCCT-операция в попытке не вернётся никогда — таких ждёт
        # sample_timeout на уровне датасета
        self._halt = time.time() + self.attempt_timeout * 4
        deadline_hits = 0
        for attempt in range(self.backtrack_attempts):
            if time.time() > self._halt:
                break
            self._reset(attempt)
            self._deadline = time.time() + self.attempt_timeout
            try:
                target = int(self.rng.integers(self.target_min, self.target_max + 1))
                self.target_ops = target
                node = self._build_solid(0, target)
                if self.config.verify_root_cold:
                    # финальная проверка корня СВЕЖИМ исполнением: тёплый кэш
                    # (self._validate(force=True) ходит через executor) может
                    # расходиться с итоговой геометрией из-за небит-детерминизма
                    # OCCT между состояниями; отфильтровываем такие деревья здесь,
                    # а не падением на этапе генерации/воспроизведения
                    cold_shape = Executor().execute(node)
                    self.validator.validate(cold_shape, is_root=True)
                else:
                    # финальная проверка корня как единого целого (тёплый путь)
                    self._validate(node, is_root=True, force=True)
                count = node.count_operations()
                if self.target_min <= count <= self.target_max:
                    return node
            except TimeoutError:
                deadline_hits += 1
                if deadline_hits >= 3:
                    raise GenerationError(
                        f"Failed to build a {self.complexity} model: repeated timeouts "
                        f"for seed {self.base_seed}")
                continue
            except (GenerationError, ValidationError):
                continue
        raise GenerationError(
            f"Failed to build a {self.complexity} model after {self.backtrack_attempts} attempts "
            f"(target ops {self.target_min}-{self.target_max}, wall-clock capped)"
        )

    def _reset(self, attempt: int):
        self.rng = np.random.default_rng(self.base_seed + attempt * 1000003)
        # масштаб модели на попытку (лог-униформ): рисуется ПОСЛЕ ресида, чтобы
        # оставаться детерминированным и стабильным внутри попытки (в т.ч. при
        # _save_state/_restore_state бэктрекинга). Геометрия инвариантна к scale:
        # все производные параметры берутся из уже масштабированных размеров.
        sl, sh = self.config.scale_low, self.config.scale_high
        self.scale = float(np.exp(self.rng.uniform(np.log(sl), np.log(sh))))
        self.total_ops = 0
        self.operation_counts = {k: 0 for k in self.operation_counts}
        self.executor.clear()
        self._validated = set()

    # ------------------------------------------------------------------ #
    # Состояние и лимиты                                                  #
    # ------------------------------------------------------------------ #
    def _save_state(self) -> BuildState:
        return BuildState(
            rng_state=self.rng.bit_generator.state,
            total_ops=self.total_ops,
            operation_counts=self.operation_counts.copy(),
        )

    def _restore_state(self, state: BuildState):
        self.rng.bit_generator.state = state.rng_state
        self.total_ops = state.total_ops
        self.operation_counts = state.operation_counts.copy()

    def _check_limits(self, op: str) -> bool:
        limits = {
            "hole": self.config.max_holes,
            "fillet": self.config.max_fillets,
            "chamfer": self.config.max_chamfers,
            "shell": self.config.max_shell,
            "split": self.config.max_splits,
            "twist_extrude": self.config.max_twist_extrudes,
            "union": self.config.max_booleans,
            "cut": self.config.max_booleans,
            "intersect": self.config.max_booleans,
            "rarray": self.config.max_patterns,
            "polarArray": self.config.max_patterns,
            "scatter": self.config.max_patterns,
        }
        if op in limits and self.operation_counts[op] >= limits[op]:
            return False
        if self.total_ops >= self.max_ops:
            return False
        return True

    def _record_op(self, op: str):
        self.total_ops += 1
        if op in self.operation_counts:
            self.operation_counts[op] += 1

    # ------------------------------------------------------------------ #
    # Рекурсивное построение                                              #
    # ------------------------------------------------------------------ #
    def _validate(self, node: ASTNode, is_root: bool = False, force: bool = False):
        """Исполняет узел и валидирует результат ровно один раз на узел."""
        if not force and node.node_id in self._validated:
            return self.executor.execute(node)
        shape = self.executor.execute(node)
        if not is_root and isinstance(node, TransformNode):
            # перемещение/поворот/отражение гарантированно сохраняют валидность
            self._validated.add(node.node_id)
            return shape
        self.validator.validate(shape, is_root=is_root)
        self._validated.add(node.node_id)
        return shape

    def _build_solid(self, depth: int, budget: int) -> ASTNode:
        if time.time() > self._deadline or time.time() > self._halt:
            raise TimeoutError("attempt time budget exceeded")
        if depth >= self.depth_limit or budget <= 1:
            return self._build_leaf_solid(depth, budget)

        saved_base = self._save_state()
        remaining_depth = self.depth_limit - depth

        binary_cands = ([op for op in self.BINARY_OPS if self._check_limits(op)]
                        if budget >= 3 else [])
        unary_cands = [op for op in self.UNARY_OPS if self._check_limits(op)]

        # ветвление обязательно, если цепочка не вмещается в оставшуюся глубину;
        # иначе — ветвление выбирается вероятностно для разнообразия
        need_binary = bool(binary_cands) and (budget - 1 > remaining_depth)
        try_binary = bool(binary_cands) and (
            need_binary or self.rng.random() < self.config.binary_probability)

        if try_binary:
            node = self._try_binary_ops(depth, budget, binary_cands)
            if node is not None:
                return node
            # все булевы операции не удались — откатываемся к унарным
            self._restore_state(saved_base)

        # --- унарные модификаторы: поддерево строится и валидируется ОДИН раз ---
        if unary_cands:
            child = self._build_solid(depth + 1, budget - 1)
            for op in self._weighted_order(unary_cands, self.config.unary_choice_weights):
                if not self._should_try_unary(op, child):
                    continue
                saved = self._save_state()
                try:
                    node = self._create_unary(op, child)
                    self._validate(node)
                    return node
                except (GenerationError, ValidationError):
                    self._restore_state(saved)
                    continue

        # ничего не получилось — откатываемся к безопасному листу
        self._restore_state(saved_base)
        return self._build_leaf_solid(depth, budget)

    def _should_try_unary(self, op: str, child: ASTNode) -> bool:
        """Вероятностный фильтр дорогих/рискованных модификаторов.

        Transform-операции гарантированно работают и почти бесплатны, но при
        вероятности 1.0 встречались чаще остальных модификаторов — теперь их
        частота ограничена config.modifier_probabilities (translate/rotate/mirror
        0.3). fillet/chamfer выводят радиус из длины рёбер и почти всегда
        успешны, hole ~0.1с, массивы — серии fuse-операций, а shell стоит
        несколько секунд на вызов OCCT. Дорогие операции пробуем реже, чтобы не
        тратить время на неудачные попытки и держать среднее время генерации
        низким. Вероятности настраиваются через config.modifier_probabilities.
        """
        probs = self.config.modifier_probabilities
        if op in ("translate", "rotate", "mirror"):
            return self.rng.random() < probs.get(op, 1.0)
        if op == "shell":
            # shell крайне дорог (секунды на вызов) и обычно работает только
            # на «простых» телах (без булевых/массивов внутри)
            return self.rng.random() < probs.get("shell", 0.08) and self._is_simple_shape(child)
        if op in ("fillet", "chamfer"):
            # выборка лишь 1-2 рёбер через Selectors делает операцию безопасной
            # даже на сложных телах (после boolean/loft/sweep/pattern)
            return self.rng.random() < probs.get("fillet", 0.5)
        if op in ("rarray", "polarArray", "scatter"):
            # массивы — серии fuse-операций; на сложных телах каждая склейка
            # становится очень дорогой, поэтому пробуем только на «простых»
            return self.rng.random() < probs.get("rarray", 0.5) and self._is_simple_shape(child)
        if op == "hole":
            return self.rng.random() < probs.get("hole", 0.45)
        if op == "split":
            # split дорогой (plane-cut+fuse, две B-Rep операции), поэтому
            # частота ограничивается вероятностью ниже 1.0
            return self.rng.random() < probs.get("split", 0.55)
        return True

    @staticmethod
    def _is_simple_shape(node: ASTNode) -> bool:
        """Нет ли внутри дерева операций, после которых shell обычно ломается."""
        stack = [node]
        while stack:
            n = stack.pop()
            if isinstance(n, (BooleanNode, PatternNode, ShellNode,
                              LoftNode, SweepNode, HoleNode)):
                return False
            stack.extend(n.children)
        return True

    def _try_binary_ops(self, depth: int, budget: int,
                        binary_cands: List[str]) -> Optional[ASTNode]:
        """Пытается применить булевы операции к общим поддеревьям left/right.

        Булевы операции над двумя «сложными» операндами (union/pattern/fillet/
        shell внутри обоих поддеревьев) — главный источник недетерминированных
        зависаний OCCT: BRepAlgoAPI_* может не завершиться вовсе. Поэтому сначала
        строится ТОЛЬКО первый («большой») операнд, а второй выбирается уже по его
        фактической сложности — так ни одно дорогое поддерево не строится впустую:
          * первый простой и маленький — второй забирает остаток бюджета
            (сохраняются сбалансированные булевы из двух средних операндов);
          * первый сложный/большой — второй дешёвая «фича»
            (`_build_simple_feature`: лист ± трансформации, гарантированно
            простая), поэтому гейт «хотя бы один простой» не срабатывает и работа
            не выбрасывается. intersect оставляем только для двух простых листов.
        """
        if time.time() > self._deadline:
            raise TimeoutError("attempt time budget exceeded")
        child_budget = budget - 1
        # Половина попыток — «основное тело + маленькая фича» (первый операнд
        # забирает почти весь бюджет), половина — сбалансированные операнды.
        # В обоих случаях второй операнд строится уже ПОСЛЕ того, как первый
        # реально построен и известно его фактическое число операций и сложность:
        #  * первый простой и занял не больше половины бюджета — второй получает
        #    остаток бюджета (сохраняются сбалансированные булевы);
        #  * первый сложный/большой — второй дешёвая «фича»
        #    (`_build_simple_feature`: лист ± трансформации, гарантированно
        #    простая), поэтому гейт «хотя бы один простой» не срабатывает и
        #    дорогие поддеревья не строятся впустую.
        if self.rng.random() < 0.5:
            first_budget = child_budget - 1
        else:
            lo = max(1, child_budget // 3)
            hi = max(lo, child_budget // 2)
            first_budget = int(self.rng.integers(lo, hi + 1))
        first = self._build_solid(depth + 1, first_budget)
        first_n = first.count_operations()
        if self._is_simple_shape(first) and first_n <= max(1, child_budget // 2):
            second = self._build_solid(depth + 1, max(1, child_budget - first_n))
        else:
            saved_second = self._save_state()
            try:
                second = self._build_simple_feature(depth + 1)
            except (GenerationError, ValidationError):
                self._restore_state(saved_second)
                return None
        if self.rng.random() < 0.5:
            left, right = first, second
        else:
            left, right = second, first
        left_simple = self._is_simple_shape(left)
        right_simple = self._is_simple_shape(right)
        # оба операнда сложными быть не могут по построению (при сложном первом
        # второй — гарантированно «простая» фича); гейт оставлен как страховка от
        # патологических комбинаций, которые вешают OCCT
        if not (left_simple or right_simple):
            return None
        for op in self._weighted_order(binary_cands, self.config.binary_choice_weights):
            if op == "intersect" and not (left_simple and right_simple):
                continue
            saved = self._save_state()
            try:
                node = self._create_boolean(op, left, right)
                self._validate(node)
                return node
            except (GenerationError, ValidationError):
                self._restore_state(saved)
                continue
        return None

    def _build_simple_feature(self, depth: int, max_ops: int = 4) -> ASTNode:
        """Маленький гарантированно «простой» операнд для булевых.

        Лист строится ТОЛЬКО из примитивов/extrude/revolve/twist (НЕ loft/sweep —
        они считаются «сложными» для булевых) и дополняется цепочкой дешёвых
        модификаторов (fillet/chamfer/split/transforms). Ни одна из них не входит
        в «сложные» операции (boolean/pattern/shell/loft/sweep/hole), поэтому
        результат никогда не делает оба операнда сложными и не требует дорогих
        retry, при этом восстанавливает богатство фичи (модификаторы на втором
        операнде), как в сбалансированных булевых. Каждая операция пробуется один
        раз и отбрасывается при неудаче — работа ограничена сверху.
        """
        choices = list(self.PRIMITIVE_OPS) + list(self.EXTRUDE_REVOLVE_OPS)
        avail = [op for op in choices if self._check_limits(op)]
        if not avail:
            avail = [op for op in self.PRIMITIVE_OPS if self._check_limits(op)]
        if not avail:
            raise GenerationError("no simple leaf available")
        op = self._weighted_order(avail, self.config.leaf_choice_weights)[0]
        saved = self._save_state()
        try:
            node = self._create_leaf_solid(op, depth, 1)
            self._validate(node)
        except (GenerationError, ValidationError):
            self._restore_state(saved)
            raise
        wraps = [op for op in ("fillet", "chamfer", "split",
                               "translate", "rotate", "mirror")
                 if self._check_limits(op)]
        order = self._weighted_order(wraps, self.config.unary_choice_weights)
        added = 0
        for op in order:
            if added >= max_ops:
                break
            if op in ("translate", "rotate", "mirror") and not self._should_try_unary(op, node):
                # трансформации фильтруются той же вероятностью (0.3), что и в
                # основном билдере — фичи не должны тащить их чаще остальных
                continue
            saved = self._save_state()
            try:
                node = self._create_unary(op, node)
                self._validate(node)
                added += 1
            except (GenerationError, ValidationError):
                self._restore_state(saved)
        return node

    # ------------------------------------------------------------------ #
    # Листовые операции                                                   #
    # ------------------------------------------------------------------ #
    def _build_leaf_solid(self, depth: int, budget: int) -> ASTNode:
        """Создаёт терминальное твёрдое тело (ровно одну операцию).

        Лист может быть примитивом, extrude/revolve или loft/sweep — это
        обеспечивает разнообразие терминальных узлов при точном учёте бюджета.
        Сначала пробуются надёжные листья (примитивы, extrude/revolve);
        рискованные (loft/sweep) добавляются в выбор с меньшей вероятностью.
        """
        choices = (list(self.PRIMITIVE_OPS) + list(self.EXTRUDE_REVOLVE_OPS)
                   + list(self.LOFT_SWEEP_OPS))
        avail = [op for op in choices if self._check_limits(op)]
        if not avail:
            avail = [op for op in self.PRIMITIVE_OPS if self._check_limits(op)]
        if not avail:
            raise GenerationError("No leaf operation available")
        safe = [op for op in avail if op in self.PRIMITIVE_OPS
                or op in self.EXTRUDE_REVOLVE_OPS]
        risky = [op for op in avail if op not in safe]
        for _ in range(2):
            pool = safe if (safe and (not risky or self.rng.random() < 0.25)) else avail
            order = self._weighted_order(pool, self.config.leaf_choice_weights)
            op = order[0]
            saved = self._save_state()
            try:
                node = self._create_leaf_solid(op, depth, budget)
                self._validate(node)
                return node
            except (GenerationError, ValidationError):
                self._restore_state(saved)
                continue
        # последний шанс — только гарантированно рабочие листья
        if safe:
            op = self._weighted_order(safe, self.config.leaf_choice_weights)[0]
            node = self._create_leaf_solid(op, depth, budget)
            self._validate(node)
            return node
        raise GenerationError("All leaf operations failed")

    def _create_leaf_solid(self, op: str, depth: int, budget: int) -> ASTNode:
        if op in self.PRIMITIVE_OPS:
            node = self._create_primitive(op)
            self._record_op(op)
            return node
        if op == "extrude":
            sk = self._create_sketch_node()
            node = ExtrudeNode(operation="extrude",
                               parameters=self._extrude_params(),
                               children=[sk])
            self._record_op("extrude")
            return node
        if op == "revolve":
            sk = self._create_sketch_node(kind="revolve")
            node = RevolveNode(operation="revolve",
                               parameters={"angle": 360.0},
                               children=[sk])
            self._record_op("revolve")
            return node
        if op == "twist_extrude":
            sk = self._create_sketch_node(kind="twist_extrude")
            node = TwistExtrudeNode(operation="twist_extrude",
                                    parameters=self._twist_extrude_params(),
                                    children=[sk])
            self._record_op("twist_extrude")
            return node
        if op == "loft":
            n_profiles = 3 if budget >= 4 else 2
            # профили одного семейства (одинаковый тип эскиза), разного размера
            base_op = str(self.rng.choice(self.SKETCH_OPS))
            base_params = self._sketch_params(base_op)
            sketches = []
            for _ in range(n_profiles):
                params = self._scale_sketch_params(base_op, base_params,
                                                   float(self.rng.uniform(0.5, 1.5)))
                params["center"] = (0.0, 0.0)
                params["workplane"] = "XY"
                sketches.append(SketchNode(operation=base_op, parameters=params))
            offsets = self._loft_offsets(n_profiles)
            node = LoftNode(operation="loft",
                            parameters={"offsets": offsets},
                            children=sketches)
            self._record_op("loft")
            return node
        if op == "sweep":
            # окружность — самый надёжный профиль для развёртки по пути
            profile = self._create_sketch_node(kind="sweep_profile")
            path = self._create_path()
            node = SweepNode(operation="sweep",
                             parameters={},
                             children=[profile, path])
            self._record_op("sweep")
            return node
        raise GenerationError(f"Unknown leaf op {op}")

    # ------------------------------------------------------------------ #
    # Модификаторы                                                        #
    # ------------------------------------------------------------------ #
    def _create_unary(self, op: str, child: ASTNode) -> ASTNode:
        if op == "fillet":
            node = self._try_fillet(child)
        elif op == "chamfer":
            node = self._try_chamfer(child)
        elif op == "shell":
            node = self._try_shell(child)
        elif op == "hole":
            node = self._try_hole(child)
        elif op == "split":
            node = self._try_split(child)
        elif op in ("translate", "rotate", "mirror"):
            node = TransformNode(operation=op,
                                 parameters=self._transform_params(op),
                                 children=[child])
            self._record_op(op)
        elif op in ("rarray", "polarArray", "scatter"):
            node = PatternNode(operation=op,
                               parameters=self._pattern_params(op, child),
                               children=[child])
            self._record_op(op)
        else:
            raise GenerationError(f"Unknown unary op {op}")
        return node

    def _edge_selection(self, shape) -> Dict[str, Any]:
        """Случайная семантическая выборка рёбер (nearest у точки / по направлению)."""
        if self.rng.random() < 0.5:
            xmin, xmax, ymin, ymax, zmin, zmax = self._bbox(shape)
            pt = (float(self.rng.uniform(xmin, xmax)),
                  float(self.rng.uniform(ymin, ymax)),
                  float(self.rng.uniform(zmin, zmax)))
            return {"kind": "nearest", "point": pt}
        axis = str(self.rng.choice(["X", "Y", "Z"]))
        sign = 1.0 if self.rng.random() < 0.5 else -1.0
        direction = [0.0, 0.0, 0.0]
        direction[{"X": 0, "Y": 1, "Z": 2}[axis]] = sign
        return {"kind": "direction", "direction": tuple(direction)}

    @staticmethod
    def _contains_split(node: ASTNode) -> bool:
        """Есть ли в дереве узел split (в т.ч. внутри поддерева)."""
        stack = [node]
        while stack:
            n = stack.pop()
            if getattr(n, "operation", None) == "split":
                return True
            stack.extend(getattr(n, "children", None) or [])
        return False

    @staticmethod
    def _edge_risk(node: ASTNode) -> bool:
        """Есть ли в дереве узлы, после fuse которых OCCT BRepFillet/BRepChamfer
        может упасть AccessViolation'ом (не исключением) — split и массивы
        (rarray/polarArray/scatter) склеивают копии/половины, и фаска по шву
        детерминированно валит процесс. Такие тела для fillet/chamfer не берём."""
        stack = [node]
        while stack:
            n = stack.pop()
            if getattr(n, "operation", None) in ("split", "rarray", "polarArray", "scatter"):
                return True
            stack.extend(getattr(n, "children", None) or [])
        return False

    def _try_fillet(self, child: ASTNode) -> FilletNode:
        if self._edge_risk(child):
            # OCCT BRepFillet по рёбрам половин/швам fuse-копий может упасть
            # AccessViolation-крашем (не исключение) — не трогаем
            raise GenerationError("fillet after split/array skipped (OCCT AV risk)")
        base = self.executor.execute(child)
        edges = list(base.edges())
        if not edges:
            raise GenerationError("no edges available for fillet")
        selection = self._edge_selection(base)
        selected = select_edges(base, selection, limit=2)
        if not selected:
            raise GenerationError("no edges selected for fillet")
        # радиус масштабируется от реальной длины выбранных рёбер: короткие
        # рёбра (после boolean/порезки) ранее давали 71% отказов в OCCT
        min_len = min(float(e.Length()) for e in selected)
        max_r = min(min_len * 0.3, self._min_dim(base) * self.config.fillet_radius_ratio, 5.0)
        if max_r < 0.02:
            raise GenerationError("edges too short for fillet")
        r = self.rng.uniform(0.02, max_r)
        node = FilletNode(operation="fillet",
                          parameters={"radius": r, "selection": selection},
                          children=[child])
        shape = self.executor.execute(node)
        self.validator.validate(shape)
        self._validated.add(node.node_id)
        self._record_op("fillet")
        return node

    def _try_chamfer(self, child: ASTNode) -> ChamferNode:
        if self._edge_risk(child):
            raise GenerationError("chamfer after split/array skipped (OCCT AV risk)")
        base = self.executor.execute(child)
        edges = list(base.edges())
        if not edges:
            raise GenerationError("no edges available for chamfer")
        selection = self._edge_selection(base)
        selected = select_edges(base, selection, limit=2)
        if not selected:
            raise GenerationError("no edges selected for chamfer")
        min_len = min(float(e.Length()) for e in selected)
        max_d = min(min_len * 0.3, self._min_dim(base) * self.config.fillet_radius_ratio, 5.0)
        if max_d < 0.02:
            raise GenerationError("edges too short for chamfer")
        d = self.rng.uniform(0.02, max_d)
        node = ChamferNode(operation="chamfer",
                           parameters={"distance": d, "selection": selection},
                           children=[child])
        shape = self.executor.execute(node)
        self.validator.validate(shape)
        self._validated.add(node.node_id)
        self._record_op("chamfer")
        return node

    def _try_shell(self, child: ASTNode) -> ShellNode:
        base = self.executor.execute(child)
        if len(list(base.faces())) == 0:
            raise GenerationError("no faces available for shell")
        base_vol = float(base.Volume())
        min_dim = self._min_dim(base)
        thicknesses = [
            max(0.01, min_dim * self.config.shell_thickness_ratio),
            max(0.005, min_dim * self.config.shell_thickness_ratio * 0.5),
        ]
        candidates = []
        for axis in ["Z", "Y", "X"]:
            for sign in (1.0, -1.0):
                direction = [0.0, 0.0, 0.0]
                direction[{"X": 0, "Y": 1, "Z": 2}[axis]] = sign
                candidates.append({"kind": "direction", "direction": tuple(direction)})
        self.rng.shuffle(candidates)
        for thickness in thicknesses:
            for selection in candidates[:2]:
                node = ShellNode(operation="shell",
                                 parameters={"thickness": thickness, "selection": selection},
                                 children=[child])
                try:
                    shape = self.executor.execute(node)
                    self.validator.validate(shape)
                    vol = float(shape.Volume())
                    if not np.isfinite(vol) or vol < self.config.min_volume:
                        continue
                    if vol > base_vol * 0.995:
                        # оболочка не удалила материал внутри — неудачная попытка
                        continue
                except (GenerationError, ValidationError):
                    continue
                self._validated.add(node.node_id)
                self._record_op("shell")
                return node
        raise GenerationError("shell failed after retries")

    def _try_hole(self, child: ASTNode) -> HoleNode:
        base = self.executor.execute(child)
        xmin, xmax, ymin, ymax, zmin, zmax = self._bbox(base)
        w = xmax - xmin
        h = ymax - ymin
        d = zmax - zmin
        if w < 0.3 * self.scale or h < 0.3 * self.scale or d < 0.3 * self.scale:
            raise GenerationError("solid too thin for hole")
        base_vol = float(base.Volume())
        r_max = min(w, h) * 0.4
        if r_max < 0.1 * self.scale:
            raise GenerationError("hole radius range too small")
        kinds = ["through", "blind", "cbore", "csk"]
        for _ in range(3):
            kind = str(self.rng.choice(kinds))
            radius = self.rng.uniform(0.1 * self.scale, r_max)
            x = self.rng.uniform(xmin + radius, xmax - radius)
            y = self.rng.uniform(ymin + radius, ymax - radius)
            params: Dict[str, Any] = {"kind": kind, "radius": radius}
            if kind == "through":
                margin = max(0.5 * self.scale, 0.2 * min(w, h, d))
                params["position"] = (x, y, zmin - margin)
                params["depth"] = d + 2.0 * margin
            else:
                # blind/cbore/csk открываются на верхней грани (position.z = zmax)
                params["position"] = (x, y, zmax)
                params["depth"] = self.rng.uniform(0.3 * d, 0.85 * d)
                if kind == "cbore":
                    if radius * 1.5 >= r_max:
                        continue
                    params["cbo_radius"] = self.rng.uniform(radius * 1.5,
                                                            min(radius * 2.2, r_max))
                    params["cbo_depth"] = self.rng.uniform(0.1 * d, 0.35 * d)
                elif kind == "csk":
                    if radius * 1.6 >= r_max:
                        continue
                    params["csk_radius"] = self.rng.uniform(radius * 1.6,
                                                            min(radius * 2.5, r_max))
                    params["csk_depth"] = self.rng.uniform(0.1 * d, 0.3 * d)
            node = HoleNode(operation="hole", parameters=params, children=[child])
            try:
                shape = self.executor.execute(node)
                self.validator.validate(shape)
                vol = float(shape.Volume())
                if vol > base_vol * 0.995:
                    raise GenerationError("hole removed no material")
            except (GenerationError, ValidationError):
                continue
            self._validated.add(node.node_id)
            self._record_op("hole")
            return node
        raise GenerationError("hole failed after retries")

    def _try_split(self, child: ASTNode) -> SplitNode:
        base = self.executor.execute(child)
        xmin, xmax, ymin, ymax, zmin, zmax = self._bbox(base)
        extents = {"X": xmax - xmin, "Y": ymax - ymin, "Z": zmax - zmin}
        # оси по убыванию размера: на длинной оси перекрытие половин надёжнее;
        # порог пропорционален масштабу модели
        axes = [ax for ax in ("X", "Y", "Z") if extents[ax] >= 0.5 * self.scale]
        if not axes:
            raise GenerationError("solid too small for split")
        axes.sort(key=lambda ax: extents[ax], reverse=True)
        base_vol = float(base.Volume())
        for axis in axes:
            # один gap на ось: plane-cut+fuse дорогой (два B-Rep вызова),
            # retry-серии из 9 исполнений давали 46-49% общего времени
            gap = float(self.rng.uniform(0.25, 0.4)) * extents[axis]
            # детерминированный разрез ДО исполнения узла: выбранные
            # plane_offset/parts_count фиксируются в параметрах, чтобы code-генератор
            # воспроизвёл тот же разрез без циклов; результат кэшируем под node_id
            # напрямую — executor.execute(node) больше не вызывается (был бы 2-й OCCT-вызов).
            try:
                plane_offset, parts_count, fused = resolve_split(base, axis, gap)
            except GenerationError:
                continue
            node = SplitNode(operation="split",
                             parameters={"axis": axis, "gap": gap,
                                         "plane_offset": float(plane_offset),
                                         "parts_count": int(parts_count)},
                             children=[child])
            shape = unwrap_solid(fused)
            self.executor.cache[node.node_id] = shape
            try:
                self.validator.validate(shape)
                vol = float(shape.Volume())
                if not np.isfinite(vol) or vol < self.config.min_volume:
                    continue
                if vol > base_vol * 0.995:
                    raise GenerationError("split removed no material")
            except (GenerationError, ValidationError):
                continue
            self._validated.add(node.node_id)
            self._record_op("split")
            return node
        raise GenerationError("split failed after retries")

    def _create_boolean(self, op: str, left: ASTNode, right: ASTNode) -> BooleanNode:
        left_shape = self.executor.execute(left)
        right_shape = self.executor.execute(right)
        lc = self._center(left_shape)
        rc = self._center(right_shape)
        left_vol = float(left_shape.Volume())
        min_dim = self._min_dim(left_shape)

        lbb = self._bbox(left_shape)
        rbb0 = self._bbox(right_shape)

        # bbox-прегейты: недорого (память/µs) отсекают «мёртвые» кандидаты до
        # честного исполнения OCCT-булевой операции (union ничего не добавляет,
        # cut ничего не вырезает, intersect — no-op)
        for _ in range(3):
            if op == "union":
                jitter = self._jitter(scale=0.5 * min_dim)
            elif op == "cut":
                jitter = self._jitter(scale=0.3 * min_dim)
            else:  # intersect
                jitter = (0.0, 0.0, 0.0)
            offset = (lc[0] - rc[0] + jitter[0],
                      lc[1] - rc[1] + jitter[1],
                      lc[2] - rc[2] + jitter[2])
            rbb = (rbb0[0] + offset[0], rbb0[1] + offset[0],
                   rbb0[2] + offset[1], rbb0[3] + offset[1],
                   rbb0[4] + offset[2], rbb0[5] + offset[2])
            if not self._boolean_bbox_ok(op, lbb, rbb):
                continue

            node = BooleanNode(operation=op, parameters={"offset": offset},
                               children=[left, right])
            shape = self.executor.execute(node)
            vol = float(shape.Volume())
            # финальные проверки объёмов после честного исполнения
            if op == "union" and vol <= left_vol * 1.01:
                continue
            if op == "cut" and vol >= left_vol * 0.995:
                continue
            if op == "intersect" and vol >= left_vol * 0.995:
                continue
            self._record_op(op)
            return node
        raise GenerationError(f"{op} failed after bbox-guided attempts")

    @staticmethod
    def _bbox_volume(bb) -> float:
        return (bb[1] - bb[0]) * (bb[3] - bb[2]) * (bb[5] - bb[4])

    @staticmethod
    def _bbox_overlap(a, b) -> float:
        x = min(a[1], b[1]) - max(a[0], b[0])
        y = min(a[3], b[3]) - max(a[2], b[2])
        z = min(a[5], b[5]) - max(a[4], b[4])
        if x <= 0.0 or y <= 0.0 or z <= 0.0:
            return 0.0
        return x * y * z

    @staticmethod
    def _bbox_contains(outer, inner) -> bool:
        return (outer[0] <= inner[0] and inner[1] <= outer[1] and
                outer[2] <= inner[2] and inner[3] <= outer[3] and
                outer[4] <= inner[4] and inner[5] <= outer[5])

    def _boolean_bbox_ok(self, op: str, lbb, rbb) -> bool:
        ov = self._bbox_overlap(lbb, rbb)
        lvol = self._bbox_volume(lbb)
        rvol = self._bbox_volume(rbb)
        min_vol = min(lvol, rvol)
        if ov <= 0.02 * max(min_vol, 1e-9):
            return False
        if op == "union":
            # правый операнд целиком в bbox левого — материал почти наверняка
            # не добавится (кроме вогнутых полостей, редких в нашей генерации)
            return not self._bbox_contains(lbb, rbb)
        if op == "cut":
            # инструмент целиком накрыл тело — вырежется почти всё, объём
            # станет меньше min_volume и валидатор отклонит; отсекаем заранее
            return not self._bbox_contains(rbb, lbb)
        # intersect: резултат не должен совпадать с левым (no-op)
        return not self._bbox_contains(rbb, lbb)

    # ------------------------------------------------------------------ #
    # Примитивы и эскизы                                                  #
    # ------------------------------------------------------------------ #
    def _dim(self, lo: float, hi: float) -> float:
        """Абсолютный размер, масштабированный глобальным scale модели."""
        return float(self.scale * self.rng.uniform(lo, hi))

    def _create_primitive(self, op: str) -> PrimitiveNode:
        params: Dict[str, Any] = {}
        if op == "box":
            params = {"length": self._dim(1, 10),
                      "width": self._dim(1, 10),
                      "height": self._dim(1, 10)}
        elif op == "cylinder":
            params = {"height": self._dim(1, 15),
                      "radius": self._dim(0.5, 5)}
        elif op == "sphere":
            params = {"radius": self._dim(0.5, 5)}
        elif op == "cone":
            params = {"height": self._dim(1, 15),
                      "radius1": self._dim(0.5, 5),
                      "radius2": self._dim(0.1, 5)}
        elif op == "wedge":
            dx = self._dim(1, 10)
            dy = self._dim(1, 10)
            dz = self._dim(1, 10)
            xmax = float(self.rng.uniform(self.scale, dx))
            zmax = float(self.rng.uniform(self.scale, dz))
            params = {"dx": dx, "dy": dy, "dz": dz,
                      "xmin": 0.0, "zmin": 0.0, "xmax": xmax, "zmax": zmax}
        elif op == "torus":
            r1 = self._dim(1, 5)
            r2 = float(self.rng.uniform(0.2 * self.scale,
                                         min(2.5 * self.scale, r1 * 0.7)))
            params = {"radius1": r1, "radius2": r2}
        return PrimitiveNode(operation=op, parameters=params)

    def _create_sketch_node(self, kind: str = "default") -> SketchNode:
        pool = list(self.SKETCH_OPS) + list(self.EXTRA_SKETCH_OPS)
        if kind == "revolve":
            pool = [op for op in pool
                    if op not in self.MULTI_WIRE_OPS and op not in self.ORIGIN_LOCKED_OPS]
        elif kind == "loft":
            # loft требует единственный замкнутый провод на профиль
            pool = [op for op in pool if op not in self.MULTI_WIRE_OPS]
        elif kind == "twist_extrude":
            # twistExtrude тоже требует единственный замкнутый провод
            pool = [op for op in pool if op not in self.MULTI_WIRE_OPS]
        op = str(self.rng.choice(pool))
        params = self._sketch_params(op)
        if kind == "revolve":
            # профиль обязан быть смещён от оси вращения
            extent = self._sketch_extent(op, params)
            params["center"] = (float(extent + self._dim(0.5, 3.0)), 0.0)
            params["workplane"] = "XY"
        elif kind == "loft":
            params["center"] = (0.0, 0.0)
            params["workplane"] = "XY"
        elif kind == "sweep_profile":
            params = {"radius": self._dim(0.3, 1.5)}
            params["center"] = (0.0, 0.0)
            params["workplane"] = "XY"
            op = "circle"
        else:
            params.setdefault("center", (0.0, 0.0))
            params["workplane"] = str(self.rng.choice(["XY", "XZ", "YZ"]))
        return SketchNode(operation=op, parameters=params)

    def _sketch_params(self, op: str) -> Dict[str, Any]:
        if op == "rect":
            return {"width": self._dim(0.5, 8),
                    "height": self._dim(0.5, 8)}
        if op == "circle":
            return {"radius": self._dim(0.5, 5)}
        if op == "ellipse":
            return {"x_radius": self._dim(0.5, 5),
                    "y_radius": self._dim(0.5, 5)}
        if op == "polygon":
            return {"n_sides": int(self.rng.integers(3, 8)),
                    "radius": self._dim(0.5, 5)}
        if op == "slot":
            return {"length": self._dim(2, 8),
                    "width": self._dim(0.5, 3)}
        if op in ("polyline", "spline"):
            # выпуклая оболочка случайных точек: контур без самопересечений,
            # поэтому extrude/revolve всегда дают корректное тело
            n = int(self.rng.integers(3, 6))
            raw = [(self._dim(-4, 4), self._dim(-4, 4))
                   for _ in range(n)]
            cx = sum(p[0] for p in raw) / n
            cy = sum(p[1] for p in raw) / n
            pts = sorted(raw, key=lambda p: math.atan2(p[1] - cy, p[0] - cx))
            return {"points": pts}
        if op == "roundrect":
            w = self._dim(2, 8)
            h = self._dim(2, 8)
            max_r = min(w, h) / 2.0 - 0.15 * self.scale
            return {"width": w, "height": h,
                    "radius": float(self.rng.uniform(0.15 * self.scale,
                                                     max(0.2 * self.scale, max_r)))}
        if op == "frame":
            base = str(self.rng.choice(["rect", "circle", "ellipse", "polygon", "slot"]))
            outer = self._sketch_params(base)
            if base == "rect":
                inset_max = max(0.1 * self.scale,
                                min(outer["width"], outer["height"]) / 4.0)
                inset = float(min(self.rng.uniform(0.05, 1.0) * inset_max,
                                  inset_max - 0.05 * self.scale))
                inner = {"width": max(0.2 * self.scale, outer["width"] - 2 * inset),
                         "height": max(0.2 * self.scale, outer["height"] - 2 * inset)}
            elif base == "circle":
                inset = float(self.rng.uniform(0.1 * self.scale, outer["radius"] / 2.0))
                inner = {"radius": max(0.15 * self.scale, outer["radius"] - inset)}
            elif base == "ellipse":
                inset = float(self.rng.uniform(0.1 * self.scale,
                                               min(outer["x_radius"], outer["y_radius"]) / 2.0))
                inner = {"x_radius": max(0.15 * self.scale, outer["x_radius"] - inset),
                         "y_radius": max(0.15 * self.scale, outer["y_radius"] - inset)}
            elif base == "polygon":
                inset = float(self.rng.uniform(0.1 * self.scale, outer["radius"] / 2.0))
                inner = {"n_sides": outer["n_sides"],
                         "radius": max(0.15 * self.scale, outer["radius"] - inset)}
            else:  # slot
                inset = float(self.rng.uniform(0.1 * self.scale,
                                               min(outer["length"], outer["width"]) / 3.0))
                inner = {"length": max(0.4 * self.scale, outer["length"] - 2 * inset),
                         "width": max(0.15 * self.scale, outer["width"] - 2 * inset)}
            return {"frame_op": base, "outer": outer, "inner": inner}
        if op == "sector":
            # угол <= 170 град: три точки дуги однозначны (минорная дуга) и OCCT
            # гарантированно строит revolution по такой дуге
            return {"radius": self._dim(1.5, 5.0),
                    "angle": float(self.rng.uniform(30, 170))}
        if op == "arc_profile":
            w = self._dim(2, 8)
            h = self._dim(2, 8)
            bow = self._dim(-1.5, 1.5)
            if abs(bow) < 0.3 * self.scale:
                bow = 0.3 * self.scale if bow >= 0 else -0.3 * self.scale
            return {"width": w, "height": h, "bow": bow}
        if op == "ellipse_arc":
            return {"x_radius": self._dim(1, 4),
                    "y_radius": self._dim(1, 4)}
        if op == "bent":
            shape = str(self.rng.choice(["L", "U"]))
            a = self._dim(4, 9)
            b = self._dim(2, 8)
            t = self._dim(0.5, 2.0)
            if shape == "L":
                b = min(b, a - 1.0 * self.scale)
                t = max(0.1 * self.scale, min(t, b - 0.1 * self.scale))
            else:  # U
                t = max(0.1 * self.scale, min(t, min(a, b) / 2.0 - 0.05 * self.scale))
            return {"shape": shape, "a": a, "b": b, "t": t}
        if op == "mirrored":
            n = int(self.rng.integers(2, 4))
            x0 = self._dim(1.0, 2.0)
            xs = [x0]
            x = x0
            for _ in range(n):
                x += self._dim(0.6, 1.6)
                xs.append(x)
            x1 = float(self.rng.uniform(x + 0.5 * self.scale, x + 2.0 * self.scale))
            mids = [(xi, self._dim(0.5, 3.0)) for xi in xs[1:-1]]
            return {"points": [(xs[0], 0.0)] + mids + [(x1, 0.0)]}
        raise GenerationError(f"Unknown sketch op {op}")

    @staticmethod
    def _scale_sketch_params(op: str, params: Dict[str, Any], scale: float) -> Dict[str, Any]:
        out = dict(params)
        for key in ("width", "height", "radius", "x_radius", "y_radius", "length"):
            if key in out:
                out[key] = float(out[key] * scale)
        return out

    @staticmethod
    def _sketch_extent(op: str, params: Dict[str, Any]) -> float:
        """Максимальный полуразмер эскиза от его центра."""
        if op == "rect":
            return max(params["width"], params["height"]) / 2.0
        if op == "circle":
            return params["radius"]
        if op == "ellipse":
            return max(params["x_radius"], params["y_radius"])
        if op == "polygon":
            return params["radius"]
        if op == "slot":
            return params["length"] / 2.0
        if op in ("polyline", "spline"):
            pts = params["points"]
            if not pts:
                return 1.0
            return max(math.hypot(p[0], p[1]) for p in pts)
        if op == "roundrect":
            return max(params["width"], params["height"]) / 2.0 + params["radius"]
        if op == "frame":
            return self._sketch_extent(params["frame_op"], params["outer"])
        if op == "sector":
            return params["radius"]
        if op == "arc_profile":
            return max(float(params["width"]) / 2.0 + abs(float(params.get("bow", 0.0))),
                       float(params["height"]) / 2.0)
        if op == "ellipse_arc":
            return params["x_radius"]
        if op == "bent":
            return max(params["a"], params["b"])
        if op == "mirrored":
            return max(math.hypot(float(p[0]), float(p[1])) for p in params["points"])
        return 5.0

    def _loft_offsets(self, n_profiles: int) -> List[float]:
        offsets = [0.0]
        z = 0.0
        for _ in range(1, n_profiles):
            z += self._dim(3, 12)
            offsets.append(z)
        return offsets

    def _create_path(self) -> PathNode:
        method = str(self.rng.choice(["spline", "polyline"]))
        n = int(self.rng.integers(3, 5))
        pts = [(0.0, 0.0, 0.0)]
        x = 0.0
        for _ in range(1, n):
            x += self._dim(2, 5)
            y = self._dim(-2, 2)
            z = self._dim(-2, 2)
            pts.append((x, y, z))
        return PathNode(operation="path",
                        parameters={"points": pts, "method": method, "workplane": "XY"})

    def _extrude_params(self) -> Dict[str, Any]:
        return {"distance": self._dim(0.5, 15)}

    def _twist_extrude_params(self) -> Dict[str, Any]:
        return {"distance": self._dim(0.5, 12),
                "angle": float(self.rng.uniform(5, 60))}

    # ------------------------------------------------------------------ #
    # Параметры трансформаций и массивов                                  #
    # ------------------------------------------------------------------ #
    def _transform_params(self, op: str) -> Dict[str, Any]:
        if op == "translate":
            return {"vector": (self._dim(-15, 15),
                               self._dim(-15, 15),
                               self._dim(-15, 15))}
        if op == "rotate":
            return {"axis": str(self.rng.choice(["X", "Y", "Z"])),
                    "angle": float(self.rng.uniform(15, 345))}
        if op == "mirror":
            return {"plane": str(self.rng.choice(["XY", "XZ", "YZ"]))}
        raise GenerationError(f"Unknown transform {op}")

    def _pattern_params(self, op: str, child: ASTNode) -> Dict[str, Any]:
        base = self.executor.execute(child)
        min_dim = self._min_dim(base)
        if op == "rarray":
            nx = int(self.rng.integers(2, 4))
            ny = int(self.rng.integers(2, 4))
            # максимум 6 копий: каждый шаг rarray делает отдельный fuse-подвызов,
            # серия из 16 копий тратила секунду на операцию
            while nx * ny > 6:
                if nx >= ny:
                    nx -= 1
                else:
                    ny -= 1
            # шаг <= 0.9*min_dim гарантирует перекрытие копий -> единый Solid
            spacing_x = float(self.rng.uniform(0.4 * min_dim, 0.9 * min_dim))
            spacing_y = float(self.rng.uniform(0.4 * min_dim, 0.9 * min_dim))
            return {"nx": nx, "ny": ny, "spacing_x": spacing_x, "spacing_y": spacing_y}
        if op == "polarArray":
            # радиус копий <= 0.45*min_dim гарантирует перекрытие с центром
            # -> единый Solid; fill=True добавляет центральную позицию
            radius = float(self.rng.uniform(0.2, 0.45) * min_dim)
            return {"count": int(self.rng.integers(3, 5)), "angle": 360.0,
                    "radius": radius, "start_angle": 0.0, "fill": True}
        if op == "scatter":
            # случайные точки в круге радиуса <= 0.45*min_dim (перекрытие с центром)
            n = int(self.rng.integers(2, 5))
            r_max = 0.45 * min_dim
            pts = []
            for _ in range(n):
                r = float(self.rng.uniform(0.0, r_max))
                a = float(self.rng.uniform(0.0, 2 * np.pi))
                pts.append((r * math.cos(a), r * math.sin(a)))
            return {"points": pts}
        raise GenerationError(f"Unknown pattern {op}")

    # ------------------------------------------------------------------ #
    # Вспомогательные геометрические утилиты                              #
    # ------------------------------------------------------------------ #
    def _jitter(self, scale: float) -> Tuple[float, float, float]:
        return (float(self.rng.uniform(-scale, scale)),
                float(self.rng.uniform(-scale, scale)),
                float(self.rng.uniform(-scale, scale)))

    def _weighted_order(self, cands: List[str], weights: Dict[str, float]) -> List[str]:
        """Перестановка кандидатов, взвешенная конфигом: чем выше вес, тем
        вероятнее операция окажется в начале (и будет применена первой)."""
        pool = list(cands)
        order: List[str] = []
        while pool:
            probs = np.array([max(weights.get(c, 1.0), 1e-9) for c in pool], dtype=float)
            probs = probs / probs.sum()
            idx = int(self.rng.choice(len(pool), p=probs))
            order.append(pool.pop(idx))
        return order

    @staticmethod
    def _bbox(shape) -> Tuple[float, float, float, float, float, float]:
        # fast_bbox строится по триангуляции и на «холодном» кеше OCCT может
        # отличаться в последних битах мантиссы между запусками. Квантуем до
        # нанометров: детерминизм параметров между прогонами одного seed
        # важнее эпсилон-различий (пороги выбора >= 2% не затрагиваются).
        xmin, xmax, ymin, ymax, zmin, zmax = fast_bbox(shape)
        return (round(xmin, 9), round(xmax, 9), round(ymin, 9),
                round(ymax, 9), round(zmin, 9), round(zmax, 9))

    @staticmethod
    def _min_dim(shape) -> float:
        xmin, xmax, ymin, ymax, zmin, zmax = fast_bbox(shape)
        return min(round(xmax - xmin, 9), round(ymax - ymin, 9),
                   round(zmax - zmin, 9))

    @staticmethod
    def _center(shape) -> Tuple[float, float, float]:
        xmin, xmax, ymin, ymax, zmin, zmax = fast_bbox(shape)
        return (round((xmin + xmax) / 2.0, 9),
                round((ymin + ymax) / 2.0, 9),
                round((zmin + zmax) / 2.0, 9))
