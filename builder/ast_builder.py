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
    ExtrudeNode, RevolveNode, LoftNode, SweepNode,
    FilletNode, ChamferNode, ShellNode, HoleNode,
    TransformNode, PatternNode, BooleanNode,
)
from ..executor.executor import Executor
from ..validators.geometry_validator import GeometryValidator, ValidationError
from ..utils.geometry_utils import fast_bbox
from ..config import GeneratorConfig
from ..exceptions import GenerationError


@dataclass
class BuildState:
    """Полное состояние генератора для отката при неудачной операции."""
    rng_state: Dict[str, Any]
    total_ops: int
    operation_counts: Dict[str, int]


class ASTBuilder:
    PRIMITIVE_OPS = ["box", "cylinder", "sphere", "cone", "wedge", "torus"]
    SKETCH_OPS = ["rect", "circle", "ellipse", "polygon", "slot"]
    EXTRA_SKETCH_OPS = ["polyline", "spline"]
    EXTRUDE_REVOLVE_OPS = ["extrude", "revolve"]
    LOFT_SWEEP_OPS = ["loft", "sweep"]
    UNARY_OPS = ["fillet", "chamfer", "shell", "hole",
                 "translate", "rotate", "mirror",
                 "rarray", "polarArray"]
    BINARY_OPS = ["union", "cut", "intersect"]

    def __init__(self, config: GeneratorConfig, seed: int = None, complexity: str = "medium"):
        self.config = config
        self.complexity = complexity
        self.base_seed = seed if seed is not None else config.global_seed
        self.rng: np.random.Generator = np.random.default_rng(self.base_seed)
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
            "union": 0, "cut": 0, "intersect": 0,
            "rarray": 0, "polarArray": 0,
            "extrude": 0, "revolve": 0, "loft": 0, "sweep": 0,
        }
        self.target_ops = self.target_min
        self._validated: set = set()

    # ------------------------------------------------------------------ #
    # Точка входа                                                         #
    # ------------------------------------------------------------------ #
    def build(self) -> ASTNode:
        deadline_hits = 0
        for attempt in range(self.backtrack_attempts):
            self._reset(attempt)
            self._deadline = time.time() + self.attempt_timeout
            try:
                target = int(self.rng.integers(self.target_min, self.target_max + 1))
                self.target_ops = target
                node = self._build_solid(0, target)
                # финальная проверка корня как единого целого (всегда полная)
                self._validate(node, is_root=True, force=True)
                count = node.count_operations()
                if self.target_min <= count <= self.target_max:
                    return node
            except TimeoutError:
                deadline_hits += 1
                if deadline_hits >= 10:
                    raise GenerationError(
                        f"Failed to build a {self.complexity} model: repeated timeouts "
                        f"for seed {self.base_seed}")
                continue
            except (GenerationError, ValidationError):
                continue
        raise GenerationError(
            f"Failed to build a {self.complexity} model after {self.backtrack_attempts} attempts "
            f"(target ops {self.target_min}-{self.target_max})"
        )

    def _reset(self, attempt: int):
        self.rng = np.random.default_rng(self.base_seed + attempt * 1000003)
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
            "union": self.config.max_booleans,
            "cut": self.config.max_booleans,
            "intersect": self.config.max_booleans,
            "rarray": self.config.max_patterns,
            "polarArray": self.config.max_patterns,
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
        if time.time() > self._deadline:
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
            self.rng.shuffle(unary_cands)
            for op in unary_cands:
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

        Transform-операции гарантированно работают и почти бесплатны — всегда
        пробуем. fillet/chamfer стоят ~0.7-1с на вызов, hole ~0.1с, массивы —
        серии fuse-операций, а shell стоит несколько секунд на вызов OCCT.
        Дорогие операции пробуем реже, чтобы не тратить время на неудачные
        попытки и держать среднее время генерации низким.
        """
        if op in ("translate", "rotate", "mirror"):
            return True
        if op == "shell":
            # shell крайне дорог (секунды на вызов) и обычно работает только
            # на «простых» телах (без булевых/массивов внутри)
            return self.rng.random() < 0.08 and self._is_simple_shape(child)
        if op in ("fillet", "chamfer"):
            # на сложных телах (после boolean/loft/sweep/pattern) fillet/chamfer
            # могут «зависнуть» в OCCT; применяем только к «простым» телам,
            # где операция хорошо обусловлена и гарантированно завершается
            return self._is_simple_shape(child) and self.rng.random() < 0.5
        if op in ("rarray", "polarArray"):
            # массивы — серии fuse-операций; на сложных телах каждая склейка
            # становится очень дорогой, поэтому пробуем только на «простых»
            return self.rng.random() < 0.5 and self._is_simple_shape(child)
        if op == "hole":
            return self.rng.random() < 0.45
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
        """Пытается применить булевы операции к общим поддеревьям left/right."""
        if time.time() > self._deadline:
            raise TimeoutError("attempt time budget exceeded")
        child_budget = budget - 1
        lo = max(1, child_budget // 3)
        hi = max(lo, child_budget // 2)
        left_budget = int(self.rng.integers(lo, hi + 1))
        right_budget = child_budget - left_budget
        left = self._build_solid(depth + 1, left_budget)
        right = self._build_solid(depth + 1, right_budget)
        self.rng.shuffle(binary_cands)
        for op in binary_cands:
            saved = self._save_state()
            try:
                node = self._create_boolean(op, left, right)
                self._validate(node)
                return node
            except (GenerationError, ValidationError):
                self._restore_state(saved)
                continue
        return None

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
        for _ in range(4):
            pool = safe if (safe and (not risky or self.rng.random() < 0.25)) else avail
            op = str(self.rng.choice(pool))
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
            op = str(self.rng.choice(safe))
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
        elif op in ("translate", "rotate", "mirror"):
            node = TransformNode(operation=op,
                                 parameters=self._transform_params(op),
                                 children=[child])
            self._record_op(op)
        elif op in ("rarray", "polarArray"):
            node = PatternNode(operation=op,
                               parameters=self._pattern_params(op, child),
                               children=[child])
            self._record_op(op)
        else:
            raise GenerationError(f"Unknown unary op {op}")
        return node

    def _try_fillet(self, child: ASTNode) -> FilletNode:
        base = self.executor.execute(child)
        if len(list(base.edges())) == 0:
            raise GenerationError("no edges available for fillet")
        max_r = min(self._min_dim(base) * self.config.fillet_radius_ratio, 5.0)
        if max_r < 0.05:
            raise GenerationError("solid too small for fillet")
        r = self.rng.uniform(0.05, max_r)
        for _ in range(4):
            node = FilletNode(operation="fillet", parameters={"radius": r}, children=[child])
            try:
                shape = self.executor.execute(node)
                self.validator.validate(shape)
            except (GenerationError, ValidationError):
                r *= 0.6
                if r < 0.02:
                    raise GenerationError("fillet radius too small")
                continue
            self._validated.add(node.node_id)
            self._record_op("fillet")
            return node
        raise GenerationError("fillet failed after retries")

    def _try_chamfer(self, child: ASTNode) -> ChamferNode:
        base = self.executor.execute(child)
        if len(list(base.edges())) == 0:
            raise GenerationError("no edges available for chamfer")
        max_d = min(self._min_dim(base) * self.config.fillet_radius_ratio, 5.0)
        if max_d < 0.05:
            raise GenerationError("solid too small for chamfer")
        d = self.rng.uniform(0.05, max_d)
        for _ in range(4):
            node = ChamferNode(operation="chamfer", parameters={"distance": d}, children=[child])
            try:
                shape = self.executor.execute(node)
                self.validator.validate(shape)
            except (GenerationError, ValidationError):
                d *= 0.6
                if d < 0.02:
                    raise GenerationError("chamfer distance too small")
                continue
            self._validated.add(node.node_id)
            self._record_op("chamfer")
            return node
        raise GenerationError("chamfer failed after retries")

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
        for thickness in thicknesses:
            for rank in range(3):
                node = ShellNode(operation="shell",
                                 parameters={"thickness": thickness, "face_rank": rank},
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
        if w < 0.3 or h < 0.3 or d < 0.3:
            raise GenerationError("solid too thin for hole")
        base_vol = float(base.Volume())
        r_max = min(w, h) * 0.4
        if r_max < 0.1:
            raise GenerationError("hole radius range too small")
        for _ in range(4):
            radius = self.rng.uniform(0.1, r_max)
            x = self.rng.uniform(xmin + radius, xmax - radius)
            y = self.rng.uniform(ymin + radius, ymax - radius)
            margin = max(0.5, 0.2 * min(w, h, d))
            position = (x, y, zmin - margin)
            depth = d + 2.0 * margin
            node = HoleNode(operation="hole",
                            parameters={"position": position, "radius": radius, "depth": depth},
                            children=[child])
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

    def _create_boolean(self, op: str, left: ASTNode, right: ASTNode) -> BooleanNode:
        left_shape = self.executor.execute(left)
        right_shape = self.executor.execute(right)
        lc = self._center(left_shape)
        rc = self._center(right_shape)
        left_vol = float(left_shape.Volume())
        min_dim = self._min_dim(left_shape)

        # валидация результата выполняется один раз в _build_solid (self._validate);
        # здесь — только проверки объёмов, требующие исполнения
        if op == "union":
            jitter = self._jitter(scale=0.5 * min_dim)
            offset = (lc[0] - rc[0] + jitter[0],
                      lc[1] - rc[1] + jitter[1],
                      lc[2] - rc[2] + jitter[2])
            node = BooleanNode(operation="union", parameters={"offset": offset},
                               children=[left, right])
            if float(self.executor.execute(node).Volume()) <= left_vol * 1.01:
                raise GenerationError("union added no material")
        elif op == "cut":
            jitter = self._jitter(scale=0.3 * min_dim)
            offset = (lc[0] - rc[0] + jitter[0],
                      lc[1] - rc[1] + jitter[1],
                      lc[2] - rc[2] + jitter[2])
            node = BooleanNode(operation="cut", parameters={"offset": offset},
                               children=[left, right])
            if float(self.executor.execute(node).Volume()) >= left_vol * 0.995:
                raise GenerationError("cut removed nothing")
        else:  # intersect
            offset = (lc[0] - rc[0], lc[1] - rc[1], lc[2] - rc[2])
            node = BooleanNode(operation="intersect", parameters={"offset": offset},
                               children=[left, right])
            if float(self.executor.execute(node).Volume()) >= left_vol * 0.995:
                raise GenerationError("intersect is a no-op")
        self._record_op(op)
        return node

    # ------------------------------------------------------------------ #
    # Примитивы и эскизы                                                  #
    # ------------------------------------------------------------------ #
    def _create_primitive(self, op: str) -> PrimitiveNode:
        params: Dict[str, Any] = {}
        if op == "box":
            params = {"length": float(self.rng.uniform(1, 10)),
                      "width": float(self.rng.uniform(1, 10)),
                      "height": float(self.rng.uniform(1, 10))}
        elif op == "cylinder":
            params = {"height": float(self.rng.uniform(1, 15)),
                      "radius": float(self.rng.uniform(0.5, 5))}
        elif op == "sphere":
            params = {"radius": float(self.rng.uniform(0.5, 5))}
        elif op == "cone":
            params = {"height": float(self.rng.uniform(1, 15)),
                      "radius1": float(self.rng.uniform(0.5, 5)),
                      "radius2": float(self.rng.uniform(0.1, 5))}
        elif op == "wedge":
            dx = float(self.rng.uniform(1, 10))
            dy = float(self.rng.uniform(1, 10))
            dz = float(self.rng.uniform(1, 10))
            xmax = float(self.rng.uniform(1, dx))
            zmax = float(self.rng.uniform(1, dz))
            params = {"dx": dx, "dy": dy, "dz": dz,
                      "xmin": 0.0, "zmin": 0.0, "xmax": xmax, "zmax": zmax}
        elif op == "torus":
            r1 = float(self.rng.uniform(1, 5))
            r2 = float(self.rng.uniform(0.2, min(2.5, r1 * 0.7)))
            params = {"radius1": r1, "radius2": r2}
        return PrimitiveNode(operation=op, parameters=params)

    def _create_sketch_node(self, kind: str = "default") -> SketchNode:
        pool = list(self.SKETCH_OPS) + list(self.EXTRA_SKETCH_OPS)
        op = str(self.rng.choice(pool))
        params = self._sketch_params(op)
        if kind == "revolve":
            # профиль обязан быть смещён от оси вращения
            extent = self._sketch_extent(op, params)
            params["center"] = (float(extent + self.rng.uniform(0.5, 3.0)), 0.0)
            params["workplane"] = "XY"
        elif kind == "loft":
            params["center"] = (0.0, 0.0)
            params["workplane"] = "XY"
        elif kind == "sweep_profile":
            params = {"radius": float(self.rng.uniform(0.3, 1.5))}
            params["center"] = (0.0, 0.0)
            params["workplane"] = "XY"
            op = "circle"
        else:
            params.setdefault("center", (0.0, 0.0))
            params["workplane"] = str(self.rng.choice(["XY", "XZ", "YZ"]))
        return SketchNode(operation=op, parameters=params)

    def _sketch_params(self, op: str) -> Dict[str, Any]:
        if op == "rect":
            return {"width": float(self.rng.uniform(0.5, 8)),
                    "height": float(self.rng.uniform(0.5, 8))}
        if op == "circle":
            return {"radius": float(self.rng.uniform(0.5, 5))}
        if op == "ellipse":
            return {"x_radius": float(self.rng.uniform(0.5, 5)),
                    "y_radius": float(self.rng.uniform(0.5, 5))}
        if op == "polygon":
            return {"n_sides": int(self.rng.integers(3, 8)),
                    "radius": float(self.rng.uniform(0.5, 5))}
        if op == "slot":
            return {"length": float(self.rng.uniform(2, 8)),
                    "width": float(self.rng.uniform(0.5, 3))}
        if op in ("polyline", "spline"):
            # выпуклая оболочка случайных точек: контур без самопересечений,
            # поэтому extrude/revolve всегда дают корректное тело
            n = int(self.rng.integers(3, 6))
            raw = [(float(self.rng.uniform(-4, 4)), float(self.rng.uniform(-4, 4)))
                   for _ in range(n)]
            cx = sum(p[0] for p in raw) / n
            cy = sum(p[1] for p in raw) / n
            pts = sorted(raw, key=lambda p: math.atan2(p[1] - cy, p[0] - cx))
            return {"points": pts}
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
        return 5.0

    def _loft_offsets(self, n_profiles: int) -> List[float]:
        offsets = [0.0]
        z = 0.0
        for _ in range(1, n_profiles):
            z += float(self.rng.uniform(3, 12))
            offsets.append(z)
        return offsets

    def _create_path(self) -> PathNode:
        method = str(self.rng.choice(["spline", "polyline"]))
        n = int(self.rng.integers(3, 5))
        pts = [(0.0, 0.0, 0.0)]
        x = 0.0
        for _ in range(1, n):
            x += float(self.rng.uniform(2, 5))
            y = float(self.rng.uniform(-2, 2))
            z = float(self.rng.uniform(-2, 2))
            pts.append((x, y, z))
        return PathNode(operation="path",
                        parameters={"points": pts, "method": method, "workplane": "XY"})

    def _extrude_params(self) -> Dict[str, Any]:
        return {"distance": float(self.rng.uniform(0.5, 15))}

    # ------------------------------------------------------------------ #
    # Параметры трансформаций и массивов                                  #
    # ------------------------------------------------------------------ #
    def _transform_params(self, op: str) -> Dict[str, Any]:
        if op == "translate":
            return {"vector": (float(self.rng.uniform(-15, 15)),
                               float(self.rng.uniform(-15, 15)),
                               float(self.rng.uniform(-15, 15)))}
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
            # шаг <= 0.9*min_dim гарантирует перекрытие копий -> единый Solid
            spacing_x = float(self.rng.uniform(0.4 * min_dim, 0.9 * min_dim))
            spacing_y = float(self.rng.uniform(0.4 * min_dim, 0.9 * min_dim))
            return {"nx": nx, "ny": ny, "spacing_x": spacing_x, "spacing_y": spacing_y}
        if op == "polarArray":
            return {"count": int(self.rng.integers(3, 6)), "angle": 360.0}
        raise GenerationError(f"Unknown pattern {op}")

    # ------------------------------------------------------------------ #
    # Вспомогательные геометрические утилиты                              #
    # ------------------------------------------------------------------ #
    def _jitter(self, scale: float) -> Tuple[float, float, float]:
        return (float(self.rng.uniform(-scale, scale)),
                float(self.rng.uniform(-scale, scale)),
                float(self.rng.uniform(-scale, scale)))

    @staticmethod
    def _bbox(shape) -> Tuple[float, float, float, float, float, float]:
        return fast_bbox(shape)

    @classmethod
    def _min_dim(cls, shape) -> float:
        xmin, xmax, ymin, ymax, zmin, zmax = cls._bbox(shape)
        return min(xmax - xmin, ymax - ymin, zmax - zmin)

    @classmethod
    def _center(cls, shape) -> Tuple[float, float, float]:
        xmin, xmax, ymin, ymax, zmin, zmax = cls._bbox(shape)
        return ((xmin + xmax) / 2.0, (ymin + ymax) / 2.0, (zmin + zmax) / 2.0)
