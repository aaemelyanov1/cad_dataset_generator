"""Генератор самодостаточного Python-кода CadQuery из AST.

Сгенерированная программа воспроизводит геометрию, идентичную Executor'у, и
состоит ТОЛЬКО из вызовов нативного API CadQuery: никаких пользовательских
функций (def) и никаких вспомогательных хелперов в теле программы. Вся
вспомогательная логика (распаковка Compound в Solid, выбор рёбер/граней,
валидация) выполняется на стороне генератора.

Консистентность между исполнителем и программой достигается тем, что обе стороны
используют одни и те же вызовы CadQuery: например, скругление выполняется через
`solid.fillet(radius, list(solid.edges()))`, оболочка — через
`solid.shell([list(solid.faces())[rank]], thickness)`. Топологический порядок
`edges()`/`faces()` детерминирован для идентично построенного BRep, поэтому
результаты совпадают. Имена переменных уникальны, порядок операций — depth-first.
"""
from .syntax_tree.nodes import (
    ASTNode, PrimitiveNode, SketchNode, ExtrudeNode, RevolveNode,
    LoftNode, SweepNode, FilletNode, ChamferNode,
    ShellNode, HoleNode, TransformNode, PatternNode, BooleanNode,
)

# Глобальный счётчик для уникальных имён переменных
_COUNTER = 0

_HEADER = "import cadquery as cq\n\n"


def _reset_counter():
    global _COUNTER
    _COUNTER = 0


def _next_var() -> str:
    global _COUNTER
    _COUNTER += 1
    return f"solid_{_COUNTER}"


def _fmt(value) -> str:
    """Детерминированное форматирование параметра в литерал Python."""
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, (list, tuple)):
        return "(" + ", ".join(_fmt(v) for v in value) + ")"
    if isinstance(value, str):
        return repr(value)
    return repr(value)


def generate_code(node: ASTNode) -> str:
    """
    Возвращает строку с Python-кодом, который воспроизводит модель.
    В конце кода переменная `result` содержит итоговый shape CadQuery.
    """
    _reset_counter()
    lines, final_var = _gen_node(node)
    lines.append(f"result = {final_var}")
    return _HEADER + "\n".join(lines) + "\n"


def _gen_node(node: ASTNode):
    """Генерирует код для узла, возвращает (список строк, имя итоговой переменной)."""
    if isinstance(node, PrimitiveNode):
        return _gen_primitive(node)
    elif isinstance(node, SketchNode):
        raise ValueError("SketchNode не должен появляться изолированно")
    elif isinstance(node, ExtrudeNode):
        return _gen_extrude(node)
    elif isinstance(node, RevolveNode):
        return _gen_revolve(node)
    elif isinstance(node, LoftNode):
        return _gen_loft(node)
    elif isinstance(node, SweepNode):
        return _gen_sweep(node)
    elif isinstance(node, FilletNode):
        return _gen_fillet(node)
    elif isinstance(node, ChamferNode):
        return _gen_chamfer(node)
    elif isinstance(node, ShellNode):
        return _gen_shell(node)
    elif isinstance(node, HoleNode):
        return _gen_hole(node)
    elif isinstance(node, TransformNode):
        return _gen_transform(node)
    elif isinstance(node, PatternNode):
        return _gen_pattern(node)
    elif isinstance(node, BooleanNode):
        return _gen_boolean(node)
    else:
        raise ValueError(f"Неизвестный тип узла: {type(node)}")


def _gen_primitive(node: PrimitiveNode):
    p = node.parameters
    op = node.operation
    var = _next_var()
    if op == "box":
        code = (f"{var} = cq.Workplane('XY').box({_fmt(p['length'])}, "
                f"{_fmt(p['width'])}, {_fmt(p['height'])}).val()")
    elif op == "cylinder":
        code = f"{var} = cq.Workplane('XY').cylinder({_fmt(p['height'])}, {_fmt(p['radius'])}).val()"
    elif op == "sphere":
        code = f"{var} = cq.Workplane('XY').sphere({_fmt(p['radius'])}).val()"
    elif op == "cone":
        code = f"{var} = cq.Solid.makeCone({_fmt(p['radius1'])}, {_fmt(p['radius2'])}, {_fmt(p['height'])})"
    elif op == "wedge":
        code = (f"{var} = cq.Workplane('XY').wedge({_fmt(p['dx'])}, {_fmt(p['dy'])}, "
                f"{_fmt(p['dz'])}, {_fmt(p['xmin'])}, {_fmt(p['zmin'])}, "
                f"{_fmt(p['xmax'])}, {_fmt(p['zmax'])}).val()")
    elif op == "torus":
        code = f"{var} = cq.Solid.makeTorus({_fmt(p['radius1'])}, {_fmt(p['radius2'])})"
    else:
        raise ValueError(f"Неизвестный примитив: {op}")
    return [code], var


def _gen_sketch(wp_var: str, op: str, params: dict) -> list:
    """Код, рисующий эскиз `op` на workplane-переменной `wp_var`."""
    center = params.get("center", (0.0, 0.0))
    cx, cy = float(center[0]), float(center[1])
    if op in ("polyline", "spline"):
        # moveTo не влияет на polyline/spline — смещение центра запекаем в точки
        pts = [(float(p[0]) + cx, float(p[1]) + cy) for p in params["points"]]
        return [f"{wp_var} = {wp_var}.{op}({_fmt(pts)}).close()"]
    lines = []
    if cx != 0.0 or cy != 0.0:
        lines.append(f"{wp_var} = {wp_var}.moveTo({_fmt(cx)}, {_fmt(cy)})")
    if op == "rect":
        lines.append(f"{wp_var} = {wp_var}.rect({_fmt(params['width'])}, {_fmt(params['height'])})")
    elif op == "circle":
        lines.append(f"{wp_var} = {wp_var}.circle({_fmt(params['radius'])})")
    elif op == "ellipse":
        lines.append(f"{wp_var} = {wp_var}.ellipse({_fmt(params['x_radius'])}, {_fmt(params['y_radius'])})")
    elif op == "polygon":
        lines.append(f"{wp_var} = {wp_var}.polygon({int(params['n_sides'])}, {_fmt(params['radius'])})")
    elif op == "slot":
        lines.append(f"{wp_var} = {wp_var}.slot2D({_fmt(params['length'])}, {_fmt(params['width'])}, 0)")
    else:
        raise ValueError(f"Неизвестный эскиз: {op}")
    return lines


def _gen_extrude(node: ExtrudeNode):
    sk = node.children[0]
    wp_var = _next_var()
    lines = [f"{wp_var} = cq.Workplane({_fmt(sk.parameters.get('workplane', 'XY'))})"]
    lines += _gen_sketch(wp_var, sk.operation, sk.parameters)
    var = _next_var()
    lines.append(f"{var} = {wp_var}.extrude({_fmt(node.parameters['distance'])}).val()")
    return lines, var


def _gen_revolve(node: RevolveNode):
    sk = node.children[0]
    wp_var = _next_var()
    lines = [f"{wp_var} = cq.Workplane({_fmt(sk.parameters.get('workplane', 'XY'))})"]
    lines += _gen_sketch(wp_var, sk.operation, sk.parameters)
    var = _next_var()
    lines.append(f"{var} = {wp_var}.revolve({_fmt(node.parameters.get('angle', 360.0))}).val()")
    return lines, var


def _gen_loft(node: LoftNode):
    offsets = node.parameters["offsets"]
    profiles = node.children
    wp_var = _next_var()
    first = profiles[0]
    lines = [f"{wp_var} = cq.Workplane('XY')"]
    lines += _gen_sketch(wp_var, first.operation, first.parameters)
    prev = offsets[0]
    for child, z in zip(profiles[1:], offsets[1:]):
        gap = z - prev
        lines.append(f"{wp_var} = {wp_var}.workplane(offset={_fmt(gap)})")
        lines += _gen_sketch(wp_var, child.operation, child.parameters)
        prev = z
    var = _next_var()
    lines.append(f"{var} = {wp_var}.loft().val()")
    return lines, var


def _gen_sweep(node: SweepNode):
    profile = node.children[0]
    path = node.children[1]
    wp_var = _next_var()
    lines = [f"{wp_var} = cq.Workplane({_fmt(profile.parameters.get('workplane', 'XY'))})"]
    lines += _gen_sketch(wp_var, profile.operation, profile.parameters)
    p = path.parameters
    pts = p["points"]
    method = p.get("method", "spline")
    wplane = p.get("workplane", "XY")
    pw = _next_var()
    lines.append(f"{pw} = cq.Workplane({_fmt(wplane)})")
    if method == "polyline":
        lines.append(f"{pw} = {pw}.polyline({_fmt(pts)}).val()")
    else:
        lines.append(f"{pw} = {pw}.spline({_fmt(pts)}).val()")
    var = _next_var()
    lines.append(f"{var} = {wp_var}.sweep({pw}).val()")
    return lines, var


def _gen_fillet(node: FilletNode):
    child_lines, child_var = _gen_node(node.children[0])
    var = _next_var()
    lines = child_lines + [
        f"{var} = {child_var}.fillet({_fmt(node.parameters['radius'])}, list({child_var}.edges())[:12])"
    ]
    return lines, var


def _gen_chamfer(node: ChamferNode):
    child_lines, child_var = _gen_node(node.children[0])
    var = _next_var()
    lines = child_lines + [
        f"{var} = {child_var}.chamfer({_fmt(node.parameters['distance'])}, None, list({child_var}.edges())[:12])"
    ]
    return lines, var


def _gen_shell(node: ShellNode):
    child_lines, child_var = _gen_node(node.children[0])
    var = _next_var()
    rank = int(node.parameters.get("face_rank", 0))
    lines = child_lines + [
        f"_faces = list({child_var}.faces())",
        f"{var} = {child_var}.shell([_faces[{rank}]], {_fmt(node.parameters['thickness'])})",
    ]
    return lines, var


def _gen_hole(node: HoleNode):
    child_lines, child_var = _gen_node(node.children[0])
    p = node.parameters
    var = _next_var()
    lines = child_lines + [
        f"_cyl = cq.Solid.makeCylinder({_fmt(p['radius'])}, {_fmt(p['depth'])}, "
        f"cq.Vector{_fmt(p['position'])}, cq.Vector(0, 0, 1))",
        f"{var} = {child_var}.cut(_cyl)",
    ]
    return lines, var


def _gen_transform(node: TransformNode):
    child_lines, child_var = _gen_node(node.children[0])
    var = _next_var()
    op = node.operation
    if op == "translate":
        v = node.parameters["vector"]
        lines = child_lines + [f"{var} = {child_var}.translate(cq.Vector{_fmt(v)})"]
    elif op == "rotate":
        axis = node.parameters["axis"]
        angle = node.parameters["angle"]
        axis_map = {"X": "cq.Vector(1, 0, 0)", "Y": "cq.Vector(0, 1, 0)", "Z": "cq.Vector(0, 0, 1)"}
        lines = child_lines + [
            f"_center = {child_var}.Center()",
            f"_axis_end = _center + {axis_map[axis]} * 10.0",
            f"{var} = {child_var}.rotate(_center, _axis_end, {_fmt(angle)})",
        ]
    elif op == "mirror":
        plane = node.parameters["plane"]
        lines = child_lines + [
            f"_center = {child_var}.Center()",
            f"{var} = {child_var}.mirror({_fmt(plane)}, _center)",
        ]
    else:
        raise ValueError(f"Неизвестная трансформация {op}")
    return lines, var


def _gen_pattern(node: PatternNode):
    child_lines, child_var = _gen_node(node.children[0])
    var = _next_var()
    op = node.operation
    if op == "rarray":
        p = node.parameters
        nx, ny = int(p["nx"]), int(p["ny"])
        lines = child_lines + [
            f"{var} = {child_var}",
            f"for _i in range({nx}):",
            f"    for _j in range({ny}):",
            f"        if _i == 0 and _j == 0:",
            f"            continue",
            f"        {var} = {var}.fuse({child_var}.translate(cq.Vector("
            f"_i * {_fmt(p['spacing_x'])}, _j * {_fmt(p['spacing_y'])}, 0)))",
        ]
    elif op == "polarArray":
        p = node.parameters
        count = int(p["count"])
        angle = p.get("angle", 360.0)
        lines = child_lines + [
            f"_center = {child_var}.Center()",
            f"_axis_end = _center + cq.Vector(0, 0, 1) * 10.0",
            f"{var} = {child_var}",
            f"for _i in range(1, {count}):",
            f"    {var} = {var}.fuse({child_var}.rotate(_center, _axis_end, "
            f"{_fmt(angle)} * _i / {count}))",
        ]
    else:
        raise ValueError(f"Неизвестный массив {op}")
    return lines, var


def _gen_boolean(node: BooleanNode):
    left_lines, left_var = _gen_node(node.children[0])
    right_lines, right_var = _gen_node(node.children[1])
    op = node.operation
    method = {"union": "fuse", "cut": "cut", "intersect": "intersect"}[op]
    var = _next_var()
    offset = node.parameters.get("offset", (0.0, 0.0, 0.0))
    lines = left_lines + right_lines
    if any(float(o) != 0.0 for o in offset):
        lines.append(f"_right = {right_var}.translate(cq.Vector{_fmt(offset)})")
        lines.append(f"{var} = {left_var}.{method}(_right)")
    else:
        lines.append(f"{var} = {left_var}.{method}({right_var})")
    return lines, var
