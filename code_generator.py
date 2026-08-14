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
import math
from .syntax_tree.nodes import (
    ASTNode, PrimitiveNode, SketchNode, ExtrudeNode, RevolveNode,
    TwistExtrudeNode, LoftNode, SweepNode, FilletNode, ChamferNode,
    ShellNode, HoleNode, SplitNode, TransformNode, PatternNode, BooleanNode,
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
    elif isinstance(node, TwistExtrudeNode):
        return _gen_twist_extrude(node)
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
    elif isinstance(node, SplitNode):
        return _gen_split(node)
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
    """Код, рисующий эскиз `op` на workplane-переменной `wp_var`.

    Воспроизводит ровно те же вызовы/математику, что и `draw_sketch`,
    поэтому геометрия совпадает с исполнителем.
    """
    center = params.get("center", (0.0, 0.0))
    cx, cy = float(center[0]), float(center[1])
    if op in ("polyline", "spline"):
        # moveTo не влияет на polyline/spline — смещение центра запекаем в точки
        pts = [(float(p[0]) + cx, float(p[1]) + cy) for p in params["points"]]
        return [f"{wp_var} = {wp_var}.{op}({_fmt(pts)}).close()"]
    if op == "mirrored":
        pts = [(float(p[0]), float(p[1])) for p in params["points"]]
        return [f"{wp_var} = {wp_var}.polyline({_fmt(pts)}).mirrorX()"]
    if op == "ellipse_arc":
        xr = _fmt(params['x_radius'])
        yr = _fmt(params['y_radius'])
        return [f"{wp_var} = {wp_var}.moveTo(-{xr}, 0.0).ellipseArc({xr}, {yr}, 0.0, 180.0)"
                f".lineTo(-{xr}, 0.0).close()"]
    if op == "arc_profile":
        w, h, bow = params['width'], params['height'], params.get('bow', 0.0)
        return [f"{wp_var} = {wp_var}.moveTo({_fmt(-float(w) / 2 + cx)}, {_fmt(-float(h) / 2 + cy)})"
                f".lineTo({_fmt(float(w) / 2 + cx)}, {_fmt(-float(h) / 2 + cy)})"
                f".threePointArc({_fmt((float(w) / 2 + float(bow) + cx, cy))}, "
                f"{_fmt((float(w) / 2 + cx, float(h) / 2 + cy))})"
                f".lineTo({_fmt(-float(w) / 2 + cx)}, {_fmt(float(h) / 2 + cy)})"
                f".close()"]
    lines = []
    if cx != 0.0 or cy != 0.0:
        lines.append(f"{wp_var} = {wp_var}.moveTo({_fmt(cx)}, {_fmt(cy)})")

    if op == "frame":
        outer, inner = params["outer"], params["inner"]
        lines += _gen_basic_sketch(wp_var, params["frame_op"], outer)
        lines += _gen_basic_sketch(wp_var, params["frame_op"], inner)
        return lines
    if op == "roundrect":
        lines.append(f"{wp_var} = {wp_var}.rect({_fmt(params['width'])}, "
                     f"{_fmt(params['height'])}).offset2D({_fmt(params['radius'])})")
        return lines
    if op == "sector":
        r, ang = params['radius'], float(params['angle'])
        half = math.radians(ang / 2.0)
        full = math.radians(ang)
        mid = (cx + r * math.cos(half), cy + r * math.sin(half))
        end = (cx + r * math.cos(full), cy + r * math.sin(full))
        lines.append(f"{wp_var} = {wp_var}.polarLineTo({_fmt(r)}, 0.0)"
                     f".threePointArc({_fmt(mid)}, {_fmt(end)}).close()")
        return lines
    if op == "bent":
        shape = params.get("shape", "L")
        a, b, t = params['a'], params['b'], params['t']
        if shape == "L":
            chain = (f".hLine({_fmt(a)}).vLine({_fmt(t)}).hLine({_fmt(-(float(a) - float(b)))})"
                     f".vLine({_fmt(b)}).hLine({_fmt(-float(t))}).vLine({_fmt(-float(b))}).close()")
        elif shape == "U":
            chain = (f".hLine({_fmt(a)}).vLine({_fmt(b)}).hLine({_fmt(-float(t))})"
                     f".vLine({_fmt(-(float(b) - 2 * float(t)))})"
                     f".hLine({_fmt(-(float(a) - 2 * float(t)))})"
                     f".vLine({_fmt(float(b) - 2 * float(t))})"
                     f".hLine({_fmt(-float(t))}).close()")
        else:
            raise ValueError(f"Неизвестный bent-профиль: {shape}")
        lines.append(f"{wp_var} = {wp_var}{chain}")
        return lines

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


def _gen_basic_sketch(wp_var: str, op: str, params: dict) -> list:
    """Код базового замкнутого контура (без обработки центра) — для frame."""
    if op == "rect":
        return [f"{wp_var} = {wp_var}.rect({_fmt(params['width'])}, {_fmt(params['height'])})"]
    if op == "circle":
        return [f"{wp_var} = {wp_var}.circle({_fmt(params['radius'])})"]
    if op == "ellipse":
        return [f"{wp_var} = {wp_var}.ellipse({_fmt(params['x_radius'])}, {_fmt(params['y_radius'])})"]
    if op == "polygon":
        return [f"{wp_var} = {wp_var}.polygon({int(params['n_sides'])}, {_fmt(params['radius'])})"]
    if op == "slot":
        return [f"{wp_var} = {wp_var}.slot2D({_fmt(params['length'])}, {_fmt(params['width'])}, 0)"]
    raise ValueError(f"Неизвестный базовый эскиз: {op}")


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


def _gen_twist_extrude(node: TwistExtrudeNode):
    sk = node.children[0]
    wp_var = _next_var()
    lines = [f"{wp_var} = cq.Workplane({_fmt(sk.parameters.get('workplane', 'XY'))})"]
    lines += _gen_sketch(wp_var, sk.operation, sk.parameters)
    var = _next_var()
    lines.append(f"{var} = {wp_var}.twistExtrude({_fmt(node.parameters['distance'])}, "
                 f"{_fmt(node.parameters.get('angle', 0.0))}, clean=False).val()")
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


def _sel_edge_expr(child_var: str, selection: dict) -> str:
    """Выражение выборки рёбер через нативный Selector CadQuery."""
    kind = selection.get("kind", "nearest")
    if kind == "nearest":
        return f"cq.NearestToPointSelector({_fmt(selection['point'])}).filter({child_var}.Edges())[:2]"
    if kind == "direction":
        return f"cq.DirectionSelector(cq.Vector{_fmt(selection['direction'])}).filter({child_var}.Edges())[:2]"
    raise ValueError(f"Неизвестная выборка рёбер: {kind}")


def _sel_face_expr(child_var: str, selection: dict) -> str:
    """Выражение выборки грани через нативный Selector CadQuery."""
    kind = selection.get("kind", "direction")
    if kind == "direction":
        return f"cq.DirectionSelector(cq.Vector{_fmt(selection['direction'])}).filter({child_var}.Faces())[0]"
    if kind == "nearest":
        return f"cq.NearestToPointSelector({_fmt(selection['point'])}).filter({child_var}.Faces())[0]"
    raise ValueError(f"Неизвестная выборка граней: {kind}")


def _gen_fillet(node: FilletNode):
    child_lines, child_var = _gen_node(node.children[0])
    sel = node.parameters.get("selection", {"kind": "nearest", "point": (0.0, 0.0, 0.0)})
    var = _next_var()
    lines = child_lines + [
        f"{var} = {child_var}.fillet({_fmt(node.parameters['radius'])}, {_sel_edge_expr(child_var, sel)})"
    ]
    return lines, var


def _gen_chamfer(node: ChamferNode):
    child_lines, child_var = _gen_node(node.children[0])
    sel = node.parameters.get("selection", {"kind": "nearest", "point": (0.0, 0.0, 0.0)})
    var = _next_var()
    lines = child_lines + [
        f"{var} = {child_var}.chamfer({_fmt(node.parameters['distance'])}, None, {_sel_edge_expr(child_var, sel)})"
    ]
    return lines, var


def _gen_shell(node: ShellNode):
    child_lines, child_var = _gen_node(node.children[0])
    sel = node.parameters.get("selection", {"kind": "direction", "direction": (0.0, 0.0, 1.0)})
    var = _next_var()
    lines = child_lines + [
        f"{var} = {child_var}.shell([{_sel_face_expr(child_var, sel)}], {_fmt(node.parameters['thickness'])})",
    ]
    return lines, var


def _gen_hole(node: HoleNode):
    child_lines, child_var = _gen_node(node.children[0])
    p = node.parameters
    kind = p.get("kind", "through")
    var = _next_var()
    pos = p["position"]
    r, depth = p["radius"], p["depth"]
    if kind == "through":
        lines = child_lines + [
            f"_cyl = cq.Solid.makeCylinder({_fmt(r)}, {_fmt(depth)}, "
            f"cq.Vector{_fmt(pos)}, cq.Vector(0, 0, 1))",
            f"{var} = {child_var}.cut(_cyl)",
        ]
    elif kind == "blind":
        lines = child_lines + [
            f"_z_top = {_fmt(float(pos[2]))}",
            f"_cyl = cq.Solid.makeCylinder({_fmt(r)}, {_fmt(depth)}, "
            f"cq.Vector({_fmt(float(pos[0]))}, {_fmt(float(pos[1]))}, _z_top - {_fmt(depth)}), "
            f"cq.Vector(0, 0, 1))",
            f"{var} = {child_var}.cut(_cyl)",
        ]
    elif kind == "cbore":
        lines = child_lines + [
            f"_cbo_top = {_fmt(float(pos[2]))}",
            f"_head = cq.Solid.makeCylinder({_fmt(p['cbo_radius'])}, {_fmt(p['cbo_depth'])}, "
            f"cq.Vector({_fmt(float(pos[0]))}, {_fmt(float(pos[1]))}, "
            f"_cbo_top - {_fmt(float(p['cbo_depth']))}), cq.Vector(0, 0, 1))",
            f"_bore = cq.Solid.makeCylinder({_fmt(r)}, {_fmt(depth)}, "
            f"cq.Vector({_fmt(float(pos[0]))}, {_fmt(float(pos[1]))}, "
            f"_cbo_top - {_fmt(float(p['cbo_depth']))}), cq.Vector(0, 0, 1))",
            f"{var} = {child_var}.cut(_head).cut(_bore)",
        ]
    elif kind == "csk":
        lines = child_lines + [
            f"_csk_top = {_fmt(float(pos[2]))}",
            f"_cone = cq.Solid.makeCone({_fmt(r)}, {_fmt(p['csk_radius'])}, {_fmt(p['csk_depth'])}, "
            f"cq.Vector({_fmt(float(pos[0]))}, {_fmt(float(pos[1]))}, "
            f"_csk_top - {_fmt(float(p['csk_depth']))}), cq.Vector(0, 0, 1))",
            f"_bore = cq.Solid.makeCylinder({_fmt(r)}, {_fmt(depth)}, "
            f"cq.Vector({_fmt(float(pos[0]))}, {_fmt(float(pos[1]))}, "
            f"_csk_top - {_fmt(float(p['csk_depth']))}), cq.Vector(0, 0, 1))",
            f"{var} = {child_var}.cut(_cone).cut(_bore)",
        ]
    else:
        raise ValueError(f"Неизвестный вид отверстия: {kind}")
    return lines, var


def _gen_split(node: SplitNode):
    child_lines, child_var = _gen_node(node.children[0])
    p = node.parameters
    axis_vec = {"X": "cq.Vector(1, 0, 0)",
                "Y": "cq.Vector(0, 1, 0)",
                "Z": "cq.Vector(0, 0, 1)"}[p["axis"]]
    gap = float(p["gap"])
    var = _next_var()
    lines = child_lines + [
        f"_plane = cq.Face.makePlane(1000000.0, 1000000.0, {child_var}.Center(), {axis_vec})",
        f"_halves = {child_var}.split(_plane)",
        f"_parts = list(_halves.Solids())",
        f"_shifted = _parts[1].translate({axis_vec} * ({_fmt(-gap)}))",
        f"{var} = _parts[0].fuse(_shifted)",
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
        lines = child_lines + [
            f"{var} = cq.Workplane('XY', origin={child_var}.Center()).rarray("
            f"{_fmt(p['spacing_x'])}, {_fmt(p['spacing_y'])}, {int(p['nx'])}, {int(p['ny'])}, "
            f"center=False).eachpoint({child_var}, combine='a').val()",
        ]
    elif op == "polarArray":
        p = node.parameters
        lines = child_lines + [
            f"{var} = cq.Workplane('XY', origin={child_var}.Center()).polarArray("
            f"{_fmt(p['radius'])}, {_fmt(p.get('start_angle', 0.0))}, "
            f"{_fmt(p.get('angle', 360.0))}, {int(p['count'])}, "
            f"fill={p.get('fill', True)}).eachpoint({child_var}, combine='a').val()",
        ]
    elif op == "scatter":
        p = node.parameters
        lines = child_lines + [
            f"{var} = cq.Workplane('XY', origin={child_var}.Center()).pushPoints("
            f"{_fmt(p['points'])}).eachpoint({child_var}, combine='a').val()",
        ]
    else:
        raise ValueError(f"Неизвестный массив {op}")
    return lines, var


def _gen_boolean(node: BooleanNode):
    left_lines, left_var = _gen_node(node.children[0])
    right_lines, right_var = _gen_node(node.children[1])
    op = node.operation
    var = _next_var()
    offset = node.parameters.get("offset", (0.0, 0.0, 0.0))
    lines = left_lines + right_lines
    if any(float(o) != 0.0 for o in offset):
        lines.append(f"_right = {right_var}.translate(cq.Vector{_fmt(offset)})")
    else:
        lines.append(f"_right = {right_var}")
    if op == "union":
        # clean=False: Shape.clean() в CadQuery 2.7 фатально ломает геометрию
        # некоторых булевых объединений (исчезающие грани, падение объёма),
        # а исполнитель делает обычный fuse — поэтому и здесь без clean().
        lines.append(f"{var} = cq.Workplane('XY').add({left_var}).union(_right, clean=False).val()")
    elif op == "cut":
        lines.append(f"{var} = {left_var}.cut(_right)")
    elif op == "intersect":
        lines.append(f"{var} = {left_var}.intersect(_right)")
    else:
        raise ValueError(f"Неизвестная булева операция {op}")
    return lines, var
