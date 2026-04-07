"""TileLang DSL v1 demo: pto.tadd (element-wise add) using Tile parameters.

Note: v1 surface only supports 1D vectorized iteration within strict_vecscope.
The canonical 2D row×col loop with dynamic masking requires v2 features.
This demo demonstrates a 1D inner-loop pattern over the tile's column extent.
"""

import sys
from pathlib import Path


def _import_tilelang_dsl():
    try:
        import tilelang_dsl as pto
        return pto
    except ModuleNotFoundError:
        repo_root = Path(__file__).resolve().parents[2]
        sys.path.insert(0, str(repo_root / "python"))
        import tilelang_dsl as pto
        return pto


pto = _import_tilelang_dsl()


@pto.vkernel(
    op="pto.tadd",
    dtypes=[(pto.f32, pto.f32, pto.f32)],
    name="template_tadd",
)
def template_tadd(src0: pto.Tile, src1: pto.Tile, dst: pto.Tile):
    # v1 strict_vecscope: all referenced values must be passed in explicitly,
    # and only scalar offsets (not 2D subscripts) are supported for vlds/vsts.
    with pto.strict_vecscope(src0, src1, dst, 0, 256, 64) as (
        a, b, c, lb, ub, step
    ):
        for j in range(lb, ub, step):
            mask = pto.make_mask(pto.f32, pto.PAT.ALL)
            vec_a = pto.vlds(a, j)
            vec_b = pto.vlds(b, j)
            result = pto.vadd(vec_a, vec_b, mask)
            pto.vsts(result, c, j, mask)


def main(argv: list[str]) -> int:
    specialized = template_tadd.specialize(
        src0=pto.TileSpecialization(
            shape=(16, 64),
            memory_space=pto.MemorySpace.UB,
        ),
        src1=pto.TileSpecialization(
            shape=(16, 64),
            memory_space=pto.MemorySpace.UB,
        ),
        dst=pto.TileSpecialization(
            shape=(16, 64),
            memory_space=pto.MemorySpace.UB,
        ),
    )

    if len(argv) == 2:
        output_path = Path(argv[1])
        specialized.emit(output_path)
        print(f"wrote MLIR to {output_path}")
        return 0

    print(specialized.mlir_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
