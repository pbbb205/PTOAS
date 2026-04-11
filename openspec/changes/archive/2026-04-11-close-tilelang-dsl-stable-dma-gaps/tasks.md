## 1. OpenSpec 契约落定

- [x] 1.1 新增 `openspec/changes/close-tilelang-dsl-stable-dma-gaps/specs/tilelang-dsl-surface/spec.md`，固定 stable DMA keyword surface、`PadMode` 和 public accepted profile。
- [x] 1.2 新增 `openspec/changes/close-tilelang-dsl-stable-dma-gaps/specs/tilelang-dsl-vpto-lowering/spec.md`，固定 stable 2D slicing、DMA inference、padded `dma_load` 和 trimmed `dma_store` 的 lowering 契约。
- [x] 1.3 在 `proposal.md` 和 `design.md` 中明确本 change 保持 frontend-only authoring path，不引入新的 backend capability。

## 2. Frontend surface 与语义模型

- [x] 2.1 在 `tilelang-dsl/python/tilelang_dsl/frontend_ast.py` 和相关 validator 中为 `pto.dma_load` / `pto.dma_store` 增加 keyword 参数承载与校验。
- [x] 2.2 在 `tilelang-dsl/python/tilelang_dsl/types.py`、`__init__.py`、`support_matrix.py` 中补齐 stable `PadMode` public surface 与 tier/unsupported 边界。
- [x] 2.3 在 `tilelang-dsl/python/tilelang_dsl/semantic.py` 中引入 normalized TensorView slice 表示，保留每轴 `start/stop/step` 与 DMA option 字段。
- [x] 2.4 基于 Tile `shape/valid_shape`、slice extent 与 padding/trim 规则实现 stable DMA shape/profile 校验，并为 unsupported 组合提供明确 diagnostics。

## 3. Stable DMA lowering

- [x] 3.1 在 `tilelang-dsl/python/tilelang_dsl/lowering.py` 中实现 non-zero/dynamic start 的 pointer offset lowering，以及基于 slice layout 的 loop stride / loop size 推导。
- [x] 3.2 实现 outer-axis static-step DMA inference，确保 `set_loop*_stride_*` 与 copy-family 参数不再固定为 full-tile contiguous 常量。
- [x] 3.3 实现 padded `dma_load` 的 frontend-only prefill + interior copy lowering，并处理 `PadNull` / `PadFirstElem` / `PadValue` 的稳定子集。
- [x] 3.4 实现 `dma_store` 的 interior trim lowering，并对 GM-side fill / 非 `PadNull` store padding 组合保持 fail-fast reject。

## 4. 回归、文档与验证

- [x] 4.1 在 `tilelang-dsl/tests/test_tilelang_dsl_v1.py` 中增加正向 regression，覆盖 dynamic start/stop、outer-axis static step、padded `dma_load` 和 trimmed `dma_store`。
- [x] 4.2 增加负向 regression，覆盖 dynamic step、第 1 轴 stepped slice、shape/padding 不匹配、unsupported store fill、unsupported init 组合等边界。
- [x] 4.3 更新 `tilelang-dsl/docs/tilelang-dsl-guide.md`、`tilelang-dsl/docs/unsupported-features.md` 和相关 migration/support 文档，使 stable DMA contract 与实现一致。
- [x] 4.4 运行并记录最小验证命令，确认新增 stable DMA regression 能通过 `tilelang-dsl/tests/` 路径下的最小相关测试集。

### 验证记录

- `PYTHONPATH=tilelang-dsl/python python3 -m unittest tilelang-dsl.tests.test_tilelang_dsl_v1`
- `openspec validate close-tilelang-dsl-stable-dma-gaps --type change --strict --json --no-interactive`
