#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
VPTO_ROOT="${VPTO_ROOT:-${ROOT_DIR}/test/vpto/cases}"
CASES_ROOT="${CASES_ROOT:-${VPTO_ROOT}}"
NPU_VALIDATION_COMMON_DIR="${NPU_VALIDATION_COMMON_DIR:-${ROOT_DIR}/test/vpto/npu_validation/common}"

WORK_SPACE="${WORK_SPACE:-}"
ASCEND_HOME_PATH="${ASCEND_HOME_PATH:-}"
PTOAS_BIN="${PTOAS_BIN:-${ROOT_DIR}/build/tools/ptoas/ptoas}"
PTOAS_FLAGS="${PTOAS_FLAGS:---pto-arch a5}"
VPTO_FLAGS="${VPTO_FLAGS:---pto-backend=vpto --vpto-emit-hivm-llvm}"
AICORE_ARCH="${AICORE_ARCH:-dav-c310-vec}"
CUBE_AICORE_ARCH="${CUBE_AICORE_ARCH:-dav-c310-cube}"
CUBE_CASES="${CUBE_CASES:-mad_bias mad_mx mad_f16f16f32 mad_f32f32f32 mad_bf16bf16f32 cube-bridge-matmul cube-bridge-store-nz2dn-nchw cube-bridge-store-nz2dn-ncdhw cube-load-frac-layouts cbuf-ubuf-roundtrip-mixed fixpipe-cc-gm-ub}"
GENERIC_AICORE_ARCH="${GENERIC_AICORE_ARCH:-dav-c310}"
GENERIC_CASES="${GENERIC_CASES:-}"
# set he HOST_RUNNER to "ssh root@localhost" if must change user to root to access the device 
HOST_RUNNER="${HOST_RUNNER:-}"
CASE_NAME="${CASE_NAME:-}"
MODULE_ID="${MODULE_ID:-a5d60abf67864aa0}"
DEVICE="${DEVICE:-SIM}"
SIM_LIB_DIR="${SIM_LIB_DIR:-}"
COMPILE_ONLY="${COMPILE_ONLY:-0}"

declare -a CUBE_CASE_LIST=()
declare -a GENERIC_CASE_LIST=()
if [[ -n "${CUBE_CASES}" ]]; then
  read -r -a CUBE_CASE_LIST <<< "${CUBE_CASES//,/ }"
fi
if [[ -n "${GENERIC_CASES}" ]]; then
  read -r -a GENERIC_CASE_LIST <<< "${GENERIC_CASES//,/ }"
fi

log() {
  echo "[$(date +'%F %T')] $*"
}

die() {
  echo "ERROR: $*" >&2
  exit 1
}

run_remote() {
  local cmd="$1"
  if [[ "${HOST_RUNNER}" == "ssh root@localhost" ]]; then
    ssh -o StrictHostKeyChecking=no root@localhost "${cmd}"
  else
    bash -lc "${cmd}"
  fi
}

require_env() {
  local name="$1"
  local value="$2"
  if [[ -z "${value}" ]]; then
    die "${name} is required"
  fi
}

require_env "WORK_SPACE" "${WORK_SPACE}"
require_env "ASCEND_HOME_PATH" "${ASCEND_HOME_PATH}"
[[ -x "${PTOAS_BIN}" ]] || die "PTOAS_BIN is not executable: ${PTOAS_BIN}"
[[ -d "${CASES_ROOT}" ]] || die "missing cases root: ${CASES_ROOT}"

if [[ -f "${ASCEND_HOME_PATH}/set_env.sh" ]]; then
  set +u
  source "${ASCEND_HOME_PATH}/set_env.sh" >/dev/null 2>&1
  set -u
fi

resolve_sim_lib_dir() {
  if [[ "${DEVICE}" != "SIM" ]]; then
    return 0
  fi

  if [[ -n "${SIM_LIB_DIR}" ]]; then
    [[ -d "${SIM_LIB_DIR}" ]] ||
      die "SIM_LIB_DIR is set but invalid: ${SIM_LIB_DIR}"
    return 0
  fi

  local -a candidates=()
  readarray -t candidates < <(
    find "${ASCEND_HOME_PATH}" -type d -path '*/simulator/dav_3510/lib' | sort
  )

  if [[ "${#candidates[@]}" -eq 1 ]]; then
    SIM_LIB_DIR="${candidates[0]}"
    log "SIM_LIB_DIR is unset; auto-selected: ${SIM_LIB_DIR}"
    return 0
  fi

  if [[ "${#candidates[@]}" -gt 1 ]]; then
    SIM_LIB_DIR="${candidates[0]}"
    log "SIM_LIB_DIR is unset; multiple dav_3510 simulator dirs found, using: ${SIM_LIB_DIR}"
    return 0
  fi

  die "SIM_LIB_DIR is required for DEVICE=SIM and no dav_3510 simulator lib dir was found under: ${ASCEND_HOME_PATH}"
}

resolve_sim_lib_dir

BISHENG_BIN="${BISHENG_BIN:-${ASCEND_HOME_PATH}/bin/bisheng}"
BISHENG_CC1_BIN="${BISHENG_CC1_BIN:-${ASCEND_HOME_PATH}/tools/bisheng_compiler/bin/bisheng}"
CCE_LD_BIN="${CCE_LD_BIN:-${ASCEND_HOME_PATH}/bin/cce-ld}"
LD_LLD_BIN="${LD_LLD_BIN:-${ASCEND_HOME_PATH}/bin/ld.lld}"
CLANG_RESOURCE_DIR="${CLANG_RESOURCE_DIR:-${ASCEND_HOME_PATH}/tools/bisheng_compiler/lib/clang/15.0.5}"
CCE_STUB_DIR="${CCE_STUB_DIR:-${CLANG_RESOURCE_DIR}/include/cce_stub}"

HOST_ARCH="$(uname -m)"
HOST_TRIPLE=""
HOST_TARGET_CPU=""
HOST_TARGET_ABI=""
HOST_FEATURE_FLAGS=()
HOST_OS_DIR=""

case "${HOST_ARCH}" in
  aarch64)
    HOST_TRIPLE="aarch64-unknown-linux-gnu"
    HOST_TARGET_CPU="generic"
    HOST_TARGET_ABI="aapcs"
    HOST_FEATURE_FLAGS=(-target-feature +neon -target-feature +v8a)
    HOST_OS_DIR="aarch64-linux"
    ;;
  x86_64)
    HOST_TRIPLE="x86_64-unknown-linux-gnu"
    HOST_TARGET_CPU="x86-64"
    HOST_OS_DIR="x86_64-linux"
    ;;
  *)
    die "unsupported host arch from uname -m: ${HOST_ARCH}"
    ;;
esac

command -v "${BISHENG_BIN}" >/dev/null 2>&1 || die "bisheng not found: ${BISHENG_BIN}"
command -v python3 >/dev/null 2>&1 || die "python3 not found"

readarray -t BISHENG_SYSTEM_INCLUDES < <(
  "${BISHENG_BIN}" -xc++ -E -v - </dev/null 2>&1 |
    awk '
      /#include <...> search starts here:/ {capture=1; next}
      /End of search list\./ {capture=0}
      capture && $0 ~ /^ / {sub(/^ +/, "", $0); print}
    '
)

[[ "${#BISHENG_SYSTEM_INCLUDES[@]}" -gt 0 ]] || die "failed to discover bisheng system include directories"

CC1_INCLUDE_FLAGS=()
for inc in "${BISHENG_SYSTEM_INCLUDES[@]}"; do
  if [[ "${inc}" == */include/c++/* || "${inc}" == */backward ]]; then
    CC1_INCLUDE_FLAGS+=(-internal-isystem "${inc}")
  elif [[ "${inc}" == "/usr/include" ]]; then
    CC1_INCLUDE_FLAGS+=(-internal-externc-isystem "${inc}")
  else
    CC1_INCLUDE_FLAGS+=(-internal-isystem "${inc}")
  fi
done

mkdir -p "${WORK_SPACE}"
WORK_SPACE="$(cd "${WORK_SPACE}" && pwd)"

discover_cases() {
  local required_files=(
    stub.cpp
    launch.cpp
    main.cpp
    golden.py
    compare.py
  )

  if [[ -n "${CASE_NAME}" ]]; then
    local requested_dir="${CASES_ROOT}/${CASE_NAME}"
    [[ -d "${requested_dir}" ]] || die "unknown case: ${CASE_NAME}"
    for f in "${required_files[@]}"; do
      [[ -f "${requested_dir}/${f}" ]] || die "case ${CASE_NAME} is missing ${f}"
    done
    [[ -f "${requested_dir}/kernel.pto" || -f "${requested_dir}/cube.pto" ]] ||
      die "case ${CASE_NAME} must provide kernel.pto and/or cube.pto"
    printf "%s\n" "${CASE_NAME#/}"
    return 0
  fi

  find "${CASES_ROOT}" -mindepth 1 -type d | sort | while read -r dir; do
    local ok=1
    for f in "${required_files[@]}"; do
      if [[ ! -f "${dir}/${f}" ]]; then
        ok=0
        break
      fi
    done
    [[ "${ok}" -eq 1 ]] || continue
    if [[ ! -f "${dir}/kernel.pto" && ! -f "${dir}/cube.pto" ]]; then
      continue
    fi
    local rel="${dir#${CASES_ROOT}/}"
    printf "%s\n" "${rel}"
  done
}

readarray -t CASES < <(discover_cases)
[[ "${#CASES[@]}" -gt 0 ]] || die "no cases found under ${CASES_ROOT}"

case_uses_cube_mode() {
  local case_name="$1"
  local case_base="${case_name##*/}"
  for item in "${CUBE_CASE_LIST[@]}"; do
    [[ -n "${item}" ]] || continue
    if [[ "${case_name}" == "${item}" || "${case_name}" == */"${item}" ||
          "${case_base}" == "${item}" ]]; then
      return 0
    fi
  done
  return 1
}

case_uses_generic_mode() {
  local case_name="$1"
  local case_base="${case_name##*/}"
  for item in "${GENERIC_CASE_LIST[@]}"; do
    [[ -n "${item}" ]] || continue
    if [[ "${case_name}" == "${item}" || "${case_name}" == */"${item}" ||
          "${case_base}" == "${item}" ]]; then
      return 0
    fi
  done
  return 1
}

build_case_vpto_flags() {
  local case_name="$1"
  local pto_name="$2"
  local base_flags="$3"
  if case_uses_generic_mode "${case_name}"; then
    echo "${base_flags} --vpto-march ${GENERIC_AICORE_ARCH} --vpto-cce-aicore-arch ${GENERIC_AICORE_ARCH}"
    return
  fi
  if [[ "${pto_name}" == "cube.pto" ]]; then
    echo "${base_flags} --vpto-march ${CUBE_AICORE_ARCH} --vpto-cce-aicore-arch ${CUBE_AICORE_ARCH}"
    return
  fi
  if [[ ! -f "${CASES_ROOT}/${case_name}/cube.pto" ]] && case_uses_cube_mode "${case_name}"; then
    echo "${base_flags} --vpto-march ${CUBE_AICORE_ARCH} --vpto-cce-aicore-arch ${CUBE_AICORE_ARCH}"
    return
  fi
  echo "${base_flags}"
}

source_aicore_arch() {
  local case_name="$1"
  local pto_name="$2"
  if case_uses_generic_mode "${case_name}"; then
    echo "${GENERIC_AICORE_ARCH}"
    return
  fi
  if [[ "${pto_name}" == "cube.pto" ]]; then
    echo "${CUBE_AICORE_ARCH}"
    return
  fi
  if [[ ! -f "${CASES_ROOT}/${case_name}/cube.pto" ]] && case_uses_cube_mode "${case_name}"; then
    echo "${CUBE_AICORE_ARCH}"
    return
  fi
  echo "${AICORE_ARCH}"
}

build_launch_object() {
  local case_dir="$1"
  local out_obj="$2"
  local case_arch="$3"

  "${BISHENG_BIN}" \
    -c -fPIC -xcce -fenable-matrix --cce-aicore-enable-tl \
    -fPIC -Xhost-start -Xhost-end \
    -mllvm -cce-aicore-stack-size=0x8000 \
    -mllvm -cce-aicore-function-stack-size=0x8000 \
    -mllvm -cce-aicore-record-overflow=true \
    -mllvm -cce-aicore-addr-transform \
    -mllvm -cce-aicore-dcci-insert-for-scalar=false \
    --cce-aicore-arch="${case_arch}" \
    -DREGISTER_BASE \
    -std=c++17 \
    -Wno-macro-redefined -Wno-ignored-attributes \
    -I "${ASCEND_HOME_PATH}/include" \
    -I "${ASCEND_HOME_PATH}/pkg_inc" \
    -I "${ASCEND_HOME_PATH}/pkg_inc/profiling" \
    -I "${ASCEND_HOME_PATH}/pkg_inc/runtime/runtime" \
    "${case_dir}/launch.cpp" \
    -o "${out_obj}"
}

merge_device_objects() {
  local out_obj="$1"
  shift
  local device_objs=("$@")
  [[ "${#device_objs[@]}" -gt 0 ]] || die "merge_device_objects requires at least one device object"

  if [[ "${#device_objs[@]}" -eq 1 ]]; then
    cp "${device_objs[0]}" "${out_obj}"
    return 0
  fi

  "${LD_LLD_BIN}" \
    -m aicorelinux \
    -Ttext 0 \
    "${device_objs[@]}" \
    -o "${out_obj}" \
    -r \
    --allow-multiple-definition
}

build_host_stub() {
  local case_dir="$1"
  local module_id="$2"
  local stub_obj="$3"
  local case_arch="$4"
  local device_obj="$5"
  [[ -f "${device_obj}" ]] || die "build_host_stub requires a valid device object: ${device_obj}"
  local host_target_args=(
    -triple "${HOST_TRIPLE}"
    -target-cpu "${HOST_TARGET_CPU}"
  )
  if [[ -n "${HOST_TARGET_ABI}" ]]; then
    host_target_args+=(-target-abi "${HOST_TARGET_ABI}")
  fi
  if [[ ${#HOST_FEATURE_FLAGS[@]} -gt 0 ]]; then
    host_target_args+=("${HOST_FEATURE_FLAGS[@]}")
  fi

  "${BISHENG_CC1_BIN}" -cc1 \
    "${host_target_args[@]}" \
    -fcce-aicpu-legacy-launch \
    -fcce-is-host \
    -cce-enable-mix \
    -mllvm -enable-mix=true \
    -cce-launch-with-flagv2-impl \
    -fcce-aicore-arch "${case_arch}" \
    -fcce-fatobj-compile \
    -emit-obj \
    --mrelax-relocations \
    -disable-free \
    -clear-ast-before-backend \
    -disable-llvm-verifier \
    -discard-value-names \
    -main-file-name "stub.cpp" \
    -mrelocation-model pic \
    -pic-level 2 \
    -fhalf-no-semantic-interposition \
    -mframe-pointer=none \
    -fmath-errno \
    -ffp-contract=on \
    -fno-rounding-math \
    -mconstructor-aliases \
    -funwind-tables=2 \
    -fallow-half-arguments-and-returns \
    -mllvm -treat-scalable-fixed-error-as-warning \
    -fcoverage-compilation-dir="${ROOT_DIR}" \
    -resource-dir "${CLANG_RESOURCE_DIR}" \
    -include __clang_cce_runtime_wrapper.h \
    -I "${ASCEND_HOME_PATH}/include" \
    -I "${ASCEND_HOME_PATH}/pkg_inc" \
    -I "${ASCEND_HOME_PATH}/pkg_inc/profiling" \
    -I "${ASCEND_HOME_PATH}/pkg_inc/runtime/runtime" \
    -D _FORTIFY_SOURCE=2 \
    -D REGISTER_BASE \
    "${CC1_INCLUDE_FLAGS[@]}" \
    -O2 \
    -Wno-macro-redefined \
    -Wno-ignored-attributes \
    -std=c++17 \
    -fdeprecated-macro \
    -fdebug-compilation-dir="${ROOT_DIR}" \
    -ferror-limit 19 \
    -stack-protector 2 \
    -fno-signed-char \
    -fgnuc-version=4.2.1 \
    -fcxx-exceptions \
    -fexceptions \
    -vectorize-loops \
    -vectorize-slp \
    -mllvm -cce-aicore-stack-size=0x8000 \
    -mllvm -cce-aicore-function-stack-size=0x8000 \
    -mllvm -cce-aicore-record-overflow=true \
    -mllvm -cce-aicore-addr-transform \
    -mllvm -cce-aicore-dcci-insert-for-scalar=false \
    -fcce-include-aibinary "${device_obj}" \
    -fcce-device-module-id "${module_id}" \
    -faddrsig \
    -D__GCC_HAVE_DWARF2_CFI_ASM=1 \
    -o "${stub_obj}" \
    -x cce "${case_dir}/stub.cpp"
}

link_kernel_so() {
  local case_name="$1"
  local host_stub_obj="$2"
  local launch_obj="$3"
  local repack_obj="$4"
  local repack_so="$5"
  local module_id="$6"
  local case_arch="$7"
  local extra_lib_dirs=()
  local extra_link_libs=()

  "${CCE_LD_BIN}" \
    "${LD_LLD_BIN}" \
    -x \
    -cce-lite-bin-module-id "${module_id}" \
    -cce-aicore-arch="${case_arch}" \
    -r \
    -o "${repack_obj}" \
    -cce-stub-dir "${CCE_STUB_DIR}" \
    -cce-install-dir "$(dirname "${BISHENG_CC1_BIN}")" \
    -cce-inputs-number 1 \
    "${host_stub_obj}"

  if [[ "${DEVICE}" == "SIM" ]]; then
    [[ -n "${SIM_LIB_DIR}" && -d "${SIM_LIB_DIR}" ]] ||
      die "SIM_LIB_DIR is not set or invalid for DEVICE=SIM: ${SIM_LIB_DIR}"
    extra_lib_dirs+=(-L "${SIM_LIB_DIR}" -Wl,-rpath,"${SIM_LIB_DIR}")
    extra_link_libs+=(-Wl,--no-as-needed -lruntime_camodel)
  else
    extra_link_libs+=(-Wl,--no-as-needed -lruntime)
  fi

  "${BISHENG_BIN}" \
    -fPIC -s -Wl,-z,relro -Wl,-z,now --cce-fatobj-link \
    -shared -Wl,-soname,"lib${case_name}_kernel.so" \
    -L "${ASCEND_HOME_PATH}/lib64" \
    "${extra_lib_dirs[@]}" \
    -Wl,-rpath,"${ASCEND_HOME_PATH}/lib64" \
    -o "${repack_so}" \
    "${repack_obj}" \
    "${launch_obj}" \
    "${extra_link_libs[@]}"
}

build_host_executable() {
  local case_token="$1"
  local case_dir="$2"
  local out_dir="$3"
  local extra_ldflags=()
  local extra_lib_dirs=()
  if [[ "${DEVICE}" == "SIM" ]]; then
    [[ -n "${SIM_LIB_DIR}" && -d "${SIM_LIB_DIR}" ]] ||
      die "SIM_LIB_DIR is not set or invalid for DEVICE=SIM: ${SIM_LIB_DIR}"
    extra_lib_dirs+=(-L "${SIM_LIB_DIR}" -Wl,-rpath,"${SIM_LIB_DIR}")
    extra_ldflags+=(-Wl,--allow-shlib-undefined -lruntime_camodel)
  else
    extra_ldflags+=(-Wl,--allow-shlib-undefined -lruntime)
  fi

  "${BISHENG_BIN}" \
    -xc++ -include stdint.h -include stddef.h -std=c++17 \
    "${case_dir}/main.cpp" \
    -I "${case_dir}" \
    -I "${NPU_VALIDATION_COMMON_DIR}" \
    -I "${ASCEND_HOME_PATH}/include" \
    -L "${out_dir}" \
    -L "${ASCEND_HOME_PATH}/lib64" \
    "${extra_lib_dirs[@]}" \
    -Wl,-rpath,"${out_dir}" \
    -Wl,-rpath,"${ASCEND_HOME_PATH}/lib64" \
    -o "${out_dir}/${case_token}" \
    -l"${case_token}_kernel" \
    "${extra_ldflags[@]}" \
    -lstdc++ -lascendcl -lm -ltiling_api -lplatform -lc_sec -ldl -lnnopbase
}

build_one_impl() {
  local case_name="$1"
  local case_dir="${CASES_ROOT}/${case_name}"
  local case_token
  case_token="$(printf '%s' "${case_name}" | sed 's#[/[:space:]]#_#g')"
  local out_dir="${WORK_SPACE}/${case_token}"
  local case_module_id
  case_module_id="$(printf '%s' "${MODULE_ID}-${case_name}" | md5sum | cut -c1-16)"
  local launch_obj="${out_dir}/launch.o"
  local merged_device_obj="${out_dir}/kernel_device_merged.o"
  local host_stub_obj="${out_dir}/kernel_host_from_llvm.o"
  local repack_obj="${out_dir}/${case_token}_stub.cpp.o"
  local repack_so="${out_dir}/lib${case_token}_kernel.so"
  local host_case_arch="${AICORE_ARCH}"
  if [[ -f "${case_dir}/cube.pto" && -f "${case_dir}/kernel.pto" ]]; then
    host_case_arch="${GENERIC_AICORE_ARCH}"
  elif [[ -f "${case_dir}/cube.pto" ]] || case_uses_cube_mode "${case_name}"; then
    host_case_arch="${CUBE_AICORE_ARCH}"
  fi

  [[ -f "${case_dir}/stub.cpp" ]] || die "missing stub.cpp for ${case_name}"
  [[ -f "${case_dir}/main.cpp" ]] || die "missing main.cpp for ${case_name}"
  [[ -f "${case_dir}/launch.cpp" ]] || die "missing launch.cpp for ${case_name}"
  [[ -f "${case_dir}/golden.py" ]] || die "missing golden.py for ${case_name}"
  [[ -f "${case_dir}/compare.py" ]] || die "missing compare.py for ${case_name}"
  [[ -f "${case_dir}/kernel.pto" || -f "${case_dir}/cube.pto" ]] ||
    die "missing kernel.pto and cube.pto for ${case_name}"

  local -a pto_sources=()
  [[ -f "${case_dir}/cube.pto" ]] && pto_sources+=("cube.pto")
  [[ -f "${case_dir}/kernel.pto" ]] && pto_sources+=("kernel.pto")

  local -a device_objs=()
  local pto_name
  for pto_name in "${pto_sources[@]}"; do
    local source_stem="${pto_name%.pto}"
    local llvm_ir="${out_dir}/${source_stem}.ll"
    local device_obj="${out_dir}/${source_stem}.o"
    local source_arch
    source_arch="$(source_aicore_arch "${case_name}" "${pto_name}")"
    local case_vpto_flags
    case_vpto_flags="$(build_case_vpto_flags "${case_name}" "${pto_name}" "${VPTO_FLAGS}")"

    log "[$case_name] compile ${pto_name} as $( [[ "${source_arch}" == "${CUBE_AICORE_ARCH}" ]] && echo cube || echo vec ) (aicore_arch=${source_arch})"
  log "[$case_name] step 1/7: lower ${pto_name} to LLVM IR"
    "${PTOAS_BIN}" ${PTOAS_FLAGS} ${case_vpto_flags} \
      "${case_dir}/${pto_name}" -o "${llvm_ir}"

    log "[$case_name] step 2/7: compile ${pto_name} LLVM IR to device object"
    "${BISHENG_BIN}" \
      --target=hiipu64-hisilicon-cce \
      -march="${source_arch}" \
      --cce-aicore-arch="${source_arch}" \
      --cce-aicore-only \
      -O2 \
      -c -x ir "${llvm_ir}" \
      -o "${device_obj}"
    device_objs+=("${device_obj}")
  done

  log "[$case_name] step 3/7: merge device objects"
  merge_device_objects "${merged_device_obj}" "${device_objs[@]}"

  log "[$case_name] step 4/7: build launch object and host fatobj stub"
  build_launch_object "${case_dir}" "${launch_obj}" "${host_case_arch}"
  build_host_stub "${case_dir}" "${case_module_id}" "${host_stub_obj}" "${host_case_arch}" "${merged_device_obj}"

  log "[$case_name] step 5/7: link kernel shared library"
  link_kernel_so "${case_token}" "${host_stub_obj}" "${launch_obj}" "${repack_obj}" "${repack_so}" "${case_module_id}" "${host_case_arch}"

  if [[ "${COMPILE_ONLY}" == "1" ]]; then
    log "[$case_name] compile-only mode: stop after kernel shared library"
    log "[$case_name] output dir: ${out_dir}"
    return 0
  fi

  log "[$case_name] step 6/7: build host executable and golden"
  build_host_executable "${case_token}" "${case_dir}" "${out_dir}"
  (
    cd "${out_dir}"
    python3 "${case_dir}/golden.py"
  )

  log "[$case_name] step 7/7: run NPU validation"
  local remote_run_cmd
  remote_run_cmd=$(cat <<EOF
cd "${out_dir}" && \
export ASCEND_HOME_PATH="${ASCEND_HOME_PATH}" && \
if [ -f "\$ASCEND_HOME_PATH/set_env.sh" ]; then source "\$ASCEND_HOME_PATH/set_env.sh" >/dev/null 2>&1; fi && \
LD_LIBRARY_PATH="${out_dir}:${SIM_LIB_DIR}:\$ASCEND_HOME_PATH/lib64:\${LD_LIBRARY_PATH:-}" "./${case_token}"
EOF
)
  run_remote "${remote_run_cmd}"

  local remote_ldd_cmd
  remote_ldd_cmd=$(cat <<EOF
cd "${out_dir}" && \
export ASCEND_HOME_PATH="${ASCEND_HOME_PATH}" && \
if [ -f "\$ASCEND_HOME_PATH/set_env.sh" ]; then source "\$ASCEND_HOME_PATH/set_env.sh" >/dev/null 2>&1; fi && \
LD_LIBRARY_PATH="${out_dir}:${SIM_LIB_DIR}:\$ASCEND_HOME_PATH/lib64:\${LD_LIBRARY_PATH:-}" ldd "./${case_token}" | grep "lib${case_token}_kernel.so"
EOF
)
  local ldd_output
  ldd_output="$(run_remote "${remote_ldd_cmd}")"
  [[ "${ldd_output}" == *"${repack_so}"* || "${ldd_output}" == *"lib${case_token}_kernel.so"* ]] || \
    die "${case_name} did not load expected kernel so: ${ldd_output}"

  (
    cd "${out_dir}"
    COMPARE_STRICT=1 python3 "${case_dir}/compare.py"
  )

  log "[$case_name] compare passed"
  log "[$case_name] output dir: ${out_dir}"
}

build_one() {
  local case_name="$1"
  local case_token
  case_token="$(printf '%s' "${case_name}" | sed 's#[/[:space:]]#_#g')"
  local out_dir="${WORK_SPACE}/${case_token}"
  local case_log="${out_dir}/validation.log"

  rm -rf "${out_dir}"
  mkdir -p "${out_dir}"

  (
    build_one_impl "${case_name}"
  ) 2>&1 | tee "${case_log}"
}

log "=== VPTO Host Validation ==="
log "WORK_SPACE=${WORK_SPACE}"
log "ASCEND_HOME_PATH=${ASCEND_HOME_PATH}"
log "PTOAS_BIN=${PTOAS_BIN}"
log "PTOAS_FLAGS=${PTOAS_FLAGS}"
log "VPTO_FLAGS=${VPTO_FLAGS}"
log "AICORE_ARCH(default)=${AICORE_ARCH}"
log "CUBE_AICORE_ARCH=${CUBE_AICORE_ARCH}"
log "CUBE_CASES=${CUBE_CASES:-<none>}"
log "COMPILE_ONLY=${COMPILE_ONLY}"
log "CASE_NAME=${CASE_NAME:-<all>}"

for case_name in "${CASES[@]}"; do
  build_one "${case_name}"
done

log "All ${#CASES[@]} VPTO case(s) passed"
